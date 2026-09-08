from __future__ import annotations

"""残局与劣势博弈专属 Deep CFR 强化学习求解器 (Endgame CFR Trainer)。

完全独立 Fork 出来，针对手牌 1~2 张、高膛压、濒死逆风博弈进行定向高密度注水训练：
1. 70% 博弈树轨迹采样自 EndgameScenarioGenerator 极端局面（单挑死斗、濒死逆风、手牌打空压迫、鬼牌反杀）；
2. 30% 保留全局全流程，防止灾难性遗忘；
3. 专项战术奖惩强化（濒死假牌暴毙极刑、残局识破抓诈特奖、绝杀免责奖励）；
4. 继承 League 快照池对抗，杜绝循环博弈。
"""

import copy
import json
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from random import Random

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from .abstractions import (
    ACTION_SPACE_SIZE,
    ActionAbstractor,
    action_to_index,
)
from .buffers import ReservoirBuffer
from .config import DeepCFRConfig
from .endgame_generator import EndgameScenarioGenerator
from .engine import SurvivalGameEngine
from .evaluation import MatchupEvaluator
from .features import FEATURE_DIM, FeatureEncoder
from .networks import (
    AdvantageNetwork,
    StrategyNetwork,
    build_advantage_net,
    build_strategy_net,
)
from .neural_policy import NeuralPolicy
from .opponents import (
    OpponentProfile,
    build_policy_from_profile,
    default_opponent_profiles,
)
from .rewards import TerminalRewardModel
from .self_play import RandomPolicy
from .types import CardKind, ChallengeAction, PlayAction, Policy


def compute_regret_matching_strategy(
    advantages: torch.Tensor,
    abstract_actions: list,
) -> tuple[dict[str, float], bool]:
    """根据优势网络预测值计算合法动作的正遗憾匹配策略。
    
    返回:
        (strategy_dict, fell_back)
        strategy_dict: {aa.label: prob}
        fell_back: 若无正遗憾值，回退为合法动作均匀分布则为 True
    """
    positive_regrets = []
    for aa in abstract_actions:
        idx = action_to_index(aa.label)
        r = max(0.0, float(advantages[idx].item()))
        positive_regrets.append((aa.label, r))

    sum_regrets = sum(r for _, r in positive_regrets)
    if sum_regrets > 1e-8:
        return {label: r / sum_regrets for label, r in positive_regrets}, False

    # 无正遗憾值：严格回退为合法动作上的均匀分布
    n = len(abstract_actions)
    uniform_p = 1.0 / max(1, n)
    return {aa.label: uniform_p for aa in abstract_actions}, True


# ── League 对手池 ─────────────────────────────────────────────────────────
class _EndgameLeaguePool:
    MAX_LEAGUE_SIZE = 25

    def __init__(
        self,
        profiles: tuple[OpponentProfile, ...],
        rng: Random | None = None,
    ) -> None:
        self.profiles = profiles
        self._rng = rng or Random()
        self._league: list[NeuralPolicy] = []
        self._frozen_stars: list[NeuralPolicy] = []
        self._current_policy: NeuralPolicy | None = None

    def add_frozen_star(self, policy: NeuralPolicy) -> None:
        self._frozen_stars.append(policy)

    def add_snapshot(self, policy: NeuralPolicy) -> None:
        if len(self._league) < self.MAX_LEAGUE_SIZE:
            self._league.append(policy)
        else:
            idx = self._rng.randint(1, self.MAX_LEAGUE_SIZE - 2)
            self._league[idx] = policy

    def update_current_policy(self, policy: NeuralPolicy) -> None:
        self._current_policy = policy

    def league_size(self) -> int:
        return len(self._league) + len(self._frozen_stars)

    def sample_opponent(self, seed: int) -> Policy:
        r = self._rng.random()
        # 40% 概率直接对抗冻结的名宿（v13/v14/v15/v16 黄金模型）
        if r < 0.40 and self._frozen_stars:
            return self._rng.choice(self._frozen_stars)
        # 35% 概率对抗动态滚动的 League 快照
        if r < 0.75 and self._league:
            return self._rng.choice(self._league)
        # 15% 概率对抗最新的自我 (Self-Play)
        if r < 0.90 and self._current_policy is not None:
            return self._current_policy
        # 8% 概率对抗固定 profile 策略
        if r < 0.98 and self.profiles:
            profile = self._rng.choice(self.profiles)
            return build_policy_from_profile(profile, seed=seed)
        # 2% 探索
def _swarm_process_worker(
    worker_id: int,
    sample_queue,
    stop_event,
    seed_base: int,
    endgame_ratio: float,
    honest_bot_injection_ratio: float,
    current_iter_val,
    frozen_star_paths: list[str],
    hidden_dim: int,
    num_layers: int,
    cmd_queue=None,
    ack_queue=None,
):
    """独立子进程工作单元：独占独立物理核与独立 GIL，以共享内存零拷贝传递张量块。"""
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import sys
    sys.modules["_wmi"] = None
    torch.set_num_threads(1)
    engine = SurvivalGameEngine()
    encoder = FeatureEncoder()
    abstractor = ActionAbstractor()
    endgame_gen = EndgameScenarioGenerator(engine)
    reward_model = TerminalRewardModel()
    worker_rng = Random(seed_base + worker_id * 10007)

    # 独立预载四大名宿黄金快照策略
    frozen_stars = []
    for sp in frozen_star_paths:
        p = Path(sp)
        if p.is_file():
            try:
                c = torch.load(p, map_location="cpu", weights_only=False)
                snap_net = build_strategy_net(
                    input_dim=FEATURE_DIM,
                    action_space_size=ACTION_SPACE_SIZE,
                    hidden_dim=c.get("hidden_dim", hidden_dim),
                    num_layers=c.get("num_layers", num_layers),
                    dropout=0.0,
                )
                snap_net.load_state_dict(c["strategy_net"])
                snap_net.eval()
                snap_policy = NeuralPolicy(
                    strategy_net=snap_net,
                    encoder=encoder,
                    abstractor=abstractor,
                    seed=worker_id * 1000 + 42,
                    prob_mode="linear_norm",
                )
                frozen_stars.append(snap_policy)
            except Exception:
                pass

    local_adv_net = build_advantage_net(
        input_dim=FEATURE_DIM,
        action_space_size=ACTION_SPACE_SIZE,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=0.0,
    )
    local_adv_net.eval()

    local_strat_net = build_strategy_net(
        input_dim=FEATURE_DIM,
        action_space_size=ACTION_SPACE_SIZE,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=0.0,
    )
    local_strat_net.eval()
    local_policy = NeuralPolicy(
        strategy_net=local_strat_net,
        encoder=encoder,
        abstractor=abstractor,
        seed=worker_id,
        prob_mode="linear_norm",
    )

    current_model_version = [0]

    chunk_adv_f: list = []
    chunk_adv_t: list = []
    chunk_adv_w: list = []
    chunk_strat_f: list = []
    chunk_strat_t: list = []
    chunk_strat_w: list = []
    diag = {
        "normal": 0,
        "endgame": 0,
        "honest_bot": 0,
        "forced": 0,
        "multi": 0,
        "rm_normal": 0,
        "rm_fallback": 0,
    }

    def _traverse(state, traverser_seat, depth, cur_iter, seed_b, game_type, hb_bot, hb_seat):
        if engine.is_terminal(state):
            return reward_model.evaluate(state).for_seat(traverser_seat)
        if depth >= 8:
            p = state.player_by_seat(traverser_seat)
            if not p.alive:
                return reward_model.score_rules.eliminated_or_last
            alive_c = sum(1 for pl in state.players if pl.alive)
            return float(max(-15.0, min(20.0, (4 - alive_c) * 4.0 - p.shots_taken * 3.5 + (5 - len(p.hand)))))

        actor = engine.current_actor(state)
        legal = engine.legal_actions(state)
        if not legal:
            return 0.0

        diag[game_type] = diag.get(game_type, 0) + 1
        if len(legal) == 1:
            diag["forced"] = diag.get("forced", 0) + 1
        else:
            diag["multi"] = diag.get("multi", 0) + 1

        obs = engine.observe(state, actor)
        feat = encoder.encode(obs)
        grouped = abstractor.abstract_legal_actions(legal, obs)
        abs_actions = list(grouped.keys())

        p_actor = state.player_by_seat(actor)
        is_endgame_hand = len(p_actor.hand) <= 2
        is_underdog = p_actor.shots_taken >= p_actor.live_round_index

        if actor == traverser_seat:
            # 1. 优势网络预测并计算正遗憾匹配策略
            with torch.no_grad():
                adv_pred = local_adv_net(feat.unsqueeze(0)).squeeze(0)
            strat_dict, fell_back = compute_regret_matching_strategy(adv_pred, abs_actions)
            if fell_back:
                diag["rm_fallback"] = diag.get("rm_fallback", 0) + 1
            else:
                diag["rm_normal"] = diag.get("rm_normal", 0) + 1

            action_utils = {}
            for aa in abs_actions:
                c = worker_rng.choice(grouped[aa])
                nxt = engine.clone_state(state)
                p_before = state.player_by_seat(traverser_seat)
                shots_before = p_before.shots_taken
                hand_len_before = len(p_before.hand)

                engine.apply_action(nxt, c)
                bonus = 0.0
                if isinstance(c, ChallengeAction):
                    last_ev = nxt.public_history[-1] if nxt.public_history else None
                    if last_ev and last_ev.event_type == "challenge":
                        outcome = last_ev.detail.get("outcome")
                        if outcome == "lie":
                            bonus += 8.0 if is_endgame_hand else 5.0
                        elif outcome == "honest":
                            bonus -= 8.0 if is_underdog else 5.0
                        elif outcome == "ghost":
                            bonus -= 4.0

                sub_u = _traverse(nxt, traverser_seat, depth + 1, cur_iter, seed_b, game_type, hb_bot, hb_seat)
                if isinstance(c, PlayAction):
                    is_bluff = False
                    has_ghost = False
                    for cid in c.card_ids:
                        card = next((cd for cd in p_before.hand if cd.card_id == cid), None)
                        if card:
                            if card.kind == CardKind.GHOST:
                                has_ghost = True
                            elif card.kind != CardKind.WILD and card.printed_rank != c.claim_rank:
                                is_bluff = True
                    p_end = nxt.player_by_seat(traverser_seat)
                    if p_end.shots_taken > shots_before:
                        if is_bluff:
                            bonus -= 7.0
                            if not p_end.alive:
                                bonus -= 15.0
                    else:
                        if not is_bluff:
                            bonus += 4.0 if (hand_len_before <= 2 and len(p_end.hand) == 0) else 2.0
                        if has_ghost:
                            bonus += 10.0

                action_utils[aa.label] = max(-25.0, min(25.0, sub_u + bonus))

            # 2. 节点期望收益严格按策略加权: v(sigma) = sum_a sigma(a) * u(a)
            ev = sum(strat_dict[aa.label] * action_utils[aa.label] for aa in abs_actions)

            # 3. 优势目标: adv_tgt[a] = u(a) - ev
            adv_tgt = torch.zeros(ACTION_SPACE_SIZE)
            for aa in abs_actions:
                idx = action_to_index(aa.label)
                adv_tgt[idx] = action_utils[aa.label] - ev

            # 4. 策略网络训练目标: strat_tgt[a] = sigma(a)
            strat_tgt = torch.zeros(ACTION_SPACE_SIZE)
            for aa in abs_actions:
                strat_tgt[action_to_index(aa.label)] = strat_dict[aa.label]

            chunk_adv_f.append(feat)
            chunk_adv_t.append(adv_tgt)
            chunk_adv_w.append(float(cur_iter))

            chunk_strat_f.append(feat)
            chunk_strat_t.append(strat_tgt)
            chunk_strat_w.append(float(cur_iter))
            return ev
        else:
            # actor != traverser_seat: 对手节点，对手根据自身独立策略行动，绝不收集为当前学习策略样本
            c = None
            if hb_bot is not None and actor == hb_seat:
                try:
                    c = hb_bot.choose_action(obs, legal)
                except Exception:
                    c = None
            elif frozen_stars and worker_rng.random() < 0.5:
                star = worker_rng.choice(frozen_stars)
                try:
                    c = star.choose_action_from_features(obs, legal, feat, grouped)
                except Exception:
                    c = None

            if c is None:
                c = local_policy.choose_action_from_features(obs, legal, feat, grouped)

            nxt = engine.clone_state(state)
            engine.apply_action(nxt, c)
            return _traverse(nxt, traverser_seat, depth + 1, cur_iter, seed_b, game_type, hb_bot, hb_seat)

    t_count = 0
    from .opponents import HeuristicProfilePolicy, OpponentProfile

    while not stop_event.is_set():
        # 1. 检查主进程指令队列，实现带版本的迭代边界权重同步
        if cmd_queue is not None:
            while not cmd_queue.empty():
                try:
                    cmd_obj = cmd_queue.get_nowait()
                    cmd_type = cmd_obj.get("cmd")
                    if cmd_type == "SYNC":
                        ver = cmd_obj["version"]
                        local_adv_net.load_state_dict(cmd_obj["adv_sd"])
                        local_strat_net.load_state_dict(cmd_obj["strat_sd"])
                        local_adv_net.eval()
                        local_strat_net.eval()
                        current_model_version[0] = ver
                        if ack_queue is not None:
                            ack_queue.put((worker_id, ver))
                    elif cmd_type == "STOP":
                        return
                except (queue.Empty, EOFError):
                    break

        cur_iter = current_iter_val[0]
        game_seed = worker_rng.randint(0, 2**31 - 1)
        seat = (t_count % 4) + 1

        if worker_rng.random() < endgame_ratio:
            init_state = endgame_gen.sample_endgame_state(
                seed=game_seed, scenario="random_mix", hero_seat=seat
            )
            game_type = "endgame"
            hb_bot = None
            hb_seat = None
        else:
            init_state = engine.new_game(seed=game_seed)
            if (
                honest_bot_injection_ratio > 0.0
                and worker_rng.random() < honest_bot_injection_ratio
            ):
                candidate_seats = [s for s in range(1, 5) if s != seat]
                hb_seat = worker_rng.choice(candidate_seats)
                hb_prof = OpponentProfile(
                    name="HonestBot_Injected",
                    bluff_tendency=worker_rng.uniform(0.0, 0.02),
                    challenge_tendency=worker_rng.uniform(0.65, 0.85),
                    single_bias=0.6,
                    pair_bias=0.2,
                    triple_bias=-0.6,
                    temperature=worker_rng.uniform(0.08, 0.15),
                    hazard_sensitivity=worker_rng.uniform(1.2, 1.8),
                )
                hb_bot = HeuristicProfilePolicy(hb_prof, seed=game_seed + 100)
                game_type = "honest_bot"
            else:
                game_type = "normal"

        try:
            _traverse(init_state, seat, 0, cur_iter, game_seed, game_type, hb_bot, hb_seat)
            t_count += 1
            if len(chunk_adv_f) >= 128 or t_count % 16 == 0:
                if chunk_adv_f or chunk_strat_f:
                    adv_tup = None
                    if chunk_adv_f:
                        adv_tup = (
                            torch.stack(chunk_adv_f).numpy(),
                            torch.stack(chunk_adv_t).numpy(),
                            np.array(chunk_adv_w, dtype=np.float32),
                        )
                    strat_tup = None
                    if chunk_strat_f:
                        strat_tup = (
                            torch.stack(chunk_strat_f).numpy(),
                            torch.stack(chunk_strat_t).numpy(),
                            np.array(chunk_strat_w, dtype=np.float32),
                        )
                    meta = {
                        "model_version": current_model_version[0],
                        "iteration": cur_iter,
                        "diag": diag.copy(),
                    }
                    sample_queue.put((adv_tup, strat_tup, meta), timeout=0.5)
                    chunk_adv_f.clear()
                    chunk_adv_t.clear()
                    chunk_adv_w.clear()
                    chunk_strat_f.clear()
                    chunk_strat_t.clear()
                    chunk_strat_w.clear()
        except Exception:
            pass


@dataclass(frozen=True)
class EndgameCFRTrainingResult:
    strategy_net: StrategyNetwork
    advantage_net: AdvantageNetwork
    output_dir: Path
    iterations_completed: int
    total_traversals: int
    final_advantage_loss: float
    final_strategy_loss: float
    eval_results: dict


class EndgameDeepCFRTrainer:
    """残局与劣势博弈专项强化学习训练系统。"""

    def __init__(
        self,
        config: DeepCFRConfig | None = None,
        engine: SurvivalGameEngine | None = None,
        reward_model: TerminalRewardModel | None = None,
        encoder: FeatureEncoder | None = None,
        abstractor: ActionAbstractor | None = None,
        endgame_ratio: float = 0.30,
        honest_bot_injection_ratio: float = 0.0,
        freeze_league: bool = True,
    ) -> None:
        self.config = config or DeepCFRConfig()
        self.engine = engine or SurvivalGameEngine()
        self.reward_model = reward_model or TerminalRewardModel()
        self.encoder = encoder or FeatureEncoder()
        self.abstractor = abstractor or ActionAbstractor()
        self.endgame_gen = EndgameScenarioGenerator(self.engine)
        self.endgame_ratio = endgame_ratio
        self.honest_bot_injection_ratio = honest_bot_injection_ratio
        self.freeze_league = freeze_league
        self._rng = Random(self.config.seed)

        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)

        hidden_dim = self.config.hidden_dim
        num_layers = self.config.num_layers
        if self.config.resume_from:
            resume_path = Path(self.config.resume_from)
            if resume_path.is_file():
                try:
                    c_probe = torch.load(resume_path, map_location="cpu", weights_only=False)
                    if "hidden_dim" in c_probe:
                        hidden_dim = c_probe["hidden_dim"]
                    if "num_layers" in c_probe:
                        num_layers = c_probe["num_layers"]
                except Exception:
                    pass

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.advantage_net: AdvantageNetwork = build_advantage_net(
            input_dim=FEATURE_DIM,
            action_space_size=ACTION_SPACE_SIZE,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=self.config.dropout,
        ).to(self.device)

        self.strategy_net: StrategyNetwork = build_strategy_net(
            input_dim=FEATURE_DIM,
            action_space_size=ACTION_SPACE_SIZE,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=self.config.dropout,
        ).to(self.device)

        self._eval_advantage_net: AdvantageNetwork = build_advantage_net(
            input_dim=FEATURE_DIM,
            action_space_size=ACTION_SPACE_SIZE,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=0.0,
        ).to(torch.device("cpu"))
        self._eval_advantage_net.load_state_dict(self.advantage_net.state_dict())
        self._eval_advantage_net.eval()

        self._eval_strategy_net: StrategyNetwork = build_strategy_net(
            input_dim=FEATURE_DIM,
            action_space_size=ACTION_SPACE_SIZE,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=0.0,
        ).to(torch.device("cpu"))
        self._eval_strategy_net.load_state_dict(self.strategy_net.state_dict())
        self._eval_strategy_net.eval()

        capacity = self.config.advantage_buffer_size
        self.advantage_buffer = ReservoirBuffer(
            capacity=capacity,
            feature_dim=FEATURE_DIM,
            action_space_size=ACTION_SPACE_SIZE,
            device=self.device,
        )
        self.strategy_buffer = ReservoirBuffer(
            capacity=capacity,
            feature_dim=FEATURE_DIM,
            action_space_size=ACTION_SPACE_SIZE,
            device=self.device,
        )

        self.advantage_optimizer = optim.Adam(
            self.advantage_net.parameters(), lr=self.config.learning_rate
        )
        self.strategy_optimizer = optim.Adam(
            self.strategy_net.parameters(), lr=self.config.learning_rate
        )

        self._opponent_pool = _EndgameLeaguePool(
            profiles=default_opponent_profiles(),
            rng=Random(self.config.seed),
        )

        self.best_win_rate = -1.0
        self.start_iteration = 1

        if self.config.resume_from:
            resume_path = Path(self.config.resume_from)
            if not resume_path.is_file():
                raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
            ckpt = torch.load(resume_path, map_location=self.device, weights_only=False)
            self.advantage_net.load_state_dict(ckpt["advantage_net"])
            self.strategy_net.load_state_dict(ckpt["strategy_net"])
            if "advantage_optimizer" in ckpt:
                self.advantage_optimizer.load_state_dict(ckpt["advantage_optimizer"])
            if "strategy_optimizer" in ckpt:
                self.strategy_optimizer.load_state_dict(ckpt["strategy_optimizer"])

            self._eval_advantage_net.load_state_dict(
                {k: v.cpu() for k, v in self.advantage_net.state_dict().items()}
            )
            self._eval_advantage_net.eval()
            self._eval_strategy_net.load_state_dict(
                {k: v.cpu() for k, v in self.strategy_net.state_dict().items()}
            )
            self._eval_strategy_net.eval()

            cpu_neural_policy = NeuralPolicy(
                strategy_net=self._eval_strategy_net,
                encoder=self.encoder,
                abstractor=self.abstractor,
                seed=self.config.seed or 42,
            )
            self._opponent_pool.update_current_policy(cpu_neural_policy)
            self._preload_league_snapshots(resume_path)

            self.start_iteration = ckpt.get("iteration", 0) + 1
            print(
                f"[Endgame CFR] 成功从基底 {resume_path} 热启动，"
                f"起始轮次: {self.start_iteration} / 目标轮次: {self.config.cfr_iterations}"
            )
            print(
                f"[Endgame CFR] 残局专属定向注水比例: {int(self.endgame_ratio * 100)}% 残局 + {int((1 - self.endgame_ratio) * 100)}% 全局"
            )
        else:
            cpu_neural_policy = NeuralPolicy(
                strategy_net=self._eval_strategy_net,
                encoder=self.encoder,
                abstractor=self.abstractor,
                seed=self.config.seed or 42,
            )
            self._opponent_pool.update_current_policy(cpu_neural_policy)
            self._preload_league_snapshots(None)
            print(
                f"[Endgame CFR] 全新冷启动（大网络高容量架构），目标轮次: {self.config.cfr_iterations}"
            )
            print(
                f"[Endgame CFR] 残局专属定向注水比例: {int(self.endgame_ratio * 100)}% 残局 + {int((1 - self.endgame_ratio) * 100)}% 全局"
            )

    def _preload_league_snapshots(self, resume_path: Path | None = None) -> None:
        base_runs = Path("runs_deep_cfr")
        # 1. 优先锁定注入四大历史黄金快照 (v13, v14, v15, v16)
        star_configs = [
            ("v16_endgame", base_runs / "deep_cfr_v16_endgame" / "policy_best.pt"),
            ("v15_league", base_runs / "deep_cfr_v15_league" / "policy_best.pt"),
            ("v14_selfplay", base_runs / "deep_cfr_v14_full_selfplay" / "policy_best.pt"),
            ("v13_extern", base_runs / "deep_cfr_v13_extern_sampling" / "policy_best.pt"),
        ]
        star_count = 0
        for name, star_path in star_configs:
            if star_path.is_file():
                try:
                    c = torch.load(star_path, map_location="cpu", weights_only=False)
                    snap_net = build_strategy_net(
                        input_dim=FEATURE_DIM,
                        action_space_size=ACTION_SPACE_SIZE,
                        hidden_dim=c.get("hidden_dim", 512),
                        num_layers=c.get("num_layers", 4),
                        dropout=0.0,
                    )
                    snap_net.load_state_dict(c["strategy_net"])
                    snap_net.eval()
                    snap_policy = NeuralPolicy(
                        strategy_net=snap_net,
                        encoder=self.encoder,
                        abstractor=self.abstractor,
                        seed=c.get("iteration", 0) + 1234,
                    )
                    self._opponent_pool.add_frozen_star(snap_policy)
                    star_count += 1
                except Exception as e:
                    print(f"[Endgame CFR] 加载明星对手 {name} 失败: {e}")

        # 2. 仅从基底目录加载代表性历史 checkpoint（均匀抽样最多 8 个）
        loaded = 0
        if resume_path is not None and resume_path.is_file():
            candidate_files = sorted(resume_path.parent.glob("checkpoint_iter_*.pt"))
            if len(candidate_files) > 8:
                step = max(1, len(candidate_files) // 8)
                selected_files = candidate_files[::step][:8]
            else:
                selected_files = candidate_files

            for ckpt_file in selected_files:
                try:
                    c = torch.load(ckpt_file, map_location="cpu", weights_only=False)
                    snap_net = build_strategy_net(
                        input_dim=FEATURE_DIM,
                        action_space_size=ACTION_SPACE_SIZE,
                        hidden_dim=c.get("hidden_dim", 512),
                        num_layers=c.get("num_layers", 4),
                        dropout=0.0,
                    )
                    snap_net.load_state_dict(c["strategy_net"])
                    snap_net.eval()
                    snap_policy = NeuralPolicy(
                        strategy_net=snap_net,
                        encoder=self.encoder,
                        abstractor=self.abstractor,
                        seed=c.get("iteration", 0),
                        prob_mode="linear_norm",
                    )
                    self._opponent_pool.add_snapshot(snap_policy)
                    loaded += 1
                except Exception:
                    pass
        print(f"[Endgame CFR] 预载历史名宿黄金对手: {star_count} 位 (v13~v16) | 历史快照: {loaded} 个 (League 规模: {self._opponent_pool.league_size()})")

    def train(self, output_dir: str | Path) -> EndgameCFRTrainingResult:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        total_traversals = 0
        last_adv_loss = 0.0
        last_strat_loss = 0.0
        eval_results: dict = {}
        log_entries: list[dict] = []

        log_file = output_path / "training_log.json"
        if log_file.is_file():
            try:
                log_entries = json.loads(log_file.read_text(encoding="utf-8"))
                for entry in log_entries:
                    ev = entry.get("eval")
                    if isinstance(ev, dict) and "win_rate" in ev:
                        self.best_win_rate = max(self.best_win_rate, ev["win_rate"])
                print(f"[Endgame CFR] 已承接历史日志 {len(log_entries)} 条，最高胜率锁定: {self.best_win_rate:.3f}")
            except Exception as e:
                print(f"[Endgame CFR] 读取历史日志警告: {e}")

        torch.set_num_threads(1)
        num_workers = max(1, getattr(self.config, "num_workers", 20))
        use_mp = getattr(self.config, "use_multiprocessing", True)

        print(f"[Endgame CFR] 设备: {self.device} | 特征: {FEATURE_DIM} | 动作: {ACTION_SPACE_SIZE}", flush=True)
        print(f"[Endgame CFR] 启动 {num_workers} 个多进程采样 Worker (Swarm Pool) | 目标: {self.config.cfr_iterations} 轮", flush=True)

        train_start = time.time()
        star_paths = [
            str(Path("runs_deep_cfr/deep_cfr_v16_endgame/policy_best.pt").resolve()),
            str(Path("runs_deep_cfr/deep_cfr_v15_league/policy_best.pt").resolve()),
            str(Path("runs_deep_cfr/deep_cfr_v14_full_selfplay/policy_best.pt").resolve()),
            str(Path("runs_deep_cfr/deep_cfr_v13_extern_sampling/policy_best.pt").resolve()),
        ]

        if use_mp:
            import torch.multiprocessing as mp
            try:
                mp.set_start_method("spawn", force=False)
            except RuntimeError:
                pass
            sample_queue = mp.Queue(maxsize=64)
            stop_event = mp.Event()
            current_iteration = mp.Array("i", [self.start_iteration])
            cmd_queues = [mp.Queue(maxsize=4) for _ in range(num_workers)]
            ack_queue = mp.Queue(maxsize=num_workers * 4)

            workers = []
            for i in range(num_workers):
                p = mp.Process(
                    target=_swarm_process_worker,
                    args=(
                        i,
                        sample_queue,
                        stop_event,
                        self.config.seed or 42,
                        self.endgame_ratio,
                        self.honest_bot_injection_ratio,
                        current_iteration,
                        star_paths,
                        self.hidden_dim,
                        self.num_layers,
                        cmd_queues[i],
                        ack_queue,
                    ),
                    daemon=True,
                )
                p.start()
                time.sleep(0.05)
                workers.append(p)

            # 初始模型快照同步 (v = self.start_iteration)
            init_adv_sd = {k: v.cpu() for k, v in self.advantage_net.state_dict().items()}
            init_strat_sd = {k: v.cpu() for k, v in self.strategy_net.state_dict().items()}
            for q in cmd_queues:
                q.put({
                    "cmd": "SYNC",
                    "version": self.start_iteration,
                    "adv_sd": init_adv_sd,
                    "strat_sd": init_strat_sd,
                })
            acks = set()
            ack_deadline = time.time() + 15.0
            while len(acks) < num_workers and time.time() < ack_deadline:
                try:
                    w_id, ver = ack_queue.get(timeout=0.2)
                    if ver == self.start_iteration:
                        acks.add(w_id)
                except (queue.Empty, EOFError):
                    pass
            print(f"[Endgame CFR] 初始模型版本 v{self.start_iteration} 成功同步至 {len(acks)}/{num_workers} 个 Worker", flush=True)

        else:
            sample_queue = queue.Queue(maxsize=64)
            stop_event = threading.Event()
            current_iteration = [self.start_iteration]
            cmd_queues = []
            ack_queue = None
            workers = []
            for i in range(num_workers):
                t = threading.Thread(
                    target=_swarm_process_worker,
                    args=(
                        i,
                        sample_queue,
                        stop_event,
                        self.config.seed or 42,
                        self.endgame_ratio,
                        self.honest_bot_injection_ratio,
                        current_iteration,
                        star_paths,
                        self.hidden_dim,
                        self.num_layers,
                    ),
                    daemon=True,
                )
                t.start()
                workers.append(t)

        try:
            if len(self.advantage_buffer) < 512:
                print("[Endgame CFR] 正在进行残局初始点火采集...", flush=True)
                while len(self.advantage_buffer) < 512 and not stop_event.is_set():
                    try:
                        item = sample_queue.get(timeout=0.1)
                        adv_tup, strat_tup = item[0], item[1]
                        if adv_tup is not None:
                            if isinstance(adv_tup[0], np.ndarray):
                                adv_tup = (
                                    torch.from_numpy(adv_tup[0]),
                                    torch.from_numpy(adv_tup[1]),
                                    torch.from_numpy(adv_tup[2]),
                                )
                            self.advantage_buffer.extend(adv_tup)
                        if strat_tup is not None:
                            if isinstance(strat_tup[0], np.ndarray):
                                strat_tup = (
                                    torch.from_numpy(strat_tup[0]),
                                    torch.from_numpy(strat_tup[1]),
                                    torch.from_numpy(strat_tup[2]),
                                )
                            self.strategy_buffer.extend(strat_tup)
                        total_traversals += 1
                    except queue.Empty:
                        pass
                print(f"[Endgame CFR] 点火完成！初始显存样本量: {len(self.advantage_buffer)}，满血开跑！\n", flush=True)

            for iteration in range(self.start_iteration, self.config.cfr_iterations + 1):
                iter_start = time.time()
                current_iteration[0] = iteration

                # ── 迭代边界带版本权重同步 (Barrier Sync) ──
                if iteration > self.start_iteration and use_mp:
                    curr_adv_sd = {k: v.cpu() for k, v in self.advantage_net.state_dict().items()}
                    curr_strat_sd = {k: v.cpu() for k, v in self.strategy_net.state_dict().items()}
                    for q in cmd_queues:
                        q.put({
                            "cmd": "SYNC",
                            "version": iteration,
                            "adv_sd": curr_adv_sd,
                            "strat_sd": curr_strat_sd,
                        })
                    acks = set()
                    ack_deadline = time.time() + 10.0
                    while len(acks) < num_workers and time.time() < ack_deadline:
                        try:
                            w_id, ver = ack_queue.get(timeout=0.2)
                            if ver == iteration:
                                acks.add(w_id)
                        except (queue.Empty, EOFError):
                            pass

                drained_adv_f: list[torch.Tensor] = []
                drained_adv_t: list[torch.Tensor] = []
                drained_adv_w: list[torch.Tensor] = []
                drained_strat_f: list[torch.Tensor] = []
                drained_strat_t: list[torch.Tensor] = []
                drained_strat_w: list[torch.Tensor] = []
                round_diag = {
                    "normal": 0,
                    "endgame": 0,
                    "honest_bot": 0,
                    "forced": 0,
                    "multi": 0,
                    "rm_normal": 0,
                    "rm_fallback": 0,
                }

                target_blocks = 16
                collected_blocks = 0
                drain_deadline = time.time() + 15.0

                while (
                    collected_blocks < target_blocks
                    and time.time() < drain_deadline
                    and not stop_event.is_set()
                ):
                    try:
                        item = sample_queue.get(timeout=0.1)
                        adv_tup, strat_tup = item[0], item[1]
                        meta = item[2] if len(item) > 2 else {}
                        # 样本版本防污染校验：丢弃过期残余样本
                        if isinstance(meta, dict):
                            s_ver = meta.get("model_version", 0)
                            if s_ver < iteration and iteration > self.start_iteration:
                                continue
                            if "diag" in meta and isinstance(meta["diag"], dict):
                                for dk, dv in meta["diag"].items():
                                    round_diag[dk] = round_diag.get(dk, 0) + dv

                        if adv_tup is not None:
                            if isinstance(adv_tup[0], np.ndarray):
                                drained_adv_f.append(torch.from_numpy(adv_tup[0]))
                                drained_adv_t.append(torch.from_numpy(adv_tup[1]))
                                drained_adv_w.append(torch.from_numpy(adv_tup[2]))
                            else:
                                drained_adv_f.append(adv_tup[0])
                                drained_adv_t.append(adv_tup[1])
                                drained_adv_w.append(adv_tup[2])
                        if strat_tup is not None:
                            if isinstance(strat_tup[0], np.ndarray):
                                drained_strat_f.append(torch.from_numpy(strat_tup[0]))
                                drained_strat_t.append(torch.from_numpy(strat_tup[1]))
                                drained_strat_w.append(torch.from_numpy(strat_tup[2]))
                            else:
                                drained_strat_f.append(strat_tup[0])
                                drained_strat_t.append(strat_tup[1])
                                drained_strat_w.append(strat_tup[2])
                        collected_blocks += 1
                        total_traversals += 1
                    except queue.Empty:
                        pass

                if drained_adv_f:
                    cat_af = torch.cat(drained_adv_f, dim=0)
                    cat_at = torch.cat(drained_adv_t, dim=0)
                    cat_aw = torch.cat(drained_adv_w, dim=0)
                    self.advantage_buffer.extend((cat_af, cat_at, cat_aw))
                if drained_strat_f:
                    cat_sf = torch.cat(drained_strat_f, dim=0)
                    cat_st = torch.cat(drained_strat_t, dim=0)
                    cat_sw = torch.cat(drained_strat_w, dim=0)
                    self.strategy_buffer.extend((cat_sf, cat_st, cat_sw))

                if len(self.advantage_buffer) >= 512:
                    adv_bs = min(self.config.batch_size, len(self.advantage_buffer))
                    last_adv_loss = self._train_network(
                        net=self.advantage_net,
                        optimizer=self.advantage_optimizer,
                        buffer=self.advantage_buffer,
                        epochs=self.config.train_epochs,
                        batch_size=adv_bs,
                    )
                    self._eval_advantage_net.load_state_dict(
                        {k: v.cpu() for k, v in self.advantage_net.state_dict().items()}
                    )
                    self._eval_advantage_net.eval()

                if len(self.strategy_buffer) >= 512:
                    strat_bs = min(self.config.batch_size, len(self.strategy_buffer))
                    last_strat_loss = self._train_network(
                        net=self.strategy_net,
                        optimizer=self.strategy_optimizer,
                        buffer=self.strategy_buffer,
                        epochs=self.config.train_epochs,
                        batch_size=strat_bs,
                    )
                    self._eval_strategy_net.load_state_dict(
                        {k: v.cpu() for k, v in self.strategy_net.state_dict().items()}
                    )
                    self._eval_strategy_net.eval()
                    cpu_neural_policy = NeuralPolicy(
                        strategy_net=self._eval_strategy_net,
                        encoder=self.encoder,
                        abstractor=self.abstractor,
                        seed=iteration,
                        prob_mode="linear_norm",
                    )
                    self._opponent_pool.update_current_policy(cpu_neural_policy)
                    if iteration % 25 == 0:
                        if not self.freeze_league:
                            snap_net = copy.deepcopy(self._eval_strategy_net)
                            snap_policy = NeuralPolicy(
                                strategy_net=snap_net,
                                encoder=self.encoder,
                                abstractor=self.abstractor,
                                seed=iteration,
                                prob_mode="linear_norm",
                            )
                            self._opponent_pool.add_snapshot(snap_policy)
                            print(f"  📸 [Endgame League] iter {iteration} 残局快照入库 (池规模: {self._opponent_pool.league_size()})")
                        else:
                            print(f"  🔒 [Endgame League] iter {iteration} 对手池保持冻结 (固定池规模: {self._opponent_pool.league_size()})")

                iter_time = time.time() - iter_start

                round_normal = round_diag.get("normal", 0)
                round_endgame = round_diag.get("endgame", 0)
                round_honest = round_diag.get("honest_bot", 0)
                round_forced = round_diag.get("forced", 0)
                round_multi = round_diag.get("multi", 0)
                round_rm_norm = round_diag.get("rm_normal", 0)
                round_rm_fb = round_diag.get("rm_fallback", 0)
                fallback_rate = round(round_rm_fb / max(1, round_rm_norm + round_rm_fb), 4)
                total_round_actions = round_forced + round_multi
                forced_ratio = round(round_forced / max(1, total_round_actions), 4)

                entry = {
                    "iteration": iteration,
                    "traversals": total_traversals,
                    "adv_buffer": len(self.advantage_buffer),
                    "strat_buffer": len(self.strategy_buffer),
                    "adv_loss": round(last_adv_loss, 6),
                    "strat_loss": round(last_strat_loss, 6),
                    "league_size": self._opponent_pool.league_size(),
                    "iter_time": round(iter_time, 2),
                    "fallback_rate": fallback_rate,
                    "sample_stats": {
                        "normal_samples": round_normal,
                        "endgame_samples": round_endgame,
                        "honest_bot_samples": round_honest,
                    },
                    "action_stats": {
                        "forced_action_samples": round_forced,
                        "multi_action_samples": round_multi,
                        "forced_ratio": forced_ratio,
                        "rm_normal": round_rm_norm,
                        "rm_fallback": round_rm_fb,
                    },
                }

                total_elapsed = time.time() - train_start
                print(
                    f"[iter {iteration:4d}/{self.config.cfr_iterations}] "
                    f"adv_loss={last_adv_loss:.5f}  "
                    f"strat_loss={last_strat_loss:.5f}  "
                    f"fb_rate={fallback_rate*100:4.1f}%  "
                    f"buf=({len(self.advantage_buffer)}/{len(self.strategy_buffer)})  "
                    f"{iter_time:.2f}s/iter  "
                    f"({total_elapsed:.0f}s total)",
                    flush=True,
                )

                if iteration % self.config.eval_interval == 0:
                    eval_results = self._evaluate(iteration)
                    entry["eval"] = eval_results
                    win_rate = eval_results.get("win_rate", 0.0)
                    endgame_wr = eval_results.get("endgame_win_rate", 0.0)
                    endgame_t2 = eval_results.get("endgame_top2_rate", 0.0)
                    print(
                        f"  >> [eval@{iteration}] "
                        f"全局胜率={win_rate:.3f} (前二={eval_results.get('top2_rate', 0.0):.3f}) | "
                        f"🔥 逆风翻盘率={endgame_wr:.3f} (残局前二={endgame_t2:.3f}) | "
                        f"场均效用={eval_results.get('mean_utility', 0.0):.2f}"
                    )
                    if win_rate > self.best_win_rate:
                        self.best_win_rate = win_rate
                        self._save_checkpoint(output_path, iteration, tag="best")
                        print(f"  🏆 [残局黄金模型] 刷新最高吃鸡胜率 ({win_rate:.3f})，已持久化 policy_best.pt")

                if iteration % self.config.checkpoint_interval == 0:
                    self._save_checkpoint(output_path, iteration)
                    try:
                        (output_path / "training_log.json").write_text(
                            json.dumps(log_entries, indent=2, ensure_ascii=False),
                            encoding="utf-8",
                        )
                    except Exception:
                        pass

                log_entries.append(entry)

        finally:
            stop_event.set()
            if use_mp:
                for q in cmd_queues:
                    try:
                        q.put({"cmd": "STOP"})
                    except Exception:
                        pass
                for p in workers:
                    try:
                        if p.is_alive():
                            p.terminate()
                            p.join(timeout=0.5)
                    except Exception:
                        pass

        self._save_checkpoint(output_path, self.config.cfr_iterations, tag="final")
        latest_path = output_path / "policy_latest.pt"
        torch.save(
            {
                "iteration": self.config.cfr_iterations,
                "strategy_net": self.strategy_net.state_dict(),
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "feature_dim": FEATURE_DIM,
                "action_space_size": ACTION_SPACE_SIZE,
            },
            latest_path,
        )

        try:
            (output_path / "training_log.json").write_text(
                json.dumps(log_entries, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            pass

        return EndgameCFRTrainingResult(
            strategy_net=self.strategy_net,
            advantage_net=self.advantage_net,
            output_dir=output_path,
            iterations_completed=self.config.cfr_iterations,
            total_traversals=total_traversals,
            final_advantage_loss=last_adv_loss,
            final_strategy_loss=last_strat_loss,
            eval_results=eval_results,
        )

    def _train_network(
        self,
        net: nn.Module,
        optimizer: optim.Optimizer,
        buffer: ReservoirBuffer,
        epochs: int,
        batch_size: int,
    ) -> float:
        net.train()
        total_loss_tensor = torch.zeros(1, device=self.device)
        actual_epochs = max(1, epochs)

        use_cuda = self.device.type == "cuda"
        for _ in range(actual_epochs):
            features, targets, weights = buffer.sample_batch(batch_size)
            if features.device != self.device:
                features = features.to(self.device)
                targets = targets.to(self.device)
                weights = weights.to(self.device)

            if use_cuda:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    predictions = net(features)
                    loss_per_sample = ((predictions.float() - targets) ** 2).mean(dim=-1)
                    loss = (loss_per_sample * weights).mean()
            else:
                predictions = net(features)
                loss_per_sample = ((predictions - targets) ** 2).mean(dim=-1)
                loss = (loss_per_sample * weights).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss_tensor += loss.detach()

        return float(total_loss_tensor.item()) / actual_epochs

    def _traverse(
        self,
        state,
        traverser_seat: int,
        iteration: int,
        depth: int,
        seed_base: int,
        adv_samples: list,
        strat_samples: list,
        worker_rng: Random | None = None,
        worker_abstractor: ActionAbstractor | None = None,
        worker_encoder: FeatureEncoder | None = None,
        worker_honest_bot: Policy | None = None,
        honest_bot_seat: int | None = None,
        game_type: str = "normal",
        diag_counters: dict | None = None,
    ) -> float:
        if self.engine.is_terminal(state):
            return self.reward_model.evaluate(state).for_seat(traverser_seat)

        if (
            self.config.max_traverse_depth is not None
            and depth >= self.config.max_traverse_depth
        ):
            return self._leaf_utility(state, traverser_seat)

        rng = worker_rng or self._rng
        abstractor = worker_abstractor or self.abstractor
        encoder = worker_encoder or self.encoder

        acting_seat = self.engine.current_actor(state)
        legal_actions = self.engine.legal_actions(state)
        if not legal_actions:
            return 0.0

        if diag_counters is not None:
            diag_counters[game_type] = diag_counters.get(game_type, 0) + 1
            if len(legal_actions) == 1:
                diag_counters["forced"] = diag_counters.get("forced", 0) + 1
            else:
                diag_counters["multi"] = diag_counters.get("multi", 0) + 1

        observation = self.engine.observe(state, acting_seat)
        grouped = abstractor.abstract_legal_actions(legal_actions, observation)
        abstract_actions = list(grouped.keys())

        features = encoder.encode(observation)
        strategy = self._get_strategy(features, abstract_actions)

        # 判定当前是否处于残局 (存活玩家手牌少于等于 2 张)
        p_actor = state.player_by_seat(acting_seat)
        is_endgame_hand = len(p_actor.hand) <= 2
        is_underdog = p_actor.shots_taken >= p_actor.live_round_index  # 濒死

        if acting_seat == traverser_seat:
            action_utilities: dict[str, float] = {}
            for aa in abstract_actions:
                concrete = rng.choice(grouped[aa])
                next_state = self.engine.clone_state(state)

                p_before = state.player_by_seat(traverser_seat)
                shots_before = p_before.shots_taken
                hand_len_before = len(p_before.hand)

                self.engine.apply_action(next_state, concrete)

                tactical_bonus = 0.0

                # ── 残局质疑精准度强化奖惩 ──
                if isinstance(concrete, ChallengeAction):
                    last_ev = next_state.public_history[-1] if next_state.public_history else None
                    if last_ev and last_ev.event_type == "challenge":
                        outcome = last_ev.detail.get("outcome")
                        if outcome == "lie":
                            # 残局关键识破谎言翻盘特奖
                            tactical_bonus += 8.0 if is_endgame_hand else 5.0
                        elif outcome == "honest":
                            # 盲目抓真牌挨枪受惩（若濒死则惩罚更重）
                            tactical_bonus -= 8.0 if is_underdog else 5.0
                        elif outcome == "ghost":
                            tactical_bonus -= 4.0

                sub_util = self._traverse(
                    next_state,
                    traverser_seat,
                    iteration,
                    depth + 1,
                    seed_base,
                    adv_samples,
                    strat_samples,
                    worker_rng=rng,
                    worker_abstractor=abstractor,
                    worker_encoder=encoder,
                    worker_honest_bot=worker_honest_bot,
                    honest_bot_seat=honest_bot_seat,
                    game_type=game_type,
                    diag_counters=diag_counters,
                )

                # ── 残局出牌与脱身奖惩 ──
                if isinstance(concrete, PlayAction):
                    is_bluff = False
                    has_ghost = False
                    for cid in concrete.card_ids:
                        c = next((card for card in p_before.hand if card.card_id == cid), None)
                        if c:
                            if c.kind == CardKind.GHOST:
                                has_ghost = True
                            elif c.kind != CardKind.WILD and c.printed_rank != concrete.claim_rank:
                                is_bluff = True

                    p_end = next_state.player_by_seat(traverser_seat)
                    if p_end.shots_taken > shots_before:
                        if is_bluff:
                            tactical_bonus -= 7.0
                            if not p_end.alive:
                                # 残局濒死盲目诈唬自杀：处以极刑
                                tactical_bonus -= 15.0
                    else:
                        if not is_bluff:
                            # 残局成功出真牌脱身奖励大幅提升！
                            tactical_bonus += 4.0 if (hand_len_before <= 2 and len(p_end.hand) == 0) else 2.0
                        if has_ghost:
                            # 成功打出魔牌反杀全场
                            tactical_bonus += 10.0

                total_action_util = max(-25.0, min(25.0, sub_util + tactical_bonus))
                action_utilities[aa.label] = total_action_util

            expected_value = sum(
                strategy.get(a.label, 0.0) * action_utilities[a.label]
                for a in abstract_actions
            )

            advantage_target = torch.zeros(ACTION_SPACE_SIZE)
            for a in abstract_actions:
                idx = action_to_index(a.label)
                advantage_target[idx] = action_utilities[a.label] - expected_value

            adv_samples.append((features.detach().cpu(), advantage_target, iteration))

            strategy_target = torch.zeros(ACTION_SPACE_SIZE)
            for aa in abstract_actions:
                strategy_target[action_to_index(aa.label)] = strategy.get(aa.label, 0.0)
            strat_samples.append((features.detach().cpu(), strategy_target, iteration))

            return expected_value

        else:
            strategy_target = torch.zeros(ACTION_SPACE_SIZE)
            for abstract_action in abstract_actions:
                idx = action_to_index(abstract_action.label)
                strategy_target[idx] = strategy.get(abstract_action.label, 0.0)
            strat_samples.append((features.detach().cpu(), strategy_target, iteration))

            concrete = None
            if worker_honest_bot is not None and acting_seat == honest_bot_seat:
                try:
                    concrete = worker_honest_bot.choose_action(observation, legal_actions)
                except Exception:
                    concrete = None
            else:
                opp_policy = self._opponent_pool.sample_opponent(
                    seed=seed_base + depth * 37 + acting_seat
                )
                try:
                    if isinstance(opp_policy, NeuralPolicy):
                        concrete = opp_policy.choose_action_from_features(
                            observation, legal_actions, features, grouped
                        )
                    else:
                        concrete = opp_policy.choose_action(observation, legal_actions)
                except Exception:
                    concrete = None

            if concrete is None:
                sampled = self._sample_from_strategy(abstract_actions, strategy, rng)
                concrete = rng.choice(grouped[sampled])

            next_state = self.engine.clone_state(state)
            self.engine.apply_action(next_state, concrete)
            return self._traverse(
                next_state,
                traverser_seat,
                iteration,
                depth + 1,
                seed_base,
                adv_samples,
                strat_samples,
                worker_rng=rng,
                worker_abstractor=abstractor,
                worker_encoder=encoder,
                worker_honest_bot=worker_honest_bot,
                honest_bot_seat=honest_bot_seat,
                game_type=game_type,
                diag_counters=diag_counters,
            )

    @torch.no_grad()
    def _get_strategy(
        self, features: torch.Tensor, abstract_actions: list
    ) -> dict[str, float]:
        device = next(self._eval_advantage_net.parameters()).device
        feat_dev = features.unsqueeze(0).to(device)
        advantages = self._eval_advantage_net(feat_dev).squeeze(0)

        positive_regrets = []
        for aa in abstract_actions:
            idx = action_to_index(aa.label)
            r = max(0.0, advantages[idx].item())
            positive_regrets.append((aa.label, r))

        sum_regrets = sum(r for _, r in positive_regrets)
        if sum_regrets > 1e-8:
            return {label: r / sum_regrets for label, r in positive_regrets}

        n = len(abstract_actions)
        return {aa.label: 1.0 / n for aa in abstract_actions}

    def _sample_from_strategy(self, abstract_actions: list, strategy: dict[str, float], rng: Random):
        if len(abstract_actions) == 1:
            return abstract_actions[0]
        probs = [strategy.get(aa.label, 0.0) for aa in abstract_actions]
        total = sum(probs)
        if total <= 1e-8:
            return rng.choice(abstract_actions)
        r = rng.random() * total
        acc = 0.0
        for aa, p in zip(abstract_actions, probs):
            acc += p
            if r <= acc:
                return aa
        return abstract_actions[-1]

    def _leaf_utility(self, state, traverser_seat: int) -> float:
        player = state.player_by_seat(traverser_seat)
        if not player.alive:
            return self.reward_model.score_rules.eliminated_or_last

        alive_count = sum(1 for p in state.players if p.alive)
        survival_bonus = (4 - alive_count) * 4.0
        shot_disadv = -player.shots_taken * 3.5
        hand_adv = (5 - len(player.hand)) * 1.0

        score = survival_bonus + shot_disadv + hand_adv
        return float(
            max(
                self.reward_model.score_rules.eliminated_or_last,
                min(self.reward_model.score_rules.first, score),
            )
        )

    def _evaluate(self, iteration: int) -> dict:
        try:
            neural_policy = NeuralPolicy(
                strategy_net=self._eval_strategy_net,
                encoder=self.encoder,
                abstractor=self.abstractor,
                seed=iteration,
            )
            evaluator = MatchupEvaluator(
                engine=self.engine,
                reward_model=self.reward_model,
            )

            def hero_factory(game_index: int, seat: int) -> Policy:
                return neural_policy

            result = evaluator.evaluate_hero_vs_pool(
                hero_policy_factory=hero_factory,
                opponent_profiles=default_opponent_profiles(),
                game_count=self.config.eval_games,
                hero_seat=1,
                seed_start=iteration * 10000,
            )
            summary = result.summary
            from .endgame_eval import EndgameBenchmark
            bench = EndgameBenchmark(self.engine, self.reward_model)
            endgame_res = bench.evaluate_policy(
                neural_policy, game_count=50, seed_start=iteration * 5000, hero_seat=1
            )

            return {
                "iteration": iteration,
                "win_rate": summary.win_rate,
                "top2_rate": summary.top2_rate,
                "mean_rank": summary.mean_rank,
                "mean_utility": summary.mean_utility,
                "endgame_win_rate": endgame_res.win_rate,
                "endgame_top2_rate": endgame_res.top2_rate,
                "endgame_challenge_acc": endgame_res.challenge_accuracy,
            }
        except Exception as e:
            return {"iteration": iteration, "error": str(e)}

    def _save_checkpoint(
        self, output_dir: Path, iteration: int, tag: str | None = None
    ) -> None:
        suffix = tag or f"iter_{iteration:05d}"
        ckpt = {
            "iteration": iteration,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "feature_dim": FEATURE_DIM,
            "action_space_size": ACTION_SPACE_SIZE,
            "output_semantics": "linear_probability",
            "advantage_net": self.advantage_net.state_dict(),
            "strategy_net": self.strategy_net.state_dict(),
            "advantage_optimizer": self.advantage_optimizer.state_dict(),
            "strategy_optimizer": self.strategy_optimizer.state_dict(),
        }
        if tag == "best":
            torch.save(ckpt, output_dir / "policy_best.pt")
            return

        torch.save(ckpt, output_dir / f"checkpoint_{suffix}.pt")
        torch.save(ckpt, output_dir / "policy_latest.pt")
