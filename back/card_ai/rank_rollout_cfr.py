from __future__ import annotations

"""全新终局目标 Deep CFR 强化学习求解器 (Rank Rollout CFR Trainer, v21)。

核心特性：
1. 目标完全对齐名次积分期望：终局积分 [+20, +10, 0, -20]，基准期望 +2.5 分/局，四人总和恒等于 10.0 分；
2. 彻底删除人工单步奖惩：无抓诈 bonus、无清牌 bonus、无出鬼牌 bonus，各动作价值纯粹由终局名次积分塑造；
3. 完整对局续演 (Rollout to Terminal) 替代手写截断公式：搜索深度到达 8 层且未终局时，启动单路径私有观测续演至整场终局；
4. 完整断点恢复 (Full Checkpoint Resume)：网络、优化器、两个缓冲区内容/权重/容量/见闻计数全部持久化；
5. 多进程 Worker 迭代边界 Barrier 版本握手同步。
"""

import copy
import json
import os
import queue
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from random import Random

# 彻底屏蔽 Windows 下极易引发多进程 RPC 挂起的 _wmi
sys.modules["_wmi"] = None

import numpy as np
import torch
import torch.multiprocessing as mp
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
from .features import FEATURE_DIM, FeatureEncoder
from .networks import (
    AdvantageNetwork,
    StrategyNetwork,
    build_advantage_net,
    build_strategy_net,
)
from .neural_policy import NeuralPolicy
from .rewards import RankRewardModel, RankScoreRules
from .types import CardKind, ChallengeAction, GameState, PlayAction, Rank


def check_v25_hard_case(state: GameState, engine: SurvivalGameEngine):
    """判定是否满足 v25 困难局面条件 (3/4人, 行动者shots>=2, 面对3张, 真牌<=2, 非强制质疑)"""
    if state.round_state is None or state.round_state.latest_play is None:
        return None
    if len(state.round_state.latest_play.cards) != 3:
        return None
    alive_count = sum(1 for p in state.players if p.alive)
    if alive_count not in (3, 4):
        return None
    actor = engine.current_actor(state)
    actor_player = state.player_by_seat(actor)
    if actor_player.shots_taken not in (2, 3, 4):
        return None
    legal = engine.legal_actions(state)
    has_challenge = any(isinstance(a, ChallengeAction) for a in legal)
    has_play = any(isinstance(a, PlayAction) for a in legal)
    if not (has_challenge and has_play):
        return None
    claim_rank = state.round_state.claim_rank
    norm_target = sum(1 for c in actor_player.hand if c.kind == CardKind.NORMAL and c.printed_rank == claim_rank)
    wild_count = sum(1 for c in actor_player.hand if c.kind == CardKind.WILD)
    if (norm_target + wild_count) not in (0, 1, 2):
        return None
    return True


def compute_regret_matching_strategy(
    advantages: torch.Tensor,
    abstract_actions: list,
) -> tuple[dict[str, float], bool]:
    """根据优势网络预测值计算合法动作的正遗憾匹配策略。"""
    regrets = {}
    for aa in abstract_actions:
        idx = action_to_index(aa.label)
        r = float(advantages[idx].item())
        regrets[aa.label] = max(0.0, r)

    sum_pos = sum(regrets.values())
    strat = {}
    if sum_pos > 1e-8:
        for aa in abstract_actions:
            strat[aa.label] = regrets[aa.label] / sum_pos
        return strat, False
    else:
        u_p = 1.0 / len(abstract_actions)
        for aa in abstract_actions:
            strat[aa.label] = u_p
        return strat, True


def _pick_best_play(sim_state: GameState, play_actions: list[PlayAction], actor: int) -> PlayAction:
    claim = sim_state.round_state.claim_rank
    my_hand = sim_state.players[actor - 1].hand
    matches = [c for c in my_hand if c.printed_rank == claim or c.kind == CardKind.WILD] if claim else []
    if matches:
        count = min(len(matches), 2)
        card_ids = tuple(c.card_id for c in matches[:count])
        for a in play_actions:
            if a.card_ids == card_ids:
                return a
    for a in play_actions:
        if len(a.card_ids) == 1:
            return a
    return play_actions[0]


def _rollout_fixed_rule(
    state: GameState,
    traverser_seat: int,
    engine: SurvivalGameEngine,
    rank_reward_model: RankRewardModel,
    worker_rng: Random,
    hard_limit: int = 300,
) -> tuple[float | None, int, bool]:
    """修正版轻量规则续演：出牌逻辑完全闭环，不质疑时严格进入优先真牌出牌逻辑。"""
    sim_state = engine.clone_state(state)
    total_steps = 0

    while not engine.is_terminal(sim_state):
        if total_steps >= hard_limit:
            return None, total_steps, False

        actor = engine.current_actor(sim_state)
        legal = engine.legal_actions(sim_state)
        if not legal:
            return None, total_steps, False

        if len(legal) == 1:
            engine.apply_action(sim_state, legal[0])
            total_steps += 1
            continue

        has_challenge = any(isinstance(a, ChallengeAction) for a in legal)
        play_actions = [a for a in legal if isinstance(a, PlayAction)]

        chosen = None
        if has_challenge:
            claim = sim_state.round_state.claim_rank
            my_hand = sim_state.players[actor - 1].hand
            match_count = sum(1 for c in my_hand if c.printed_rank == claim or c.kind in (CardKind.WILD, CardKind.GHOST))
            latest_play = sim_state.round_state.latest_play
            last_count = len(latest_play.cards) if latest_play else 1

            prob_challenge = 0.15 + 0.15 * last_count + 0.08 * match_count
            if worker_rng.random() < min(0.70, prob_challenge):
                chosen = next((a for a in legal if isinstance(a, ChallengeAction)), None)
            
            # 严格修正：若决定不质疑，且有出牌选项，严格进入出牌逻辑（杜绝按 legal 列表顺序退化）
            if chosen is None and play_actions:
                chosen = _pick_best_play(sim_state, play_actions, actor)

        if chosen is None:
            if play_actions:
                chosen = _pick_best_play(sim_state, play_actions, actor)
            else:
                chosen = legal[0]

        engine.apply_action(sim_state, chosen)
        total_steps += 1

    reward_vec = rank_reward_model.evaluate(sim_state)
    score = reward_vec.for_seat(traverser_seat)
    return score, total_steps, True


def _rollout_frozen_net(
    state: GameState,
    traverser_seat: int,
    engine: SurvivalGameEngine,
    rank_reward_model: RankRewardModel,
    frozen_policy: NeuralPolicy,
    hard_limit: int = 300,
) -> tuple[float | None, int, bool]:
    """原计划纯冻结神经网络续演：无手写出牌规则，完全由神经网络自博弈推进到终局。"""
    sim_state = engine.clone_state(state)
    total_steps = 0

    while not engine.is_terminal(sim_state):
        if total_steps >= hard_limit:
            return None, total_steps, False

        actor = engine.current_actor(sim_state)
        legal = engine.legal_actions(sim_state)
        if not legal:
            return None, total_steps, False

        if len(legal) == 1:
            engine.apply_action(sim_state, legal[0])
            total_steps += 1
            continue

        obs = engine.observe(sim_state, actor)
        action = frozen_policy.choose_action(obs, legal)
        engine.apply_action(sim_state, action)
        total_steps += 1

    reward_vec = rank_reward_model.evaluate(sim_state)
    score = reward_vec.for_seat(traverser_seat)
    return score, total_steps, True


def _worker_sampling_process(
    worker_id: int,
    traversals_per_worker: int,
    endgame_ratio: float,
    hidden_dim: int,
    num_layers: int,
    max_depth: int,
    data_queue: mp.Queue,
    cmd_queue: mp.Queue,
    ack_queue: mp.Queue,
    stop_event: mp.Event,
    seed_base: int,
    frozen_stars_paths: list[str],
    rollout_mode: str = "fixed_rule",
    score_rules: RankScoreRules | None = None,
    engine_variant: str = "new_exempt",
    hard_cases_pool: list[GameState] | None = None,
    hard_case_ratio: float = 0.0,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    torch.set_num_threads(1)

    worker_rng = Random(seed_base + worker_id * 10007)
    torch.manual_seed(seed_base + worker_id * 10007)

    if engine_variant == "legacy_all":
        from .legacy_engine import LegacySurvivalGameEngine
        engine = LegacySurvivalGameEngine()
    else:
        engine = SurvivalGameEngine()
    rank_reward_model = RankRewardModel(score_rules=score_rules)
    endgame_gen = EndgameScenarioGenerator()
    encoder = FeatureEncoder()
    abstractor = ActionAbstractor()

    local_adv_net = build_advantage_net(
        input_dim=FEATURE_DIM,
        action_space_size=ACTION_SPACE_SIZE,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    )
    local_strat_net = build_strategy_net(
        input_dim=FEATURE_DIM,
        action_space_size=ACTION_SPACE_SIZE,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    )
    local_adv_net.eval()
    local_strat_net.eval()

    local_policy = NeuralPolicy(
        strategy_net=local_strat_net,
        encoder=encoder,
        abstractor=abstractor,
        temperature=1.0,
        greedy=False,
        prob_mode="linear_norm",
    )

    frozen_stars = []
    for star_path in frozen_stars_paths:
        try:
            if Path(star_path).is_file():
                star_p = NeuralPolicy.load(star_path, prob_mode="linear_norm")
                frozen_stars.append(star_p)
        except Exception:
            pass

    current_model_version = [0]
    current_iter_val = [1]

    chunk_adv_f: list = []
    chunk_adv_t: list = []
    chunk_adv_w: list = []
    chunk_strat_f: list = []
    chunk_strat_t: list = []
    chunk_strat_w: list = []
    diag = {
        "rm_normal": 0,
        "rm_fallback": 0,
        "rollouts": 0,
        "rollout_steps": 0,
        "sources": {"endgame": 0, "hard_case": 0, "normal": 0},
        "actual_hard_cases": 0,
    }

    def _traverse(state: GameState, traverser_seat: int, depth: int, cur_iter: int):
        if engine.is_terminal(state):
            return rank_reward_model.evaluate(state).for_seat(traverser_seat)
        if depth >= max_depth:
            # 完整对局续演 (Rollout to Terminal)
            diag["rollouts"] += 1
            if rollout_mode == "frozen_net":
                sc, steps, ok = _rollout_frozen_net(
                    state=state,
                    traverser_seat=traverser_seat,
                    engine=engine,
                    rank_reward_model=rank_reward_model,
                    frozen_policy=local_policy,
                )
            else:
                sc, steps, ok = _rollout_fixed_rule(
                    state=state,
                    traverser_seat=traverser_seat,
                    engine=engine,
                    rank_reward_model=rank_reward_model,
                    worker_rng=worker_rng,
                )
            diag["rollout_steps"] += steps
            if ok and sc is not None:
                return sc
            return 2.5  # 极罕见异常兜底基准期望

        actor = engine.current_actor(state)
        legal = engine.legal_actions(state)
        if not legal:
            return 2.5

        obs = engine.observe(state, actor)
        feat = encoder.encode(obs)
        grouped = abstractor.abstract_legal_actions(legal, obs)
        abs_actions = list(grouped.keys())

        if actor == traverser_seat:
            with torch.no_grad():
                adv_pred = local_adv_net(feat.unsqueeze(0)).squeeze(0)
            strat_dict, fell_back = compute_regret_matching_strategy(adv_pred, abs_actions)
            if fell_back:
                diag["rm_fallback"] += 1
            else:
                diag["rm_normal"] += 1

            action_utils = {}
            for aa in abs_actions:
                c = worker_rng.choice(grouped[aa])
                nxt = engine.clone_state(state)
                engine.apply_action(nxt, c)
                # 动作价值完全由终局名次积分回传，bonus 彻底置 0.0
                sub_u = _traverse(nxt, traverser_seat, depth + 1, cur_iter)
                action_utils[aa.label] = float(sub_u)

            # 节点期望收益严格按策略加权: v(sigma) = sum_a sigma(a) * u(a)
            ev = sum(strat_dict[aa.label] * action_utils[aa.label] for aa in abs_actions)

            # 优势目标: adv_tgt[a] = u(a) - ev
            adv_tgt = torch.zeros(ACTION_SPACE_SIZE)
            for aa in abs_actions:
                idx = action_to_index(aa.label)
                adv_tgt[idx] = action_utils[aa.label] - ev

            # 策略网络训练目标: strat_tgt[a] = sigma(a)
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
            # 对手节点：使用当前冻结策略或名宿自对弈
            c = None
            if frozen_stars and worker_rng.random() < 0.4:
                star = worker_rng.choice(frozen_stars)
                try:
                    c = star.choose_action_from_features(obs, legal, feat, grouped)
                except Exception:
                    c = None
            if c is None:
                c = local_policy.choose_action_from_features(obs, legal, feat, grouped)

            nxt = engine.clone_state(state)
            engine.apply_action(nxt, c)
            return _traverse(nxt, traverser_seat, depth + 1, cur_iter)

    t_count = 0
    iter_done = False
    while not stop_event.is_set():
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
                    elif cmd_type == "SET_ITER":
                        current_iter_val[0] = cmd_obj["iteration"]
                        t_count = 0
                        iter_done = False
                    elif cmd_type == "STOP":
                        return
                except (queue.Empty, EOFError):
                    break

        if iter_done or t_count >= traversals_per_worker:
            time.sleep(0.01)
            continue

        cur_iter = current_iter_val[0]
        game_seed = worker_rng.randint(0, 2**31 - 1)
        seat = (t_count % 4) + 1

        r_val = worker_rng.random()
        if hard_cases_pool and r_val < hard_case_ratio:
            picked = worker_rng.choice(hard_cases_pool)
            init_state = engine.clone_state(picked)
            # 关键要求：确保相应行动者确实作为待更新玩家参与遍历
            seat = engine.current_actor(init_state)
            source_tag = "hard_case"
        elif r_val < (hard_case_ratio + endgame_ratio):
            init_state = endgame_gen.sample_endgame_state(
                seed=game_seed, scenario="random_mix", hero_seat=seat
            )
            source_tag = "endgame"
        else:
            init_state = engine.new_game(seed=game_seed)
            source_tag = "normal"

        diag["sources"][source_tag] += 1
        if check_v25_hard_case(init_state, engine) is not None:
            diag["actual_hard_cases"] += 1

        _traverse(init_state, seat, 0, cur_iter)
        t_count += 1
        is_last = t_count >= traversals_per_worker

        if len(chunk_adv_f) >= 200 or is_last:
            if chunk_adv_f:
                b_adv_f = torch.stack(chunk_adv_f)
                b_adv_t = torch.stack(chunk_adv_t)
                b_adv_w = torch.tensor(chunk_adv_w, dtype=torch.float32)

                b_strat_f = torch.stack(chunk_strat_f)
                b_strat_t = torch.stack(chunk_strat_t)
                b_strat_w = torch.tensor(chunk_strat_w, dtype=torch.float32)

                payload = {
                    "worker_id": worker_id,
                    "iteration": cur_iter,
                    "model_version": current_model_version[0],
                    "adv": (b_adv_f, b_adv_t, b_adv_w),
                    "strat": (b_strat_f, b_strat_t, b_strat_w),
                    "diag": copy.copy(diag),
                    "is_last": is_last,
                }
                data_queue.put(payload)
                chunk_adv_f.clear()
                chunk_adv_t.clear()
                chunk_adv_w.clear()
                chunk_strat_f.clear()
                chunk_strat_t.clear()
                chunk_strat_w.clear()
                diag = {
                    "rm_normal": 0,
                    "rm_fallback": 0,
                    "rollouts": 0,
                    "rollout_steps": 0,
                    "sources": {"endgame": 0, "hard_case": 0, "normal": 0},
                    "actual_hard_cases": 0,
                }

            if is_last:
                iter_done = True



class RankRolloutDeepCFRTrainer:
    """终局目标与名次积分对局续演 Deep CFR 训练器。"""

    def __init__(
        self,
        config: DeepCFRConfig,
        endgame_ratio: float = 0.70,
        max_depth: int = 8,
        rollout_mode: str = "fixed_rule",
        score_rules: RankScoreRules | None = None,
        initial_weights: dict | str | Path | None = None,
        engine_variant: str = "new_exempt",
        deadline_time: float | None = None,
        hard_cases_pool: list[GameState] | None = None,
        hard_case_ratio: float = 0.0,
    ) -> None:
        self.config = config
        self.endgame_ratio = endgame_ratio
        self.hard_cases_pool = hard_cases_pool
        self.hard_case_ratio = hard_case_ratio
        self.max_depth = max_depth
        self.rollout_mode = rollout_mode
        self.score_rules = score_rules or RankScoreRules()
        self.engine_variant = engine_variant
        self.deadline_time = deadline_time or float("inf")
        self.objective_version = f"rank_rollout_{rollout_mode}_v22_{int(self.score_rules.first)}_{int(self.score_rules.second)}_{int(self.score_rules.third)}_{int(self.score_rules.fourth)}"

        self.device = torch.device(
            "cuda" if (config.device == "auto" and torch.cuda.is_available()) or config.device == "cuda" else "cpu"
        )
        print(f"[Rank Rollout CFR] 训练设备: {self.device} | 目标版本: {self.objective_version} | 续演模式: {self.rollout_mode} | 截断深度: {self.max_depth}")
        print(f"[Rank Rollout CFR] 奖励规则配置: [{self.score_rules.first}, {self.score_rules.second}, {self.score_rules.third}, {self.score_rules.fourth}]")

        self.advantage_net = build_advantage_net(
            input_dim=FEATURE_DIM,
            action_space_size=ACTION_SPACE_SIZE,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
        ).to(self.device)

        self.strategy_net = build_strategy_net(
            input_dim=FEATURE_DIM,
            action_space_size=ACTION_SPACE_SIZE,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
        ).to(self.device)

        if initial_weights:
            if isinstance(initial_weights, (str, Path)):
                init_sd = torch.load(initial_weights, map_location="cpu", weights_only=False)
            else:
                init_sd = initial_weights
            if "advantage_net" in init_sd:
                self.advantage_net.load_state_dict(init_sd["advantage_net"])
                self.advantage_net.to(self.device)
            if "strategy_net" in init_sd:
                self.strategy_net.load_state_dict(init_sd["strategy_net"])
                self.strategy_net.to(self.device)
            print(f"[Rank Rollout CFR] 成功注入受控初始网络张量！")

        self.advantage_optimizer = optim.Adam(
            self.advantage_net.parameters(),
            lr=config.learning_rate,
            weight_decay=1e-4,
        )
        self.strategy_optimizer = optim.Adam(
            self.strategy_net.parameters(),
            lr=config.learning_rate,
            weight_decay=1e-4,
        )

        self.advantage_buffer = ReservoirBuffer(
            capacity=config.advantage_buffer_size,
            feature_dim=FEATURE_DIM,
            action_space_size=ACTION_SPACE_SIZE,
            device=self.device,
        )
        self.strategy_buffer = ReservoirBuffer(
            capacity=config.strategy_buffer_size,
            feature_dim=FEATURE_DIM,
            action_space_size=ACTION_SPACE_SIZE,
            device=self.device,
        )

        self.start_iteration = 1
        self.best_rank_score = -999.0
        self.training_log: list[dict] = []

        self.league_roster = [
            "runs_deep_cfr/deep_cfr_v14_full_selfplay/policy_best.pt",
            "runs_deep_cfr/deep_cfr_v15_league/policy_best.pt",
            "runs_deep_cfr/deep_cfr_v18_control/policy_best.pt",
            "runs_deep_cfr/deep_cfr_v18_honest_adaptive/policy_best.pt",
        ]

        if config.resume_from:
            self._load_checkpoint(config.resume_from)

    def _load_checkpoint(self, path: str | Path) -> None:
        p = Path(path)
        if not p.is_file():
            print(f"[Resume] 找不到检查点文件: {p}，从第 1 轮冷启动")
            return

        print(f"[Resume] 正在加载完整检查点: {p}")
        ckpt = torch.load(p, map_location="cpu", weights_only=False)

        # 校验目标版本隔离
        ckpt_obj = ckpt.get("objective_version", "legacy_score")
        if ckpt_obj != self.objective_version:
            print(f"[Resume] 警告: 检查点目标版本为 '{ckpt_obj}'，与当前 '{self.objective_version}' 不同！仅继承网络初始化，丢弃旧缓冲区。")
            if "strategy_net" in ckpt:
                self.strategy_net.load_state_dict(ckpt["strategy_net"])
                self.strategy_net.to(self.device)
            return

        self.start_iteration = ckpt.get("iteration", 0) + 1
        self.best_rank_score = ckpt.get("best_rank_score", -999.0)
        self.training_log = ckpt.get("training_log", [])

        if "advantage_net" in ckpt:
            self.advantage_net.load_state_dict(ckpt["advantage_net"])
            self.advantage_net.to(self.device)
        if "strategy_net" in ckpt:
            self.strategy_net.load_state_dict(ckpt["strategy_net"])
            self.strategy_net.to(self.device)
        if "advantage_optimizer" in ckpt:
            self.advantage_optimizer.load_state_dict(ckpt["advantage_optimizer"])
        if "strategy_optimizer" in ckpt:
            self.strategy_optimizer.load_state_dict(ckpt["strategy_optimizer"])

        # 完整恢复两个缓冲区及其见闻计数
        if "advantage_buffer" in ckpt:
            self.advantage_buffer.load_state_dict(ckpt["advantage_buffer"])
            print(f"[Resume] 优势池恢复成功: 样本 {len(self.advantage_buffer)}, 历史计数 {self.advantage_buffer.total_seen}")
        if "strategy_buffer" in ckpt:
            self.strategy_buffer.load_state_dict(ckpt["strategy_buffer"])
            print(f"[Resume] 策略池恢复成功: 样本 {len(self.strategy_buffer)}, 历史计数 {self.strategy_buffer.total_seen}")

        print(f"[Resume] 恢复完毕，将从第 {self.start_iteration} 轮继续训练！")

    def _save_checkpoint(self, output_dir: Path, iteration: int, is_final: bool = False) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        ckpt_filename = f"checkpoint_iter_{iteration:05d}.pt"
        ckpt_path = output_dir / ckpt_filename

        # 完整续训检查点
        payload = {
            "objective_version": self.objective_version,
            "rollout_mode": self.rollout_mode,
            "score_rules": {
                "first": self.score_rules.first,
                "second": self.score_rules.second,
                "third": self.score_rules.third,
                "fourth": self.score_rules.fourth,
            },
            "iteration": iteration,
            "hidden_dim": self.config.hidden_dim,
            "num_layers": self.config.num_layers,
            "feature_dim": FEATURE_DIM,
            "action_space_size": ACTION_SPACE_SIZE,
            "best_rank_score": self.best_rank_score,
            "training_log": self.training_log,
            "advantage_net": self.advantage_net.state_dict(),
            "strategy_net": self.strategy_net.state_dict(),
            "advantage_optimizer": self.advantage_optimizer.state_dict(),
            "strategy_optimizer": self.strategy_optimizer.state_dict(),
            "advantage_buffer": self.advantage_buffer.state_dict(),
            "strategy_buffer": self.strategy_buffer.state_dict(),
        }
        torch.save(payload, ckpt_path)
        if is_final:
            torch.save(payload, output_dir / "checkpoint_final.pt")

        # 轻量部署文件
        policy_best_path = output_dir / "policy_best.pt"
        torch.save({
            "objective_version": self.objective_version,
            "rollout_mode": self.rollout_mode,
            "score_rules": {
                "first": self.score_rules.first,
                "second": self.score_rules.second,
                "third": self.score_rules.third,
                "fourth": self.score_rules.fourth,
            },
            "iteration": iteration,
            "hidden_dim": self.config.hidden_dim,
            "num_layers": self.config.num_layers,
            "feature_dim": FEATURE_DIM,
            "action_space_size": ACTION_SPACE_SIZE,
            "strategy_net": self.strategy_net.state_dict(),
            "prob_mode": "linear_norm",
        }, policy_best_path)

    def train(self, output_dir: str | Path) -> None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        num_workers = max(1, getattr(self.config, "num_workers", 6))
        traversals_per_worker = max(1, self.config.traversals_per_iteration // num_workers)

        print(f"[Rank Rollout CFR] 启动 {num_workers} 个 Worker 进程池 | 模式: {self.rollout_mode} | 目标轮次: {self.config.cfr_iterations}")

        data_queue = mp.Queue(maxsize=120)
        cmd_queues = [mp.Queue(maxsize=10) for _ in range(num_workers)]
        ack_queue = mp.Queue(maxsize=num_workers * 2)
        stop_event = mp.Event()

        workers = []
        for w_idx in range(num_workers):
            p = mp.Process(
                target=_worker_sampling_process,
                args=(
                    w_idx,
                    traversals_per_worker,
                    self.endgame_ratio,
                    self.config.hidden_dim,
                    self.config.num_layers,
                    self.max_depth,
                    data_queue,
                    cmd_queues[w_idx],
                    ack_queue,
                    stop_event,
                    self.config.seed + w_idx * 1000,
                    self.league_roster,
                    self.rollout_mode,
                    self.score_rules,
                    self.engine_variant,
                    self.hard_cases_pool,
                    self.hard_case_ratio,
                ),
            )
            p.daemon = True
            p.start()
            workers.append(p)
            time.sleep(0.35)

        # 广播初始版本
        adv_cpu = {k: v.cpu() for k, v in self.advantage_net.state_dict().items()}
        strat_cpu = {k: v.cpu() for k, v in self.strategy_net.state_dict().items()}
        for q in cmd_queues:
            q.put({"cmd": "SYNC", "version": self.start_iteration, "adv_sd": adv_cpu, "strat_sd": strat_cpu})

        synced = set()
        while len(synced) < num_workers:
            wid, v = ack_queue.get()
            if v == self.start_iteration:
                synced.add(wid)

        print(f"[Rank Rollout CFR] 初始模型版本 v{self.start_iteration} 成功同步至 {num_workers}/{num_workers} 个 Worker，开跑！")

        criterion = nn.MSELoss(reduction="none")
        total_train_time = 0.0

        for iteration in range(self.start_iteration, self.config.cfr_iterations + 1):
            t_iter_start = time.time()

            # 通知 Worker 当前轮次
            for q in cmd_queues:
                q.put({"cmd": "SET_ITER", "iteration": iteration})

            # 收集样本：等待所有 Worker 完成本轮采样 (全部报告 is_last=True)
            samples_collected = 0
            fb_count = 0
            rm_count = 0
            rollouts_count = 0
            rollouts_steps_total = 0
            iter_sources = defaultdict(int)
            iter_actual_hard = 0
            workers_finished = set()

            collect_deadline = time.time() + 120.0
            while len(workers_finished) < num_workers and time.time() < collect_deadline:
                try:
                    payload = data_queue.get(timeout=2.0)
                    if payload["model_version"] >= iteration - 1:
                        self.advantage_buffer.extend(payload["adv"])
                        self.strategy_buffer.extend(payload["strat"])
                        samples_collected += len(payload["adv"][0])
                        d = payload["diag"]
                        fb_count += d.get("rm_fallback", 0)
                        rm_count += d.get("rm_normal", 0)
                        rollouts_count += d.get("rollouts", 0)
                        rollouts_steps_total += d.get("rollout_steps", 0)
                        srcs = d.get("sources", {})
                        for sk, sv in srcs.items():
                            iter_sources[sk] += sv
                        iter_actual_hard += d.get("actual_hard_cases", 0)
                    if payload.get("is_last", False):
                        workers_finished.add(payload["worker_id"])
                except queue.Empty:
                    pass

            # 训练网络
            self.advantage_net.train()
            self.strategy_net.train()
            adv_losses = []
            strat_losses = []

            batch_size = min(self.config.batch_size, len(self.advantage_buffer))
            if batch_size >= 64:
                for _ in range(self.config.train_epochs):
                    f_b, t_b, w_b = self.advantage_buffer.sample_batch(batch_size)
                    pred_adv = self.advantage_net(f_b)
                    loss_adv = (w_b.unsqueeze(1) * criterion(pred_adv, t_b)).mean()
                    self.advantage_optimizer.zero_grad()
                    loss_adv.backward()
                    nn.utils.clip_grad_norm_(self.advantage_net.parameters(), 1.0)
                    self.advantage_optimizer.step()
                    adv_losses.append(loss_adv.item())

                    f_s, t_s, w_s = self.strategy_buffer.sample_batch(batch_size)
                    pred_strat = self.strategy_net(f_s)
                    loss_strat = (w_s.unsqueeze(1) * criterion(pred_strat, t_s)).mean()
                    self.strategy_optimizer.zero_grad()
                    loss_strat.backward()
                    nn.utils.clip_grad_norm_(self.strategy_net.parameters(), 1.0)
                    self.strategy_optimizer.step()
                    strat_losses.append(loss_strat.item())

            mean_adv_loss = float(np.mean(adv_losses)) if adv_losses else 0.0
            mean_strat_loss = float(np.mean(strat_losses)) if strat_losses else 0.0
            fb_rate = fb_count / (fb_count + rm_count) if (fb_count + rm_count) > 0 else 0.0
            avg_rollout_steps = rollouts_steps_total / rollouts_count if rollouts_count > 0 else 0.0

            # 跨进程权重同步
            next_ver = iteration + 1
            adv_sd_cpu = {k: v.cpu() for k, v in self.advantage_net.state_dict().items()}
            strat_sd_cpu = {k: v.cpu() for k, v in self.strategy_net.state_dict().items()}
            for q in cmd_queues:
                q.put({"cmd": "SYNC", "version": next_ver, "adv_sd": adv_sd_cpu, "strat_sd": strat_sd_cpu})

            synced_w = set()
            sync_deadline = time.time() + 10.0
            while len(synced_w) < num_workers and time.time() < sync_deadline:
                try:
                    wid, v = ack_queue.get(timeout=0.5)
                    if v == next_ver:
                        synced_w.add(wid)
                except queue.Empty:
                    pass

            iter_sec = time.time() - t_iter_start
            total_train_time += iter_sec

            log_entry = {
                "iteration": iteration,
                "adv_loss": round(mean_adv_loss, 5),
                "strat_loss": round(mean_strat_loss, 5),
                "fb_rate": round(fb_rate, 4),
                "rollouts_count": rollouts_count,
                "avg_rollout_steps": round(avg_rollout_steps, 1),
                "buf_adv_size": len(self.advantage_buffer),
                "buf_adv_count": self.advantage_buffer.total_seen,
                "sources": dict(iter_sources),
                "actual_hard_cases": iter_actual_hard,
                "iter_time_sec": round(iter_sec, 2),
            }
            self.training_log.append(log_entry)

            print(
                f"[iter {iteration:4d}/{self.config.cfr_iterations}] "
                f"adv_loss={mean_adv_loss:.5f}  strat_loss={mean_strat_loss:.5f}  "
                f"sources={dict(iter_sources)}  actual_hard={iter_actual_hard}  "
                f"buf=({len(self.advantage_buffer)}/{len(self.strategy_buffer)})  "
                f"{iter_sec:.2f}s/iter  ({int(total_train_time)}s total)"
            )

            # 检查点保存
            if iteration % self.config.checkpoint_interval == 0 or iteration == self.config.cfr_iterations:
                self._save_checkpoint(out_dir, iteration, is_final=(iteration == self.config.cfr_iterations))

            # 逼近时间预算边界时平稳收尾退出 (预留一轮耗时+15秒)
            if time.time() + iter_sec + 15.0 >= self.deadline_time:
                print(f"[TimeBudget] 逼近预算截止时间，在第 {iteration} 轮迭代完整结束后平稳收尾保存并退出。")
                self._save_checkpoint(out_dir, iteration, is_final=True)
                break

        # 结束 Worker
        stop_event.set()
        for q in cmd_queues:
            q.put({"cmd": "STOP"})
        for p in workers:
            p.join(timeout=1.0)
            if p.is_alive():
                p.terminate()

        # 保存训练日志
        log_path = out_dir / "training_log.json"
        log_path.write_text(json.dumps(self.training_log, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n[Done] 训练全部完成！最终检查点与日志已落盘至: {out_dir}")
