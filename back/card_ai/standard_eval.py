"""
逆水寒卡牌 AI 标准化评测基准模块 (Standard Benchmark Evaluation Framework)

核心规范：
1. 显式白名单对手匹配：只允许 "v14", "v15", "v18", "HonestBot"，未知名称直接抛出 ValueError，严禁隐式回退。
2. 步骤种子局部派生与随机隔离：
   - 显式先手传入：engine.new_game(seed=game_seed, starting_seat=starting_seat)
   - 动作采样确定性：在每次决策前使用纯算术整数派生局部种子，绑定 torch.manual_seed、random.seed
     以及 pol._rng / pol.policy._rng，彻底杜绝跨模型执行顺序污染。
3. 候选策略统一封装 StandardCandidatePolicy：
   - 支持 104 维 (control/profiling) 和 80 维模型自动适配；
   - 局内记忆 reset_for_new_game() 清空；
   - 严格在 eval() 模式和 torch.no_grad() 下推理。
4. 公共单局执行 run_standard_match 与全量评测流水线。
"""

from __future__ import annotations

import os
import sys
import json
import time
import random
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import stats

from card_ai.engine import SurvivalGameEngine
from card_ai.rewards import RankRewardModel, RankScoreRules
from card_ai.neural_policy import NeuralPolicy
from card_ai.opponents import HeuristicProfilePolicy
from card_ai.honest_bot_diagnostic import HONEST_PROFILE
from card_ai.profiling_features import ProfilingFeatureEncoder, MatchBehaviorTracker, PROFILING_FEATURE_DIM
from card_ai.features import FeatureEncoder, FEATURE_DIM
from card_ai.types import PrivateObservation, GameAction, PlayAction, CardKind

# 标准基准名宿对手路径
OPPONENT_PATHS: dict[str, str] = {
    "v14": "runs_deep_cfr/deep_cfr_v14_full_selfplay/policy_best.pt",
    "v15": "runs_deep_cfr/deep_cfr_v15_league/policy_best.pt",
    "v18": "runs_deep_cfr/deep_cfr_v18_honest_adaptive/policy_best.pt",
}

ALLOWED_OPPONENTS: set[str] = {"v14", "v15", "v18", "HonestBot"}


def load_benchmark_opponent(name: str, seed: int | None = None) -> Any:
    """严格匹配并加载基准对手，未知名称直接抛出异常，严禁任何降级回退。"""
    if name not in ALLOWED_OPPONENTS:
        raise ValueError(
            f"[Fatal Error] 未知的基准对手名称: '{name}'！"
            f"允许的对手仅限: {sorted(ALLOWED_OPPONENTS)}。严禁静默回退为规则机器人！"
        )

    if name == "HonestBot":
        return HeuristicProfilePolicy(profile=HONEST_PROFILE, seed=seed)
    else:
        path = OPPONENT_PATHS[name]
        if not Path(path).is_file():
            raise FileNotFoundError(f"[Fatal Error] 找不到名宿模型权重: {path}")
        pol = NeuralPolicy.load(path, prob_mode="linear_norm")
        pol.strategy_net.eval()
        if seed is not None:
            pol._rng = random.Random(seed)
        return pol


class StandardCandidatePolicy:
    """标准候选策略封装类：严格绑定观察者席位，支持 104 维或 80 维模型。"""

    def __init__(
        self,
        checkpoint_path: str | Path,
        mode: str = "control",
        observer_seat: int = 1,
        prob_mode: str = "linear_norm",
    ) -> None:
        self.checkpoint_path = str(checkpoint_path)
        self.mode = mode
        self.observer_seat = observer_seat
        self.prob_mode = prob_mode

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        if "actor_net" in ckpt:
            # PPO 强化学习候选模型：严格使用 104 维 control 特征与 masked_softmax
            from card_ai.ppo_networks import PPOActor
            actor_net = PPOActor(
                input_dim=ckpt.get("input_dim", 104),
                action_space_size=ckpt.get("action_space_size", 58),
                hidden_dim=ckpt.get("hidden_dim", 512),
                num_layers=ckpt.get("num_layers", 4),
            )
            actor_net.load_state_dict(ckpt["actor_net"])
            actor_net.eval()
            self.is_104dim = True
            self.encoder = ProfilingFeatureEncoder(mode=mode)
            self.tracker = None
            self.prob_mode = "masked_softmax"
            self.policy = NeuralPolicy(
                strategy_net=actor_net,
                encoder=self.encoder,
                prob_mode="masked_softmax",
            )
        else:
            # 传统 CFR / Deep CFR 候选模型
            feat_dim = ckpt.get("feature_dim", 80)
            self.is_104dim = (feat_dim == 104 or feat_dim == PROFILING_FEATURE_DIM)

            if self.is_104dim:
                self.encoder = ProfilingFeatureEncoder(mode=mode)
                self.tracker = MatchBehaviorTracker(observer_seat=observer_seat)
                self.policy = NeuralPolicy.load(checkpoint_path, encoder=self.encoder, prob_mode=prob_mode)
            else:
                self.encoder = FeatureEncoder()
                self.tracker = None
                self.policy = NeuralPolicy.load(checkpoint_path, encoder=self.encoder, prob_mode=prob_mode)

        self.policy.strategy_net.eval()
        self._rng = self.policy._rng

    def reset_for_new_game(self) -> None:
        if self.tracker is not None:
            self.tracker.reset()

    def choose_action(self, obs: PrivateObservation, legal_actions: tuple[GameAction, ...]) -> GameAction:
        if not legal_actions:
            raise ValueError("no legal actions")
        if len(legal_actions) == 1:
            return legal_actions[0]

        if self.tracker is not None:
            self.tracker.update_from_history(obs.public_history)
            feat = self.encoder.encode(obs, self.tracker)
            grouped = self.policy.abstractor.abstract_legal_actions(legal_actions, obs)
            chosen_action = self.policy.choose_action_from_features(obs, legal_actions, feat, grouped)

            if isinstance(chosen_action, PlayAction):
                claim_rank = obs.round_claim_rank
                is_bluff = any(
                    c.kind == CardKind.NORMAL and c.printed_rank != claim_rank
                    for c in obs.hero_hand if c.card_id in chosen_action.card_ids
                )
                rel_opps = [p for p in self.encoder._relative_players(obs) if p.seat != obs.hero_seat and p.alive and p.hand_count > 0]
                next_seat = rel_opps[0].seat if rel_opps else None
                self.tracker.record_my_play(is_bluff, next_seat)

            return chosen_action
        else:
            return self.policy.choose_action(obs, legal_actions)


def run_standard_match(
    engine: SurvivalGameEngine,
    candidate_policy: StandardCandidatePolicy,
    opponents: dict[int, Any],
    rank_model: RankRewardModel,
    game_seed: int,
    starting_seat: int,
) -> tuple[float, int, int]:
    """运行一场标准对局：

    - 显式传入 starting_seat；
    - 清空所有策略的局内记忆；
    - 每步基于 (game_seed, step_count, actor) 派生独立步骤种子，完全解耦跨候选和全局状态；
    - 返回 (hero_score, hero_rank, step_count)。
    """
    # 1. 真实传入先手与种子
    state = engine.new_game(seed=game_seed, starting_seat=starting_seat)

    # 2. 清空局内记忆
    candidate_policy.reset_for_new_game()
    policies: dict[int, Any] = {1: candidate_policy, **opponents}
    for p in opponents.values():
        if hasattr(p, "reset_for_new_game"):
            p.reset_for_new_game()

    step_count = 0
    while not engine.is_terminal(state) and step_count < 250:
        step_count += 1
        actor = engine.current_actor(state)
        legal = engine.legal_actions(state)
        if not legal:
            break

        if len(legal) == 1:
            act = legal[0]
        else:
            pol = policies[actor]
            obs = engine.observe(state, actor)

            # 3. 步骤种子派生：纯整数运算，解耦跨候选和全局状态
            step_seed = (game_seed * 1000003 + step_count * 1009 + actor * 37) & 0x7FFFFFFF
            torch.manual_seed(step_seed)
            random.seed(step_seed)
            if hasattr(pol, "_rng"):
                pol._rng = random.Random(step_seed)
            if hasattr(pol, "policy") and hasattr(pol.policy, "_rng"):
                pol.policy._rng = random.Random(step_seed)

            with torch.no_grad():
                act = pol.choose_action(obs, legal)

        engine.apply_action(state, act)

        # 广播给有 tracker 的 policy 顺序消费公开历史
        for seat, p in policies.items():
            if hasattr(p, "tracker") and p.tracker is not None:
                p.tracker.update_from_history(state.public_history)

    # 4. 名次评分结算
    payoffs = rank_model.evaluate(state)
    hero_score = float(payoffs.for_seat(1))
    if hero_score >= 19.9:
        hero_rank = 1
    elif hero_score >= 14.9:
        hero_rank = 2
    elif hero_score >= -5.1:
        hero_rank = 3
    else:
        hero_rank = 4

    return hero_score, hero_rank, step_count


def generate_benchmark_matches(
    seed_base_history: int = 981000,
    seed_base_honest: int = 982000,
    num_seeds: int = 10,
) -> list[dict[str, Any]]:
    """预先生成确定的评测对局清单，完整覆盖 6 种对手排列与 4 个先手席位。

    - 历史场景: 6 排列 x 4 先手 x 10 种子 = 240 局
    - 诚实场景: 6 排列 x 4 先手 x 10 种子 = 240 局
    合计 480 局。
    """
    history_opps = ["v14", "v15", "v18"]
    honest_opps = ["HonestBot", "v18", "v14"]

    matches = []
    opp_perms = list(itertools.permutations(range(3)))  # 6 种排列

    for scen_name, pool, base_seed in [("history", history_opps, seed_base_history), ("honest", honest_opps, seed_base_honest)]:
        match_idx = 0
        for perm in opp_perms:
            perm_opp_names = [pool[i] for i in perm]
            for dealer_seat in range(1, 5):
                for s_i in range(num_seeds):
                    match_idx += 1
                    game_seed = base_seed + match_idx * 43
                    matches.append({
                        "scenario": scen_name,
                        "match_idx": match_idx,
                        "game_seed": game_seed,
                        "starting_seat": dealer_seat,
                        "opponents": {
                            2: perm_opp_names[0],
                            3: perm_opp_names[1],
                            4: perm_opp_names[2],
                        },
                    })
    return matches
