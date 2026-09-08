"""
v28「通用＋劣势双专家」条件路由模块 (Dual Expert Conditional Routing Module)

核心规范：
1. 固定规则路由判定：is_disadvantage_state(obs)
   - 存活人数 >= 3
   - 自己普通目标牌 + Wild 数量 <= 1 (Ghost 不计入普通真牌数，保留在输入特征中)
   - 当前小轮自己首发 (latest_play_seat is None 或 round_claim_rank is None)，或自己空枪次数 >= 2
   - 仅依赖私有可见观测 PrivateObservation，不访问隐藏真实信息或未来状态
2. 双专家策略封装 DualExpertPolicy：
   - 封装冻结通用专家 G (Generalist) 与可训练劣势专家 E (Disadvantage Specialist)
   - 在决策时自动路由，并统计路由触发计数与动作分布差异 (TVD)
   - 严格 eval 模式推理与 masked softmax
"""

from __future__ import annotations

from pathlib import Path
from random import Random
from typing import Any

import torch

from .types import PrivateObservation, GameAction, CardKind, Rank
from .abstractions import ActionAbstractor, action_to_index, ACTION_SPACE_SIZE
from .profiling_features import ProfilingFeatureEncoder
from .ppo_networks import PPOActor


def is_disadvantage_state(obs: PrivateObservation) -> bool:
    """第一版固定路由规则：

    仅当以下三个条件同时满足时，判定为劣势局面并选择劣势专家 E：
    1. 存活人数 >= 3
    2. 自己普通目标牌 + Wild 数量 <= 1 (Ghost 不计入上述普通真牌数，但完整保留在输入中)
    3. 当前小轮自己首发，或自己空枪次数 >= 2
    其他状态选择通用专家 G。
    """
    # 1. 存活人数 >= 3
    alive_count = sum(1 for p in obs.players if p.alive)
    if alive_count < 3:
        return False

    # 2. 当前小轮自己首发，或自己空枪次数 >= 2
    # 当前小轮首发：obs.latest_play_seat is None 或 obs.round_claim_rank is None
    is_round_leader = (obs.latest_play_seat is None or obs.round_claim_rank is None)
    hero_p = next((p for p in obs.players if p.seat == obs.hero_seat), None)
    shots_taken = hero_p.shots_taken if hero_p is not None else 0
    high_hazard = (shots_taken >= 2)

    if not (is_round_leader or high_hazard):
        return False

    # 3. 自己普通目标牌 + Wild 数量 <= 1
    wild_count = sum(1 for c in obs.hero_hand if c.kind == CardKind.WILD)
    if obs.round_claim_rank is not None:
        # 已有目标声称点数
        target_normal_count = sum(
            1 for c in obs.hero_hand
            if c.kind == CardKind.NORMAL and c.printed_rank == obs.round_claim_rank
        )
        real_cards = target_normal_count + wild_count
    else:
        # 尚未声称点数（当前小轮首发）：看手牌中是否存在任何 rank 使得 (普通牌 + Wild) > 1
        max_normal = 0
        for r in (Rank.A, Rank.K, Rank.Q):
            cnt = sum(1 for c in obs.hero_hand if c.kind == CardKind.NORMAL and c.printed_rank == r)
            if cnt > max_normal:
                max_normal = cnt
        real_cards = max_normal + wild_count

    return real_cards <= 1


class DualExpertPolicy:
    """双专家条件路由策略封装类。

    通用专家 G 全程冻结，劣势专家 E 负责在触发劣势状态时代替 G 决策。
    """

    def __init__(
        self,
        generalist_actor: PPOActor,
        specialist_actor: PPOActor,
        encoder: ProfilingFeatureEncoder | None = None,
        abstractor: ActionAbstractor | None = None,
        seed: int | None = None,
    ) -> None:
        self.actor_g = generalist_actor
        self.actor_e = specialist_actor
        self.actor_g.eval()
        self.actor_e.eval()

        self.encoder = encoder or ProfilingFeatureEncoder(mode="control")
        self.abstractor = abstractor or ActionAbstractor()
        self._rng = Random(seed)

        # 统计计数器
        self.stats = {
            "total_decisions": 0,
            "expert_e_triggers": 0,
            "matches_with_e": 0,
            "current_match_had_e": False,
            "distribution_tvds": [],
            "top1_agreements": 0,
        }

    def reset_for_new_game(self) -> None:
        """在新对局开始时调用。"""
        if self.stats["current_match_had_e"]:
            self.stats["matches_with_e"] += 1
        self.stats["current_match_had_e"] = False

    def finish_evaluation_stats(self) -> dict[str, Any]:
        """评测结束时结算统计。"""
        if self.stats["current_match_had_e"]:
            self.stats["matches_with_e"] += 1
            self.stats["current_match_had_e"] = False

        total_dec = max(1, self.stats["total_decisions"])
        e_trig = self.stats["expert_e_triggers"]
        tvds = self.stats["distribution_tvds"]
        mean_tvd = float(sum(tvds) / len(tvds)) if tvds else 0.0
        top1_agree = float(self.stats["top1_agreements"] / max(1, len(tvds))) if tvds else 1.0

        return {
            "total_decisions": total_dec,
            "expert_e_triggers": e_trig,
            "e_trigger_rate": round(e_trig / total_dec, 4),
            "matches_with_e": self.stats["matches_with_e"],
            "mean_tvd_on_trigger": round(mean_tvd, 4),
            "top1_agreement_rate": round(top1_agree, 4),
        }

    @torch.no_grad()
    def choose_action(
        self,
        observation: PrivateObservation,
        legal_actions: tuple[GameAction, ...],
    ) -> GameAction:
        if not legal_actions:
            raise ValueError("no legal actions")
        if len(legal_actions) == 1:
            return legal_actions[0]

        self.stats["total_decisions"] += 1
        is_e = is_disadvantage_state(observation)

        device = next(self.actor_g.parameters()).device
        feat = self.encoder.encode(observation).unsqueeze(0).to(device)
        grouped = self.abstractor.abstract_legal_actions(legal_actions, observation)
        legal_indices = [action_to_index(aa.label) for aa in grouped]

        mask = torch.full((1, ACTION_SPACE_SIZE), -1e9, dtype=torch.float32, device=device)
        for idx in legal_indices:
            mask[0, idx] = 0.0

        # 获取 G 的概率分布
        logits_g = self.actor_g(feat) + mask
        probs_g = torch.softmax(logits_g, dim=-1).squeeze(0)

        # 获取 E 的概率分布
        logits_e = self.actor_e(feat) + mask
        probs_e = torch.softmax(logits_e, dim=-1).squeeze(0)

        if is_e:
            self.stats["expert_e_triggers"] += 1
            self.stats["current_match_had_e"] = True

            # 诊断：记录触发状态下 G 与 E 的分布差异 (TVD = 0.5 * sum |p_g - p_e|)
            p_g_legal = probs_g[legal_indices]
            p_e_legal = probs_e[legal_indices]
            tvd = 0.5 * float(torch.abs(p_g_legal - p_e_legal).sum().item())
            self.stats["distribution_tvds"].append(tvd)
            if p_g_legal.argmax().item() == p_e_legal.argmax().item():
                self.stats["top1_agreements"] += 1

            active_probs = probs_e
        else:
            active_probs = probs_g

        probs_legal = active_probs[legal_indices]
        sum_p = probs_legal.sum()
        if sum_p > 1e-8:
            norm_p = probs_legal / sum_p
        else:
            norm_p = torch.full_like(probs_legal, 1.0 / len(legal_indices))

        chosen_local = torch.multinomial(norm_p, 1).item()
        chosen_idx = legal_indices[chosen_local]

        # 映射到具体动作
        for aa, concrete_group in grouped.items():
            if action_to_index(aa.label) == chosen_idx:
                return self._rng.choice(concrete_group)

        return self._rng.choice(legal_actions)
