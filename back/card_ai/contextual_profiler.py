from __future__ import annotations

"""
上下文动作偏好追踪与预估器 (Contextual Action Profiler & Predictor)

功能：
1. 精准记录每场对局中，每个模型在特定上下文（面对特定上家、特定下家、特定手牌张数）下的每一个具体动作。
2. 聚合统计：
   - 整体行为特征：出牌张数 (1/2/3)、真牌/诈唬/鬼牌分布、主动质疑率、安全脱手率；
   - 面对特定下家 (Successor) 的出牌倾向：出几张牌、诈唬率、被抓率；
   - 面对特定上家 (Predecessor) 的反应偏好：质疑率、抓假成功率、接牌出牌习惯。
3. 偏好预估 (Action Predictor)：
   - 提供实时概率估计接口，供人机对战界面提示或 AI 运行时策略使用。
"""

import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from card_ai.types import (
    Card,
    CardKind,
    ChallengeAction,
    GameAction,
    GameState,
    PlayAction,
    Rank,
)


@dataclass
class ActionRecord:
    """单个动作上下文详细记录。"""
    game_id: int
    round_idx: int
    turn_idx: int
    actor_name: str
    predecessor_name: str | None
    successor_name: str | None
    actor_hand_count_before: int
    action_type: str  # 'play' or 'challenge'
    # 出牌相关
    play_count: int | None = None
    claim_rank: str | None = None
    card_nature: str | None = None  # 'honest', 'bluff', 'ghost'
    # 质疑相关
    challenged_target_name: str | None = None
    challenge_outcome: str | None = None  # 'honest', 'lie', 'ghost'
    is_forced_challenge: bool = False


class ContextualActionProfiler:
    """多模型上下文动作偏好收集与统计分析器。"""

    def __init__(self) -> None:
        self.records: list[ActionRecord] = []

        # 累计统计矩阵
        # 1. 整体统计: model -> stats
        self.overall_plays = defaultdict(lambda: {1: 0, 2: 0, 3: 0})
        self.overall_nature = defaultdict(lambda: {"honest": 0, "bluff": 0, "ghost": 0})
        self.voluntary_challenges = defaultdict(lambda: {"oppty": 0, "count": 0, "caught_lie": 0})
        self.escapes = defaultdict(lambda: {"rounds_alive": 0, "success": 0})

        # 2. 面对特定下家 (Actor -> Successor)
        # self.vs_successor[actor][succ] = {...}
        self.vs_successor = defaultdict(lambda: defaultdict(lambda: {
            "total_plays": 0,
            "play_counts": {1: 0, 2: 0, 3: 0},
            "nature": {"honest": 0, "bluff": 0, "ghost": 0},
            "challenged_by_succ": 0,
            "caught_by_succ": 0,
        }))

        # 3. 面对特定上家 (Actor -> Predecessor)
        # self.vs_predecessor[actor][pred] = {...}
        self.vs_predecessor = defaultdict(lambda: defaultdict(lambda: {
            "oppty_challenge": 0,
            "voluntary_challenge": 0,
            "challenge_success": 0,
            "pass_and_play": 0,
            "pass_play_counts": {1: 0, 2: 0, 3: 0},
        }))

    def record_step(
        self,
        game_id: int,
        round_idx: int,
        turn_idx: int,
        actor_name: str,
        predecessor_name: str | None,
        successor_name: str | None,
        actor_hand_before: list[Card],
        action: GameAction,
        legal_actions: tuple[GameAction, ...],
        latest_play_cards: list[Card] | None = None,
        latest_play_claim: Rank | None = None,
    ) -> None:
        can_chal = any(isinstance(a, ChallengeAction) for a in legal_actions)
        can_ply = any(isinstance(a, PlayAction) for a in legal_actions)
        is_voluntary_oppty = can_chal and can_ply

        if isinstance(action, PlayAction):
            cards = [c for c in actor_hand_before if c.card_id in action.card_ids]
            num_cards = len(cards)
            is_ghost = any(c.kind == CardKind.GHOST for c in cards)
            is_bluff = any(c.kind == CardKind.NORMAL and c.printed_rank != action.claim_rank for c in cards)

            if is_ghost:
                nature = "ghost"
            elif is_bluff:
                nature = "bluff"
            else:
                nature = "honest"

            # 记录全局
            self.overall_plays[actor_name][num_cards] += 1
            self.overall_nature[actor_name][nature] += 1

            # 面对下家
            if successor_name:
                succ_stat = self.vs_successor[actor_name][successor_name]
                succ_stat["total_plays"] += 1
                succ_stat["play_counts"][num_cards] += 1
                succ_stat["nature"][nature] += 1

            # 面对上家（有质疑机会但选择出牌放行）
            if predecessor_name and is_voluntary_oppty:
                pred_stat = self.vs_predecessor[actor_name][predecessor_name]
                pred_stat["oppty_challenge"] += 1
                pred_stat["pass_and_play"] += 1
                pred_stat["pass_play_counts"][num_cards] += 1

            rec = ActionRecord(
                game_id=game_id,
                round_idx=round_idx,
                turn_idx=turn_idx,
                actor_name=actor_name,
                predecessor_name=predecessor_name,
                successor_name=successor_name,
                actor_hand_count_before=len(actor_hand_before),
                action_type="play",
                play_count=num_cards,
                claim_rank=action.claim_rank.value,
                card_nature=nature,
            )
            self.records.append(rec)

        elif isinstance(action, ChallengeAction):
            target_name = predecessor_name
            # 判断 outcome
            outcome = "honest"
            if latest_play_cards is not None and latest_play_claim is not None:
                if len(latest_play_cards) == 1 and latest_play_cards[0].kind == CardKind.GHOST:
                    outcome = "ghost"
                elif any(c.kind == CardKind.NORMAL and c.printed_rank != latest_play_claim for c in latest_play_cards):
                    outcome = "lie"

            is_forced = not is_voluntary_oppty
            if is_voluntary_oppty:
                self.voluntary_challenges[actor_name]["oppty"] += 1
                self.voluntary_challenges[actor_name]["count"] += 1
                if outcome == "lie":
                    self.voluntary_challenges[actor_name]["caught_lie"] += 1

                if predecessor_name:
                    pred_stat = self.vs_predecessor[actor_name][predecessor_name]
                    pred_stat["oppty_challenge"] += 1
                    pred_stat["voluntary_challenge"] += 1
                    if outcome == "lie":
                        pred_stat["challenge_success"] += 1

            # 对应上家被该下家质疑
            if predecessor_name:
                succ_stat = self.vs_successor[predecessor_name][actor_name]
                succ_stat["challenged_by_succ"] += 1
                if outcome == "lie":
                    succ_stat["caught_by_succ"] += 1

            rec = ActionRecord(
                game_id=game_id,
                round_idx=round_idx,
                turn_idx=turn_idx,
                actor_name=actor_name,
                predecessor_name=predecessor_name,
                successor_name=successor_name,
                actor_hand_count_before=len(actor_hand_before),
                action_type="challenge",
                challenged_target_name=target_name,
                challenge_outcome=outcome,
                is_forced_challenge=is_forced,
            )
            self.records.append(rec)

    def record_escape_result(self, hero_name: str, escaped: bool) -> None:
        self.escapes[hero_name]["rounds_alive"] += 1
        if escaped:
            self.escapes[hero_name]["success"] += 1

    def build_summary_report(self) -> dict[str, Any]:
        """提炼完整画像报表与预估参数。"""
        all_models = sorted(set(list(self.overall_plays.keys()) + list(self.vs_successor.keys())))
        model_profiles = {}

        for m in all_models:
            # 1. 总体出牌分布
            p_counts = self.overall_plays[m]
            tot_plays = sum(p_counts.values())
            play_dist = {
                "1_card": round(p_counts[1] / max(1, tot_plays), 4),
                "2_cards": round(p_counts[2] / max(1, tot_plays), 4),
                "3_cards": round(p_counts[3] / max(1, tot_plays), 4),
            }

            # 真实牌型分布
            nat = self.overall_nature[m]
            tot_nat = sum(nat.values())
            nature_dist = {
                "honest": round(nat["honest"] / max(1, tot_nat), 4),
                "bluff": round(nat["bluff"] / max(1, tot_nat), 4),
                "ghost": round(nat["ghost"] / max(1, tot_nat), 4),
            }

            # 主动质疑
            v_chal = self.voluntary_challenges[m]
            vol_rate = round(v_chal["count"] / max(1, v_chal["oppty"]), 4)
            vol_succ_rate = round(v_chal["caught_lie"] / max(1, v_chal["count"]), 4)

            # 安全脱手
            esc = self.escapes[m]
            esc_rate = round(esc["success"] / max(1, esc["rounds_alive"]), 4)

            # 2. 面对不同下家的偏好
            succ_matrix = {}
            for succ, s_data in self.vs_successor[m].items():
                s_tot = s_data["total_plays"]
                if s_tot == 0:
                    continue
                s_p_dist = {
                    "1_card": round(s_data["play_counts"][1] / s_tot, 4),
                    "2_cards": round(s_data["play_counts"][2] / s_tot, 4),
                    "3_cards": round(s_data["play_counts"][3] / s_tot, 4),
                }
                s_nat_tot = sum(s_data["nature"].values())
                s_nat_dist = {
                    "honest": round(s_data["nature"]["honest"] / max(1, s_nat_tot), 4),
                    "bluff": round(s_data["nature"]["bluff"] / max(1, s_nat_tot), 4),
                    "ghost": round(s_data["nature"]["ghost"] / max(1, s_nat_tot), 4),
                }
                be_challenged_rate = round(s_data["challenged_by_succ"] / max(1, s_tot), 4)
                be_caught_rate = round(s_data["caught_by_succ"] / max(1, s_data["challenged_by_succ"]), 4)
                succ_matrix[succ] = {
                    "samples": s_tot,
                    "play_count_dist": s_p_dist,
                    "nature_dist": s_nat_dist,
                    "challenged_by_successor_rate": be_challenged_rate,
                    "caught_by_successor_rate": be_caught_rate,
                }

            # 3. 面对不同上家的偏好
            pred_matrix = {}
            for pred, p_data in self.vs_predecessor[m].items():
                p_oppty = p_data["oppty_challenge"]
                if p_oppty == 0:
                    continue
                chal_rate = round(p_data["voluntary_challenge"] / p_oppty, 4)
                chal_succ = round(p_data["challenge_success"] / max(1, p_data["voluntary_challenge"]), 4)
                pass_tot = p_data["pass_and_play"]
                pass_p_dist = {
                    "1_card": round(p_data["pass_play_counts"][1] / max(1, pass_tot), 4),
                    "2_cards": round(p_data["pass_play_counts"][2] / max(1, pass_tot), 4),
                    "3_cards": round(p_data["pass_play_counts"][3] / max(1, pass_tot), 4),
                }
                pred_matrix[pred] = {
                    "samples_face_play": p_oppty,
                    "challenge_rate": chal_rate,
                    "challenge_success_rate": chal_succ,
                    "pass_play_count_dist": pass_p_dist,
                }

            model_profiles[m] = {
                "total_plays": tot_plays,
                "overall_play_dist": play_dist,
                "overall_nature_dist": nature_dist,
                "voluntary_challenge_rate": vol_rate,
                "voluntary_challenge_success_rate": vol_succ_rate,
                "escape_rate": esc_rate,
                "preference_against_successor": succ_matrix,
                "preference_against_predecessor": pred_matrix,
            }

        return {
            "total_action_records": len(self.records),
            "model_profiles": model_profiles,
        }


class ContextualActionPredictor:
    """基于聚合分析资产的动作偏好预测与推理器。"""

    def __init__(self, profile_data: dict[str, Any] | None = None, profile_path: str | Path | None = None) -> None:
        if profile_path and Path(profile_path).is_file():
            with open(profile_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
                self.profiles = raw.get("model_profiles", raw.get("profiling_dossier", {}).get("model_profiles", {}))
        elif profile_data:
            self.profiles = profile_data.get("model_profiles", {})
        else:
            self.profiles = {}

    def predict_play_style(self, actor_name: str, successor_name: str | None = None) -> dict[str, Any]:
        """预测行动者出牌倾向（1/2/3张概率，诚实/诈唬概率）。"""
        prof = self.profiles.get(actor_name)
        if not prof:
            return {
                "play_count_prob": {1: 0.90, 2: 0.08, 3: 0.02},
                "nature_prob": {"honest": 0.75, "bluff": 0.20, "ghost": 0.05},
                "summary": "默认保守估计：倾向单张真牌",
            }

        # 优先使用面对特定下家的画像
        if successor_name and successor_name in prof.get("preference_against_successor", {}):
            succ_data = prof["preference_against_successor"][successor_name]
            p_dist = {int(k[0]): v for k, v in succ_data["play_count_dist"].items()}
            n_dist = succ_data["nature_dist"]
            summary = (
                f"针对下家【{successor_name}】：出单张率 {p_dist.get(1, 0)*100:.1f}%，"
                f"诈唬率 {n_dist.get('bluff', 0)*100:.1f}%，诚实率 {n_dist.get('honest', 0)*100:.1f}%"
            )
            return {"play_count_prob": p_dist, "nature_prob": n_dist, "summary": summary}

        # 回退全局统计
        p_dist = {int(k[0]): v for k, v in prof["overall_play_dist"].items()}
        n_dist = prof["overall_nature_dist"]
        summary = (
            f"全局习惯：出单张率 {p_dist.get(1, 0)*100:.1f}%，"
            f"诈唬率 {n_dist.get('bluff', 0)*100:.1f}%，诚实率 {n_dist.get('honest', 0)*100:.1f}%"
        )
        return {"play_count_prob": p_dist, "nature_prob": n_dist, "summary": summary}

    def predict_challenge_tendency(self, actor_name: str, predecessor_name: str | None = None) -> dict[str, Any]:
        """预测行动者面对上家出牌时的质疑倾向。"""
        prof = self.profiles.get(actor_name)
        if not prof:
            return {"challenge_prob": 0.25, "success_prob": 0.50, "summary": "默认常规质疑倾向"}

        if predecessor_name and predecessor_name in prof.get("preference_against_predecessor", {}):
            p_data = prof["preference_against_predecessor"][predecessor_name]
            c_rate = p_data["challenge_rate"]
            c_succ = p_data["challenge_success_rate"]
            summary = f"面对上家【{predecessor_name}】：主动质疑率 {c_rate*100:.1f}%，抓假命中率 {c_succ*100:.1f}%"
            return {"challenge_prob": c_rate, "success_prob": c_succ, "summary": summary}

        c_rate = prof.get("voluntary_challenge_rate", 0.25)
        c_succ = prof.get("voluntary_challenge_success_rate", 0.50)
        summary = f"全局主动质疑率 {c_rate*100:.1f}%，抓假命中率 {c_succ*100:.1f}%"
        return {"challenge_prob": c_rate, "success_prob": c_succ, "summary": summary}
