from __future__ import annotations

from dataclasses import dataclass

from .config import ScoreRules
from .types import GameState


@dataclass(frozen=True)
class RewardVector:
    utilities: dict[int, float]

    def for_seat(self, seat: int) -> float:
        return self.utilities.get(seat, 0.0)


class TerminalRewardModel:
    def __init__(self, score_rules: ScoreRules | None = None) -> None:
        self.score_rules = score_rules or ScoreRules()

    def evaluate(self, state: GameState) -> RewardVector:
        if not self._looks_terminal(state):
            raise ValueError("terminal rewards require a terminal or completed state")

        utilities = {player.seat: self.score_rules.eliminated_or_last for player in state.players}
        successful_order = self._successful_finish_order(state)
        rank_scores = [self.score_rules.first, self.score_rules.second, self.score_rules.third]

        for index, seat in enumerate(successful_order[: len(rank_scores)]):
            utilities[seat] = rank_scores[index]

        return RewardVector(utilities=utilities)

    def _successful_finish_order(self, state: GameState) -> list[int]:
        # 存活者排最前（吃鸡获胜者）
        survivors = [p.seat for p in state.players if p.alive]
        # 淘汰者按死亡顺序倒序（越晚被淘汰名次越前）
        eliminated_reversed = list(
            reversed([s for s in state.finish_order if not state.player_by_seat(s).alive])
        )
        return survivors + eliminated_reversed

    def _looks_terminal(self, state: GameState) -> bool:
        alive_players = [player for player in state.players if player.alive]
        return len(alive_players) <= 1


@dataclass(frozen=True)
class RankScoreRules:
    first: float = 20.0
    second: float = 10.0
    third: float = 0.0
    fourth: float = -20.0


class RankRewardModel:
    """终局名次积分模型 (四人总分严格恒等于 10.0 分)。

    第一名: +20
    第二名: +10
    第三名: 0
    第四名: -20
    基准期望: +2.5 分/局。
    若发生同次判定（如鬼牌反杀）导致的多人并列淘汰，名次积分按并列席位求算术平均，确保总分守恒。
    """

    def __init__(self, score_rules: RankScoreRules | None = None) -> None:
        self.score_rules = score_rules or RankScoreRules()

    def evaluate(self, state: GameState) -> RewardVector:
        if not self._looks_terminal(state):
            raise ValueError("rank rewards require a terminal or completed state")

        tier_scores = [
            self.score_rules.first,
            self.score_rules.second,
            self.score_rules.third,
            self.score_rules.fourth,
        ]

        # 1. 存活者组 (第 1 名)
        survivors = [p.seat for p in state.players if p.alive]

        # 2. 从 public_history 提取按挑战/行动分组的死亡批次
        death_batches: list[list[int]] = []
        current_batch: list[int] = []
        for ev in state.public_history:
            if ev.event_type in ("play", "challenge"):
                if current_batch:
                    death_batches.append(current_batch)
                    current_batch = []
            elif ev.event_type == "shot":
                if ev.detail.get("died"):
                    if ev.seat not in current_batch:
                        current_batch.append(ev.seat)
        if current_batch:
            death_batches.append(current_batch)

        # 按照死亡时间倒序（越晚死亡批次名次越前）
        reversed_death_batches = list(reversed(death_batches))

        ranked_groups: list[list[int]] = []
        if survivors:
            ranked_groups.append(survivors)
        for batch in reversed_death_batches:
            ranked_groups.append(batch)

        # 容错：如果有淘汰者未被 batch 捕捉，按 finish_order 倒序补齐
        accounted = {s for group in ranked_groups for s in group}
        missing = [p.seat for p in state.players if p.seat not in accounted]
        if missing:
            # 优先看 finish_order
            fo_missing = [s for s in reversed(state.finish_order) if s in missing]
            other_missing = [s for s in missing if s not in fo_missing]
            for s in fo_missing + other_missing:
                ranked_groups.append([s])

        # 3. 为每个席位分配积分（并列时平分区间分）
        utilities: dict[int, float] = {}
        current_rank_idx = 0
        for group in ranked_groups:
            k = len(group)
            if k == 0:
                continue
            assigned_scores = tier_scores[current_rank_idx : current_rank_idx + k]
            mean_score = sum(assigned_scores) / k
            for seat in group:
                utilities[seat] = float(mean_score)
            current_rank_idx += k

        # 4. 严格校验四人总和守恒于当前规则理论值
        expected_sum = float(sum(tier_scores))
        total_score = sum(utilities.values())
        assert abs(total_score - expected_sum) < 1e-4, (
            f"Rank reward sum must equal {expected_sum}, got {total_score} (utilities: {utilities})"
        )

        return RewardVector(utilities=utilities)

    def _looks_terminal(self, state: GameState) -> bool:
        alive_players = [player for player in state.players if player.alive]
        return len(alive_players) <= 1


