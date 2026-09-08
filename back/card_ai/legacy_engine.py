from __future__ import annotations

"""独立实验版本：旧规则生存游戏引擎 (LegacySurvivalGameEngine)。

规则特性（实验用 Legacy 引擎）：
- 仅保留实验所需的空手轮转差异：中间空手存活玩家仍按顺时针参与轮转，若面临上家出牌则强制质疑替全场挨枪；
- 两人残局末手规则与生产真实引擎完全一致：仅剩两名存活玩家时，上家刚打空最后一手，下家必须且只能质疑刚打空的上家（返回唯一 ChallengeAction）。
"""

from itertools import combinations

from card_ai.engine import SurvivalGameEngine
from card_ai.types import (
    CardKind,
    ChallengeAction,
    GameAction,
    GameState,
    PlayAction,
    Rank,
)


class LegacySurvivalGameEngine(SurvivalGameEngine):
    """v35 实验用旧规则引擎：
    仅保留实验所需的空手轮转差异（中间空手存活玩家仍按顺时针参与轮转，若面临上家出牌则强制质疑替全场挨枪）。
    在两人残局末手等核心逻辑上与生产引擎完全保持一致。
    """

    def current_actor(self, state: GameState) -> int:
        player = state.player_by_seat(state.current_seat)
        if player.alive:
            return state.current_seat
        return self._next_active_seat(state, state.current_seat)

    def legal_actions(self, state: GameState) -> tuple[GameAction, ...]:
        if self.is_terminal(state):
            return ()

        acting_seat = self.current_actor(state)
        player = state.player_by_seat(acting_seat)
        latest_play = state.round_state.latest_play

        # 核心规则 1：两人残局末手强制质疑！
        # 仅剩两名存活玩家时，上家刚打空最后一手，下家必须且只能质疑这手牌，即使下家仍有手牌
        alive_players = [p for p in state.players if p.alive]
        if len(alive_players) == 2 and latest_play is not None and latest_play.seat != player.seat:
            challenged_player = state.player_by_seat(latest_play.seat)
            if len(challenged_player.hand) == 0:
                return (
                    ChallengeAction(seat=player.seat, challenged_seat=latest_play.seat),
                )

        # 核心规则 2：若当前存活玩家手里已无牌（Legacy 轮转机制仍轮转到他）：
        # 面对上家出牌，只能强制质疑替全场挨枪
        if len(player.hand) == 0:
            if latest_play is not None and latest_play.seat != player.seat:
                return (
                    ChallengeAction(seat=player.seat, challenged_seat=latest_play.seat),
                )
            return ()

        # 正常手牌动作
        actions: list[GameAction] = []
        claim_ranks = (state.round_state.claim_rank,) if state.round_state.claim_rank else tuple(Rank)
        for count in range(1, self.rules.max_play_count + 1):
            if count > len(player.hand):
                break
            for combo in combinations(player.hand, count):
                if any(card.kind == CardKind.GHOST for card in combo) and count > 1:
                    continue
                card_ids = tuple(card.card_id for card in combo)
                for claim_rank in claim_ranks:
                    actions.append(
                        PlayAction(seat=player.seat, card_ids=card_ids, claim_rank=claim_rank)
                    )

        if latest_play is not None and latest_play.seat != player.seat:
            actions.append(
                ChallengeAction(seat=player.seat, challenged_seat=latest_play.seat)
            )

        return tuple(actions)

    def _next_active_seat(
        self,
        state: GameState,
        current_seat: int,
        allow_self_if_active: bool = False,
    ) -> int:
        seats = [player.seat for player in state.players]
        start_index = seats.index(current_seat)
        offsets = range(0, len(seats)) if allow_self_if_active else range(1, len(seats) + 1)
        latest_play = state.round_state.latest_play
        for offset in offsets:
            seat = seats[(start_index + offset) % len(seats)]
            player = state.player_by_seat(seat)
            if player.alive:
                if latest_play is not None and latest_play.seat == seat:
                    continue
                return seat
        return current_seat
