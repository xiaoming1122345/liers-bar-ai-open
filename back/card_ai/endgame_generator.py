from __future__ import annotations

"""残局与劣势博弈状态生成器 (Endgame & Underdog Scenario Generator)。

专门生成高质量、符合骗子酒馆/逆水寒卡牌规则的高压残局局面：
1. heads_up_duel (单挑决死局): 仅剩 2 人存活，手牌各 1~2 张，膛压高悬；
2. underdog_one_shot_away (濒死劣势翻盘局): Hero 濒死 (差一枪必死)，手牌劣势 (无真牌散牌)，对手状态健康；
3. last_card_clash (手牌打空决胜局): 全员各剩 1 张牌，上家刚出完或轮到自己出最后手牌；
4. ghost_trap_ambush (鬼牌诱杀局): Hero 手握魔牌/鬼牌，手牌 1~2 张，诱敌质疑反杀全场。
"""

from random import Random
from typing import Literal

from .config import GameRules
from .engine import SurvivalGameEngine
from .types import (
    Card,
    CardKind,
    GameState,
    PlayAction,
    PlayedSet,
    PlayerState,
    PublicEvent,
    Rank,
    RoundState,
)

ScenarioType = Literal[
    "heads_up_duel",
    "underdog_one_shot_away",
    "last_card_clash",
    "ghost_trap_ambush",
    "three_player_clash",
    "random_mix",
]


class EndgameScenarioGenerator:
    """极端残局与劣势生死局生成器。"""

    def __init__(self, engine: SurvivalGameEngine | None = None) -> None:
        self.engine = engine or SurvivalGameEngine()
        self.rules = self.engine.rules

    def sample_endgame_state(
        self,
        seed: int | None = None,
        scenario: ScenarioType = "random_mix",
        hero_seat: int = 1,
    ) -> GameState:
        """采样一个残局/劣势博弈状态。"""
        rng = Random(seed)
        if scenario == "random_mix":
            weights = [0.25, 0.25, 0.20, 0.15, 0.15]
            choices: list[ScenarioType] = [
                "heads_up_duel",
                "underdog_one_shot_away",
                "last_card_clash",
                "ghost_trap_ambush",
                "three_player_clash",
            ]
            scenario = rng.choices(choices, weights=weights, k=1)[0]

        if scenario == "heads_up_duel":
            return self._gen_heads_up_duel(rng, hero_seat)
        elif scenario == "underdog_one_shot_away":
            return self._gen_underdog(rng, hero_seat)
        elif scenario == "last_card_clash":
            return self._gen_last_card_clash(rng, hero_seat)
        elif scenario == "ghost_trap_ambush":
            return self._gen_ghost_trap(rng, hero_seat)
        elif scenario == "three_player_clash":
            return self._gen_three_player_clash(rng, hero_seat)
        else:
            return self._gen_heads_up_duel(rng, hero_seat)

    # ── 1. 单挑决死局 ────────────────────────────────────────────────────────
    def _gen_heads_up_duel(self, rng: Random, hero_seat: int) -> GameState:
        """2 人单挑局，另 2 人已阵亡，手牌各 1~2 张。"""
        claim_rank = rng.choice(list(Rank))
        deck, _ = self.engine._build_deck(rng, ghost_rank=claim_rank)
        rng.shuffle(deck)

        opp_seat = rng.choice([s for s in range(1, 5) if s != hero_seat])

        players: list[PlayerState] = []
        card_idx = 0
        for s in range(1, 5):
            if s in (hero_seat, opp_seat):
                hand_len = rng.randint(1, 2)
                hand = deck[card_idx : card_idx + hand_len]
                card_idx += hand_len
                shots = rng.randint(0, 3)
                live_idx = rng.randint(shots, 4)
                players.append(
                    PlayerState(
                        seat=s,
                        hand=list(hand),
                        live_round_index=live_idx,
                        shots_taken=shots,
                        alive=True,
                    )
                )
            else:
                players.append(
                    PlayerState(
                        seat=s,
                        hand=[],
                        live_round_index=rng.randint(0, 2),
                        shots_taken=rng.randint(1, 4),
                        alive=False,
                    )
                )

        round_state = RoundState(claim_rank=claim_rank, leader_seat=opp_seat)
        current_seat = hero_seat

        history = [
            PublicEvent(turn_index=0, event_type="new_round", seat=None, detail={"round_index": 2, "claim_rank": claim_rank.value}),
        ]

        if rng.random() < 0.5:
            prior_cards = (deck[card_idx],)
            card_idx += 1
            round_state.latest_play = PlayedSet(seat=opp_seat, claim_rank=claim_rank, cards=prior_cards)
            history.append(
                PublicEvent(turn_index=1, event_type="play", seat=opp_seat, detail={"count": 1, "claim_rank": claim_rank.value})
            )

        return GameState(
            players=players,
            current_seat=current_seat,
            round_state=round_state,
            round_index=2,
            turn_index=len(history),
            public_history=history,
        )

    # ── 2. 濒死劣势翻盘局 ────────────────────────────────────────────────────
    def _gen_underdog(self, rng: Random, hero_seat: int) -> GameState:
        """Hero 濒死（shots_taken 达到 live_round_index，再吃一枪当场死亡），手牌为 1~2 张散牌。"""
        claim_rank = rng.choice(list(Rank))
        deck, _ = self.engine._build_deck(rng, ghost_rank=claim_rank)
        rng.shuffle(deck)

        bad_cards = [c for c in deck if c.kind == CardKind.NORMAL and c.printed_rank != claim_rank]
        rng.shuffle(bad_cards)
        hero_hand = bad_cards[: rng.randint(1, 2)]
        used_ids = {c.card_id for c in hero_hand}
        rem_deck = [c for c in deck if c.card_id not in used_ids]

        alive_count = rng.choice([3, 4])
        alive_seats = {hero_seat}
        other_seats = [s for s in range(1, 5) if s != hero_seat]
        rng.shuffle(other_seats)
        for s in other_seats[: alive_count - 1]:
            alive_seats.add(s)

        players: list[PlayerState] = []
        card_ptr = 0
        for s in range(1, 5):
            if s == hero_seat:
                live_idx = rng.randint(1, 4)
                players.append(
                    PlayerState(
                        seat=s,
                        hand=list(hero_hand),
                        live_round_index=live_idx,
                        shots_taken=live_idx,
                        alive=True,
                    )
                )
            elif s in alive_seats:
                hand_len = rng.randint(1, 3)
                opp_h = rem_deck[card_ptr : card_ptr + hand_len]
                card_ptr += hand_len
                live_idx = rng.randint(2, 4)
                players.append(
                    PlayerState(
                        seat=s,
                        hand=list(opp_h),
                        live_round_index=live_idx,
                        shots_taken=rng.choice([0, 0, 1]),
                        alive=True,
                    )
                )
            else:
                players.append(
                    PlayerState(
                        seat=s,
                        hand=[],
                        live_round_index=1,
                        shots_taken=2,
                        alive=False,
                    )
                )

        active_opps = [s for s in alive_seats if s != hero_seat]
        last_actor = rng.choice(active_opps)

        round_state = RoundState(claim_rank=claim_rank, leader_seat=last_actor)
        history = [
            PublicEvent(turn_index=0, event_type="new_round", seat=None, detail={"round_index": 3, "claim_rank": claim_rank.value})
        ]

        if rng.random() < 0.70:
            play_cnt = rng.randint(1, 2)
            played_c = tuple(rem_deck[card_ptr : card_ptr + play_cnt])
            round_state.latest_play = PlayedSet(seat=last_actor, claim_rank=claim_rank, cards=played_c)
            history.append(
                PublicEvent(turn_index=1, event_type="play", seat=last_actor, detail={"count": play_cnt, "claim_rank": claim_rank.value})
            )

        return GameState(
            players=players,
            current_seat=hero_seat,
            round_state=round_state,
            round_index=3,
            turn_index=len(history),
            public_history=history,
        )

    # ── 3. 全员仅剩 1 张手牌决胜残局 ──────────────────────────────────────
    def _gen_last_card_clash(self, rng: Random, hero_seat: int) -> GameState:
        """存活 2~3 人，所有人手里都只有 1 张手牌！生死一锤定音。"""
        claim_rank = rng.choice(list(Rank))
        deck, _ = self.engine._build_deck(rng, ghost_rank=claim_rank)
        rng.shuffle(deck)

        alive_count = rng.choice([2, 3])
        alive_seats = {hero_seat}
        other_seats = [s for s in range(1, 5) if s != hero_seat]
        rng.shuffle(other_seats)
        for s in other_seats[: alive_count - 1]:
            alive_seats.add(s)

        players: list[PlayerState] = []
        card_ptr = 0
        for s in range(1, 5):
            if s in alive_seats:
                players.append(
                    PlayerState(
                        seat=s,
                        hand=[deck[card_ptr]],
                        live_round_index=rng.randint(2, 4),
                        shots_taken=rng.randint(0, 2),
                        alive=True,
                    )
                )
                card_ptr += 1
            else:
                players.append(
                    PlayerState(
                        seat=s,
                        hand=[],
                        live_round_index=1,
                        shots_taken=2,
                        alive=False,
                    )
                )

        round_state = RoundState(claim_rank=claim_rank)
        history = [
            PublicEvent(turn_index=0, event_type="new_round", seat=None, detail={"round_index": 2, "claim_rank": claim_rank.value})
        ]

        if rng.random() < 0.5:
            last_opp = rng.choice([s for s in alive_seats if s != hero_seat])
            round_state.latest_play = PlayedSet(
                seat=last_opp, claim_rank=claim_rank, cards=(deck[card_ptr],)
            )
            history.append(
                PublicEvent(turn_index=1, event_type="play", seat=last_opp, detail={"count": 1, "claim_rank": claim_rank.value})
            )

        return GameState(
            players=players,
            current_seat=hero_seat,
            round_state=round_state,
            round_index=2,
            turn_index=len(history),
            public_history=history,
        )

    # ── 4. 鬼牌反杀陷阱局 ──────────────────────────────────────────────────
    def _gen_ghost_trap(self, rng: Random, hero_seat: int) -> GameState:
        """Hero 手握珍贵鬼牌，手牌 1~2 张，训练如何在残局假装诈唬引诱对手开枪。"""
        claim_rank = rng.choice(list(Rank))
        deck, ghost_rank = self.engine._build_deck(rng, ghost_rank=claim_rank)

        ghost_card = next(c for c in deck if c.kind == CardKind.GHOST)
        rem_deck = [c for c in deck if c.card_id != ghost_card.card_id]
        rng.shuffle(rem_deck)

        hero_cards = [ghost_card]
        if rng.random() < 0.5:
            hero_cards.append(rem_deck.pop(0))

        alive_count = rng.choice([2, 3, 4])
        alive_seats = {hero_seat}
        other_seats = [s for s in range(1, 5) if s != hero_seat]
        rng.shuffle(other_seats)
        for s in other_seats[: alive_count - 1]:
            alive_seats.add(s)

        players: list[PlayerState] = []
        for s in range(1, 5):
            if s == hero_seat:
                players.append(
                    PlayerState(
                        seat=s,
                        hand=hero_cards,
                        live_round_index=rng.randint(2, 4),
                        shots_taken=rng.randint(0, 2),
                        alive=True,
                    )
                )
            elif s in alive_seats:
                hand_len = rng.randint(1, 2)
                players.append(
                    PlayerState(
                        seat=s,
                        hand=[rem_deck.pop(0) for _ in range(hand_len)],
                        live_round_index=rng.randint(1, 3),
                        shots_taken=rng.randint(1, 3),
                        alive=True,
                    )
                )
            else:
                players.append(
                    PlayerState(seat=s, hand=[], live_round_index=1, shots_taken=2, alive=False)
                )

        round_state = RoundState(claim_rank=claim_rank)
        history = [
            PublicEvent(turn_index=0, event_type="new_round", seat=None, detail={"round_index": 2, "claim_rank": claim_rank.value})
        ]

        return GameState(
            players=players,
            current_seat=hero_seat,
            round_state=round_state,
            round_index=2,
            turn_index=len(history),
            public_history=history,
        )

    # ── 5. 三人中盘拉锯残局 ──────────────────────────────────────────────────
    def _gen_three_player_clash(self, rng: Random, hero_seat: int) -> GameState:
        """3 人存活（1人已阵亡），手牌 2~3 张，枪膛吃过 0~2 枪，处于中期转残局的关键分水岭。"""
        claim_rank = rng.choice(list(Rank))
        deck, _ = self.engine._build_deck(rng, ghost_rank=claim_rank)
        rng.shuffle(deck)

        other_seats = [s for s in range(1, 5) if s != hero_seat]
        rng.shuffle(other_seats)
        alive_seats = {hero_seat, other_seats[0], other_seats[1]}
        dead_seat = other_seats[2]

        players: list[PlayerState] = []
        card_ptr = 0
        for s in range(1, 5):
            if s in alive_seats:
                hand_len = rng.randint(2, 3)
                hand = deck[card_ptr : card_ptr + hand_len]
                card_ptr += hand_len
                shots = rng.randint(0, 2)
                live_idx = rng.randint(max(shots, 2), 4)
                players.append(
                    PlayerState(
                        seat=s,
                        hand=list(hand),
                        live_round_index=live_idx,
                        shots_taken=shots,
                        alive=True,
                    )
                )
            else:
                players.append(
                    PlayerState(
                        seat=dead_seat,
                        hand=[],
                        live_round_index=1,
                        shots_taken=2,
                        alive=False,
                    )
                )

        # 随机决定谁先手或是否有上家出牌
        leader = rng.choice(list(alive_seats))
        round_state = RoundState(claim_rank=claim_rank, leader_seat=leader)
        history = [
            PublicEvent(turn_index=0, event_type="new_round", seat=None, detail={"round_index": 2, "claim_rank": claim_rank.value})
        ]

        if rng.random() < 0.5 and leader != hero_seat:
            play_count = rng.randint(1, 2)
            prior_cards = tuple(deck[card_ptr : card_ptr + play_count])
            card_ptr += play_count
            round_state.latest_play = PlayedSet(seat=leader, claim_rank=claim_rank, cards=prior_cards)
            history.append(
                PublicEvent(turn_index=1, event_type="play", seat=leader, detail={"count": play_count, "claim_rank": claim_rank.value})
            )
            current_seat = hero_seat
        else:
            current_seat = leader

        return GameState(
            players=players,
            current_seat=current_seat,
            round_state=round_state,
            round_index=2,
            turn_index=len(history),
            public_history=history,
        )

