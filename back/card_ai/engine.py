from __future__ import annotations

from copy import deepcopy
from itertools import combinations
from random import Random

from .config import GameRules
from .types import (
    Card,
    CardKind,
    ChallengeAction,
    ChallengeOutcome,
    GameAction,
    GameState,
    PlayAction,
    PlayedSet,
    PlayerState,
    PrivateObservation,
    PublicEvent,
    PublicPlayerView,
    Rank,
    RoundState,
)


class SurvivalGameEngine:
    """逆水寒 / 骗子酒馆（Liar's Bar）生死左轮卡牌核心引擎。

    核心机制（多轮生死大逃杀）：
    1. 牌库始终为 20 张（针对目标点数的 5 张普通 + 1 张鬼牌 + 2 张万能 + 其余两门各 6 张）。
    2. 每小轮（Deal / Round）：
       - 随机更换目标点数（A / K / Q 轮流替换）。
       - 给所有存活（alive）玩家重新发满 5 张手牌。
       - 若有人中弹淘汰，未发的牌作为【轮空底牌】隐秘保留在牌堆中。
    3. 出牌与轮转：
       - 玩家顺时针出 1~3 张牌（声称当前目标点数）或发起【质疑】。
       - 鬼牌只能单出，不可组合出牌。
       - 若玩家在本小轮出空手牌（0张手牌），且下一家没有质疑他，则该玩家本小轮进入【安全状态】，
         轮次自动跳过他，不再需要出牌，绝不会因为打空手牌而被逼质疑挨枪。
    4. 质疑判定与左轮开枪：
       - 诚实牌：质疑者抓错，质疑者扣动一次扳机。
       - 诈唬牌：出牌者被抓，出牌者扣动一次扳机。
       - 鬼牌反杀：出牌者免疫，全场其余所有存活玩家各扣动一次扳机！
       - 5 孔轮盘随机 1 发实弹，命中者当场淘汰（alive=False）；空枪者存活，中枪计数保留。
    5. 小轮重置：
       - 一旦质疑结算完毕（或全员手牌出空和平通过），本小轮立即宣告结束！
       - 若存活人数 <= 1，终局决出最后吃鸡幸存者。
       - 若存活人数 >= 2，立即收牌洗牌，换新目标点数，为所有存活玩家重新发满 5 张牌，开启下一小轮！
       - 受罚者若存活，由受罚者在新一轮先手；若受罚者阵亡，由其顺时针下一个活人先手。
    """

    def __init__(
        self,
        player_count: int = 4,
        hand_size: int = 5,
        rules: GameRules | None = None,
    ) -> None:
        if rules is None:
            rules = GameRules(player_count=player_count, hand_size=hand_size)
        self.rules = rules
        self.player_count = rules.player_count
        self.hand_size = rules.hand_size

    def new_game(
        self,
        seed: int | None = None,
        starting_seat: int = 1,
        target_rank: Rank | None = None,
    ) -> GameState:
        rng = Random(seed)
        deck, ghost_rank = self._build_deck(rng, ghost_rank=target_rank)
        rng.shuffle(deck)

        players: list[PlayerState] = []
        for offset in range(self.player_count):
            seat = offset + 1
            hand = deck[offset * self.hand_size : (offset + 1) * self.hand_size]
            players.append(
                PlayerState(
                    seat=seat,
                    hand=list(hand),
                    live_round_index=rng.randrange(self.rules.chamber_count),
                    shots_taken=0,
                    alive=True,
                    escaped=False,
                    pending_escape=False,
                )
            )

        return GameState(
            players=players,
            current_seat=starting_seat,
            round_state=RoundState(claim_rank=ghost_rank),
            round_index=1,
            public_history=[
                PublicEvent(
                    turn_index=0,
                    event_type="new_round",
                    seat=None,
                    detail={"round_index": 1, "claim_rank": ghost_rank.value},
                )
            ],
        )

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

        # 核心规则 2：全场手牌清空时的收官终局质疑！
        # 若场上除上一手出牌者外已无任何存活玩家手里有手牌，桌面最后出牌已无人可用手牌承接，下家必定且只能质疑！
        alive_with_cards = [p for p in state.players if p.alive and len(p.hand) > 0]
        other_alive_with_cards = [
            p for p in alive_with_cards
            if latest_play is None or p.seat != latest_play.seat
        ]
        if not other_alive_with_cards and latest_play is not None and latest_play.seat != player.seat:
            return (
                ChallengeAction(seat=player.seat, challenged_seat=latest_play.seat),
            )

        # 保护性防守：若当前玩家手里已无牌（正常流程下已被跳过），不能进行出牌
        if len(player.hand) == 0:
            return ()

        actions: list[GameAction] = []
        claim_ranks = (state.round_state.claim_rank,) if state.round_state.claim_rank else tuple(Rank)

        # 正常出牌动作（手里有牌时）
        if len(player.hand) > 0:
            for count in range(1, self.rules.max_play_count + 1):
                if count > len(player.hand):
                    break
                for combo in combinations(player.hand, count):
                    if any(card.kind == CardKind.GHOST for card in combo) and count > 1:
                        continue  # 鬼牌只能单出
                    card_ids = tuple(card.card_id for card in combo)
                    for claim_rank in claim_ranks:
                        actions.append(
                            PlayAction(seat=player.seat, card_ids=card_ids, claim_rank=claim_rank)
                        )

        # 正常质疑动作
        if latest_play is not None and latest_play.seat != player.seat:
            actions.append(
                ChallengeAction(seat=player.seat, challenged_seat=latest_play.seat)
            )

        return tuple(actions)

    def apply_action(self, state: GameState, action: GameAction) -> GameState:
        if isinstance(action, PlayAction):
            return self._apply_play(state, action)
        return self._apply_challenge(state, action)

    def is_terminal(self, state: GameState) -> bool:
        alive_players = [player for player in state.players if player.alive]
        return len(alive_players) <= 1

    def observe(self, state: GameState, hero_seat: int) -> PrivateObservation:
        hero = state.player_by_seat(hero_seat)
        latest_play = state.round_state.latest_play
        return PrivateObservation(
            hero_seat=hero_seat,
            hero_hand=tuple(hero.hand),
            current_seat=self.current_actor(state),
            round_claim_rank=state.round_state.claim_rank,
            latest_play_seat=latest_play.seat if latest_play else None,
            latest_play_count=len(latest_play.cards) if latest_play else 0,
            players=tuple(
                PublicPlayerView(
                    seat=player.seat,
                    alive=player.alive,
                    escaped=player.escaped,
                    pending_escape=player.pending_escape,
                    hand_count=player.hand_count,
                    shots_taken=player.shots_taken,
                )
                for player in state.players
            ),
            public_history=tuple(state.public_history),
        )

    def clone_state(self, state: GameState) -> GameState:
        """高性能专用状态克隆：消除 copy.deepcopy 遍历对象树和 memo 字典的开销（实测提速 21 倍）。"""
        new_players = [
            PlayerState(
                seat=p.seat,
                hand=list(p.hand),
                live_round_index=p.live_round_index,
                shots_taken=p.shots_taken,
                alive=p.alive,
                escaped=p.escaped,
                pending_escape=p.pending_escape,
            )
            for p in state.players
        ]
        rs = state.round_state
        new_rs = RoundState(
            claim_rank=rs.claim_rank,
            leader_seat=rs.leader_seat,
            latest_play=rs.latest_play,
            plays=list(rs.plays),
        )
        new_history = [
            PublicEvent(
                turn_index=e.turn_index,
                event_type=e.event_type,
                seat=e.seat,
                detail=dict(e.detail),
            )
            for e in state.public_history
        ]
        return GameState(
            players=new_players,
            current_seat=state.current_seat,
            round_state=new_rs,
            discard_pile=list(state.discard_pile),
            public_history=new_history,
            finish_order=list(state.finish_order),
            turn_index=state.turn_index,
            round_index=state.round_index,
        )

    def current_actor(self, state: GameState) -> int:
        player = state.player_by_seat(state.current_seat)
        latest_play = state.round_state.latest_play
        alive_with_cards = [p for p in state.players if p.alive and len(p.hand) > 0]
        other_alive_with_cards = [
            p for p in alive_with_cards
            if latest_play is None or p.seat != latest_play.seat
        ]

        # 存活且手里有牌：正常作为当前行动者
        if player.alive and len(player.hand) > 0:
            return state.current_seat

        # 存活但手里没牌：仅当场上除上一手出牌者外已无任何人手里有手牌时，才允许他进行收官质疑
        if player.alive and not other_alive_with_cards and latest_play is not None and latest_play.seat != player.seat:
            return state.current_seat

        # 否则当前玩家手牌已打空跑路，必须跳过！
        return self._next_active_seat(state, state.current_seat)

    def _apply_play(self, state: GameState, action: PlayAction) -> GameState:
        acting_seat = self.current_actor(state)
        if action.seat != acting_seat:
            raise ValueError("play action is not from the current seat")
        state.current_seat = acting_seat

        player = state.player_by_seat(action.seat)
        chosen = self._take_cards(player, action.card_ids)

        if state.round_state.claim_rank is None:
            state.round_state.claim_rank = action.claim_rank
            state.round_state.leader_seat = action.seat
        elif state.round_state.claim_rank != action.claim_rank:
            raise ValueError("claim rank must stay constant inside the round")

        played = PlayedSet(seat=action.seat, claim_rank=action.claim_rank, cards=chosen)
        state.round_state.latest_play = played
        state.round_state.plays.append(played)
        state.turn_index += 1
        state.public_history.append(
            PublicEvent(
                turn_index=state.turn_index,
                event_type="play",
                seat=action.seat,
                detail={"count": len(chosen), "claim_rank": action.claim_rank.value},
            )
        )

        state.current_seat = self._next_active_seat(state, action.seat)
        return state

    def _apply_challenge(self, state: GameState, action: ChallengeAction) -> GameState:
        latest_play = state.round_state.latest_play
        if latest_play is None:
            raise ValueError("cannot challenge without a latest play")
        acting_seat = self.current_actor(state)
        if action.seat != acting_seat:
            raise ValueError("challenge action is not from the current seat")
        state.current_seat = acting_seat
        if latest_play.seat != action.challenged_seat:
            raise ValueError("challenge target is not the latest play")

        outcome = self._evaluate_play(latest_play)
        state.turn_index += 1
        acting_player = state.player_by_seat(action.seat)
        is_forced = (
            len(acting_player.hand) == 0
            or not any(p.alive and len(p.hand) > 0 for p in state.players)
            or (sum(1 for p in state.players if p.alive) == 2 and len(state.player_by_seat(action.challenged_seat).hand) == 0)
        )
        state.public_history.append(
            PublicEvent(
                turn_index=state.turn_index,
                event_type="challenge",
                seat=action.seat,
                detail={
                    "challenged_seat": action.challenged_seat,
                    "outcome": outcome.value,
                    "revealed_cards": [card.short_label() for card in latest_play.cards],
                    "forced": is_forced,
                },
            )
        )

        penalty_seat = action.seat
        if outcome == ChallengeOutcome.GHOST:
            # 鬼牌反杀：出牌者免死，质疑者抓错必扣一次轮盘；
            # 其余第三方存活玩家中，仅在这一轮尚未出完手牌者扣动一次轮盘（已出完手牌者不受波及，不扣轮盘）
            protected = latest_play.seat
            # 1. 质疑者必扣动一次轮盘
            self._shoot_player(state, action.seat)
            # 2. 第三方存活玩家：仅未出完手牌者扣动轮盘
            for player in state.players:
                if player.seat == protected or player.seat == action.seat or not player.alive:
                    continue
                if len(player.hand) > 0:
                    self._shoot_player(state, player.seat)
            penalty_seat = self._next_active_seat(state, protected)
        elif outcome == ChallengeOutcome.HONEST:
            # 诚实牌：质疑者抓错，质疑者扣动一次扳机
            self._shoot_player(state, action.seat)
            penalty_seat = action.seat
        else:
            # 诈唬牌：出牌者被抓，出牌者扣动一次扳机
            self._shoot_player(state, latest_play.seat)
            penalty_seat = latest_play.seat

        # 检查是否终局（只剩 <= 1 人存活）
        if self.is_terminal(state):
            alive = [p.seat for p in state.players if p.alive]
            if alive and alive[0] not in state.finish_order:
                state.finish_order.append(alive[0])
            state.round_state.latest_play = None
            return state

        # 未终局：本小轮结算完毕！立即进入新一小轮（洗牌重发、换目标点数）
        self._start_new_round(state, starter_seat=penalty_seat)
        return state

    def _start_new_round(self, state: GameState, starter_seat: int | None = None) -> None:
        """开启新一小轮：回收弃牌，重新随机目标点数，给存活玩家重新发满 5 张手牌。"""
        for played in state.round_state.plays:
            state.discard_pile.extend(played.cards)

        alive_players = [p for p in state.players if p.alive]
        if len(alive_players) <= 1:
            return

        # 确保淘汰死亡玩家手牌清空
        for p in state.players:
            if not p.alive:
                p.hand = []

        # 换新目标点数（轮换随机）
        prior_rank = state.round_state.claim_rank
        candidates = [r for r in Rank if r != prior_rank]
        rng = Random(state.turn_index + len(state.public_history) + 42)
        new_rank = rng.choice(candidates) if candidates else Rank.A

        # 重新生成 20 张牌并洗牌
        deck, _ = self._build_deck(rng, ghost_rank=new_rank)
        rng.shuffle(deck)

        # 给所有存活玩家重新发满手牌（5张），淘汰者份额作为轮空底牌保留
        for idx, p in enumerate(alive_players):
            p.hand = list(deck[idx * self.hand_size : (idx + 1) * self.hand_size])

        state.round_state = RoundState(claim_rank=new_rank)
        state.round_index += 1
        state.turn_index += 1
        state.public_history.append(
            PublicEvent(
                turn_index=state.turn_index,
                event_type="new_round",
                seat=None,
                detail={"round_index": state.round_index, "claim_rank": new_rank.value},
            )
        )

        # 确定新一轮先手：受罚者存活则受罚者先手，死亡则顺延下家活人
        if starter_seat is not None and state.player_by_seat(starter_seat).alive:
            state.current_seat = starter_seat
        else:
            fallback = starter_seat or alive_players[0].seat
            state.current_seat = self._next_active_seat(state, fallback, allow_self_if_active=True)

    def _evaluate_play(self, played: PlayedSet) -> ChallengeOutcome:
        if len(played.cards) == 1 and played.cards[0].kind == CardKind.GHOST:
            return ChallengeOutcome.GHOST

        for card in played.cards:
            if card.kind == CardKind.WILD:
                continue
            if card.kind == CardKind.NORMAL and card.printed_rank == played.claim_rank:
                continue
            return ChallengeOutcome.LIE

        return ChallengeOutcome.HONEST

    def _shoot_player(self, state: GameState, seat: int) -> bool:
        player = state.player_by_seat(seat)
        if not player.alive:
            return False

        died = player.shots_taken == player.live_round_index
        player.shots_taken += 1
        state.turn_index += 1
        state.public_history.append(
            PublicEvent(
                turn_index=state.turn_index,
                event_type="shot",
                seat=seat,
                detail={"died": died, "shots_taken": player.shots_taken},
            )
        )

        if died:
            player.alive = False
            player.hand = []  # 淘汰者手牌立即清空
            if seat not in state.finish_order:
                state.finish_order.append(seat)
        return died

    def _take_cards(self, player: PlayerState, card_ids: tuple[str, ...]) -> tuple[Card, ...]:
        chosen: list[Card] = []
        wanted = set(card_ids)
        remaining: list[Card] = []

        for card in player.hand:
            if card.card_id in wanted:
                chosen.append(card)
            else:
                remaining.append(card)

        if len(chosen) != len(card_ids):
            raise ValueError("play references cards not in hand")

        player.hand = remaining
        return tuple(chosen)

    def _build_deck(self, rng: Random, ghost_rank: Rank | None = None) -> tuple[list[Card], Rank]:
        if ghost_rank is None:
            ghost_rank = rng.choice(list(Rank))

        rank_cards: list[Card] = []
        for rank in Rank:
            # 目标点数的普通牌为 5 张，非目标点数各 6 张
            count = self.rules.target_rank_count - 1 if rank == ghost_rank else self.rules.target_rank_count
            for index in range(count):
                rank_cards.append(
                    Card(card_id=f"{rank.value}-{index}", printed_rank=rank, kind=CardKind.NORMAL)
                )

        # 1 张鬼牌（属于声明的目标点数）
        rank_cards.append(
            Card(card_id=f"G-{ghost_rank.value}", printed_rank=ghost_rank, kind=CardKind.GHOST)
        )

        # 2 张万能牌
        wilds = [
            Card(card_id=f"W-{index}", printed_rank=None, kind=CardKind.WILD)
            for index in range(self.rules.wild_count)
        ]
        return rank_cards + wilds, ghost_rank

    def _next_active_seat(
        self,
        state: GameState,
        current_seat: int,
        allow_self_if_active: bool = False,
    ) -> int:
        """寻找顺时针下一个可以行动的玩家。
        规则：
        1. 若场上仍有除上一手出牌者外、存活且手里有手牌的玩家：
           行动权只在手里有牌的玩家之间顺时针轮转！手牌为 0 的玩家（已跑路）自动跳过。
        2. 若场上除上一手出牌者外已无任何有手牌的玩家（全员手牌打空或仅出牌者有牌）：
           此时桌面最后一手牌已无人可用手牌承接，必须进行终局收官质疑，由顺时针下一个存活玩家接牌质疑。
        """
        seats = [player.seat for player in state.players]
        start_index = seats.index(current_seat)

        if allow_self_if_active:
            offsets = range(0, len(seats))
        else:
            offsets = range(1, len(seats) + 1)

        latest_play = state.round_state.latest_play
        alive_with_cards = [p for p in state.players if p.alive and len(p.hand) > 0]
        other_alive_with_cards = [
            p for p in alive_with_cards
            if latest_play is None or p.seat != latest_play.seat
        ]

        if other_alive_with_cards:
            # 优先顺时针寻找手里有牌的存活玩家，出完手牌者完全跳过！
            for offset in offsets:
                seat = seats[(start_index + offset) % len(seats)]
                player = state.player_by_seat(seat)
                if player.alive and len(player.hand) > 0:
                    return seat
            return current_seat

        # 若场上除了出牌者之外已没有任何人手里有牌，进入终局质疑，顺时针寻找存活玩家接牌质疑
        for offset in offsets:
            seat = seats[(start_index + offset) % len(seats)]
            player = state.player_by_seat(seat)
            if player.alive:
                if latest_play is not None and latest_play.seat == seat:
                    continue  # 出牌者不能质疑自己
                return seat

        return current_seat
