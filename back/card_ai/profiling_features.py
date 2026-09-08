from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any
import torch

from .features import FeatureEncoder, FEATURE_DIM
from .types import (
    Card, CardKind, PrivateObservation, PublicEvent, Rank, GameAction, PlayAction, ChallengeAction
)
from .pomdp import POMDPParticleFilter
from .opponent_modeling import OnlineOpponentModeler

PROFILING_FEATURE_DIM: int = 104
PRIOR_N0: float = 10.0
PRIOR_MEANS: list[float] = [0.25, 0.50, 0.50, 0.65]


@dataclass
class MatchBehaviorTracker:
    """局内对手行为画像追踪器。

    严格规则约束：
    1. 明确归属于某个观察者 (observer_seat)。只有观察者自己出牌时，才能使用自己的私有手牌记录“我的诈唬”；
       其他玩家未揭示的牌绝不能进入该观察者的“自身诈唬”统计。
    2. 顺序消费 public_history 事件，消费进度随状态分支独立克隆，不重复统计，不跨分支串记录。
    3. 主动质疑需确认非强制 (forced is False)。若缺少 forced 信息，严禁默认当作主动质疑。
    4. 换小轮 (new_round) 保留整场对局的行为累计，新整场对局 (reset) 清空。
    """
    observer_seat: int = 1
    consumed_events: int = 0
    o_active: dict[int, int] = field(default_factory=lambda: {s: 0 for s in range(1, 5)})
    c_active: dict[int, int] = field(default_factory=lambda: {s: 0 for s in range(1, 5)})
    l_caught: dict[int, int] = field(default_factory=lambda: {s: 0 for s in range(1, 5)})
    t_rev: dict[int, int] = field(default_factory=lambda: {s: 0 for s in range(1, 5)})
    l_rev: dict[int, int] = field(default_factory=lambda: {s: 0 for s in range(1, 5)})
    b_my_bluff: dict[int, int] = field(default_factory=lambda: {s: 0 for s in range(1, 5)})
    p_pass: dict[int, int] = field(default_factory=lambda: {s: 0 for s in range(1, 5)})

    _last_play_seat: int | None = None
    _my_pending_bluff_target: int | None = None

    def reset(self) -> None:
        """整场新对局重置：清空全部行为统计与临时状态。"""
        self.consumed_events = 0
        self.o_active = {s: 0 for s in range(1, 5)}
        self.c_active = {s: 0 for s in range(1, 5)}
        self.l_caught = {s: 0 for s in range(1, 5)}
        self.t_rev = {s: 0 for s in range(1, 5)}
        self.l_rev = {s: 0 for s in range(1, 5)}
        self.b_my_bluff = {s: 0 for s in range(1, 5)}
        self.p_pass = {s: 0 for s in range(1, 5)}
        self._last_play_seat = None
        self._my_pending_bluff_target = None

    def clone(self) -> MatchBehaviorTracker:
        """深拷贝追踪器副本，供搜索或 CFR 分支独立演化，互不污染。"""
        t = MatchBehaviorTracker(observer_seat=self.observer_seat)
        t.consumed_events = self.consumed_events
        t.o_active = dict(self.o_active)
        t.c_active = dict(self.c_active)
        t.l_caught = dict(self.l_caught)
        t.t_rev = dict(self.t_rev)
        t.l_rev = dict(self.l_rev)
        t.b_my_bluff = dict(self.b_my_bluff)
        t.p_pass = dict(self.p_pass)
        t._last_play_seat = self._last_play_seat
        t._my_pending_bluff_target = self._my_pending_bluff_target
        return t

    def record_my_play(self, is_bluff: bool, target_opp: int | None) -> None:
        """仅当观察者自己出牌时调用，记录自身私有手牌已知真伪与预期接牌对手。"""
        if is_bluff and target_opp is not None and target_opp != self.observer_seat:
            self._my_pending_bluff_target = target_opp
        else:
            self._my_pending_bluff_target = None

    def update_from_history(self, public_history: list[PublicEvent] | tuple[PublicEvent, ...]) -> None:
        """顺序消费 public_history 中从 consumed_events 开始的全部新增事件。"""
        total_len = len(public_history)
        if total_len <= self.consumed_events:
            return
        for ev in public_history[self.consumed_events:total_len]:
            self._process_single_event(ev)
        self.consumed_events = total_len

    def _process_single_event(self, event: PublicEvent) -> None:
        ev_type = event.event_type
        ev_seat = event.seat
        detail = event.detail or {}

        if ev_type == 'new_round':
            # 换小轮结算：若观察者此前出了诈唬且本小轮平安结束无人质疑，计为放行
            if self._my_pending_bluff_target is not None:
                self.b_my_bluff[self._my_pending_bluff_target] += 1
                self.p_pass[self._my_pending_bluff_target] += 1
                self._my_pending_bluff_target = None
            self._last_play_seat = None

        elif ev_type == 'play':
            # 1. 对手主动质疑机会更新：若有人面对上一手牌选择了出牌，则其放弃了一次主动质疑机会
            if self._last_play_seat is not None and ev_seat is not None and ev_seat != self._last_play_seat:
                self.o_active[ev_seat] += 1

            # 2. 观察者自身诈唬获放行判定：若观察者正有待决诈唬，且有其他存活对手出牌（未发生质疑），说明诈唬成功通过
            if self._my_pending_bluff_target is not None and ev_seat != self.observer_seat:
                self.b_my_bluff[self._my_pending_bluff_target] += 1
                self.p_pass[self._my_pending_bluff_target] += 1
                self._my_pending_bluff_target = None

            self._last_play_seat = ev_seat

        elif ev_type == 'challenge':
            challenger = ev_seat
            challenged = detail.get('challenged_seat', self._last_play_seat)
            outcome = detail.get('outcome')
            # 缺少 forced 信息时，严禁默认主动质疑 (必须 forced is False 才能算主动)
            is_forced = detail.get('forced')

            # 1. 质疑者主动质疑统计 (排除强制质疑和缺少 forced 信息的事件)
            if challenger is not None and is_forced is False:
                self.o_active[challenger] += 1
                self.c_active[challenger] += 1
                # 抓假成功：outcome == 'lie' (鬼牌算反杀不算抓假)
                if outcome == 'lie':
                    self.l_caught[challenger] += 1

            # 2. 被质疑者揭牌统计 (客观公开事实，强制质疑也揭牌)
            if challenged is not None:
                self.t_rev[challenged] += 1
                if outcome == 'lie':
                    self.l_rev[challenged] += 1

            # 3. 观察者自身诈唬被质疑判定：若被质疑者是观察者自己，则诈唬被抓 (尝试+1, 放行不加)
            if self._my_pending_bluff_target is not None:
                if challenged == self.observer_seat:
                    self.b_my_bluff[self._my_pending_bluff_target] += 1
                    self._my_pending_bluff_target = None
                else:
                    # 被质疑的是别人，观察者此前的诈唬实际上早已放行
                    self.b_my_bluff[self._my_pending_bluff_target] += 1
                    self.p_pass[self._my_pending_bluff_target] += 1
                    self._my_pending_bluff_target = None

            self._last_play_seat = None
            self._my_pending_bluff_target = None

        elif ev_type in ('shot', 'player_eliminated', 'player_escaped'):
            # 开枪与淘汰事件，安全消费，不重置上一手出牌
            pass

    def build_from_observation(self, obs: PrivateObservation) -> None:
        """从公开历史重建。注意：公开历史无私有记忆，自身诈唬项保持分母为0（无证据）。"""
        self.reset()
        self.observer_seat = obs.hero_seat
        self.update_from_history(obs.public_history)

    def get_profiling_vector(self, rel_opponents: list) -> list[float]:
        """针对相对座次对手输出 24 维对手画像与证据量特征。"""
        profiling_24: list[float] = []
        N0 = PRIOR_N0
        for opp in rel_opponents:
            s = opp.seat
            # 1. 主动质疑率与证据量
            n1 = self.o_active.get(s, 0)
            c1 = self.c_active.get(s, 0)
            v1 = (c1 + N0 * PRIOR_MEANS[0]) / (n1 + N0)
            e1 = n1 / (n1 + N0)

            # 2. 主动质疑成功率与证据量
            n2 = self.c_active.get(s, 0)
            l2 = self.l_caught.get(s, 0)
            v2 = (l2 + N0 * PRIOR_MEANS[1]) / (n2 + N0)
            e2 = n2 / (n2 + N0)

            # 3. 被质疑后的假牌比例与证据量
            n3 = self.t_rev.get(s, 0)
            l3 = self.l_rev.get(s, 0)
            v3 = (l3 + N0 * PRIOR_MEANS[2]) / (n3 + N0)
            e3 = n3 / (n3 + N0)

            # 4. 我向该对手诈唬的获放行率与证据量
            n4 = self.b_my_bluff.get(s, 0)
            p4 = self.p_pass.get(s, 0)
            v4 = (p4 + N0 * PRIOR_MEANS[3]) / (n4 + N0)
            e4 = n4 / (n4 + N0)

            profiling_24.extend([float(v1), float(e1), float(v2), float(e2), float(v3), float(e3), float(v4), float(e4)])

        while len(profiling_24) < 24:
            profiling_24.extend([0.25, 0.0, 0.50, 0.0, 0.50, 0.0, 0.65, 0.0])

        return profiling_24[:24]


class ProfilingFeatureEncoder:
    def __init__(self, mode: str = 'profiling') -> None:
        self.mode = mode
        self.base_encoder = FeatureEncoder()
        self._cache: dict = {}

    def _relative_players(self, obs: PrivateObservation) -> list:
        hero_seat = obs.hero_seat
        players = list(obs.players)
        players.sort(key=lambda p: (p.seat - hero_seat) % 4)
        return players

    def encode(self, obs: PrivateObservation, tracker: MatchBehaviorTracker | None = None) -> torch.Tensor:
        hand_feat = self.base_encoder._hand_features(obs)
        player_feat = self.base_encoder._player_features(obs)
        round_feat = self.base_encoder._round_features(obs)

        round_start_idx = 0
        for idx, ev in enumerate(obs.public_history):
            if ev.event_type == 'new_round':
                round_start_idx = idx

        obs_curr = PrivateObservation(
            hero_seat=obs.hero_seat,
            hero_hand=obs.hero_hand,
            current_seat=obs.current_seat,
            round_claim_rank=obs.round_claim_rank,
            latest_play_seat=obs.latest_play_seat,
            latest_play_count=obs.latest_play_count,
            players=obs.players,
            public_history=obs.public_history[round_start_idx:],
        )

        rev_counts_curr = {Rank.A: 0, Rank.K: 0, Rank.Q: 0}
        ghost_rev_curr = 0.0
        for ev in obs_curr.public_history:
            if ev.event_type == 'challenge':
                if ev.detail.get('outcome') == 'ghost':
                    ghost_rev_curr = 1.0
                for cl in ev.detail.get('revealed_cards', []):
                    if cl == 'A': rev_counts_curr[Rank.A] += 1
                    elif cl == 'K': rev_counts_curr[Rank.K] += 1
                    elif cl == 'Q': rev_counts_curr[Rank.Q] += 1
        phys_hist_curr = [
            rev_counts_curr[Rank.A] / 6.0,
            rev_counts_curr[Rank.K] / 6.0,
            rev_counts_curr[Rank.Q] / 6.0,
            ghost_rev_curr,
        ]

        rel_players = self._relative_players(obs)
        seed = obs.hero_seat + len(obs.public_history)
        pf_curr = POMDPParticleFilter(num_particles=16, seed=seed)
        res_curr = pf_curr.infer(obs_curr)
        pomdp_curr = []
        if res_curr.last_play:
            pomdp_curr.extend([
                res_curr.last_play.honest_prob,
                res_curr.last_play.bluff_prob,
                res_curr.last_play.ghost_trap_prob,
            ])
        else:
            pomdp_curr.extend([0.33, 0.33, 0.0])
        for p in rel_players:
            est = res_curr.seat_estimates.get(p.seat)
            pomdp_curr.append((est.target_count_mean / 5.0) if est else 0.25)
        for p in rel_players:
            est = res_curr.seat_estimates.get(p.seat)
            pomdp_curr.append(est.has_ghost_prob if est else 0.25)

        turn_idx = obs.public_history[-1].turn_index if obs.public_history else 0
        game_prog = min(1.0, turn_idx / 50.0)

        shot_by_seat = {p.seat: 0 for p in obs.players}
        for ev in obs.public_history:
            if ev.event_type == 'shot' and ev.seat is not None:
                shot_by_seat[ev.seat] += 1
        actual_shot_feats = [shot_by_seat.get(p.seat, 0) / 5.0 for p in rel_players]

        neutral_chal_feats = [0.0] * 4
        neutral_revealed_rates = [0.0, 0.0, 0.0]
        neutral_opp_aggr = [0.35, 0.35, 0.35]

        hero_p = next((p for p in obs.players if p.seat == obs.hero_seat), None)
        shots = hero_p.shots_taken if hero_p else 0
        hero_hazard = 1.0 / max(1, 5 - shots)
        min_cards = min((p.hand_count for p in obs.players if p.alive and p.seat != obs.hero_seat), default=5)
        escape_pres = 1.0 if min_cards <= 1 else (0.5 if min_cards == 2 else 0.0)
        urgency = [hero_hazard, escape_pres]

        base_80 = (
            hand_feat
            + player_feat
            + round_feat
            + [game_prog]
            + phys_hist_curr
            + neutral_chal_feats
            + actual_shot_feats
            + neutral_revealed_rates
            + pomdp_curr
            + neutral_opp_aggr
            + urgency
        )

        rel_opponents = [p for p in rel_players if p.seat != obs.hero_seat][:3]
        profiling_24: list[float] = []

        if self.mode == 'control':
            for _ in rel_opponents:
                profiling_24.extend([
                    PRIOR_MEANS[0], 0.0,
                    PRIOR_MEANS[1], 0.0,
                    PRIOR_MEANS[2], 0.0,
                    PRIOR_MEANS[3], 0.0,
                ])
            while len(profiling_24) < 24:
                profiling_24.extend([0.25, 0.0, 0.50, 0.0, 0.50, 0.0, 0.65, 0.0])
        else:
            if tracker is None:
                # 训练人工残局起点或无传入追踪器时：基于公开历史重建公开统计，自身诈唬保持分母0（无证据）
                tracker = MatchBehaviorTracker(observer_seat=obs.hero_seat)
                tracker.update_from_history(obs.public_history)
            else:
                # 确保当前消费进度与最新公开事件同步
                if len(obs.public_history) > tracker.consumed_events:
                    tracker.update_from_history(obs.public_history)

            profiling_24 = tracker.get_profiling_vector(rel_opponents)

        total_104 = base_80 + profiling_24[:24]
        return torch.tensor(total_104, dtype=torch.float32)
