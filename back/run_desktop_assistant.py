#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
逆水寒·小丑牌 桌面原生置顶悬浮窗 AI 决策辅助系统 (Windows 原生 Tkinter 版)

功能架构：
1. 纯 Windows 原生 GUI：基于内置 tkinter + ttk，纯暗黑现代化电竞/辅助风格，严禁 HTML / Electron / Webview；
2. 伸缩双形态置顶悬浮窗：
   - 微型胶囊条 (Mini Bar) 模式 (~340x52)：角落极简悬浮，实时高亮 AI 核心决策与概率，支持拖拽与一键展开；
   - 完整控制面板 (Full Panel) 模式 (~420x680)：开局设置、方案B手牌输入、4席位状态与挂机开关、局势动作记录、AI深度概率推演；
3. AI 决策与后端模型桥接：
   - 自动加载生产主力 C_long (runs_ppo/v29_long_ppo/checkpoint_iter_00120.pt)；
   - 支持动态下拉热切换 v37 争胜模型等备选权重；
   - 采用 ProfilingFeatureEncoder(mode="control") 提取 104 维特征；
   - 桥接 ContextualActionPredictor 画像预测器与 POMDP 粒子滤波，推演上家虚报/诈唬率；
   - 神经网络 masked softmax 实时计算合法动作分布并输出最优建议；
4. 自动化测试支持：提供 --test 命令行模式，可在无屏幕环境下自动验证全流程功能。
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass, field
from itertools import combinations
import json
import os
from pathlib import Path
import random
import sys
from typing import Any, Callable

# 解决 Windows 终端编码问题
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# 确保工作目录与模块导入路径
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
os.chdir(ROOT_DIR)

import torch
import tkinter as tk
from tkinter import ttk, messagebox

# 导入底层小丑牌 AI 引擎与类型
from card_ai.abstractions import ActionAbstractor, action_to_index
from card_ai.contextual_profiler import ContextualActionPredictor
from card_ai.ppo_networks import PPOActor
from card_ai.profiling_features import ProfilingFeatureEncoder
from card_ai.types import (
    Card,
    CardKind,
    ChallengeAction,
    GameAction,
    PlayAction,
    PrivateObservation,
    PublicEvent,
    PublicPlayerView,
    Rank,
)


# ===========================================================================
# 1. 局势状态跟踪器 (GameStateTracker)
# ===========================================================================

@dataclass
class Snapshot:
    """局势历史快照，用于支持无限制撤销 (Undo)。"""
    hero_seat: int
    player_count: int
    claim_rank: Rank
    round_index: int
    turn_index: int
    current_seat: int
    hero_target_count: int
    hero_nontarget_count: int
    hero_has_ghost: bool
    seat_hand_counts: dict[int, int]
    player_alive: dict[int, bool]
    player_afk: dict[int, bool]
    shots_taken: dict[int, int]
    latest_play: dict[str, Any] | None
    public_history: list[PublicEvent]
    pending_penalty_seat: int | None
    round_starter_seat: int
    hero_exact_hand: tuple[Card, ...] | None = None


class GameStateTracker:
    """内部局势状态跟踪器：严格维护多轮小丑牌对局物理规则与信息集。"""

    def __init__(
        self,
        hero_seat: int = 1,
        player_count: int = 4,
        claim_rank: Rank = Rank.A,
        starting_seat: int = 1,
    ) -> None:
        self.hero_seat = hero_seat
        self.player_count = player_count
        self.claim_rank = claim_rank
        self.starting_seat = starting_seat
        self.round_starter_seat = starting_seat

        self.round_index = 1
        self.turn_index = 0
        self.current_seat = starting_seat

        # 方案B手牌：主角私有手牌
        self.hero_target_count = 2
        self.hero_nontarget_count = 3
        self.hero_has_ghost = False
        self.hero_exact_hand: tuple[Card, ...] | None = None

        # 各席位估算状态
        self.seat_hand_counts: dict[int, int] = {s: 5 for s in range(1, 5)}
        self.player_alive: dict[int, bool] = {s: (s <= player_count) for s in range(1, 5)}
        self.player_afk: dict[int, bool] = {s: False for s in range(1, 5)}
        self.shots_taken: dict[int, int] = {s: 0 for s in range(1, 5)}

        # 上一手出牌记录
        self.latest_play: dict[str, Any] | None = None
        self.pending_penalty_seat: int | None = None

        # 公开事件历史
        self.public_history: list[PublicEvent] = [
            PublicEvent(
                turn_index=0,
                event_type="new_round",
                seat=None,
                detail={"round_index": 1, "claim_rank": claim_rank.value},
            )
        ]

        # 撤销栈
        self.undo_stack: list[Snapshot] = []

    def save_snapshot(self) -> None:
        """保存当前局势快照以供撤销。"""
        snap = Snapshot(
            hero_seat=self.hero_seat,
            player_count=self.player_count,
            claim_rank=self.claim_rank,
            round_index=self.round_index,
            turn_index=self.turn_index,
            current_seat=self.current_seat,
            hero_target_count=self.hero_target_count,
            hero_nontarget_count=self.hero_nontarget_count,
            hero_has_ghost=self.hero_has_ghost,
            seat_hand_counts=dict(self.seat_hand_counts),
            player_alive=dict(self.player_alive),
            player_afk=dict(self.player_afk),
            shots_taken=dict(self.shots_taken),
            latest_play=copy.deepcopy(self.latest_play),
            public_history=copy.deepcopy(self.public_history),
            pending_penalty_seat=self.pending_penalty_seat,
            round_starter_seat=self.round_starter_seat,
            hero_exact_hand=self.hero_exact_hand,
        )
        self.undo_stack.append(snap)
        if len(self.undo_stack) > 50:
            self.undo_stack.pop(0)

    def undo(self) -> bool:
        """撤销上一步操作。"""
        if not self.undo_stack:
            return False
        snap = self.undo_stack.pop()
        self.hero_seat = snap.hero_seat
        self.player_count = snap.player_count
        self.claim_rank = snap.claim_rank
        self.round_index = snap.round_index
        self.turn_index = snap.turn_index
        self.current_seat = snap.current_seat
        self.hero_target_count = snap.hero_target_count
        self.hero_nontarget_count = snap.hero_nontarget_count
        self.hero_has_ghost = snap.hero_has_ghost
        self.hero_exact_hand = snap.hero_exact_hand
        self.seat_hand_counts = snap.seat_hand_counts
        self.player_alive = snap.player_alive
        self.player_afk = snap.player_afk
        self.shots_taken = snap.shots_taken
        self.latest_play = snap.latest_play
        self.public_history = snap.public_history
        self.pending_penalty_seat = snap.pending_penalty_seat
        self.round_starter_seat = snap.round_starter_seat
        return True

    def reset_game(
        self,
        hero_seat: int | None = None,
        claim_rank: Rank | None = None,
        starting_seat: int | None = None,
        player_count: int | None = None,
    ) -> None:
        """整场对局重新开局。"""
        self.save_snapshot()
        if hero_seat is not None:
            self.hero_seat = hero_seat
        if claim_rank is not None:
            self.claim_rank = claim_rank
        if player_count is not None:
            self.player_count = player_count
        if starting_seat is not None:
            self.starting_seat = starting_seat
            self.round_starter_seat = starting_seat

        self.round_index = 1
        self.turn_index = 0
        self.current_seat = self.starting_seat

        self.hero_target_count = 2
        self.hero_nontarget_count = 3
        self.hero_has_ghost = False

        self.seat_hand_counts = {s: 5 for s in range(1, 5)}
        self.player_alive = {s: (s <= self.player_count) for s in range(1, 5)}
        self.shots_taken = {s: 0 for s in range(1, 5)}
        self.latest_play = None
        self.pending_penalty_seat = None

        self.public_history = [
            PublicEvent(
                turn_index=0,
                event_type="new_round",
                seat=None,
                detail={"round_index": 1, "claim_rank": self.claim_rank.value},
            )
        ]

    def reset_round(self, new_claim_rank: Rank | None = None, starter_seat: int | None = None) -> None:
        """开启新一小轮：存活玩家手牌重置为5张，重新发牌。"""
        self.save_snapshot()
        if new_claim_rank is not None:
            self.claim_rank = new_claim_rank

        self.round_index += 1
        self.turn_index += 1
        self.latest_play = None
        self.pending_penalty_seat = None

        # 存活且未离场的玩家手牌重新补满 5 张
        for s in range(1, 5):
            if self.player_alive[s]:
                self.seat_hand_counts[s] = 5
            else:
                self.seat_hand_counts[s] = 0

        # 主角默认初始手牌分配
        self.hero_target_count = 2
        self.hero_nontarget_count = 3
        self.hero_has_ghost = False

        if starter_seat is not None and self.is_seat_active(starter_seat):
            self.current_seat = starter_seat
        else:
            self.current_seat = self.get_next_active_seat(self.current_seat, allow_self=True)
        self.round_starter_seat = self.current_seat

        self.public_history.append(
            PublicEvent(
                turn_index=self.turn_index,
                event_type="new_round",
                seat=None,
                detail={"round_index": self.round_index, "claim_rank": self.claim_rank.value},
            )
        )

    def is_seat_active(self, seat: int) -> bool:
        """判断席位是否仍存活参与（存活且在总人数范围内，挂机不影响物理合法性）。"""
        if seat > self.player_count:
            return False
        return self.player_alive.get(seat, False)

    def get_next_active_seat(self, from_seat: int, allow_self: bool = False) -> int:
        """顺时针寻找下一个存活且有手牌的玩家（严格遵循真实物理规则）。"""
        all_seats = [s for s in range(1, self.player_count + 1)]
        if not all_seats:
            return from_seat

        start_idx = (from_seat - 1) % len(all_seats)
        offsets = range(0 if allow_self else 1, len(all_seats) + (1 if allow_self else 0))

        # 1. 优先寻找有手牌的存活玩家
        for off in offsets:
            s = all_seats[(start_idx + off) % len(all_seats)]
            if self.is_seat_active(s) and self.seat_hand_counts.get(s, 0) > 0:
                return s

        # 2. 次选：两人局/终局收官质疑接盘（存活玩家）
        for off in offsets:
            s = all_seats[(start_idx + off) % len(all_seats)]
            if self.is_seat_active(s):
                if self.latest_play and self.latest_play.get("seat") == s:
                    continue  # 出牌者自己不能接盘质疑自己
                return s

        return from_seat

    def toggle_afk(self, seat: int) -> bool:
        """切换某席位的挂机标记（仅用于战术偏好预估提示，绝对不篡改物理轮转与合法规则）。"""
        self.save_snapshot()
        self.player_afk[seat] = not self.player_afk.get(seat, False)
        return self.player_afk[seat]

    def set_hero_hand(
        self,
        target_count: int | None = None,
        nontarget_count: int | None = None,
        has_ghost: bool | None = None,
        exact_hand: tuple[Card, ...] | list[Card] | None = None,
    ) -> None:
        """更新主角私有手牌计数（方案B）或设置精确手牌（回放与高保真模式）。"""
        self.save_snapshot()
        if exact_hand is not None:
            self.hero_exact_hand = tuple(exact_hand)
            self.hero_target_count = sum(
                1 for c in exact_hand
                if c.kind != CardKind.GHOST and (c.printed_rank == self.claim_rank or c.kind == CardKind.WILD)
            )
            self.hero_nontarget_count = sum(
                1 for c in exact_hand
                if c.kind == CardKind.NORMAL and c.printed_rank != self.claim_rank
            )
            self.hero_has_ghost = any(c.kind == CardKind.GHOST for c in exact_hand)
        else:
            self.hero_exact_hand = None
            if target_count is not None:
                self.hero_target_count = max(0, min(5, target_count))
            if nontarget_count is not None:
                self.hero_nontarget_count = max(0, min(5, nontarget_count))
            if has_ghost is not None:
                self.hero_has_ghost = bool(has_ghost)

        total = self.hero_target_count + self.hero_nontarget_count + (1 if self.hero_has_ghost else 0)
        self.seat_hand_counts[self.hero_seat] = total

    def record_play(
        self,
        seat: int,
        count: int,
        hero_used_target: int = 0,
        hero_used_nontarget: int = 0,
        hero_used_ghost: bool = False,
        hero_played_card_ids: tuple[str, ...] | list[str] | None = None,
    ) -> None:
        """记录出牌动作。"""
        self.save_snapshot()
        count = max(1, min(3, count))
        self.turn_index += 1

        # 若是主角出牌，扣减主角的具体手牌
        if seat == self.hero_seat:
            if self.hero_exact_hand is not None:
                curr_hand = list(self.hero_exact_hand)
                if hero_played_card_ids:
                    curr_hand = [c for c in curr_hand if c.card_id not in hero_played_card_ids]
                elif hero_used_target > 0 or hero_used_nontarget > 0 or hero_used_ghost:
                    if hero_used_ghost:
                        g_card = next((c for c in curr_hand if c.kind == CardKind.GHOST), None)
                        if g_card:
                            curr_hand.remove(g_card)
                    for _ in range(hero_used_target):
                        t_card = next((c for c in curr_hand if c.kind != CardKind.GHOST and (c.printed_rank == self.claim_rank or c.kind == CardKind.WILD)), None)
                        if t_card:
                            curr_hand.remove(t_card)
                    for _ in range(hero_used_nontarget):
                        nt_card = next((c for c in curr_hand if c.kind == CardKind.NORMAL and c.printed_rank != self.claim_rank), None)
                        if nt_card:
                            curr_hand.remove(nt_card)
                else:
                    curr_hand = curr_hand[count:]
                self.hero_exact_hand = tuple(curr_hand)
                self.hero_target_count = sum(
                    1 for c in curr_hand
                    if c.kind != CardKind.GHOST and (c.printed_rank == self.claim_rank or c.kind == CardKind.WILD)
                )
                self.hero_nontarget_count = sum(
                    1 for c in curr_hand
                    if c.kind == CardKind.NORMAL and c.printed_rank != self.claim_rank
                )
                self.hero_has_ghost = any(c.kind == CardKind.GHOST for c in curr_hand)
            elif hero_used_target > 0 or hero_used_nontarget > 0 or hero_used_ghost:
                self.hero_target_count = max(0, self.hero_target_count - hero_used_target)
                self.hero_nontarget_count = max(0, self.hero_nontarget_count - hero_used_nontarget)
                if hero_used_ghost:
                    self.hero_has_ghost = False
            else:
                # 默认扣减逻辑：优先扣目标牌，不足扣杂牌
                rem = count
                take_t = min(rem, self.hero_target_count)
                self.hero_target_count -= take_t
                rem -= take_t
                take_n = min(rem, self.hero_nontarget_count)
                self.hero_nontarget_count -= take_n
                rem -= take_n
                if rem > 0 and self.hero_has_ghost:
                    self.hero_has_ghost = False

            total = self.hero_target_count + self.hero_nontarget_count + (1 if self.hero_has_ghost else 0)
            self.seat_hand_counts[self.hero_seat] = total
        else:
            self.seat_hand_counts[seat] = max(0, self.seat_hand_counts.get(seat, 5) - count)

        self.latest_play = {
            "seat": seat,
            "count": count,
            "claim_rank": self.claim_rank,
        }

        self.public_history.append(
            PublicEvent(
                turn_index=self.turn_index,
                event_type="play",
                seat=seat,
                detail={"count": count, "claim_rank": self.claim_rank.value},
            )
        )

        # 顺延到下一个行动者
        self.current_seat = self.get_next_active_seat(seat, allow_self=False)

    def record_challenge(self, challenging_seat: int, outcome: str) -> int:
        """记录质疑动作与验牌结果。
        
        outcome: 'honest' (真牌, 质疑者输), 'lie' (假牌, 出牌者输), 'ghost' (鬼牌, 质疑者输)
        返回受罚需要扣动扳机的席位。
        """
        self.save_snapshot()
        self.turn_index += 1
        challenged_seat = self.latest_play["seat"] if self.latest_play else (challenging_seat % self.player_count + 1)

        revealed_cards = [self.claim_rank.value] * (self.latest_play["count"] if self.latest_play else 1)
        if outcome == "lie":
            penalty_seat = challenged_seat
            revealed_cards = ["K" if self.claim_rank == Rank.A else "A"] * len(revealed_cards)
        elif outcome == "ghost":
            penalty_seat = challenging_seat
            revealed_cards = ["G"]
        else:  # 'honest'
            penalty_seat = challenging_seat

        self.pending_penalty_seat = penalty_seat

        self.public_history.append(
            PublicEvent(
                turn_index=self.turn_index,
                event_type="challenge",
                seat=challenging_seat,
                detail={
                    "challenged_seat": challenged_seat,
                    "outcome": outcome,
                    "revealed_cards": revealed_cards,
                },
            )
        )
        return penalty_seat

    def record_shot(self, seat: int, died: bool) -> None:
        """记录开枪结算：幸存或淘汰，并自动推进新一轮。"""
        self.save_snapshot()
        self.turn_index += 1
        self.shots_taken[seat] = self.shots_taken.get(seat, 0) + 1

        self.public_history.append(
            PublicEvent(
                turn_index=self.turn_index,
                event_type="shot",
                seat=seat,
                detail={"died": died, "shots_taken": self.shots_taken[seat]},
            )
        )

        if died:
            self.player_alive[seat] = False
            self.seat_hand_counts[seat] = 0

        # 小轮结束，开启新小轮
        # 新一轮点数顺延循环：A -> K -> Q -> A
        next_rank_map = {Rank.A: Rank.K, Rank.K: Rank.Q, Rank.Q: Rank.A}
        new_rank = next_rank_map.get(self.claim_rank, Rank.A)
        starter = seat if not died and self.is_seat_active(seat) else self.get_next_active_seat(seat, allow_self=False)
        self.reset_round(new_claim_rank=new_rank, starter_seat=starter)

    def build_private_observation(self) -> PrivateObservation:
        """构造供神经网络与特征提取器使用的精准 PrivateObservation。"""
        # 1. 构造主角私有手牌列表
        if getattr(self, "hero_exact_hand", None) is not None:
            hero_hand = list(self.hero_exact_hand)
        else:
            hero_hand = []
            for i in range(self.hero_target_count):
                hero_hand.append(Card(card_id=f"hero_t_{i}", printed_rank=self.claim_rank, kind=CardKind.NORMAL))
            other_rank = Rank.K if self.claim_rank != Rank.K else Rank.Q
            for i in range(self.hero_nontarget_count):
                hero_hand.append(Card(card_id=f"hero_nt_{i}", printed_rank=other_rank, kind=CardKind.NORMAL))
            if self.hero_has_ghost:
                hero_hand.append(Card(card_id="hero_g_0", printed_rank=self.claim_rank, kind=CardKind.GHOST))

        # 2. 构造公共玩家视角 (真实物理存活，挂机不篡改物理状态)
        players_view: list[PublicPlayerView] = []
        for s in range(1, 5):
            is_alive = self.player_alive.get(s, False)
            h_count = self.seat_hand_counts.get(s, 0) if is_alive else 0
            players_view.append(
                PublicPlayerView(
                    seat=s,
                    alive=is_alive,
                    escaped=False,
                    pending_escape=False,
                    hand_count=h_count,
                    shots_taken=self.shots_taken.get(s, 0),
                )
            )

        latest_seat = self.latest_play["seat"] if self.latest_play else None
        latest_count = self.latest_play["count"] if self.latest_play else 0

        return PrivateObservation(
            hero_seat=self.hero_seat,
            hero_hand=tuple(hero_hand),
            current_seat=self.current_seat,
            round_claim_rank=self.claim_rank,
            latest_play_seat=latest_seat,
            latest_play_count=latest_count,
            players=tuple(players_view),
            public_history=tuple(self.public_history),
        )

    def generate_legal_actions(self) -> tuple[GameAction, ...]:
        """生成当前轮到主角时的所有合法动作组合（严格对齐真实物理规则）。"""
        actions: list[GameAction] = []
        obs = self.build_private_observation()
        hand = list(obs.hero_hand)

        alive_seats = [s for s in range(1, self.player_count + 1) if self.player_alive.get(s, False)]

        # 两人局特殊规则：仅剩两人存活且上家出完手牌打空，接牌者唯一合法动作只能是质疑！
        if len(alive_seats) == 2 and self.latest_play and self.latest_play["seat"] != self.hero_seat:
            opp_seat = self.latest_play["seat"]
            if self.seat_hand_counts.get(opp_seat, 0) == 0:
                return (ChallengeAction(seat=self.hero_seat, challenged_seat=opp_seat),)

        # 1. 允许质疑：当前有上家出牌且上家不是自己
        if self.latest_play and self.latest_play["seat"] != self.hero_seat:
            actions.append(ChallengeAction(seat=self.hero_seat, challenged_seat=self.latest_play["seat"]))

        # 2. 允许出牌：手里有手牌时
        if hand:
            for count in (1, 2, 3):
                if count > len(hand):
                    break
                for combo in combinations(hand, count):
                    # 鬼牌只能单出，严禁组合出牌
                    if any(c.kind == CardKind.GHOST for c in combo) and count > 1:
                        continue
                    card_ids = tuple(c.card_id for c in combo)
                    actions.append(
                        PlayAction(
                            seat=self.hero_seat,
                            card_ids=card_ids,
                            claim_rank=self.claim_rank,
                        )
                    )

        # 极端情况兜底：无手牌且有上家出牌时强制质疑
        if not actions and self.latest_play and self.latest_play["seat"] != self.hero_seat:
            actions.append(ChallengeAction(seat=self.hero_seat, challenged_seat=self.latest_play["seat"]))

        return tuple(actions)


# ===========================================================================
# 2. AI 决策引擎 (AIDecisionEngine)
# ===========================================================================

class AIDecisionEngine:
    """深度强化学习与上下文画像桥接推演引擎。"""

    DEFAULT_MODEL_KEY: str = "C_long (主力 PPO v29)"

    SUPPORTED_MODELS: dict[str, str] = {
        "C_long (主力 PPO v29)": "runs_ppo/v29_long_ppo/checkpoint_iter_00120.pt",
        "v39_B10 (强 PPO 对抗候选，真人表现待验证)": "runs_ppo/v39_strong_opponent_pool/checkpoint_B_iter10.pt",
        "v37_B (争胜优先 PPO v37)": "runs_ppo/v37_win_priority/checkpoint_B_iter10.pt",
        "v35_B (课程迁移 PPO v35)": "runs_ppo/v35_curriculum_transfer/checkpoint_B_iter10.pt",
        "v38_Ext (长线微调 PPO v38)": "runs_ppo/v38_clong_extended/checkpoint_iter_00150_add30.pt",
    }

    def __init__(self, default_model_key: str = DEFAULT_MODEL_KEY) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.current_model_key = default_model_key
        self.actor: PPOActor | None = None
        self.encoder = ProfilingFeatureEncoder(mode="control")
        self.abstractor = ActionAbstractor()

        # 加载偏好画像预测器
        profile_path = ROOT_DIR / "runs_ppo" / "league_profiling_results_2800.json"
        if profile_path.is_file():
            self.profiler = ContextualActionPredictor(profile_path=str(profile_path))
        else:
            self.profiler = ContextualActionPredictor()

        # 同局面动作采样缓存 (避免窗口折叠/展开或刷新时抽取的建议跳动)
        self._cached_signature: Any = None
        self._cached_sampled_action: dict[str, Any] | None = None

        self.load_model(default_model_key)

    def get_available_models(self) -> list[str]:
        """获取本地实际存在的模型键名列表。"""
        available: list[str] = []
        for name, rel_path in self.SUPPORTED_MODELS.items():
            if (ROOT_DIR / rel_path).is_file():
                available.append(name)
        return available if available else list(self.SUPPORTED_MODELS.keys())

    def load_model(self, model_key: str) -> bool:
        """加载或热切换指定的神经网络权重；遇到缺失或异常时安全自动回退至默认主力模型 C_long。"""
        rel_path = self.SUPPORTED_MODELS.get(model_key)
        target_key = model_key

        if not rel_path or not (ROOT_DIR / rel_path).is_file():
            print(f"[AIDecisionEngine] 找不到模型检查点: {model_key} (路径: {rel_path})")
            if model_key != self.DEFAULT_MODEL_KEY:
                print(f"[AIDecisionEngine] 触发安全回退 -> 回退至默认生产主力: 【{self.DEFAULT_MODEL_KEY}】")
                return self.load_model(self.DEFAULT_MODEL_KEY)
            return False

        full_path = ROOT_DIR / rel_path
        try:
            ckpt = torch.load(str(full_path), map_location="cpu", weights_only=False)
            actor = PPOActor(
                input_dim=ckpt.get("input_dim", 104),
                action_space_size=ckpt.get("action_space_size", 58),
                hidden_dim=ckpt.get("hidden_dim", 512),
                num_layers=ckpt.get("num_layers", 4),
            )
            actor.load_state_dict(ckpt["actor_net"])
            actor.to(self.device)
            actor.eval()
            self.actor = actor
            self.current_model_key = target_key
            print(f"[AIDecisionEngine] 成功加载模型: 【{target_key}】 | 设备: {self.device}")
            return True
        except Exception as e:
            print(f"[AIDecisionEngine] 模型加载失败 ({target_key}): {e}")
            if target_key != self.DEFAULT_MODEL_KEY:
                print(f"[AIDecisionEngine] 触发安全回退 -> 尝试载入默认生产主力: 【{self.DEFAULT_MODEL_KEY}】")
                return self.load_model(self.DEFAULT_MODEL_KEY)
            return False

    def translate_abstract_action(self, label: str, claim_rank: Rank) -> tuple[str, str]:
        """将抽象动作标签翻译为人类友好的中文描述与标签图标。"""
        if label.startswith("challenge"):
            return "抓假质疑", "⚔️ 发起质疑 (抓上家虚报)"

        # 示例: play:A:1:target / play:A:2:non_target,target
        parts = label.split(":")
        if len(parts) >= 4:
            count = parts[2]
            comp = parts[3].split(",")
            t_cnt = comp.count("target")
            nt_cnt = comp.count("non_target")
            g_cnt = comp.count("ghost")

            if g_cnt > 0:
                return "奇袭鬼牌", f"👻 出1张鬼牌 (点数锁定 {claim_rank.value})"
            elif nt_cnt == 0:
                return "真实跟牌", f"🟢 跟牌出 {count} 张目标牌 ({claim_rank.value})"
            elif t_cnt == 0:
                return "虚报诈唬", f"🎭 诈唬虚报 {count} 张杂牌 (冒充 {claim_rank.value})"
            else:
                return "混合半诈唬", f"🌗 混合出 {count} 张 ({t_cnt}目标 + {nt_cnt}杂牌)"

        return "常规动作", label

    @staticmethod
    def parse_abstract_action_payload(label: str) -> dict[str, Any]:
        """解析抽象动作标签，提取动作类别与真假手牌构成。"""
        if label.startswith("challenge"):
            return {"type": "challenge"}
        parts = label.split(":")
        if len(parts) >= 4:
            comp = parts[3].split(",")
            t_cnt = comp.count("target")
            nt_cnt = comp.count("non_target")
            g_cnt = comp.count("ghost")
            return {
                "type": "play",
                "count": len(comp),
                "target": t_cnt,
                "nontarget": nt_cnt,
                "ghost": (g_cnt > 0),
            }
        return {"type": "play", "count": 1, "target": 1, "nontarget": 0, "ghost": False}

    @torch.no_grad()
    def evaluate(self, tracker: GameStateTracker) -> dict[str, Any]:
        """推演当前局势，返回 AI 核心推荐、动作概率分布、上家虚报分析及战术提示。"""
        obs = tracker.build_private_observation()
        legal_actions = tracker.generate_legal_actions()

        # 1. 上家行为分析与虚报率预估 (明确标为启发式估计，未针对真人校准)
        bluff_info = {
            "has_target": False,
            "target_seat": None,
            "bluff_prob": 0.25,
            "honest_prob": 0.70,
            "summary": "当前局首发，尚无上家出牌",
            "is_heuristic": True,
            "disclaimer": "启发式参考（尚未针对真人校准，仅供观察）",
        }
        if tracker.latest_play and tracker.latest_play["seat"] != tracker.hero_seat:
            lp_seat = tracker.latest_play["seat"]
            lp_count = tracker.latest_play["count"]
            bluff_info["has_target"] = True
            bluff_info["target_seat"] = lp_seat

            # 结合 ContextualActionPredictor 与出牌张数调整先验
            pred = self.profiler.predict_play_style("C_long")
            base_bluff = pred.get("nature_prob", {}).get("bluff", 0.265)
            # 出牌张数越多，诈唬嫌疑相对上升；若对手手牌较少，诈唬率亦上升
            opp_cards = tracker.seat_hand_counts.get(lp_seat, 5)
            count_factor = 1.0 + (lp_count - 1) * 0.35 + (1.0 if opp_cards <= 1 else 0.0)
            est_bluff = min(0.92, max(0.08, base_bluff * count_factor))

            bluff_info["bluff_prob"] = round(est_bluff, 3)
            bluff_info["honest_prob"] = round(1.0 - est_bluff, 3)
            bluff_info["summary"] = f"席位 {lp_seat} 出 {lp_count} 张 | 启发式参考: {est_bluff*100:.0f}% (尚未针对真人校准，仅供观察)"

        # 2. 神经网络动作概率计算
        action_dist: list[dict[str, Any]] = []
        greedy_action = {"desc": "等待局势变化...", "tag": "观望", "prob": 0.0}
        sampled_action = {"desc": "等待局势变化...", "tag": "观望", "prob": 0.0}

        if legal_actions and self.actor is not None:
            grouped = self.abstractor.abstract_legal_actions(legal_actions, obs)
            features = self.encoder.encode(obs).unsqueeze(0).to(self.device)
            logits = self.actor(features).squeeze(0)  # (58,)

            mask = torch.full((58,), float("-inf"), device=self.device)
            for aa in grouped:
                mask[action_to_index(aa.label)] = 0.0

            probs = torch.softmax(logits + mask, dim=0)

            # 聚合抽象动作
            for aa in grouped:
                idx = action_to_index(aa.label)
                p = probs[idx].item()
                tag, desc = self.translate_abstract_action(aa.label, tracker.claim_rank)
                action_dist.append({
                    "label": aa.label,
                    "tag": tag,
                    "desc": desc,
                    "prob": p,
                    "is_challenge": aa.label.startswith("challenge"),
                })

            # 按概率降序排序
            action_dist.sort(key=lambda x: x["prob"], reverse=True)

            if action_dist:
                # 贪心最高权重动作 (Argmax)
                greedy_action = action_dist[0]

                # 实战推荐：按策略多项式分布采样 (带同局面缓存，避免刷新时跳动)
                curr_sig = self._get_state_signature(tracker)
                if curr_sig == self._cached_signature and self._cached_sampled_action is not None:
                    if any(a["label"] == self._cached_sampled_action["label"] for a in action_dist):
                        sampled_action = self._cached_sampled_action
                    else:
                        sampled_action = self._sample_from_dist(action_dist)
                        self._cached_signature = curr_sig
                        self._cached_sampled_action = sampled_action
                else:
                    sampled_action = self._sample_from_dist(action_dist)
                    self._cached_signature = curr_sig
                    self._cached_sampled_action = sampled_action

        # 3. 战术小贴士
        tips: list[str] = []
        is_hero_turn = (tracker.current_seat == tracker.hero_seat)
        if is_hero_turn:
            tips.append("👉 当前轮到【我方】行动！")
            if tracker.hero_target_count >= 1:
                tips.append("手中握有目标牌，安全垫充足。")
            else:
                tips.append("⚠️ 手中缺乏目标牌，需权衡虚报诈唬或果断质疑！")
        else:
            tips.append(f"⏳ 正在等待 席位 {tracker.current_seat} 行动...")

        if tracker.shots_taken.get(tracker.hero_seat, 0) >= 3:
            tips.append("🚨 自身枪膛子弹已较深，开枪风险极高，尽量避免冒险质疑！")

        return {
            "greedy_action": greedy_action,
            "sampled_action": sampled_action,
            "best_action_desc": sampled_action["desc"],
            "best_action_tag": sampled_action["tag"],
            "best_prob": sampled_action["prob"],
            "greedy_desc": greedy_action["desc"],
            "greedy_prob": greedy_action["prob"],
            "action_dist": action_dist,
            "bluff_info": bluff_info,
            "tactical_tip": " ".join(tips),
        }

    def _get_state_signature(self, tracker: GameStateTracker) -> tuple:
        """生成当前局面的不可变指纹签名，用于动作采样缓存。"""
        return (
            tracker.round_index,
            tracker.turn_index,
            tracker.current_seat,
            tracker.hero_seat,
            tracker.claim_rank.value,
            tracker.hero_target_count,
            tracker.hero_nontarget_count,
            tracker.hero_has_ghost,
            tuple(sorted(tracker.seat_hand_counts.items())),
            tuple(sorted(tracker.player_alive.items())),
            tracker.latest_play["seat"] if tracker.latest_play else None,
            tracker.latest_play["count"] if tracker.latest_play else None,
            len(tracker.public_history),
            self.current_model_key,
        )

    def _sample_from_dist(self, action_dist: list[dict[str, Any]]) -> dict[str, Any]:
        """按照 Softmax 策略分布进行多项式抽样。"""
        probs = [a["prob"] for a in action_dist]
        tot = sum(probs)
        if tot <= 1e-8:
            return action_dist[0]
        norm_p = [p / tot for p in probs]
        chosen_idx = random.choices(range(len(action_dist)), weights=norm_p, k=1)[0]
        return action_dist[chosen_idx]


# ===========================================================================
# 3. 桌面原生 Tkinter 悬浮窗界面 (DesktopAssistantUI)
# ===========================================================================

class ModernDarkTheme:
    """纯 Windows 原生现代化暗黑主题色板与视觉规范。"""
    BG_MAIN = "#151619"         # 整体主背景
    BG_PANEL = "#1c1e24"        # 卡片/容器背景
    BG_SUBPANEL = "#252830"     # 子区块背景
    BORDER = "#323642"          # 边框微光线
    BORDER_LIGHT = "#444958"

    TEXT_MAIN = "#f0f2f5"        # 主要白字
    TEXT_MUTED = "#8b949e"       # 次要灰字
    TEXT_DIM = "#586069"         # 极暗提示

    ACCENT_CYAN = "#00e5ff"      # 核心重点/青蓝
    ACCENT_GREEN = "#00e676"     # 真实/成功/绿
    ACCENT_ORANGE = "#ff9100"    # 警告/虚报/橙
    ACCENT_RED = "#ff3d00"       # 危险/淘汰/红
    ACCENT_PURPLE = "#d500f9"    # 鬼牌/紫
    ACCENT_BLUE = "#2979ff"      # 操作按钮/蓝


class DesktopAssistantUI:
    """逆水寒小丑牌双形态置顶桌面悬浮窗 UI。"""

    def __init__(self, root: tk.Tk, tracker: GameStateTracker, engine: AIDecisionEngine) -> None:
        self.root = root
        self.tracker = tracker
        self.engine = engine

        # 窗口基础特性：置顶、透明度、无边框
        self.is_mini_mode = False
        self.is_topmost = True
        self.root.title("逆水寒·小丑牌 决策助手")
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", 0.93)
        self.root.configure(bg=ModernDarkTheme.BG_MAIN)
        self.root.overrideredirect(True)

        # 拖拽窗口位移记录
        self.drag_x = 0
        self.drag_y = 0

        # 初始化窗口尺寸与居中位置 (默认展开完整面板: 420x680)
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        default_x = max(20, screen_w - 440)
        default_y = max(40, (screen_h - 700) // 2)
        self.full_geometry = f"420x680+{default_x}+{default_y}"
        self.mini_geometry = f"340x52+{default_x+40}+{default_y+100}"
        self.root.geometry(self.full_geometry)

        # 主容器：分为 mini_bar_frame 与 full_panel_frame
        self.mini_bar_frame = tk.Frame(self.root, bg=ModernDarkTheme.BG_MAIN)
        self.full_panel_frame = tk.Frame(self.root, bg=ModernDarkTheme.BG_MAIN)

        self._build_mini_bar_ui()
        self._build_full_panel_ui()

        # 默认展示展开面板
        self.show_full_panel()

        # 绑定快捷键与事件
        self.root.bind("<Escape>", lambda e: self.toggle_mode())

        # 首次执行推演并渲染
        self.refresh_ui()

    # -----------------------------------------------------------------------
    # 窗口拖拽与形态切换
    # -----------------------------------------------------------------------

    def _start_drag(self, event: tk.Event) -> None:
        # 使用全局屏幕绝对坐标，避免不同子组件相对坐标差异导致的跳跃抖动
        self.drag_start_x = event.x_root - self.root.winfo_x()
        self.drag_start_y = event.y_root - self.root.winfo_y()

    def _do_drag(self, event: tk.Event) -> None:
        new_x = event.x_root - self.drag_start_x
        new_y = event.y_root - self.drag_start_y
        self.root.geometry(f"+{new_x}+{new_y}")

    def show_mini_bar(self) -> None:
        """缩回微型胶囊条 (Mini Bar) 模式。"""
        curr_x = self.root.winfo_x()
        curr_y = self.root.winfo_y()
        self.full_panel_frame.pack_forget()
        self.mini_bar_frame.pack(fill=tk.BOTH, expand=True)
        # 防止缩回时跑出屏幕可视区域
        scr_w = self.root.winfo_screenwidth()
        scr_h = self.root.winfo_screenheight()
        target_x = max(0, min(curr_x, scr_w - 380))
        target_y = max(0, min(curr_y, scr_h - 60))
        self.root.geometry(f"360x52+{target_x}+{target_y}")
        self.is_mini_mode = True
        self.refresh_ui()

    def show_full_panel(self) -> None:
        """展开完整面板 (Full Panel) 模式。"""
        curr_x = self.root.winfo_x()
        curr_y = self.root.winfo_y()
        self.mini_bar_frame.pack_forget()
        self.full_panel_frame.pack(fill=tk.BOTH, expand=True)
        # 边界自适应防护：若展开后超出屏幕右下边界，自动向左/向上收拢以完整呈现
        scr_w = self.root.winfo_screenwidth()
        scr_h = self.root.winfo_screenheight()
        target_x = max(0, min(curr_x, scr_w - 430))
        target_y = max(0, min(curr_y, scr_h - 700))
        self.root.geometry(f"420x680+{target_x}+{target_y}")
        self.is_mini_mode = False
        self.refresh_ui()

    def toggle_mode(self) -> None:
        """一键双形态切换。"""
        if self.is_mini_mode:
            self.show_full_panel()
        else:
            self.show_mini_bar()

    def toggle_topmost(self) -> None:
        """切换是否置顶。"""
        self.is_topmost = not self.is_topmost
        self.root.attributes("-topmost", self.is_topmost)
        if hasattr(self, "topmost_btn"):
            self.topmost_btn.config(
                text="📌" if self.is_topmost else "📍",
                fg=ModernDarkTheme.ACCENT_CYAN if self.is_topmost else ModernDarkTheme.TEXT_MUTED,
            )

    # -----------------------------------------------------------------------
    # 模式 1：微型胶囊条 (Mini Bar) 构建
    # -----------------------------------------------------------------------

    def _build_mini_bar_ui(self) -> None:
        bar = tk.Frame(
            self.mini_bar_frame,
            bg=ModernDarkTheme.BG_PANEL,
            highlightthickness=1,
            highlightbackground=ModernDarkTheme.ACCENT_CYAN,
        )
        bar.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        # 拖拽绑定
        bar.bind("<ButtonPress-1>", self._start_drag)
        bar.bind("<B1-Motion>", self._do_drag)
        bar.bind("<Double-Button-1>", lambda e: self.show_full_panel())

        # 拖拽指示手柄
        grip = tk.Label(
            bar,
            text="⠿",
            font=("Segoe UI Symbol", 12),
            bg=ModernDarkTheme.BG_PANEL,
            fg=ModernDarkTheme.ACCENT_CYAN,
            cursor="fleur",
        )
        grip.pack(side=tk.LEFT, padx=(6, 2))
        grip.bind("<ButtonPress-1>", self._start_drag)
        grip.bind("<B1-Motion>", self._do_drag)

        # 核心推荐标签 (支持按住随心拖拽，双击一键展开)
        self.mini_text_label = tk.Label(
            bar,
            text="🎯 AI推演准备中...",
            font=("Microsoft YaHei UI", 9, "bold"),
            bg=ModernDarkTheme.BG_PANEL,
            fg=ModernDarkTheme.ACCENT_CYAN,
            anchor="w",
            cursor="fleur",
        )
        self.mini_text_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
        self.mini_text_label.bind("<ButtonPress-1>", self._start_drag)
        self.mini_text_label.bind("<B1-Motion>", self._do_drag)
        self.mini_text_label.bind("<Double-Button-1>", lambda e: self.show_full_panel())

        # 展开按钮
        expand_btn = tk.Button(
            bar,
            text="展开▾",
            font=("Microsoft YaHei UI", 8, "bold"),
            bg=ModernDarkTheme.BG_SUBPANEL,
            fg=ModernDarkTheme.ACCENT_CYAN,
            activebackground=ModernDarkTheme.BORDER,
            activeforeground="#ffffff",
            relief=tk.FLAT,
            bd=0,
            command=self.show_full_panel,
            cursor="hand2",
            padx=4,
        )
        expand_btn.pack(side=tk.RIGHT, padx=3, pady=4)

        # 关闭按钮
        close_btn = tk.Button(
            bar,
            text="✕",
            font=("Microsoft YaHei UI", 8),
            bg=ModernDarkTheme.BG_SUBPANEL,
            fg=ModernDarkTheme.TEXT_MUTED,
            activebackground=ModernDarkTheme.ACCENT_RED,
            activeforeground="#ffffff",
            relief=tk.FLAT,
            bd=0,
            command=self.root.destroy,
            width=2,
            cursor="hand2",
        )
        close_btn.pack(side=tk.RIGHT, padx=(0, 2), pady=4)

    # -----------------------------------------------------------------------
    # 模式 2：完整面板 (Full Panel) 构建
    # -----------------------------------------------------------------------

    def _build_full_panel_ui(self) -> None:
        outer = tk.Frame(
            self.full_panel_frame,
            bg=ModernDarkTheme.BG_MAIN,
            highlightthickness=1,
            highlightbackground=ModernDarkTheme.BORDER_LIGHT,
        )
        outer.pack(fill=tk.BOTH, expand=True)

        # 1. 顶部标题栏
        self._build_header(outer)

        # 2. 开局与席位设置区
        self._build_game_setup_section(outer)

        # 3. 手牌输入区 (方案B)
        self._build_hero_hand_section(outer)

        # 4. 4席位状态与挂机开关
        self._build_seats_status_section(outer)

        # 5. 局势动作记录区
        self._build_actions_control_section(outer)

        # 6. AI 决策推演展示区
        self._build_ai_inference_section(outer)

    def _build_header(self, parent: tk.Frame) -> None:
        """顶部精巧标题栏。"""
        header = tk.Frame(parent, bg=ModernDarkTheme.BG_PANEL, height=36)
        header.pack(fill=tk.X, padx=1, pady=1)

        header.bind("<ButtonPress-1>", self._start_drag)
        header.bind("<B1-Motion>", self._do_drag)

        # 标题与图标
        title_lbl = tk.Label(
            header,
            text="🃏 小丑牌 AI 决策助手",
            font=("Microsoft YaHei UI", 10, "bold"),
            bg=ModernDarkTheme.BG_PANEL,
            fg=ModernDarkTheme.ACCENT_CYAN,
        )
        title_lbl.pack(side=tk.LEFT, padx=8)
        title_lbl.bind("<ButtonPress-1>", self._start_drag)
        title_lbl.bind("<B1-Motion>", self._do_drag)

        # 关闭按钮
        close_btn = tk.Button(
            header,
            text="✕",
            font=("Microsoft YaHei UI", 9),
            bg=ModernDarkTheme.BG_PANEL,
            fg=ModernDarkTheme.TEXT_MUTED,
            activebackground=ModernDarkTheme.ACCENT_RED,
            activeforeground="#ffffff",
            relief=tk.FLAT,
            bd=0,
            command=self.root.destroy,
            width=3,
            cursor="hand2",
        )
        close_btn.pack(side=tk.RIGHT, padx=2, pady=4)

        # 收起折叠按钮
        fold_btn = tk.Button(
            header,
            text="收起▴",
            font=("Microsoft YaHei UI", 8, "bold"),
            bg=ModernDarkTheme.BG_PANEL,
            fg=ModernDarkTheme.ACCENT_CYAN,
            activebackground=ModernDarkTheme.BORDER,
            activeforeground="#ffffff",
            relief=tk.FLAT,
            bd=0,
            command=self.show_mini_bar,
            cursor="hand2",
            padx=4,
        )
        fold_btn.pack(side=tk.RIGHT, padx=2, pady=4)

        # 置顶切换按钮
        self.topmost_btn = tk.Button(
            header,
            text="📌",
            font=("Segoe UI Symbol", 9),
            bg=ModernDarkTheme.BG_PANEL,
            fg=ModernDarkTheme.ACCENT_CYAN,
            relief=tk.FLAT,
            bd=0,
            command=self.toggle_topmost,
            width=2,
            cursor="hand2",
        )
        self.topmost_btn.pack(side=tk.RIGHT, padx=2, pady=4)

        # 模型下拉切换
        available_models = self.engine.get_available_models()
        self.model_var = tk.StringVar(value=self.engine.current_model_key)
        self.model_combo = ttk.Combobox(
            header,
            textvariable=self.model_var,
            values=available_models,
            state="readonly",
            width=24,
            font=("Microsoft YaHei UI", 8),
        )
        self.model_combo.pack(side=tk.RIGHT, padx=6, pady=4)
        self.model_combo.bind("<<ComboboxSelected>>", self._on_model_selected)

    def _on_model_selected(self, event: Any) -> None:
        selected = self.model_var.get()
        if selected != self.engine.current_model_key:
            self.engine.load_model(selected)
            # 无论成功加载还是触发安全回退，均显示实际生效的模型版本
            self.model_var.set(self.engine.current_model_key)
            self.refresh_ui()

    def _build_game_setup_section(self, parent: tk.Frame) -> None:
        """Section 1: 开局与席位设置区。"""
        sec = tk.LabelFrame(
            parent,
            text=" ⚙️ 对局与席位设置 ",
            font=("Microsoft YaHei UI", 8, "bold"),
            bg=ModernDarkTheme.BG_MAIN,
            fg=ModernDarkTheme.TEXT_MUTED,
            relief=tk.GROOVE,
            bd=1,
        )
        sec.pack(fill=tk.X, padx=6, pady=(4, 2))

        f1 = tk.Frame(sec, bg=ModernDarkTheme.BG_MAIN)
        f1.pack(fill=tk.X, padx=4, pady=2)

        # 我所在的席位
        tk.Label(f1, text="我方席位:", font=("Microsoft YaHei UI", 8), bg=ModernDarkTheme.BG_MAIN, fg=ModernDarkTheme.TEXT_MAIN).pack(side=tk.LEFT)
        self.hero_seat_btns: dict[int, tk.Button] = {}
        for s in range(1, 5):
            b = tk.Button(
                f1,
                text=f"{s}号位",
                font=("Microsoft YaHei UI", 8, "bold"),
                relief=tk.FLAT,
                bd=0,
                width=4,
                cursor="hand2",
                command=lambda seat=s: self._on_set_hero_seat(seat),
            )
            b.pack(side=tk.LEFT, padx=2)
            self.hero_seat_btns[s] = b

        # 新局与重置按钮
        rst_btn = tk.Button(
            f1,
            text="新对局",
            font=("Microsoft YaHei UI", 8),
            bg=ModernDarkTheme.BG_SUBPANEL,
            fg=ModernDarkTheme.ACCENT_CYAN,
            activebackground=ModernDarkTheme.BORDER,
            relief=tk.FLAT,
            bd=0,
            command=self._on_click_new_game,
            cursor="hand2",
            padx=4,
        )
        rst_btn.pack(side=tk.RIGHT, padx=2)

        # 锁定点数 A / K / Q
        f2 = tk.Frame(sec, bg=ModernDarkTheme.BG_MAIN)
        f2.pack(fill=tk.X, padx=4, pady=2)

        tk.Label(f2, text="轮次点数:", font=("Microsoft YaHei UI", 8), bg=ModernDarkTheme.BG_MAIN, fg=ModernDarkTheme.TEXT_MAIN).pack(side=tk.LEFT)
        self.rank_btns: dict[Rank, tk.Button] = {}
        for r in (Rank.A, Rank.K, Rank.Q):
            rb = tk.Button(
                f2,
                text=f"【 {r.value} 】",
                font=("Microsoft YaHei UI", 8, "bold"),
                relief=tk.FLAT,
                bd=0,
                width=5,
                cursor="hand2",
                command=lambda rk=r: self._on_set_rank(rk),
            )
            rb.pack(side=tk.LEFT, padx=3)
            self.rank_btns[r] = rb

        # 新小轮按钮
        round_btn = tk.Button(
            f2,
            text="新小轮 (发牌)",
            font=("Microsoft YaHei UI", 8),
            bg=ModernDarkTheme.BG_SUBPANEL,
            fg=ModernDarkTheme.ACCENT_GREEN,
            activebackground=ModernDarkTheme.BORDER,
            relief=tk.FLAT,
            bd=0,
            command=self._on_click_new_round,
            cursor="hand2",
            padx=4,
        )
        round_btn.pack(side=tk.RIGHT, padx=2)

    def _on_set_hero_seat(self, seat: int) -> None:
        self.tracker.hero_seat = seat
        self.refresh_ui()

    def _on_set_rank(self, rank: Rank) -> None:
        self.tracker.claim_rank = rank
        self.refresh_ui()

    def _on_click_new_game(self) -> None:
        self.tracker.reset_game(hero_seat=self.tracker.hero_seat, claim_rank=self.tracker.claim_rank)
        self.refresh_ui()

    def _on_click_new_round(self) -> None:
        # 下一轮点数
        next_rank_map = {Rank.A: Rank.K, Rank.K: Rank.Q, Rank.Q: Rank.A}
        self.tracker.reset_round(new_claim_rank=next_rank_map[self.tracker.claim_rank])
        self.refresh_ui()

    def _build_hero_hand_section(self, parent: tk.Frame) -> None:
        """Section 2: 手牌输入区 (方案B)。"""
        sec = tk.LabelFrame(
            parent,
            text=" 🎴 我方手牌配置 (方案B) ",
            font=("Microsoft YaHei UI", 8, "bold"),
            bg=ModernDarkTheme.BG_MAIN,
            fg=ModernDarkTheme.TEXT_MUTED,
            relief=tk.GROOVE,
            bd=1,
        )
        sec.pack(fill=tk.X, padx=6, pady=2)

        row = tk.Frame(sec, bg=ModernDarkTheme.BG_MAIN)
        row.pack(fill=tk.X, padx=4, pady=3)

        # 1. 目标牌数量
        tk.Label(row, text="目标牌:", font=("Microsoft YaHei UI", 8), bg=ModernDarkTheme.BG_MAIN, fg=ModernDarkTheme.ACCENT_GREEN).pack(side=tk.LEFT)
        btn_t_minus = tk.Button(row, text="-", font=("Microsoft YaHei UI", 8, "bold"), width=2, relief=tk.FLAT, bd=0, bg=ModernDarkTheme.BG_SUBPANEL, fg=ModernDarkTheme.TEXT_MAIN, command=lambda: self._adj_hand("target", -1))
        btn_t_minus.pack(side=tk.LEFT, padx=2)
        self.lbl_target_cnt = tk.Label(row, text="2", font=("Microsoft YaHei UI", 9, "bold"), bg=ModernDarkTheme.BG_PANEL, fg=ModernDarkTheme.ACCENT_GREEN, width=2)
        self.lbl_target_cnt.pack(side=tk.LEFT, padx=1)
        btn_t_plus = tk.Button(row, text="+", font=("Microsoft YaHei UI", 8, "bold"), width=2, relief=tk.FLAT, bd=0, bg=ModernDarkTheme.BG_SUBPANEL, fg=ModernDarkTheme.TEXT_MAIN, command=lambda: self._adj_hand("target", 1))
        btn_t_plus.pack(side=tk.LEFT, padx=2)

        # 2. 杂牌数量
        tk.Label(row, text="  杂牌:", font=("Microsoft YaHei UI", 8), bg=ModernDarkTheme.BG_MAIN, fg=ModernDarkTheme.ACCENT_ORANGE).pack(side=tk.LEFT)
        btn_nt_minus = tk.Button(row, text="-", font=("Microsoft YaHei UI", 8, "bold"), width=2, relief=tk.FLAT, bd=0, bg=ModernDarkTheme.BG_SUBPANEL, fg=ModernDarkTheme.TEXT_MAIN, command=lambda: self._adj_hand("nontarget", -1))
        btn_nt_minus.pack(side=tk.LEFT, padx=2)
        self.lbl_nontarget_cnt = tk.Label(row, text="3", font=("Microsoft YaHei UI", 9, "bold"), bg=ModernDarkTheme.BG_PANEL, fg=ModernDarkTheme.ACCENT_ORANGE, width=2)
        self.lbl_nontarget_cnt.pack(side=tk.LEFT, padx=1)
        btn_nt_plus = tk.Button(row, text="+", font=("Microsoft YaHei UI", 8, "bold"), width=2, relief=tk.FLAT, bd=0, bg=ModernDarkTheme.BG_SUBPANEL, fg=ModernDarkTheme.TEXT_MAIN, command=lambda: self._adj_hand("nontarget", 1))
        btn_nt_plus.pack(side=tk.LEFT, padx=2)

        # 3. 鬼牌开关
        self.btn_ghost = tk.Button(
            row,
            text="👻 鬼牌: 无",
            font=("Microsoft YaHei UI", 8, "bold"),
            relief=tk.FLAT,
            bd=0,
            cursor="hand2",
            padx=4,
            command=self._toggle_ghost,
        )
        self.btn_ghost.pack(side=tk.LEFT, padx=8)

        # 手牌总数指示
        self.lbl_hand_total = tk.Label(
            row,
            text="共 5 张",
            font=("Microsoft YaHei UI", 8),
            bg=ModernDarkTheme.BG_MAIN,
            fg=ModernDarkTheme.TEXT_MUTED,
        )
        self.lbl_hand_total.pack(side=tk.RIGHT, padx=4)

    def _adj_hand(self, kind: str, delta: int) -> None:
        if kind == "target":
            self.tracker.set_hero_hand(target_count=self.tracker.hero_target_count + delta)
        elif kind == "nontarget":
            self.tracker.set_hero_hand(nontarget_count=self.tracker.hero_nontarget_count + delta)
        self.refresh_ui()

    def _toggle_ghost(self) -> None:
        self.tracker.set_hero_hand(has_ghost=not self.tracker.hero_has_ghost)
        self.refresh_ui()

    def _build_seats_status_section(self, parent: tk.Frame) -> None:
        """Section 3: 4席位状态卡片与挂机开关。"""
        sec = tk.LabelFrame(
            parent,
            text=" 👥 席位局势与挂机标记 ",
            font=("Microsoft YaHei UI", 8, "bold"),
            bg=ModernDarkTheme.BG_MAIN,
            fg=ModernDarkTheme.TEXT_MUTED,
            relief=tk.GROOVE,
            bd=1,
        )
        sec.pack(fill=tk.X, padx=6, pady=2)

        self.seat_card_widgets: dict[int, dict[str, Any]] = {}
        for s in range(1, 5):
            row = tk.Frame(sec, bg=ModernDarkTheme.BG_SUBPANEL, highlightthickness=1, highlightbackground=ModernDarkTheme.BORDER)
            row.pack(fill=tk.X, padx=3, pady=1)

            # 行动者指示 / 席位名
            lbl_name = tk.Label(
                row,
                text=f"Seat {s}",
                font=("Microsoft YaHei UI", 8, "bold"),
                width=9,
                bg=ModernDarkTheme.BG_SUBPANEL,
                fg=ModernDarkTheme.TEXT_MAIN,
                anchor="w",
            )
            lbl_name.pack(side=tk.LEFT, padx=4)

            # 存活与枪膛状态
            lbl_status = tk.Label(
                row,
                text="🔫 0/6 | 存活",
                font=("Microsoft YaHei UI", 8),
                width=11,
                bg=ModernDarkTheme.BG_SUBPANEL,
                fg=ModernDarkTheme.TEXT_MUTED,
                anchor="w",
            )
            lbl_status.pack(side=tk.LEFT, padx=2)

            # 估算手牌
            lbl_cards = tk.Label(
                row,
                text="手牌: 5",
                font=("Microsoft YaHei UI", 8),
                width=7,
                bg=ModernDarkTheme.BG_SUBPANEL,
                fg=ModernDarkTheme.ACCENT_CYAN,
                anchor="w",
            )
            lbl_cards.pack(side=tk.LEFT, padx=2)

            # 设为当前行动者按钮
            btn_act = tk.Button(
                row,
                text="指定行动",
                font=("Microsoft YaHei UI", 7),
                bg=ModernDarkTheme.BG_PANEL,
                fg=ModernDarkTheme.TEXT_MUTED,
                relief=tk.FLAT,
                bd=0,
                cursor="hand2",
                command=lambda seat=s: self._on_set_actor(seat),
            )
            btn_act.pack(side=tk.LEFT, padx=2)

            # 挂机开关按钮
            btn_afk = tk.Button(
                row,
                text="挂机: 否",
                font=("Microsoft YaHei UI", 7, "bold"),
                relief=tk.FLAT,
                bd=0,
                width=6,
                cursor="hand2",
                command=lambda seat=s: self._on_toggle_afk(seat),
            )
            btn_afk.pack(side=tk.RIGHT, padx=4, pady=2)

            self.seat_card_widgets[s] = {
                "frame": row,
                "name": lbl_name,
                "status": lbl_status,
                "cards": lbl_cards,
                "btn_act": btn_act,
                "btn_afk": btn_afk,
            }

    def _on_set_actor(self, seat: int) -> None:
        self.tracker.save_snapshot()
        self.tracker.current_seat = seat
        self.refresh_ui()

    def _on_toggle_afk(self, seat: int) -> None:
        self.tracker.toggle_afk(seat)
        self.refresh_ui()

    def _build_actions_control_section(self, parent: tk.Frame) -> None:
        """Section 4: 局势动作记录区 (自适应我方自主出牌 vs 对手出牌记录)。"""
        sec = tk.LabelFrame(
            parent,
            text=" 🎯 动作记录与实际出牌 ",
            font=("Microsoft YaHei UI", 8, "bold"),
            bg=ModernDarkTheme.BG_MAIN,
            fg=ModernDarkTheme.TEXT_MUTED,
            relief=tk.GROOVE,
            bd=1,
        )
        sec.pack(fill=tk.X, padx=6, pady=2)

        # 1. 状态横幅
        self.lbl_action_banner = tk.Label(
            sec,
            text="轮到 席位 1 行动",
            font=("Microsoft YaHei UI", 8, "bold"),
            bg=ModernDarkTheme.BG_PANEL,
            fg=ModernDarkTheme.ACCENT_CYAN,
            padx=4,
            pady=2,
        )
        self.lbl_action_banner.pack(fill=tk.X, padx=4, pady=2)

        # 2. 我方行动专属面板 (Hero Actions Frame)
        self.frame_hero_actions = tk.Frame(sec, bg=ModernDarkTheme.BG_MAIN)

        # 2.1 采纳 AI 建议快捷按钮
        row_adopt = tk.Frame(self.frame_hero_actions, bg=ModernDarkTheme.BG_MAIN)
        row_adopt.pack(fill=tk.X, padx=2, pady=1)

        self.btn_adopt_ai = tk.Button(
            row_adopt,
            text="⚡ 一键采纳 AI 实战建议",
            font=("Microsoft YaHei UI", 8, "bold"),
            bg="#1b4332",
            fg="#74c69d",
            activebackground="#2d6a4f",
            activeforeground="#d8f3dc",
            relief=tk.FLAT,
            bd=0,
            cursor="hand2",
            command=self._on_adopt_ai_action,
            padx=6,
            pady=2,
        )
        self.btn_adopt_ai.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        undo_btn_hero = tk.Button(
            row_adopt,
            text="↩️ 撤销",
            font=("Microsoft YaHei UI", 8),
            bg=ModernDarkTheme.BG_SUBPANEL,
            fg=ModernDarkTheme.ACCENT_ORANGE,
            relief=tk.FLAT,
            bd=0,
            command=self._on_undo,
            cursor="hand2",
            padx=4,
        )
        undo_btn_hero.pack(side=tk.RIGHT)

        # 2.2 我方自主出真牌行
        self.row_hero_true = tk.Frame(self.frame_hero_actions, bg=ModernDarkTheme.BG_MAIN)
        self.row_hero_true.pack(fill=tk.X, padx=2, pady=1)
        tk.Label(self.row_hero_true, text="🟢出真牌:", font=("Microsoft YaHei UI", 8, "bold"), bg=ModernDarkTheme.BG_MAIN, fg=ModernDarkTheme.ACCENT_GREEN, width=8, anchor="w").pack(side=tk.LEFT)

        self.btn_hero_t1 = tk.Button(self.row_hero_true, text="1张真", font=("Microsoft YaHei UI", 8, "bold"), bg=ModernDarkTheme.BG_SUBPANEL, fg="#ffffff", relief=tk.FLAT, bd=0, width=6, cursor="hand2", command=lambda: self._on_record_hero_play(target_cnt=1))
        self.btn_hero_t1.pack(side=tk.LEFT, padx=2)

        self.btn_hero_t2 = tk.Button(self.row_hero_true, text="2张真", font=("Microsoft YaHei UI", 8, "bold"), bg=ModernDarkTheme.BG_SUBPANEL, fg="#ffffff", relief=tk.FLAT, bd=0, width=6, cursor="hand2", command=lambda: self._on_record_hero_play(target_cnt=2))
        self.btn_hero_t2.pack(side=tk.LEFT, padx=2)

        self.btn_hero_t3 = tk.Button(self.row_hero_true, text="3张真", font=("Microsoft YaHei UI", 8, "bold"), bg=ModernDarkTheme.BG_SUBPANEL, fg="#ffffff", relief=tk.FLAT, bd=0, width=6, cursor="hand2", command=lambda: self._on_record_hero_play(target_cnt=3))
        self.btn_hero_t3.pack(side=tk.LEFT, padx=2)

        # 2.3 我方自主虚报诈唬行
        self.row_hero_bluff = tk.Frame(self.frame_hero_actions, bg=ModernDarkTheme.BG_MAIN)
        self.row_hero_bluff.pack(fill=tk.X, padx=2, pady=1)
        tk.Label(self.row_hero_bluff, text="🎭出杂牌:", font=("Microsoft YaHei UI", 8, "bold"), bg=ModernDarkTheme.BG_MAIN, fg=ModernDarkTheme.ACCENT_ORANGE, width=8, anchor="w").pack(side=tk.LEFT)

        self.btn_hero_nt1 = tk.Button(self.row_hero_bluff, text="1张杂", font=("Microsoft YaHei UI", 8), bg=ModernDarkTheme.BG_SUBPANEL, fg=ModernDarkTheme.TEXT_MAIN, relief=tk.FLAT, bd=0, width=6, cursor="hand2", command=lambda: self._on_record_hero_play(nontarget_cnt=1))
        self.btn_hero_nt1.pack(side=tk.LEFT, padx=2)

        self.btn_hero_nt2 = tk.Button(self.row_hero_bluff, text="2张杂", font=("Microsoft YaHei UI", 8), bg=ModernDarkTheme.BG_SUBPANEL, fg=ModernDarkTheme.TEXT_MAIN, relief=tk.FLAT, bd=0, width=6, cursor="hand2", command=lambda: self._on_record_hero_play(nontarget_cnt=2))
        self.btn_hero_nt2.pack(side=tk.LEFT, padx=2)

        self.btn_hero_nt3 = tk.Button(self.row_hero_bluff, text="3张杂", font=("Microsoft YaHei UI", 8), bg=ModernDarkTheme.BG_SUBPANEL, fg=ModernDarkTheme.TEXT_MAIN, relief=tk.FLAT, bd=0, width=6, cursor="hand2", command=lambda: self._on_record_hero_play(nontarget_cnt=3))
        self.btn_hero_nt3.pack(side=tk.LEFT, padx=2)

        self.btn_hero_mix = tk.Button(self.row_hero_bluff, text="1真1杂", font=("Microsoft YaHei UI", 8), bg=ModernDarkTheme.BG_SUBPANEL, fg=ModernDarkTheme.TEXT_MAIN, relief=tk.FLAT, bd=0, width=6, cursor="hand2", command=lambda: self._on_record_hero_play(target_cnt=1, nontarget_cnt=1))
        self.btn_hero_mix.pack(side=tk.LEFT, padx=2)

        self.btn_hero_ghost = tk.Button(self.row_hero_bluff, text="👻出鬼", font=("Microsoft YaHei UI", 8), bg=ModernDarkTheme.BG_SUBPANEL, fg=ModernDarkTheme.ACCENT_PURPLE, relief=tk.FLAT, bd=0, width=5, cursor="hand2", command=lambda: self._on_record_hero_play(ghost=True))
        self.btn_hero_ghost.pack(side=tk.LEFT, padx=2)

        # 2.4 我方质疑上家行 (有上家且非我方时可用)
        self.row_hero_chal = tk.Frame(self.frame_hero_actions, bg=ModernDarkTheme.BG_MAIN)
        self.row_hero_chal.pack(fill=tk.X, padx=2, pady=1)
        tk.Label(self.row_hero_chal, text="⚔️我质疑:", font=("Microsoft YaHei UI", 8, "bold"), bg=ModernDarkTheme.BG_MAIN, fg=ModernDarkTheme.ACCENT_RED, width=8, anchor="w").pack(side=tk.LEFT)

        btn_hero_chal_honest = tk.Button(self.row_hero_chal, text="验出真(我受罚)", font=("Microsoft YaHei UI", 7), bg=ModernDarkTheme.BG_SUBPANEL, fg=ModernDarkTheme.ACCENT_GREEN, relief=tk.FLAT, bd=0, cursor="hand2", command=lambda: self._on_record_challenge("honest"), padx=2)
        btn_hero_chal_honest.pack(side=tk.LEFT, padx=2)

        btn_hero_chal_lie = tk.Button(self.row_hero_chal, text="验出假(抓假赢)", font=("Microsoft YaHei UI", 7, "bold"), bg="#5c1d1d", fg="#ffcccc", relief=tk.FLAT, bd=0, cursor="hand2", command=lambda: self._on_record_challenge("lie"), padx=2)
        btn_hero_chal_lie.pack(side=tk.LEFT, padx=2)

        btn_hero_chal_gh = tk.Button(self.row_hero_chal, text="验出鬼(我受罚)", font=("Microsoft YaHei UI", 7), bg=ModernDarkTheme.BG_SUBPANEL, fg=ModernDarkTheme.ACCENT_PURPLE, relief=tk.FLAT, bd=0, cursor="hand2", command=lambda: self._on_record_challenge("ghost"), padx=2)
        btn_hero_chal_gh.pack(side=tk.LEFT, padx=2)

        # 3. 对手行动专属面板 (Opponent Actions Frame)
        self.frame_opp_actions = tk.Frame(sec, bg=ModernDarkTheme.BG_MAIN)

        row_opp_play = tk.Frame(self.frame_opp_actions, bg=ModernDarkTheme.BG_MAIN)
        row_opp_play.pack(fill=tk.X, padx=2, pady=1)
        tk.Label(row_opp_play, text="对手出牌:", font=("Microsoft YaHei UI", 8), bg=ModernDarkTheme.BG_MAIN, fg=ModernDarkTheme.TEXT_MAIN, width=8, anchor="w").pack(side=tk.LEFT)

        for cnt in (1, 2, 3):
            b = tk.Button(
                row_opp_play,
                text=f"出 {cnt} 张",
                font=("Microsoft YaHei UI", 8, "bold"),
                bg=ModernDarkTheme.BG_SUBPANEL,
                fg=ModernDarkTheme.TEXT_MAIN,
                activebackground=ModernDarkTheme.BORDER,
                relief=tk.FLAT,
                bd=0,
                width=6,
                cursor="hand2",
                command=lambda c=cnt: self._on_record_play(c),
            )
            b.pack(side=tk.LEFT, padx=2)

        undo_btn_opp = tk.Button(
            row_opp_play,
            text="↩️ 撤销",
            font=("Microsoft YaHei UI", 8),
            bg=ModernDarkTheme.BG_SUBPANEL,
            fg=ModernDarkTheme.ACCENT_ORANGE,
            relief=tk.FLAT,
            bd=0,
            command=self._on_undo,
            cursor="hand2",
            padx=4,
        )
        undo_btn_opp.pack(side=tk.RIGHT)

        row_opp_chal = tk.Frame(self.frame_opp_actions, bg=ModernDarkTheme.BG_MAIN)
        row_opp_chal.pack(fill=tk.X, padx=2, pady=1)
        tk.Label(row_opp_chal, text="有人质疑:", font=("Microsoft YaHei UI", 8), bg=ModernDarkTheme.BG_MAIN, fg=ModernDarkTheme.TEXT_MAIN, width=8, anchor="w").pack(side=tk.LEFT)

        btn_honest = tk.Button(
            row_opp_chal,
            text="验出真牌",
            font=("Microsoft YaHei UI", 8),
            bg=ModernDarkTheme.BG_SUBPANEL,
            fg=ModernDarkTheme.ACCENT_GREEN,
            relief=tk.FLAT,
            bd=0,
            command=lambda: self._on_record_challenge("honest"),
            cursor="hand2",
            padx=3,
        )
        btn_honest.pack(side=tk.LEFT, padx=2)

        btn_lie = tk.Button(
            row_opp_chal,
            text="验出假牌",
            font=("Microsoft YaHei UI", 8),
            bg=ModernDarkTheme.BG_SUBPANEL,
            fg=ModernDarkTheme.ACCENT_RED,
            relief=tk.FLAT,
            bd=0,
            command=lambda: self._on_record_challenge("lie"),
            cursor="hand2",
            padx=2,
        )
        btn_lie.pack(side=tk.LEFT, padx=2)

        btn_gh = tk.Button(
            row_opp_chal,
            text="验出鬼牌",
            font=("Microsoft YaHei UI", 8),
            bg=ModernDarkTheme.BG_SUBPANEL,
            fg=ModernDarkTheme.ACCENT_PURPLE,
            relief=tk.FLAT,
            bd=0,
            command=lambda: self._on_record_challenge("ghost"),
            cursor="hand2",
            padx=2,
        )
        btn_gh.pack(side=tk.LEFT, padx=2)

        # 4. 底层开枪结算条 (任何人触发质疑后使用，常驻)
        self.row_shot_bar = tk.Frame(sec, bg=ModernDarkTheme.BG_MAIN)
        self.row_shot_bar.pack(fill=tk.X, padx=2, pady=(2, 1))

        self.lbl_shot_hint = tk.Label(
            self.row_shot_bar,
            text="开枪结算:",
            font=("Microsoft YaHei UI", 8),
            bg=ModernDarkTheme.BG_MAIN,
            fg=ModernDarkTheme.TEXT_MUTED,
            width=8,
            anchor="w",
        )
        self.lbl_shot_hint.pack(side=tk.LEFT)

        self.btn_shot_survive = tk.Button(
            self.row_shot_bar,
            text="🔫 扣扳机·幸存",
            font=("Microsoft YaHei UI", 8),
            bg=ModernDarkTheme.BG_SUBPANEL,
            fg=ModernDarkTheme.ACCENT_CYAN,
            relief=tk.FLAT,
            bd=0,
            command=lambda: self._on_record_shot(False),
            cursor="hand2",
            padx=4,
        )
        self.btn_shot_survive.pack(side=tk.LEFT, padx=2)

        self.btn_shot_die = tk.Button(
            self.row_shot_bar,
            text="💥 扣扳机·中弹淘汰",
            font=("Microsoft YaHei UI", 8),
            bg=ModernDarkTheme.BG_SUBPANEL,
            fg=ModernDarkTheme.ACCENT_RED,
            relief=tk.FLAT,
            bd=0,
            command=lambda: self._on_record_shot(True),
            cursor="hand2",
            padx=4,
        )
        self.btn_shot_die.pack(side=tk.LEFT, padx=2)

    def _update_btn_state(self, btn: tk.Button, enabled: bool, active_fg: str) -> None:
        """动态更新按钮可用与禁用外观。"""
        if enabled:
            btn.config(state=tk.NORMAL, fg=active_fg, bg=ModernDarkTheme.BG_SUBPANEL)
        else:
            btn.config(state=tk.DISABLED, fg=ModernDarkTheme.TEXT_DIM, bg=ModernDarkTheme.BG_PANEL)

    def _on_record_hero_play(self, target_cnt: int = 0, nontarget_cnt: int = 0, ghost: bool = False) -> None:
        """记录我方实际打出的具体手牌构成（真/假/鬼）。"""
        count = target_cnt + nontarget_cnt + (1 if ghost else 0)
        if count == 0:
            return
        self.tracker.record_play(
            seat=self.tracker.hero_seat,
            count=count,
            hero_used_target=target_cnt,
            hero_used_nontarget=nontarget_cnt,
            hero_used_ghost=ghost,
        )
        self.refresh_ui()

    def _on_adopt_ai_action(self) -> None:
        """一键采纳 AI 当前实战策略建议。"""
        eval_result = self.engine.evaluate(self.tracker)
        sampled = eval_result.get("sampled_action", {})
        label = sampled.get("label", "")
        if not label:
            return

        payload = AIDecisionEngine.parse_abstract_action_payload(label)
        if payload["type"] == "challenge":
            # 引导质疑验牌
            if self.tracker.latest_play and self.tracker.latest_play["seat"] != self.tracker.hero_seat:
                self.lbl_action_banner.config(
                    text="⚔️【已采纳质疑】请看游戏内验牌结果，点击下方验牌按钮！",
                    bg="#4a2e00",
                    fg="#ffcc00",
                )
        else:
            self._on_record_hero_play(
                target_cnt=payload.get("target", 0),
                nontarget_cnt=payload.get("nontarget", 0),
                ghost=payload.get("ghost", False),
            )

    def _on_record_play(self, count: int) -> None:
        acting_seat = self.tracker.current_seat
        self.tracker.record_play(acting_seat, count)
        self.refresh_ui()

    def _on_record_challenge(self, outcome: str) -> None:
        acting_seat = self.tracker.current_seat
        penalty_seat = self.tracker.record_challenge(acting_seat, outcome)
        self.refresh_ui()

    def _on_record_shot(self, died: bool) -> None:
        target_seat = self.tracker.pending_penalty_seat or self.tracker.current_seat
        self.tracker.record_shot(target_seat, died)
        self.refresh_ui()

    def _on_undo(self) -> None:
        ok = self.tracker.undo()
        if ok:
            self.refresh_ui()

    def _build_ai_inference_section(self, parent: tk.Frame) -> None:
        """Section 5: AI 决策推演展示区 (核心推荐 + 概率分布条 + 画像分析)。"""
        sec = tk.LabelFrame(
            parent,
            text=" 🧠 AI 决策推演与博弈分析 ",
            font=("Microsoft YaHei UI", 8, "bold"),
            bg=ModernDarkTheme.BG_MAIN,
            fg=ModernDarkTheme.TEXT_MUTED,
            relief=tk.GROOVE,
            bd=1,
        )
        sec.pack(fill=tk.BOTH, expand=True, padx=6, pady=(2, 6))

        # 当前加载模型版本指示条 (清晰展示实际生效的神经网络版本与候选定位)
        self.lbl_active_model_badge = tk.Label(
            sec,
            text=f"🤖 当前决策网络: {self.engine.current_model_key}",
            font=("Microsoft YaHei UI", 8),
            bg=ModernDarkTheme.BG_SUBPANEL,
            fg=ModernDarkTheme.ACCENT_CYAN,
            anchor="w",
            padx=6,
            pady=2,
        )
        self.lbl_active_model_badge.pack(fill=tk.X, padx=4, pady=(2, 2))

        # 核心高亮推荐卡片：清晰并列展示实战策略建议与策略最高项 (Argmax)
        self.rec_card = tk.Frame(
            sec,
            bg=ModernDarkTheme.BG_PANEL,
            highlightthickness=1,
            highlightbackground=ModernDarkTheme.ACCENT_CYAN,
        )
        self.rec_card.pack(fill=tk.X, padx=4, pady=4)

        # 1. 实战建议行 (按策略分布随机抽取，纳什混合博弈行为)
        row_sampled = tk.Frame(self.rec_card, bg=ModernDarkTheme.BG_PANEL)
        row_sampled.pack(fill=tk.X, padx=6, pady=(4, 2))
        self.lbl_sampled_tag = tk.Label(
            row_sampled,
            text="🎲 实战建议 (按策略抽取):",
            font=("Microsoft YaHei UI", 8, "bold"),
            bg=ModernDarkTheme.BG_PANEL,
            fg=ModernDarkTheme.ACCENT_CYAN,
        )
        self.lbl_sampled_tag.pack(side=tk.LEFT)

        self.lbl_sampled_desc = tk.Label(
            row_sampled,
            text="跟牌出 1 张目标牌 (A)",
            font=("Microsoft YaHei UI", 9, "bold"),
            bg=ModernDarkTheme.BG_PANEL,
            fg="#ffffff",
        )
        self.lbl_sampled_desc.pack(side=tk.LEFT, padx=4)

        # 2. 策略最高权重项 (Argmax)
        row_greedy = tk.Frame(self.rec_card, bg=ModernDarkTheme.BG_PANEL)
        row_greedy.pack(fill=tk.X, padx=6, pady=(1, 4))
        self.lbl_greedy_tag = tk.Label(
            row_greedy,
            text="⭐ 策略最高权重 (Argmax):",
            font=("Microsoft YaHei UI", 8),
            bg=ModernDarkTheme.BG_PANEL,
            fg=ModernDarkTheme.TEXT_MUTED,
        )
        self.lbl_greedy_tag.pack(side=tk.LEFT)

        self.lbl_greedy_desc = tk.Label(
            row_greedy,
            text="跟牌出 1 张目标牌 (A)",
            font=("Microsoft YaHei UI", 8),
            bg=ModernDarkTheme.BG_PANEL,
            fg=ModernDarkTheme.TEXT_MAIN,
        )
        self.lbl_greedy_desc.pack(side=tk.LEFT, padx=4)

        # 上家虚报分析横幅
        self.lbl_bluff_analysis = tk.Label(
            sec,
            text="📊 上家虚报分析: 暂无",
            font=("Microsoft YaHei UI", 8),
            bg=ModernDarkTheme.BG_SUBPANEL,
            fg=ModernDarkTheme.ACCENT_ORANGE,
            anchor="w",
            padx=6,
            pady=2,
        )
        self.lbl_bluff_analysis.pack(fill=tk.X, padx=4, pady=2)

        # 备选动作概率分布 Canvas 列表
        self.prob_canvas = tk.Canvas(
            sec,
            bg=ModernDarkTheme.BG_MAIN,
            highlightthickness=0,
            height=130,
        )
        self.prob_canvas.pack(fill=tk.BOTH, expand=True, padx=4, pady=2)

        # 底部战术小贴士
        self.lbl_tactical_tips = tk.Label(
            sec,
            text="💡 战术提示: 手中有真牌，安全度高",
            font=("Microsoft YaHei UI", 8),
            bg=ModernDarkTheme.BG_MAIN,
            fg=ModernDarkTheme.TEXT_MUTED,
            anchor="w",
            padx=2,
        )
        self.lbl_tactical_tips.pack(fill=tk.X, padx=4, pady=(2, 1))

        # 概率性质提示声明（杜绝误标为胜率或正确率）
        self.lbl_prob_disclaimer = tk.Label(
            sec,
            text="注：概率为当前局面下 AI 神经网络策略偏好权重，非胜率或正确率",
            font=("Microsoft YaHei UI", 7),
            bg=ModernDarkTheme.BG_MAIN,
            fg=ModernDarkTheme.TEXT_MUTED,
            anchor="w",
            padx=2,
        )
        self.lbl_prob_disclaimer.pack(fill=tk.X, padx=4, pady=(0, 4))

    # -----------------------------------------------------------------------
    # 数据刷新与 UI 状态同步
    # -----------------------------------------------------------------------

    def refresh_ui(self) -> None:
        """从局势状态与 AI 引擎中提取最新推演，全量刷新 UI。"""
        # 1. 刷新开局设置按钮高亮
        for s, btn in self.hero_seat_btns.items():
            if s == self.tracker.hero_seat:
                btn.config(bg=ModernDarkTheme.ACCENT_BLUE, fg="#ffffff")
            else:
                btn.config(bg=ModernDarkTheme.BG_SUBPANEL, fg=ModernDarkTheme.TEXT_MAIN)

        for r, btn in self.rank_btns.items():
            if r == self.tracker.claim_rank:
                btn.config(bg=ModernDarkTheme.ACCENT_CYAN, fg="#000000")
            else:
                btn.config(bg=ModernDarkTheme.BG_SUBPANEL, fg=ModernDarkTheme.TEXT_MAIN)

        # 2. 刷新手牌区计数
        self.lbl_target_cnt.config(text=str(self.tracker.hero_target_count))
        self.lbl_nontarget_cnt.config(text=str(self.tracker.hero_nontarget_count))
        if self.tracker.hero_has_ghost:
            self.btn_ghost.config(text="👻 鬼牌: 持有", bg=ModernDarkTheme.ACCENT_PURPLE, fg="#ffffff")
        else:
            self.btn_ghost.config(text="👻 鬼牌: 无", bg=ModernDarkTheme.BG_SUBPANEL, fg=ModernDarkTheme.TEXT_MUTED)

        total_hero_cards = self.tracker.hero_target_count + self.tracker.hero_nontarget_count + (1 if self.tracker.hero_has_ghost else 0)
        self.lbl_hand_total.config(text=f"共 {total_hero_cards} 张")

        # 3. 刷新席位状态卡片
        for s, w in self.seat_card_widgets.items():
            alive = self.tracker.player_alive.get(s, False)
            afk = self.tracker.player_afk.get(s, False)
            shots = self.tracker.shots_taken.get(s, 0)
            cards = self.tracker.seat_hand_counts.get(s, 0)
            is_actor = (s == self.tracker.current_seat)
            is_hero = (s == self.tracker.hero_seat)

            # 行动者高亮
            if is_actor:
                w["frame"].config(highlightbackground=ModernDarkTheme.ACCENT_CYAN, highlightthickness=1)
            else:
                w["frame"].config(highlightbackground=ModernDarkTheme.BORDER, highlightthickness=1)

            # 名称
            hero_tag = " (我)" if is_hero else ""
            turn_tag = " ▶" if is_actor else ""
            w["name"].config(
                text=f"Seat {s}{hero_tag}{turn_tag}",
                fg=ModernDarkTheme.ACCENT_CYAN if is_actor else ModernDarkTheme.TEXT_MAIN,
            )

            # 存活/中弹
            if not alive:
                w["status"].config(text=f"💥 淘汰 ({shots}枪)", fg=ModernDarkTheme.ACCENT_RED)
                w["cards"].config(text="出局", fg=ModernDarkTheme.TEXT_DIM)
            elif afk:
                w["status"].config(text=f"💤 挂机中", fg=ModernDarkTheme.TEXT_DIM)
                w["cards"].config(text=f"手牌: {cards}", fg=ModernDarkTheme.TEXT_DIM)
            else:
                w["status"].config(text=f"🔫 {shots}/6 枪", fg=ModernDarkTheme.TEXT_MUTED)
                w["cards"].config(text=f"手牌: {cards}", fg=ModernDarkTheme.ACCENT_GREEN if cards > 0 else ModernDarkTheme.ACCENT_ORANGE)

            # 挂机按钮
            if afk:
                w["btn_afk"].config(text="挂机: 是", bg=ModernDarkTheme.ACCENT_RED, fg="#ffffff")
            else:
                w["btn_afk"].config(text="挂机: 否", bg=ModernDarkTheme.BG_PANEL, fg=ModernDarkTheme.TEXT_MUTED)

        # 4. 刷新动作横幅与开枪状态引导
        lp_desc = ""
        if self.tracker.latest_play:
            lp_s = self.tracker.latest_play["seat"]
            lp_c = self.tracker.latest_play["count"]
            lp_desc = f" | 上手: 席位{lp_s} 出{lp_c}张"

        is_hero_turn = (self.tracker.current_seat == self.tracker.hero_seat)

        if self.tracker.pending_penalty_seat:
            p_seat = self.tracker.pending_penalty_seat
            banner_text = f"🚨【待开枪: 席位 {p_seat} 受罚】请看游戏内开枪结果，点击右侧 [🔫幸存] 或 [💥淘汰]！"
            self.lbl_action_banner.config(text=banner_text, bg="#5c1d1d", fg="#ffdddd")
            self.btn_shot_survive.config(bg=ModernDarkTheme.ACCENT_CYAN, fg="#000000", font=("Microsoft YaHei UI", 8, "bold"))
            self.btn_shot_die.config(bg=ModernDarkTheme.ACCENT_RED, fg="#ffffff", font=("Microsoft YaHei UI", 8, "bold"))
        elif is_hero_turn:
            banner_text = f"👉【轮到我方行动】选择你实际打出的牌或采纳AI建议{lp_desc}"
            self.lbl_action_banner.config(text=banner_text, bg="#1a3a2a", fg="#a7f3d0")
            self.btn_shot_survive.config(bg=ModernDarkTheme.BG_SUBPANEL, fg=ModernDarkTheme.ACCENT_CYAN, font=("Microsoft YaHei UI", 8))
            self.btn_shot_die.config(bg=ModernDarkTheme.BG_SUBPANEL, fg=ModernDarkTheme.ACCENT_RED, font=("Microsoft YaHei UI", 8))
        else:
            banner_text = f"第 {self.tracker.round_index} 轮 (点数 {self.tracker.claim_rank.value}) · 轮到 席位 {self.tracker.current_seat} 行动{lp_desc}"
            self.lbl_action_banner.config(text=banner_text, bg=ModernDarkTheme.BG_PANEL, fg=ModernDarkTheme.ACCENT_CYAN)
            self.btn_shot_survive.config(bg=ModernDarkTheme.BG_SUBPANEL, fg=ModernDarkTheme.ACCENT_CYAN, font=("Microsoft YaHei UI", 8))
            self.btn_shot_die.config(bg=ModernDarkTheme.BG_SUBPANEL, fg=ModernDarkTheme.ACCENT_RED, font=("Microsoft YaHei UI", 8))

        # 4.1 自适应切换我方行动 vs 对手行动面板
        if is_hero_turn:
            self.frame_opp_actions.pack_forget()
            self.frame_hero_actions.pack(fill=tk.X, padx=2, pady=1)

            t_cnt = self.tracker.hero_target_count
            nt_cnt = self.tracker.hero_nontarget_count
            has_g = self.tracker.hero_has_ghost

            self._update_btn_state(self.btn_hero_t1, t_cnt >= 1, ModernDarkTheme.ACCENT_GREEN)
            self._update_btn_state(self.btn_hero_t2, t_cnt >= 2, ModernDarkTheme.ACCENT_GREEN)
            self._update_btn_state(self.btn_hero_t3, t_cnt >= 3, ModernDarkTheme.ACCENT_GREEN)

            self._update_btn_state(self.btn_hero_nt1, nt_cnt >= 1, ModernDarkTheme.TEXT_MAIN)
            self._update_btn_state(self.btn_hero_nt2, nt_cnt >= 2, ModernDarkTheme.TEXT_MAIN)
            self._update_btn_state(self.btn_hero_nt3, nt_cnt >= 3, ModernDarkTheme.TEXT_MAIN)
            self._update_btn_state(self.btn_hero_mix, (t_cnt >= 1 and nt_cnt >= 1), ModernDarkTheme.TEXT_MAIN)
            self._update_btn_state(self.btn_hero_ghost, has_g, ModernDarkTheme.ACCENT_PURPLE)

            can_chal = (self.tracker.latest_play is not None and self.tracker.latest_play["seat"] != self.tracker.hero_seat)
            if can_chal:
                self.row_hero_chal.pack(fill=tk.X, padx=2, pady=1)
            else:
                self.row_hero_chal.pack_forget()
        else:
            self.frame_hero_actions.pack_forget()
            self.frame_opp_actions.pack(fill=tk.X, padx=2, pady=1)

        # 5. 调用 AI 引擎计算并渲染推演卡片
        eval_result = self.engine.evaluate(self.tracker)
        if hasattr(self, "lbl_active_model_badge"):
            self.lbl_active_model_badge.config(text=f"🤖 当前决策网络: {self.engine.current_model_key}")

        sampled_act = eval_result["sampled_action"]
        greedy_act = eval_result["greedy_action"]

        s_desc = sampled_act.get("desc", "等待局势变化...")
        s_prob = sampled_act.get("prob", 0.0)
        s_pct = f"{s_prob * 100:.1f}%" if s_prob > 0 else ""

        g_desc = greedy_act.get("desc", "等待局势变化...")
        g_prob = greedy_act.get("prob", 0.0)
        g_pct = f"{g_prob * 100:.1f}%" if g_prob > 0 else ""

        self.lbl_sampled_desc.config(text=f"{s_desc}  (概率: {s_pct})" if s_pct else s_desc)
        self.lbl_greedy_desc.config(text=f"{g_desc}  (权重: {g_pct})" if g_pct else g_desc)

        # 同步更新采纳按钮文本
        if is_hero_turn:
            adopt_pct = f" ({s_pct})" if s_pct else ""
            self.btn_adopt_ai.config(text=f"⚡ 一键采纳建议: {s_desc}{adopt_pct}")

        # Mini Bar 同步更新
        mini_summary = f"🎲 建议: {s_desc} ({s_pct})"
        if g_desc != s_desc:
            mini_summary += f" | ⭐ Top1: {g_desc} ({g_pct})"
        if eval_result["bluff_info"]["has_target"]:
            mini_summary += f" | 启发虚报 {eval_result['bluff_info']['bluff_prob']*100:.0f}%"
        self.mini_text_label.config(text=mini_summary)

        # 上家分析与提示
        self.lbl_bluff_analysis.config(text=f"📊 {eval_result['bluff_info']['summary']}")
        self.lbl_tactical_tips.config(text=f"💡 战术: {eval_result['tactical_tip']}")

        # 绘制动作概率 Canvas 条状图
        self._render_probability_bars(eval_result["action_dist"])

    def _render_probability_bars(self, action_dist: list[dict[str, Any]]) -> None:
        """在 Canvas 上自绘制高质感暗黑概率分布条。"""
        self.prob_canvas.delete("all")
        if not action_dist:
            self.prob_canvas.create_text(
                190, 60,
                text="当前暂无可执行动作或非我方回合",
                fill=ModernDarkTheme.TEXT_MUTED,
                font=("Microsoft YaHei UI", 8),
            )
            return

        w = self.prob_canvas.winfo_width()
        if w < 100:
            w = 390

        # 取 Top 4 动作展示
        display_acts = action_dist[:4]
        bar_h = 24
        start_y = 6

        for i, item in enumerate(display_acts):
            y = start_y + i * (bar_h + 6)
            prob = item["prob"]
            prob_pct = f"{prob * 100:.1f}%"
            desc = item["desc"]
            is_chal = item["is_challenge"]

            # 背景条
            self.prob_canvas.create_rectangle(
                8, y, w - 8, y + bar_h,
                fill=ModernDarkTheme.BG_SUBPANEL,
                outline=ModernDarkTheme.BORDER,
                width=1,
            )

            # 填充进度条
            fill_w = max(4, int((w - 16) * prob))
            fill_color = ModernDarkTheme.ACCENT_RED if is_chal else ModernDarkTheme.ACCENT_CYAN
            if i == 0 and not is_chal:
                fill_color = ModernDarkTheme.ACCENT_GREEN

            self.prob_canvas.create_rectangle(
                8, y, 8 + fill_w, y + bar_h,
                fill=fill_color,
                outline="",
            )

            # 动作文本 (左对齐)
            self.prob_canvas.create_text(
                16, y + bar_h // 2,
                text=desc,
                fill=ModernDarkTheme.TEXT_MAIN,
                font=("Microsoft YaHei UI", 8, "bold" if i == 0 else "normal"),
                anchor="w",
            )

            # 百分比文本 (右对齐)
            self.prob_canvas.create_text(
                w - 16, y + bar_h // 2,
                text=prob_pct,
                fill="#ffffff" if i == 0 else ModernDarkTheme.TEXT_MUTED,
                font=("Microsoft YaHei UI", 8, "bold"),
                anchor="e",
            )


# ===========================================================================
# 4. 自动化测试套件 (--test)
# ===========================================================================

def run_headless_tests() -> bool:
    """全自动功能与推演测试套件：验证各模块逻辑闭环与稳定性。"""
    print("\n" + "=" * 60)
    print("🚀 [小丑牌桌面AI助手] 开始自动化功能与模型推演测试...")
    print("=" * 60)

    # 1. 测试 AIDecisionEngine 与模型加载
    print("\n[Test 1/6] 测试模型发现、热加载与回退保护...")
    engine = AIDecisionEngine()
    models = engine.get_available_models()
    print(f" -> 本地可用模型列表: {models}")
    assert len(models) >= 1, "未找到任何可用模型！"
    assert engine.actor is not None, "默认模型加载失败！"
    assert engine.current_model_key == AIDecisionEngine.DEFAULT_MODEL_KEY
    print(f" -> 默认生产主力模型加载成功 ({engine.current_model_key}): OK")

    # 测试热加载 v39_B10 候选模型
    b10_key = "v39_B10 (强 PPO 对抗候选，真人表现待验证)"
    if b10_key in models:
        ok_b10 = engine.load_model(b10_key)
        assert ok_b10 is True, "v39_B10 热加载失败！"
        assert engine.current_model_key == b10_key, "当前加载版本未正确更新为 v39_B10！"
        print(f" -> 成功热切换至候选模型 ({engine.current_model_key}): OK")

    # 测试非法模型安全回退至默认生产主力 C_long
    engine.load_model("Invalid_Non_Existent_Model_Key")
    assert engine.current_model_key == AIDecisionEngine.DEFAULT_MODEL_KEY, "非法模型未能成功安全回退到默认主力 C_long！"
    print(f" -> 非法模型安全回退机制验证成功 (已自动恢复为默认主力: {engine.current_model_key}): OK")

    # 2. 测试 GameStateTracker 状态初始化与手牌调整
    print("\n[Test 2/6] 测试 GameStateTracker 局势初始化与方案B手牌...")
    tracker = GameStateTracker(hero_seat=1, claim_rank=Rank.A, player_count=4)
    tracker.set_hero_hand(target_count=2, nontarget_count=2, has_ghost=True)
    assert tracker.hero_target_count == 2
    assert tracker.hero_nontarget_count == 2
    assert tracker.hero_has_ghost is True
    assert tracker.seat_hand_counts[1] == 5
    print(" -> 手牌配置方案B更新: OK")

    # 3. 测试合法动作生成与神经网络推演
    print("\n[Test 3/6] 测试动作生成与神经网络推演...")
    eval_res = engine.evaluate(tracker)
    assert "best_action_desc" in eval_res
    assert "action_dist" in eval_res
    assert len(eval_res["action_dist"]) > 0
    total_prob = sum(a["prob"] for a in eval_res["action_dist"])
    print(f" -> 最优推荐: {eval_res['best_action_desc']} ({eval_res['best_prob']*100:.1f}%)")
    print(f" -> 动作分布总概率和: {total_prob:.4f}")
    assert abs(total_prob - 1.0) < 1e-3, "Softmax 动作概率和不为 1！"
    print(" -> 决策推演与 Softmax 归一化: OK")

    # 4. 测试对局流程演进：出牌、质疑与开枪结算
    print("\n[Test 4/6] 测试对局动作记录与轮转流转...")
    # Seat 1 出 1 张目标牌
    tracker.record_play(seat=1, count=1, hero_used_target=1)
    assert tracker.hero_target_count == 1
    assert tracker.latest_play["seat"] == 1
    assert tracker.current_seat == 2

    # Seat 2 出 2 张牌
    tracker.record_play(seat=2, count=2)
    assert tracker.seat_hand_counts[2] == 3
    assert tracker.current_seat == 3

    # Seat 3 发起质疑 -> 验出假牌
    penalty_seat = tracker.record_challenge(challenging_seat=3, outcome="lie")
    assert penalty_seat == 2, "验出假牌时出牌者(Seat 2)应受罚！"

    # Seat 2 开枪幸存
    tracker.record_shot(seat=2, died=False)
    assert tracker.shots_taken[2] == 1
    print(" -> 出牌、质疑、开枪与换轮流转: OK")

    # 5. 测试挂机功能 (AFK) 与撤销 (Undo)
    print("\n[Test 5/6] 测试挂机标记与撤销功能...")
    tracker.toggle_afk(3)
    assert tracker.player_afk[3] is True
    # 严格落实用户铁律：挂机状态不得自行改变行动规则！
    next_s = tracker.get_next_active_seat(from_seat=2, allow_self=False)
    assert next_s == 3, f"挂机玩家 Seat 3 仍应遵循物理顺时针轮转，实际为: {next_s}"
    assert tracker.is_seat_active(3) is True, "挂机玩家存活时物理上仍为活跃席位！"

    # 测试撤销
    undo_ok = tracker.undo()
    assert undo_ok is True
    assert tracker.player_afk[3] is False
    print(" -> 挂机状态不改变物理规则 & 快照撤销: OK")

    # 5.1 测试两人局末手打空强制质疑真实物理规则
    print("\n[Test 5.1] 测试两人局末手打空强制质疑真实物理规则...")
    two_p_tracker = GameStateTracker(hero_seat=1, player_count=2, claim_rank=Rank.A, starting_seat=2)
    # Seat 2 出牌打空
    two_p_tracker.record_play(seat=2, count=1)
    two_p_tracker.seat_hand_counts[2] = 0
    # 此时仅剩 2 人且 Seat 2 打空，Seat 1 唯一合法动作必须且只能是质疑！
    legals = two_p_tracker.generate_legal_actions()
    assert len(legals) == 1, f"两人残局对方打空，合法动作应唯一，实际有: {len(legals)}"
    assert isinstance(legals[0], ChallengeAction), f"两人残局对方打空，合法动作必须为质疑，实际为: {type(legals[0])}"
    print(" -> 两人局末手打空强制质疑断言: OK")

    # 6. 测试 Tkinter UI 初始化与形态切换 (无头模式)
    print("\n[Test 6/6] 测试 Tkinter 悬浮窗组件、我方自主出牌与采纳推演...")
    root = tk.Tk()
    root.withdraw()  # 无头隐藏
    ui = DesktopAssistantUI(root=root, tracker=tracker, engine=engine)
    ui.show_mini_bar()
    assert ui.is_mini_mode is True
    ui.show_full_panel()
    assert ui.is_mini_mode is False

    # 测试我方实际自主出牌：出 1 张真牌
    tracker.current_seat = 1
    tracker.hero_target_count = 2
    tracker.hero_nontarget_count = 3
    ui._on_record_hero_play(target_cnt=1)
    assert tracker.hero_target_count == 1, f"出真牌后目标牌应剩余 1，实际: {tracker.hero_target_count}"
    assert tracker.hero_nontarget_count == 3, f"出真牌后杂牌应仍为 3，实际: {tracker.hero_nontarget_count}"
    assert tracker.latest_play["count"] == 1
    assert tracker.current_seat == 2

    # 测试我方实际自主诈唬：出 2 张杂牌
    tracker.current_seat = 1
    ui._on_record_hero_play(nontarget_cnt=2)
    assert tracker.hero_target_count == 1, f"出杂牌后目标牌应不变，实际: {tracker.hero_target_count}"
    assert tracker.hero_nontarget_count == 1, f"出2张杂牌后杂牌应剩余 1，实际: {tracker.hero_nontarget_count}"
    assert tracker.latest_play["count"] == 2

    # 测试一键采纳 AI 实战建议
    tracker.current_seat = 1
    ui._on_adopt_ai_action()
    assert tracker.current_seat != 1, "采纳 AI 建议后应自动推进至下家行动！"

    ui.refresh_ui()
    root.update()
    root.destroy()
    print(" -> Tkinter 悬浮窗自主出牌、一键采纳与双形态切换: OK")

    print("\n" + "=" * 60)
    print("🎉 [SUCCESS] 所有 6 项功能与推演测试全部通过！系统运行稳定健壮！")
    print("=" * 60 + "\n")
    return True


# ===========================================================================
# 5. 主程序入口
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="逆水寒小丑牌 桌面原生置顶悬浮窗 AI 决策辅助工具")
    parser.add_argument("--test", action="store_true", help="运行无头自动化测试套件并退出")
    parser.add_argument("--hero-seat", type=int, default=1, choices=[1, 2, 3, 4], help="我方所在的席位 (默认 1)")
    parser.add_argument("--rank", type=str, default="A", choices=["A", "K", "Q"], help="初始轮次点数 (默认 A)")
    parser.add_argument("--players", type=int, default=4, choices=[2, 3, 4], help="对局总人数 (默认 4)")
    parser.add_argument("--mini", action="store_true", help="启动时直接以微型胶囊条 (Mini Bar) 模式悬浮")
    args = parser.parse_args()

    if args.test:
        success = run_headless_tests()
        sys.exit(0 if success else 1)

    print("\n[Desktop Assistant] 正在初始化逆水寒小丑牌桌面决策助手...")
    claim_rank = Rank(args.rank)
    tracker = GameStateTracker(
        hero_seat=args.hero_seat,
        player_count=args.players,
        claim_rank=claim_rank,
    )
    engine = AIDecisionEngine()

    root = tk.Tk()
    ui = DesktopAssistantUI(root=root, tracker=tracker, engine=engine)

    if args.mini:
        ui.show_mini_bar()

    print("[Desktop Assistant] 桌面置顶悬浮窗启动成功！开始事件循环...")
    root.mainloop()


if __name__ == "__main__":
    main()
