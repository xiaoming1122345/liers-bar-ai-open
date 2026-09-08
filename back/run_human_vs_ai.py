from __future__ import annotations

"""
本地人机对弈终端客户端 (Human vs AI Showdown)

选手阵容：
- 1 号位：你 [人类玩家]
- 2 号位：AI-C (v28_C_single 终点模型, 104维 control, masked_softmax)
- 3 号位：AI-D50 (v22_D50 第50轮, 80维, linear_norm)
- 4 号位：AI-v21 (v21 第25轮规则续演, 80维, linear_norm)

规则与计分：
- 规则：修正后标准 SurvivalGameEngine，4人各5张手牌，包含万能 Wild 与鬼牌反杀。
- 计分：标准 [20, 15, -5, -30] 终局名次积分。
- 两局制：第 1 局玩家先手发牌 (Seat 1)，第 2 局 AI-C 先手发牌 (Seat 2) 玩家后手接牌。
- 信息边界：仅公开已消耗子弹 shots_taken 与公开牌桌历史，严格隔离对手私有手牌与弹仓实弹位置。
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

# Windows 终端 UTF-8 强制
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# 切换并引入当前项目根路径
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
os.chdir(ROOT_DIR)

import numpy as np
import torch
from card_ai.contextual_profiler import ContextualActionPredictor
from card_ai.engine import SurvivalGameEngine
from card_ai.rewards import RankRewardModel, RankScoreRules
from card_ai.standard_eval import StandardCandidatePolicy
from card_ai.types import (
    Card,
    CardKind,
    ChallengeAction,
    ChallengeOutcome,
    GameAction,
    GameState,
    PlayAction,
    PrivateObservation,
    Rank,
)

# 加载全局动作偏好预估器 (若已有联赛数据则载入，否则使用经典先验)
_GLOBAL_PREDICTOR = ContextualActionPredictor(profile_path=ROOT_DIR / "runs_ppo" / "league_profiling_results_2800.json")

RULE_VERSION_TAG = "逆水寒官方真实物理规则 (两人局末手打空强制质疑 | 3/4人局安全脱手跳过 | 鬼牌单出免死反杀)"

DEFAULT_CLONG_PATH = "runs_ppo/v29_long_ppo/checkpoint_iter_00120.pt"

MODEL_SPECS = {
    2: {
        "name": "AI-C (C_long 生产主力)",
        "desc": "120轮 PPO 生产绝对主力 (104dim PPO)",
        "path": DEFAULT_CLONG_PATH,
        "mode": "control",
        "prob_mode": "masked_softmax",
    },
    3: {
        "name": "AI-D50 (CFR高压反制)",
        "desc": "极度敏锐抓假型专家 (80dim CFR)",
        "path": "runs_deep_cfr/v22_D_top2_penalty_512/checkpoint_iter_00050.pt",
        "mode": "control",
        "prob_mode": "linear_norm",
    },
    4: {
        "name": "AI-v21 (CFR均衡基准)",
        "desc": "规则续演防守稳健型 (80dim CFR)",
        "path": "runs_deep_cfr/deep_cfr_v21_fixed_rule_rollout_512/policy_iter_00025_baseline.pt",
        "mode": "control",
        "prob_mode": "linear_norm",
    },
}

HUMAN_SEAT = 1
ROLE_NAMES = {
    1: "你 [人类玩家]",
    2: "AI-C (C_long 生产主力)",
    3: "AI-D50 (CFR高压反制)",
    4: "AI-v21 (CFR均衡基准)",
}


def configure_candidate_model(model_type: str = "clong", candidate_path: str | None = None) -> None:
    """动态切换 2 号位对阵模型，支持一键回退到原版 C_long。"""
    if model_type == "candidate" and candidate_path:
        p = Path(candidate_path)
        if not p.is_file():
            raise FileNotFoundError(f"找不到指定的候选模型权重: {candidate_path}")
        MODEL_SPECS[2]["name"] = f"AI-C ({p.stem} 续训候选)"
        MODEL_SPECS[2]["desc"] = f"实验续训候选模型 ({p.name})"
        MODEL_SPECS[2]["path"] = str(p)
        ROLE_NAMES[2] = f"AI-C ({p.stem} 续训候选)"
    else:
        # 回退至原版 C_long
        MODEL_SPECS[2]["name"] = "AI-C (C_long 生产主力)"
        MODEL_SPECS[2]["desc"] = "120轮 PPO 生产绝对主力 (104dim PPO)"
        MODEL_SPECS[2]["path"] = DEFAULT_CLONG_PATH
        ROLE_NAMES[2] = "AI-C (C_long 生产主力)"



def card_display_str(c: Card) -> str:
    """卡牌显示文本。"""
    if c.kind == CardKind.WILD:
        return "[万能 Wild]"
    if c.kind == CardKind.GHOST:
        rk = c.printed_rank.value if c.printed_rank else "?"
        return f"[鬼牌 Ghost({rk})]"
    return f"[{c.printed_rank.value}]"


def card_type_desc(c: Card, claim_rank: Rank | None) -> str:
    """卡牌在当前目标点数下的性质描述。"""
    if c.kind == CardKind.WILD:
        return "万能真牌"
    if c.kind == CardKind.GHOST:
        return "鬼牌 (单出庇护反杀)"
    if claim_rank is None:
        return f"普通牌 {c.printed_rank.value}"
    if c.printed_rank == claim_rank:
        return "目标真牌"
    return "非目标牌 (打出算诈唬)"


def load_ai_policies() -> dict[int, StandardCandidatePolicy]:
    """严格加载三大指定模型。"""
    print("\n" + "=" * 65)
    print("  📦 正在加载参赛 AI 神经网络模型...")
    print("=" * 65)
    policies = {}
    for seat, spec in MODEL_SPECS.items():
        p = Path(spec["path"])
        if not p.is_file():
            raise FileNotFoundError(f"[致命错误] 找不到模型检查点: {p.resolve()}")
        pol = StandardCandidatePolicy(
            checkpoint_path=p,
            mode=spec["mode"],
            observer_seat=seat,
            prob_mode=spec["prob_mode"],
        )
        pol.policy.strategy_net.eval()
        policies[seat] = pol
        print(f"  [OK] 席位 {seat} | {spec['name']:20s} | 权重: {spec['path']}")
        print(f"       特征: {'104维' if pol.is_104dim else '80维'} | 决策模式: {pol.prob_mode} | 风格: {spec['desc']}")
    print("=" * 65 + "\n")
    return policies


# ---------------------------------------------------------------------------
# 界面渲染与展示
# ---------------------------------------------------------------------------

def render_table(state: GameState, hero_obs: PrivateObservation, current_actor: int) -> None:
    """渲染牌桌公共信息（严格隔离对手手牌和弹仓实弹位置）。"""
    claim = state.round_state.claim_rank.value if state.round_state.claim_rank else "未定"
    played_cards = sum(len(ps.cards) for ps in state.round_state.plays)
    latest = state.round_state.latest_play

    print("\n" + "━" * 68)
    print(f"  🔥【第 {state.round_index} 轮】 本轮锁定目标点数: 【 {claim} 】 | 桌面叠牌池: {played_cards} 张")
    if latest is not None:
        last_name = ROLE_NAMES.get(latest.seat, f"{latest.seat}号位")
        print(f"  🎴 上一手出牌: 【{last_name}】 盖放了 {len(latest.cards)} 张牌，声称是【{latest.claim_rank.value}】")
    else:
        print("  🎴 上一手出牌: 无 (新一轮首发，等待出牌)")
    print("━" * 68)

    print("  席位   选手名称                状态    手牌数  轮盘已消耗子弹 (最多6发)")
    print("  " + "─" * 64)

    for p in state.players:
        name = ROLE_NAMES.get(p.seat, f"{p.seat}号位")
        is_acting = (p.seat == current_actor)
        mark = " 👈【行动中】" if is_acting else ""
        if not p.alive:
            status = "💀 淘汰"
        elif len(p.hand) == 0:
            status = "✨ 已脱手"
        else:
            status = "🟢 存活"

        # 轮盘消耗子弹展示 (shots_taken 是公开信息，实弹孔位是绝对机密！)
        # 弹膛共 6 格，💥 表示已扣动开火的次数（即已消耗空枪），○ 表示尚未触及的弹膛
        bullet_bar = "💥" * p.shots_taken + "○" * max(0, 6 - p.shots_taken)
        bullet_text = f"{bullet_bar} (已扣 {p.shots_taken} 次)" if p.alive else f"{bullet_bar} (已淘汰)"

        # 手牌数展示（公开信息只有张数）
        hand_count_str = f"{len(p.hand)} 张"

        print(f"  {p.seat}号位  {name:<22}  {status}   {hand_count_str:<6}  {bullet_text}{mark}")
    print("━" * 68)


def render_my_hand(hero_obs: PrivateObservation) -> None:
    """渲染人类玩家自己的手牌。"""
    claim_rank = hero_obs.round_claim_rank
    print(f"\n  🃏 你的私有手牌（共 {len(hero_obs.hero_hand)} 张）：")
    for i, c in enumerate(hero_obs.hero_hand, 1):
        desc = card_type_desc(c, claim_rank)
        print(f"    [{i}] {card_display_str(c):<16} ({desc})")


def format_action_label(act: GameAction, hero_obs: PrivateObservation) -> tuple[str, str]:
    """格式化动作显示文本并标明性质。"""
    if isinstance(act, ChallengeAction):
        opp_name = ROLE_NAMES.get(act.challenged_seat, f"{act.challenged_seat}号位")
        return "⚡ 拍案质疑", f"质疑 {act.challenged_seat} 号位【{opp_name}】上一手的出牌！"

    if isinstance(act, PlayAction):
        cards = [c for c in hero_obs.hero_hand if c.card_id in act.card_ids]
        cnames = " + ".join(card_display_str(c) for c in cards)
        claim_val = act.claim_rank.value

        # 判断是否为纯诚实
        is_ghost = any(c.kind == CardKind.GHOST for c in cards)
        is_bluff = any(c.kind == CardKind.NORMAL and c.printed_rank != act.claim_rank for c in cards)

        if is_ghost:
            tag = "👻 鬼牌特出"
        elif is_bluff:
            tag = "🎭 诈唬虚报"
        else:
            tag = "✨ 诚实真牌"

        return tag, f"盖放 {len(cards)} 张（{cnames}），声称是【{claim_val}】"

    return "未知操作", str(act)


# ---------------------------------------------------------------------------
# 玩家交互与输入解析
# ---------------------------------------------------------------------------

def prompt_human_action(
    legal_actions: tuple[GameAction, ...],
    hero_obs: PrivateObservation,
    simulate: bool = False,
) -> GameAction:
    """提示玩家操作并支持智能解析输入。"""
    # 模拟模式（供自动化测试）
    if simulate:
        # 如果有质疑，以 25% 概率质疑，其余出牌
        challenges = [a for a in legal_actions if isinstance(a, ChallengeAction)]
        plays = [a for a in legal_actions if isinstance(a, PlayAction)]
        if challenges and random.random() < 0.25:
            return challenges[0]
        return random.choice(plays if plays else legal_actions)

    # 仅 1 个操作（通常是最后强制质疑）
    if len(legal_actions) == 1:
        act = legal_actions[0]
        tag, desc = format_action_label(act, hero_obs)
        print(f"\n  ⚠️  根据规则，当前为唯一合法动作（强制触发）：{tag} -> {desc}")
        input("  👉 请按回车键执行确认...")
        return act

    print(f"\n  ⚔️  可执行操作列表（共 {len(legal_actions)} 项）：")

    # 分组显示操作：1. 质疑操作；2. 出牌操作
    challenge_act = None
    play_actions: list[tuple[int, GameAction, str, str]] = []

    for idx, act in enumerate(legal_actions, 1):
        tag, desc = format_action_label(act, hero_obs)
        if isinstance(act, ChallengeAction):
            challenge_act = (idx, act, tag, desc)
        else:
            play_actions.append((idx, act, tag, desc))

    if challenge_act is not None:
        c_idx, _, tag, desc = challenge_act
        print(f"    [{c_idx:>2}] (或按 'c')  {tag:<10}  {desc}")
        print("    " + "─" * 58)

    for p_idx, _, tag, desc in play_actions:
        print(f"    [{p_idx:>2}]              {tag:<10}  {desc}")

    print("\n  💡 输入提示：")
    print("     - 直接输入操作序号（如 1、2、3...）")
    if challenge_act:
        print("     - 输入 'c' 快速发起【拍案质疑】")
    print("     - 输入要出的手牌序号组合（例如输入 '1' 或 '1 2'）自动匹配出牌")

    while True:
        try:
            raw = input(f"\n  👉 请选择你的行动 (1-{len(legal_actions)}): ").strip().lower()
            if not raw:
                continue

            # 快捷质疑指令
            if raw in ("c", "challenge", "zy") and challenge_act is not None:
                return challenge_act[1]

            # 直接输入编号
            if raw.isdigit():
                val = int(raw)
                if 1 <= val <= len(legal_actions):
                    return legal_actions[val - 1]

            # 尝试手牌序号匹配（如 "1 2" 或 "2"）
            parts = raw.split()
            if all(p.isdigit() for p in parts):
                hand_indices = [int(p) - 1 for p in parts]
                if all(0 <= hi < len(hero_obs.hero_hand) for hi in hand_indices):
                    target_card_ids = set(hero_obs.hero_hand[hi].card_id for hi in hand_indices)
                    # 匹配 legal_actions 中的出牌
                    matching_plays = [
                        a for a in legal_actions
                        if isinstance(a, PlayAction) and set(a.card_ids) == target_card_ids
                    ]
                    if len(matching_plays) == 1:
                        return matching_plays[0]
                    elif len(matching_plays) > 1:
                        # 多个 claim_rank 可选
                        print(f"  👉 所选卡牌支持声明多种点数：")
                        for mi, ma in enumerate(matching_plays, 1):
                            print(f"     [{mi}] 声称 {ma.claim_rank.value}")
                        c_raw = input("     请选择声称序号: ").strip()
                        if c_raw.isdigit() and 1 <= int(c_raw) <= len(matching_plays):
                            return matching_plays[int(c_raw) - 1]

            print(f"  ⚠️  输入无法识别，请输入 1 到 {len(legal_actions)} 之间的编号，或输入 'c' 质疑！")
        except (KeyboardInterrupt, EOFError):
            print("\n\n  🚪 游戏已退出。")
            sys.exit(0)


# ---------------------------------------------------------------------------
# 单局执行流程
# ---------------------------------------------------------------------------

def run_single_match(
    match_index: int,
    starting_seat: int,
    ai_policies: dict[int, StandardCandidatePolicy],
    delay_sec: float = 1.2,
    simulate_human: bool = False,
) -> dict[str, Any]:
    """运行一场完整人机对局并生成详细日志。"""
    engine = SurvivalGameEngine()
    game_seed = int(time.time() * 1000) % 10000000 + match_index * 137
    state = engine.new_game(seed=game_seed, starting_seat=starting_seat)

    # 重置 AI 策略的局内记忆
    for p in ai_policies.values():
        p.reset_for_new_game()

    starter_name = ROLE_NAMES[starting_seat]
    print("\n" + "█" * 68)
    print(f"  🏆【第 {match_index} 局】 开局！本局发牌先手：【{starter_name}】")
    if starting_seat == HUMAN_SEAT:
        print("  💡 本局提示：你是先手！拥有首发定点数特权，掌控战局进攻节奏！")
    else:
        print("  💡 本局提示：AI 率先发牌！请保持警惕，注意后手接牌与反诈抓假！")
    print("█" * 68)

    step_count = 0
    match_log: list[dict[str, Any]] = []

    while not engine.is_terminal(state) and step_count < 250:
        step_count += 1
        actor = engine.current_actor(state)
        legal_actions = engine.legal_actions(state)

        if not legal_actions:
            break

        hero_obs = engine.observe(state, hero_seat=HUMAN_SEAT)

        # 牌桌展示
        render_table(state, hero_obs, current_actor=actor)

        # 决策执行
        if actor == HUMAN_SEAT:
            render_my_hand(hero_obs)
            action = prompt_human_action(legal_actions, hero_obs, simulate=simulate_human)
        else:
            ai_name = ROLE_NAMES[actor]
            print(f"\n  ⏳ 【{ai_name}】 正在观察局势与暗中推演...")

            # 寻找该 AI 的下家（接牌者）
            seats_all = [p.seat for p in state.players]
            start_idx = seats_all.index(actor)
            succ_seat = None
            for off in range(1, len(seats_all)):
                s = seats_all[(start_idx + off) % len(seats_all)]
                if state.player_by_seat(s).alive:
                    succ_seat = s
                    break
            succ_name = ROLE_NAMES.get(succ_seat, "未知")

            # 战术偏好画像预估
            pred_key = "C_long" if actor == 2 else ("v22_D50" if actor == 3 else "v21")
            succ_key = "人类" if succ_seat == 1 else ("C_long" if succ_seat == 2 else ("v22_D50" if succ_seat == 3 else "v21"))
            pred_info = _GLOBAL_PREDICTOR.predict_play_style(pred_key, succ_key)
            if pred_info and "summary" in pred_info:
                print(f"  💡 【动作偏好预估】 {pred_info['summary']}")

            if delay_sec > 0:
                time.sleep(delay_sec)

            ai_obs = engine.observe(state, hero_seat=actor)
            step_seed = (game_seed * 1000003 + step_count * 1009 + actor * 37) & 0x7FFFFFFF
            torch.manual_seed(step_seed)
            random.seed(step_seed)
            if hasattr(ai_policies[actor], "_rng"):
                ai_policies[actor]._rng = random.Random(step_seed)

            with torch.no_grad():
                action = ai_policies[actor].choose_action(ai_obs, legal_actions)

        # 记录行动
        prev_len = len(state.public_history)
        action_desc = ""

        if isinstance(action, PlayAction):
            act_name = ROLE_NAMES[actor]
            action_desc = f"{act_name} 盖放了 {len(action.card_ids)} 张牌，声称是【{action.claim_rank.value}】"
            print(f"\n  🎴 【出牌】 {action_desc}！")
        elif isinstance(action, ChallengeAction):
            act_name = ROLE_NAMES[actor]
            target_name = ROLE_NAMES[action.challenged_seat]
            action_desc = f"{act_name} 拍案发难，质疑 {target_name} 刚刚出的牌！"
            print(f"\n  ⚡ 【质疑】 {action_desc}")

        # 应用动作
        engine.apply_action(state, action)

        # 捕捉并公开播报质疑与枪决结果
        new_events = state.public_history[prev_len:]
        for ev in new_events:
            if ev.event_type == "challenge":
                outcome = ev.detail.get("outcome", "")
                revealed = ev.detail.get("revealed_cards", [])
                rev_str = " + ".join(f"[{c}]" for c in revealed)
                print(f"  🔍 【公开验牌】 翻开底牌：{rev_str}")

                if outcome == "honest":
                    print("  ✅ 【判定结果】 诚实无虚！出牌完全属实！质疑者判断失误，必须扣动扳机接受惩罚！")
                elif outcome == "lie":
                    print("  ❌ 【判定结果】 假牌当场被抓！出牌者涉嫌诈唬！出牌者必须扣动扳机接受惩罚！")
                elif outcome == "ghost":
                    print("  👻 【判定结果】 鬼牌庇护反杀！出牌者受神力护体免除惩罚！质疑者及未出完手牌者皆受连带惩罚！")

            elif ev.event_type == "shot":
                shot_seat = ev.seat
                shot_name = ROLE_NAMES.get(shot_seat, f"{shot_seat}号位")
                died = ev.detail.get("died", False)
                shots_taken = ev.detail.get("shots_taken", 0)

                print(f"  🔫 【轮盘开火】 轮到 {shot_name} 扣动轮盘扳机...")
                if delay_sec > 0:
                    time.sleep(delay_sec * 0.8)

                if died:
                    print(f"     💥💥💥【砰！！！实弹击发！！！】")
                    print(f"     💀 {shot_name} 不幸中弹身亡，当场淘汰！(弹仓累计扣动 {shots_taken}/6)")
                else:
                    print(f"     💨💨💨【咔…… 空枪击发！】")
                    print(f"     🟢 {shot_name} 侥幸生还！(弹仓累计扣动 {shots_taken}/6)")

            elif ev.event_type == "new_round":
                new_rk = ev.detail.get("claim_rank", "?")
                new_idx = ev.detail.get("round_index", "?")
                print(f"\n  📢【洗牌换轮】 本小轮结束！进入第 {new_idx} 轮！新目标点数更换为: 【 {new_rk} 】！")
                print("     存活玩家重新补满 5 张手牌，弃牌堆已清空！")
                if delay_sec > 0:
                    time.sleep(delay_sec * 0.5)

        match_log.append({
            "step": step_count,
            "actor": actor,
            "action": str(action),
            "events": [
                {"type": e.event_type, "seat": e.seat, "detail": e.detail}
                for e in new_events
            ],
        })

    # 终局计分结算
    reward_model = RankRewardModel(RankScoreRules(first=20.0, second=15.0, third=-5.0, fourth=-30.0))
    rewards = reward_model.evaluate(state)

    # 确定名次
    survivors = [p.seat for p in state.players if p.alive]
    eliminated = [p.seat for p in state.players if not p.alive]

    # 按分数高低排序名次
    seat_scores = {s: rewards.for_seat(s) for s in range(1, 5)}
    sorted_seats = sorted(range(1, 5), key=lambda s: seat_scores[s], reverse=True)

    print("\n" + "🏁" * 34)
    print(f"  🏁🏁🏁【第 {match_index} 局 对局终局！】🏁🏁🏁")
    print("🏁" * 34)
    print("  最终胜者: " + (ROLE_NAMES[survivors[0]] if survivors else "无幸存者"))
    print("\n  本局结算积分榜（计分标准 [20, 15, -5, -30]）：")
    print("  " + "─" * 58)
    print("  名次   席位   选手名称                局终状态    本局积分")
    print("  " + "─" * 58)

    rank_map = {}
    for r_idx, seat in enumerate(sorted_seats, 1):
        rank_map[seat] = r_idx
        name = ROLE_NAMES[seat]
        st = "🟢 幸存吃鸡" if seat in survivors else "💀 中弹淘汰"
        sc = seat_scores[seat]
        print(f"  第 {r_idx} 名  {seat}号位  {name:<22}  {st}   {sc:+6.1f} 分")
    print("  " + "─" * 58)

    return {
        "match_index": match_index,
        "starting_seat": starting_seat,
        "game_seed": game_seed,
        "steps": step_count,
        "scores": seat_scores,
        "ranks": rank_map,
        "survivors": survivors,
        "eliminated": eliminated,
        "log": match_log,
    }


# ---------------------------------------------------------------------------
# 主流程：两局对战与复盘保存
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="逆水寒生死左轮卡牌 —— 人机本地终极对弈与人工监督试用")
    parser.add_argument("--model", type=str, choices=["clong", "candidate"], default="clong", help="2号位模型选择: clong (默认原版) 或 candidate (续训候选)")
    parser.add_argument("--candidate-path", type=str, default=None, help="自定义续训候选模型权重路径 (仅当 --model candidate 时生效)")
    parser.add_argument("--delay", type=float, default=1.0, help="AI 思考与播报节奏延时秒数 (默认 1.0 秒)")
    parser.add_argument("--fast", action="store_true", help="极速模式 (延时设为 0)")
    parser.add_argument("--simulate-human", action="store_true", help="自动化模拟玩家决策 (用于快速测试或无人值守核验)")
    parser.add_argument("--trial-mode", action="store_true", help="启用人工监督试用模式 (连续推进并结构化记录对局数据)")
    parser.add_argument("--trial-games", type=int, default=20, help="人工监督试用局数 (默认 20 局)")
    args = parser.parse_args()

    # 1. 配置对阵模型与规则标识
    configure_candidate_model(model_type=args.model, candidate_path=args.candidate_path)
    delay = 0.0 if args.fast else args.delay

    print("\n" + "╔" + "═" * 68 + "╗")
    print("║        🎮 逆水寒 · 生死左轮卡牌 · 人类玩家与 AI 本地对决战          ║")
    print("╚" + "═" * 68 + "╝")
    print(f"  📜 物理规则: {RULE_VERSION_TAG}")
    print(f"  🤖 2号位加载: 【{MODEL_SPECS[2]['name']}】")
    print(f"     权重路径: {MODEL_SPECS[2]['path']}")
    print("  对弈阵容：")
    print(f"    - 1 号位：{ROLE_NAMES[1]}")
    print(f"    - 2 号位：{ROLE_NAMES[2]} ({MODEL_SPECS[2]['desc']})")
    print(f"    - 3 号位：{ROLE_NAMES[3]} ({MODEL_SPECS[3]['desc']})")
    print(f"    - 4 号位：{ROLE_NAMES[4]} ({MODEL_SPECS[4]['desc']})")
    print("═" * 70)

    ai_policies = load_ai_policies()

    # -----------------------------------------------------------------------
    # 模式 A: 20 局人工监督试用模式 (Trial Mode)
    # -----------------------------------------------------------------------
    if args.trial_mode:
        num_trials = args.trial_games
        print("\n" + "█" * 70)
        print(f"  🔬 【人工监督试用模式启动】 计划进行 {num_trials} 局连续实战对决")
        print("  💡 目的定位：排查操作录入体验、UI 反馈、异常阻断及具体局势策略短板。")
        print("  💡 不以 20 局宣称长期统计胜率，数据将沉淀至 runs_ppo/v38_human_trial_logs_20.json")
        print("█" * 70 + "\n")

        trial_records = []
        human_scores = []
        human_ranks = []
        ai_c_scores = []
        ai_c_ranks = []

        for g_idx in range(1, num_trials + 1):
            starting_seat = ((g_idx - 1) % 4) + 1
            if not args.simulate_human:
                input(f"\n  👉 请按回车键开始【第 {g_idx}/{num_trials} 局】(本局发牌先手: {ROLE_NAMES[starting_seat]})...")

            res = run_single_match(
                match_index=g_idx,
                starting_seat=starting_seat,
                ai_policies=ai_policies,
                delay_sec=delay,
                simulate_human=args.simulate_human,
            )

            h_sc = res["scores"][1]
            h_rk = res["ranks"][1]
            c_sc = res["scores"][2]
            c_rk = res["ranks"][2]

            human_scores.append(h_sc)
            human_ranks.append(h_rk)
            ai_c_scores.append(c_sc)
            ai_c_ranks.append(c_rk)

            trial_records.append({
                "match_index": g_idx,
                "starting_seat": starting_seat,
                "game_seed": res["game_seed"],
                "human_score": h_sc,
                "human_rank": h_rk,
                "ai_c_score": c_sc,
                "ai_c_rank": c_rk,
                "steps": res["steps"],
                "match_log": res["log"],
            })

        # 试用总结
        trial_log_path = Path("runs_ppo/v38_human_trial_logs_20.json")
        trial_log_path.parent.mkdir(parents=True, exist_ok=True)

        h_top1 = sum(1 for r in human_ranks if r <= 1.0) / len(human_ranks)
        h_top2 = sum(1 for r in human_ranks if r <= 2.0) / len(human_ranks)
        c_top1 = sum(1 for r in ai_c_ranks if r <= 1.0) / len(ai_c_ranks)
        c_top2 = sum(1 for r in ai_c_ranks if r <= 2.0) / len(ai_c_ranks)

        summary_payload = {
            "trial_type": "human_supervised_trial",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "rule_version": RULE_VERSION_TAG,
            "tested_model": {
                "name": MODEL_SPECS[2]["name"],
                "path": MODEL_SPECS[2]["path"],
            },
            "num_games": num_trials,
            "human_summary": {
                "mean_score": float(np.mean(human_scores)),
                "top1_rate": h_top1,
                "top2_rate": h_top2,
            },
            "ai_c_summary": {
                "mean_score": float(np.mean(ai_c_scores)),
                "top1_rate": c_top1,
                "top2_rate": c_top2,
            },
            "records": trial_records,
        }

        with open(trial_log_path, "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, ensure_ascii=False, indent=2)

        print("\n\n" + "╔" + "═" * 68 + "╗")
        print("║                   🏁 人工监督试用 20 局总评测完成                   ║")
        print("╚" + "═" * 68 + "╝")
        print(f"  人类玩家: 均分 {np.mean(human_scores):+.2f} | 胜率(Top1) {h_top1*100:.1f}% | 前二率 {h_top2*100:.1f}%")
        print(f"  AI-C模型: 均分 {np.mean(ai_c_scores):+.2f} | 胜率(Top1) {c_top1*100:.1f}% | 前二率 {c_top2*100:.1f}%")
        print(f"  💾 详细实战对局明细与动作记录已保存至: {trial_log_path}")
        print("═" * 70)
        return

    # -----------------------------------------------------------------------
    # 模式 B: 常规双局对决模式 (Classic 2-match showdown)
    # -----------------------------------------------------------------------
    if not args.simulate_human:
        input("  👉 准备就绪，请按回车键开始【第 1 局】(你先手)...")

    m1_result = run_single_match(
        match_index=1,
        starting_seat=1,
        ai_policies=ai_policies,
        delay_sec=delay,
        simulate_human=args.simulate_human,
    )

    if not args.simulate_human:
        print("\n" + "=" * 68)
        input("  👉 第 1 局结束！请按回车键进入【第 2 局】(换由 AI-C 先手发牌)...")

    m2_result = run_single_match(
        match_index=2,
        starting_seat=2,
        ai_policies=ai_policies,
        delay_sec=delay,
        simulate_human=args.simulate_human,
    )

    print("\n\n" + "╔" + "═" * 68 + "╗")
    print("║                   🏆 双局总决赛 最终总积分榜                      ║")
    print("╚" + "═" * 68 + "╝")

    total_scores = {s: m1_result["scores"][s] + m2_result["scores"][s] for s in range(1, 5)}
    sorted_total = sorted(range(1, 5), key=lambda s: total_scores[s], reverse=True)

    print("  总名次   席位   选手名称                第1局得分  第2局得分   总积分")
    print("  " + "─" * 68)
    for rank_idx, seat in enumerate(sorted_total, 1):
        name = ROLE_NAMES[seat]
        s1 = m1_result["scores"][seat]
        s2 = m2_result["scores"][seat]
        st = total_scores[seat]
        crown = " 👑【总冠军】" if rank_idx == 1 else ""
        print(f"  第 {rank_idx} 名   {seat}号位  {name:<24}  {s1:+8.1f}   {s2:+8.1f}  {st:+8.1f} 分{crown}")
    print("  " + "─" * 68)

    record_path = Path("runs_ppo/human_vs_ai_match_record.json")
    record_path.parent.mkdir(parents=True, exist_ok=True)
    summary_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "rule_version": RULE_VERSION_TAG,
        "model_loaded": MODEL_SPECS[2]["path"],
        "total_scores": total_scores,
        "matches": [m1_result, m2_result],
    }
    with open(record_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)
    print(f"\n  💾 对局战报已保存至: {record_path.resolve()}")
    print("  🎮 感谢体验逆水寒生死左轮人机对战系统！\n")


if __name__ == "__main__":
    main()
