from __future__ import annotations

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from random import Random

from .engine import SurvivalGameEngine
from .neural_policy import NeuralPolicy
from .opponents import HeuristicProfilePolicy, OpponentProfile
from .rewards import TerminalRewardModel
from .types import CardKind, ChallengeAction, GameAction, GameState, PlayAction, Policy

DIAGNOSTIC_SEEDS: tuple[int, ...] = tuple(60000 + i * 43 for i in range(25))

# 定义三种与训练生态有鲜明差异的极端风格对手
STYLE_PROFILES: dict[str, OpponentProfile] = {
    # 风格 1：偏诚实、极低诈唬（检验模型会不会在对方诚实时过度盲目质疑而暴毙）
    "honest_conservative": OpponentProfile(
        name="honest_conservative",
        description="偏诚实、低诈唬：手里只要有真牌坚决不出假牌，极度少诈唬，质疑谨慎。",
        variability="low",
        style="honest",
        honesty_bias=2.2,
        bluff_bias=-2.8,
        challenge_bias=-0.8,
        tempo_bias=0.2,
        escape_bias=0.0,
        single_bias=0.6,
        pair_bias=0.2,
        triple_bias=-0.6,
        temperature=0.10,
        hazard_sensitivity=1.5,
    ),
    # 风格 2：高频诈唬、极低质疑（检验模型能不能洞察破绽、高频抓诈并惩罚软柿子）
    "heavy_bluffer_gullible": OpponentProfile(
        name="heavy_bluffer_gullible",
        description="高频诈唬、低质疑：手里没牌也硬吹，出牌极激进，但极少质疑别人，容易被白嫖。",
        variability="medium",
        style="wild_bluffer",
        honesty_bias=-1.8,
        bluff_bias=2.6,
        challenge_bias=-2.2,
        tempo_bias=1.1,
        escape_bias=0.0,
        single_bias=-0.4,
        pair_bias=0.5,
        triple_bias=0.8,
        temperature=0.45,
        hazard_sensitivity=0.2,
    ),
    # 风格 3：高频狂热质疑（检验模型会不会调整出牌策略，收敛诈唬，合理打出真牌与鬼牌设伏）
    "suspicious_challenger": OpponentProfile(
        name="suspicious_challenger",
        description="高频质疑型：怀疑心极重，见牌就想抓，给全场制造恐怖质疑压迫。",
        variability="low",
        style="paranoid",
        honesty_bias=0.6,
        bluff_bias=-0.4,
        challenge_bias=2.6,
        tempo_bias=0.6,
        escape_bias=0.0,
        single_bias=0.0,
        pair_bias=0.3,
        triple_bias=0.4,
        temperature=0.12,
        hazard_sensitivity=0.5,
    ),
}


@dataclass
class RoleSummary:
    role: str
    wins: int = 0
    top2: int = 0
    rank_counts: dict[int, int] = field(default_factory=lambda: {1: 0, 2: 0, 3: 0, 4: 0})
    total_utility: float = 0.0
    win_rate: float = 0.0
    top2_rate: float = 0.0
    mean_utility: float = 0.0


@dataclass
class StyleDiagnosticResult:
    candidate_name: str
    style_key: str
    total_games: int
    table_summary: dict[str, dict]
    cand_challenges_on_opponents: dict[str, dict]
    opponents_challenges_on_cand: dict[str, dict]
    ghost_stats: dict[str, int]
    cand_seat_wins: dict[int, int]
    per_game_records: list[dict] = field(default_factory=list)


class StyleDiagnosticBenchmark:
    def __init__(
        self,
        v14_path: str = "runs_deep_cfr/deep_cfr_v14_full_selfplay/policy_best.pt",
        v15_path: str = "runs_deep_cfr/deep_cfr_v15_league/policy_best.pt",
    ) -> None:
        self.engine = SurvivalGameEngine()
        self.reward_model = TerminalRewardModel()
        self.v14 = NeuralPolicy.load(v14_path)
        self.v15 = NeuralPolicy.load(v15_path)

    def run_style_benchmark(
        self,
        candidate_policy: Policy,
        candidate_name: str,
        style_key: str,
        seeds: tuple[int, ...] = DIAGNOSTIC_SEEDS,
    ) -> StyleDiagnosticResult:
        profile = STYLE_PROFILES[style_key]

        roles = ["candidate", "style_bot", "v14", "v15"]
        role_stats = {r: RoleSummary(role=r) for r in roles}

        # 细分动作统计：考生对各对手的质疑情况
        cand_challenges = {
            "style_bot": {"total": 0, "correct": 0, "lie": 0, "ghost": 0, "wrong_honest": 0},
            "v14": {"total": 0, "correct": 0, "lie": 0, "ghost": 0, "wrong_honest": 0},
            "v15": {"total": 0, "correct": 0, "lie": 0, "ghost": 0, "wrong_honest": 0},
        }

        # 细分动作统计：各对手对考生的质疑情况
        opp_challenges = {
            "style_bot": {"total": 0, "cand_lied": 0, "cand_honest": 0, "cand_ghost": 0},
            "v14": {"total": 0, "cand_lied": 0, "cand_honest": 0, "cand_ghost": 0},
            "v15": {"total": 0, "cand_lied": 0, "cand_honest": 0, "cand_ghost": 0},
        }

        ghost_stats = {
            "cand_ghost_played": 0,
            "cand_ghost_countered": 0,
            "cand_ghost_uncalled": 0,
            "cand_hit_by_others_ghost": 0,
        }

        cand_seat_wins = {1: 0, 2: 0, 3: 0, 4: 0}
        records: list[dict] = []

        total_games = len(seeds) * 4
        print(f"[Style Benchmark] 开始测试: {candidate_name} vs 【{style_key}】 | 局数={total_games}")

        for s_idx, seed in enumerate(seeds):
            for cand_seat in range(1, 5):
                # 安排座位：cand_seat 坐考生；其余 3 位按顺序分配：风格Bot、v14、v15
                other_seats = [s for s in range(1, 5) if s != cand_seat]
                style_bot = HeuristicProfilePolicy(profile, seed=seed + 100)

                seat_map: dict[int, Policy] = {
                    cand_seat: candidate_policy,
                    other_seats[0]: style_bot,
                    other_seats[1]: self.v14,
                    other_seats[2]: self.v15,
                }

                seat_to_role: dict[int, str] = {
                    cand_seat: "candidate",
                    other_seats[0]: "style_bot",
                    other_seats[1]: "v14",
                    other_seats[2]: "v15",
                }

                state = self.engine.new_game(seed=seed, starting_seat=1)
                step = 0
                active_cand_ghost_play = False

                while not self.engine.is_terminal(state) and step < 200:
                    actor = self.engine.current_actor(state)
                    legal = self.engine.legal_actions(state)
                    if not legal:
                        break

                    obs = self.engine.observe(state, actor)
                    act = seat_map[actor].choose_action(obs, legal)

                    cand_p_before = state.player_by_seat(cand_seat)
                    cand_shots_before = cand_p_before.shots_taken

                    # 追踪出牌事件
                    if isinstance(act, PlayAction):
                        p_obj = state.player_by_seat(actor)
                        played_cards = [c for c in p_obj.hand if c.card_id in act.card_ids]
                        is_ghost_play = any(c.kind == CardKind.GHOST for c in played_cards)

                        if actor == cand_seat:
                            if active_cand_ghost_play:
                                # 上一次鬼牌没人质疑，安全过关
                                ghost_stats["cand_ghost_uncalled"] += 1
                                active_cand_ghost_play = False
                            if is_ghost_play:
                                ghost_stats["cand_ghost_played"] += 1
                                active_cand_ghost_play = True
                        else:
                            if active_cand_ghost_play:
                                # 对手没有质疑考生，而是选择了出牌，考生的鬼牌安全溜走
                                ghost_stats["cand_ghost_uncalled"] += 1
                                active_cand_ghost_play = False

                    # 追踪质疑事件
                    if isinstance(act, ChallengeAction):
                        target_play = state.round_state.latest_play
                        if target_play:
                            target_seat = target_play.seat
                            target_role = seat_to_role[target_seat]
                            actor_role = seat_to_role[actor]
                            ev_res = self.engine._evaluate_play(target_play).value

                            # 1. 考生质疑对手
                            if actor == cand_seat and target_seat != cand_seat:
                                t_entry = cand_challenges[target_role]
                                t_entry["total"] += 1
                                if ev_res in ("lie", "ghost"):
                                    t_entry["correct"] += 1
                                    if ev_res == "lie":
                                        t_entry["lie"] += 1
                                    else:
                                        t_entry["ghost"] += 1
                                else:
                                    t_entry["wrong_honest"] += 1

                            # 2. 对手质疑考生
                            elif actor != cand_seat and target_seat == cand_seat:
                                o_entry = opp_challenges[actor_role]
                                o_entry["total"] += 1
                                if ev_res == "lie":
                                    o_entry["cand_lied"] += 1
                                elif ev_res == "honest":
                                    o_entry["cand_honest"] += 1
                                elif ev_res == "ghost":
                                    o_entry["cand_ghost"] += 1
                                    ghost_stats["cand_ghost_countered"] += 1

                        if active_cand_ghost_play:
                            active_cand_ghost_play = False

                    self.engine.apply_action(state, act)

                    # 检查考生是否被其他人的鬼牌反杀波及中弹
                    cand_p_after = state.player_by_seat(cand_seat)
                    if (
                        cand_p_after.shots_taken > cand_shots_before
                        and state.public_history
                        and state.public_history[-1].event_type == "challenge"
                        and state.public_history[-1].detail.get("outcome") == "ghost"
                        and state.public_history[-1].detail.get("target_seat") != cand_seat
                    ):
                        ghost_stats["cand_hit_by_others_ghost"] += 1

                    step += 1

                if active_cand_ghost_play:
                    ghost_stats["cand_ghost_uncalled"] += 1
                    active_cand_ghost_play = False

                # 局终名次结算
                finish_order = self.reward_model._successful_finish_order(state)
                rewards = self.reward_model.evaluate(state)

                game_ranks = {}
                game_utils = {}

                for s, role in seat_to_role.items():
                    rank = finish_order.index(s) + 1
                    util = rewards.for_seat(s)
                    stat = role_stats[role]
                    stat.rank_counts[rank] += 1
                    stat.total_utility += util
                    if rank == 1:
                        stat.wins += 1
                        stat.top2 += 1
                        if role == "candidate":
                            cand_seat_wins[cand_seat] += 1
                    elif rank == 2:
                        stat.top2 += 1

                    game_ranks[role] = rank
                    game_utils[role] = round(util, 2)

                records.append({
                    "seed": seed,
                    "cand_seat": cand_seat,
                    "ranks": game_ranks,
                    "utilities": game_utils,
                    "finish_order": finish_order,
                })

        # 汇总各角色比率
        table_summary = {}
        for r in roles:
            st = role_stats[r]
            st.win_rate = round(st.wins / total_games, 3)
            st.top2_rate = round(st.top2 / total_games, 3)
            st.mean_utility = round(st.total_utility / total_games, 2)
            table_summary[r] = {
                "wins": st.wins,
                "win_rate": st.win_rate,
                "top2": st.top2,
                "top2_rate": st.top2_rate,
                "ranks": st.rank_counts,
                "mean_utility": st.mean_utility,
            }

        return StyleDiagnosticResult(
            candidate_name=candidate_name,
            style_key=style_key,
            total_games=total_games,
            table_summary=table_summary,
            cand_challenges_on_opponents=cand_challenges,
            opponents_challenges_on_cand=opp_challenges,
            ghost_stats=ghost_stats,
            cand_seat_wins=cand_seat_wins,
            per_game_records=records,
        )


def print_detailed_style_result(res: StyleDiagnosticResult) -> None:
    print("=" * 72)
    print(f"  风格场景: 【 {res.style_key} 】 | 参测模型: {res.candidate_name}")
    print("=" * 72)
    print(f"{'角色名':<16} | {'吃鸡数 (胜率)':<16} | {'保底前二率':<12} | {'名次分布 [1/2/3/4]':<20} | {'场均得分':<8}")
    print("-" * 72)
    for role, data in res.table_summary.items():
        rk = data["ranks"]
        rk_str = f"[{rk[1]:2d}, {rk[2]:2d}, {rk[3]:2d}, {rk[4]:2d}]"
        print(
            f"{role:<16} | {data['wins']:2d}/100 ({data['win_rate']*100:4.1f}%) | "
            f"{data['top2']:2d}/100 ({data['top2_rate']*100:4.1f}%) | "
            f"{rk_str:<20} | "
            f"{data['mean_utility']:+6.2f}"
        )
    print("-" * 72)
    print("【考生主动定向质疑细分】:")
    for target, dat in res.cand_challenges_on_opponents.items():
        acc = (dat["correct"] / dat["total"] * 100) if dat["total"] > 0 else 0.0
        print(
            f"  • 抓 {target:<10}: 共 {dat['total']:2d} 次 | 命中率 {acc:4.1f}% "
            f"(识破谎言: {dat['lie']}, 命中鬼牌: {dat['ghost']}, 误抓真牌: {dat['wrong_honest']})"
        )
    print("【各对手对考生定向质疑细分】:")
    for opp, dat in res.opponents_challenges_on_cand.items():
        print(
            f"  • {opp:<10} 质疑考生: 共 {dat['total']:2d} 次 | "
            f"抓到考生诈唬: {dat['cand_lied']}, 抓错考生真牌: {dat['cand_honest']}, 被考生鬼牌反杀: {dat['cand_ghost']}"
        )
    gh = res.ghost_stats
    print(
        f"【鬼牌全局互动】: 考生出鬼牌 {gh['cand_ghost_played']} 次 "
        f"(反杀对手: {gh['cand_ghost_countered']} 次, 无人质疑安全过: {gh['cand_ghost_uncalled']} 次) | "
        f"考生被他人鬼牌波及扣血: {gh['cand_hit_by_others_ghost']} 次"
    )
    print(f"【考生座位分布】 (1-4号位胜场): [{res.cand_seat_wins[1]}, {res.cand_seat_wins[2]}, {res.cand_seat_wins[3]}, {res.cand_seat_wins[4]}] / 25")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 3-style diagnostic benchmark with 4-player tracking")
    parser.add_argument("--v16-checkpoint", type=str, default="runs_deep_cfr/deep_cfr_v16_endgame/policy_best.pt")
    parser.add_argument("--v17-iter1000", type=str, default="runs_deep_cfr/deep_cfr_v17_balanced/checkpoint_iter_01000.pt")
    parser.add_argument("--v17-iter1075", type=str, default="runs_deep_cfr/deep_cfr_v17_balanced/policy_latest.pt")
    parser.add_argument("--output-json", type=str, default="runs_deep_cfr/deep_cfr_v17_balanced/style_diagnostic_detailed_900games.json")
    args = parser.parse_args()

    bench = StyleDiagnosticBenchmark()
    v16_policy = NeuralPolicy.load(args.v16_checkpoint)
    v17_1000_policy = NeuralPolicy.load(args.v17_iter1000)
    v17_1075_policy = NeuralPolicy.load(args.v17_iter1075)

    candidates = [
        ("v16_baseline", v16_policy),
        ("v17_iter1000", v17_1000_policy),
        ("v17_iter1075_final", v17_1075_policy),
    ]

    styles = ["honest_conservative", "heavy_bluffer_gullible", "suspicious_challenger"]

    all_results = {}

    print("*" * 72)
    print("     🧪 四家胜负流向与定向博弈全景诊断测试 (3风格 × 3模型 × 100局) 🧪")
    print("*" * 72)

    for style_key in styles:
        for cand_name, cand_policy in candidates:
            res = bench.run_style_benchmark(cand_policy, cand_name, style_key)
            print_detailed_style_result(res)
            all_results[f"{cand_name}_{style_key}"] = asdict(res)

    out_p = Path(args.output_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[Done] 全景诊断报告已持久化至: {out_p}")

