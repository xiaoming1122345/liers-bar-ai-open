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

from .engine import SurvivalGameEngine
from .neural_policy import NeuralPolicy
from .opponents import HeuristicProfilePolicy, OpponentProfile
from .rewards import TerminalRewardModel
from .types import CardKind, ChallengeAction, GameAction, GameState, PlayAction, Policy

HONEST_PROFILE = OpponentProfile(
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
)


@dataclass
class HonestDiagnosticResult:
    adapter_mode: str
    total_games: int
    table_wins: dict[str, int]
    table_win_rates: dict[str, float]
    table_top2: dict[str, int]
    table_top2_rates: dict[str, float]
    table_ranks: dict[str, dict[int, int]]
    table_mean_utils: dict[str, float]
    cand_active_challenges_on_bot: dict[str, int]  # total, correct, lie, ghost, wrong_honest, acc
    cand_forced_challenges_on_bot: dict[str, int]  # total, correct, wrong_honest
    bot_challenges_on_cand: dict[str, int]         # total, cand_lied, cand_honest, cand_ghost
    cand_play_honesty: dict[str, int]              # total_plays, honest_plays, bluff_plays, ghost_plays
    cand_seat_wins: dict[int, int]
    diagnostics: dict[str, float]
    per_game_records: list[dict] = field(default_factory=list)


class HonestBotDiagnosticBenchmark:
    def __init__(
        self,
        v14_path: str = "runs_deep_cfr/deep_cfr_v14_full_selfplay/policy_best.pt",
        v15_path: str = "runs_deep_cfr/deep_cfr_v15_league/policy_best.pt",
    ) -> None:
        self.engine = SurvivalGameEngine()
        self.reward_model = TerminalRewardModel()
        self.v14 = NeuralPolicy.load(v14_path)
        self.v15 = NeuralPolicy.load(v15_path)

    def run_benchmark(
        self,
        checkpoint_path: str,
        prob_mode: str,
        num_seeds: int = 50,
    ) -> HonestDiagnosticResult:
        candidate = NeuralPolicy.load(checkpoint_path, prob_mode=prob_mode)
        candidate.reset_diagnostics()

        seeds = tuple(60000 + i * 43 for i in range(num_seeds))
        total_games = len(seeds) * 4
        roles = ["candidate", "honest_bot", "v14", "v15"]

        table_wins = {r: 0 for r in roles}
        table_top2 = {r: 0 for r in roles}
        table_ranks = {r: {1: 0, 2: 0, 3: 0, 4: 0} for r in roles}
        table_utils = {r: 0.0 for r in roles}
        cand_seat_wins = {1: 0, 2: 0, 3: 0, 4: 0}

        # 考生对 Bot 的主动质疑 (手里有牌)
        cand_active_on_bot = {"total": 0, "correct": 0, "lie": 0, "ghost": 0, "wrong_honest": 0}
        # 考生对 Bot 的强制质疑 (手里0张牌)
        cand_forced_on_bot = {"total": 0, "correct": 0, "wrong_honest": 0}
        # Bot 抓考生的细分
        bot_on_cand = {"total": 0, "cand_lied": 0, "cand_honest": 0, "cand_ghost": 0}
        # 考生出牌动作构成
        cand_play_honesty = {"total_plays": 0, "honest_plays": 0, "bluff_plays": 0, "ghost_plays": 0}

        records: list[dict] = []

        print(f"[Honest Bot Benchmark] 开始测试: prob_mode={prob_mode} | 总局数={total_games} (种子数={num_seeds} × 4轮)")

        for s_idx, seed in enumerate(seeds):
            for cand_seat in range(1, 5):
                other_seats = [s for s in range(1, 5) if s != cand_seat]
                style_bot = HeuristicProfilePolicy(HONEST_PROFILE, seed=seed + 100)

                seat_map: dict[int, Policy] = {
                    cand_seat: candidate,
                    other_seats[0]: style_bot,
                    other_seats[1]: self.v14,
                    other_seats[2]: self.v15,
                }

                seat_to_role: dict[int, str] = {
                    cand_seat: "candidate",
                    other_seats[0]: "honest_bot",
                    other_seats[1]: "v14",
                    other_seats[2]: "v15",
                }

                state = self.engine.new_game(seed=seed, starting_seat=1)
                step = 0

                while not self.engine.is_terminal(state) and step < 200:
                    actor = self.engine.current_actor(state)
                    legal = self.engine.legal_actions(state)
                    if not legal:
                        break

                    obs = self.engine.observe(state, actor)
                    act = seat_map[actor].choose_action(obs, legal)

                    cand_p = state.player_by_seat(cand_seat)

                    # 记录考生出牌类型
                    if actor == cand_seat and isinstance(act, PlayAction):
                        cand_play_honesty["total_plays"] += 1
                        played_cards = [c for c in cand_p.hand if c.card_id in act.card_ids]
                        if any(c.kind == CardKind.GHOST for c in played_cards):
                            cand_play_honesty["ghost_plays"] += 1
                        else:
                            is_bluff = any(c.kind != CardKind.WILD and c.printed_rank != act.claim_rank for c in played_cards)
                            if is_bluff:
                                cand_play_honesty["bluff_plays"] += 1
                            else:
                                cand_play_honesty["honest_plays"] += 1

                    # 记录质疑事件
                    if isinstance(act, ChallengeAction):
                        target_play = state.round_state.latest_play
                        if target_play:
                            target_seat = target_play.seat
                            target_role = seat_to_role[target_seat]
                            actor_role = seat_to_role[actor]
                            ev_res = self.engine._evaluate_play(target_play).value

                            # 考生抓 Bot
                            if actor == cand_seat and target_role == "honest_bot":
                                is_forced = (len(cand_p.hand) == 0)
                                if is_forced:
                                    cand_forced_on_bot["total"] += 1
                                    if ev_res in ("lie", "ghost"):
                                        cand_forced_on_bot["correct"] += 1
                                    else:
                                        cand_forced_on_bot["wrong_honest"] += 1
                                else:
                                    cand_active_on_bot["total"] += 1
                                    if ev_res in ("lie", "ghost"):
                                        cand_active_on_bot["correct"] += 1
                                        if ev_res == "lie":
                                            cand_active_on_bot["lie"] += 1
                                        else:
                                            cand_active_on_bot["ghost"] += 1
                                    else:
                                        cand_active_on_bot["wrong_honest"] += 1

                            # Bot 抓考生
                            elif actor_role == "honest_bot" and target_seat == cand_seat:
                                bot_on_cand["total"] += 1
                                if ev_res == "lie":
                                    bot_on_cand["cand_lied"] += 1
                                elif ev_res == "honest":
                                    bot_on_cand["cand_honest"] += 1
                                elif ev_res == "ghost":
                                    bot_on_cand["cand_ghost"] += 1

                    self.engine.apply_action(state, act)
                    step += 1

                finish_order = self.reward_model._successful_finish_order(state)
                rewards = self.reward_model.evaluate(state)

                for s, r in seat_to_role.items():
                    rk = finish_order.index(s) + 1
                    u = rewards.for_seat(s)
                    table_ranks[r][rk] += 1
                    table_utils[r] += u
                    if rk == 1:
                        table_wins[r] += 1
                        table_top2[r] += 1
                        if r == "candidate":
                            cand_seat_wins[cand_seat] += 1
                    elif rk == 2:
                        table_top2[r] += 1

                records.append({
                    "seed": seed,
                    "cand_seat": cand_seat,
                    "ranks": {seat_to_role[s]: finish_order.index(s) + 1 for s in seat_to_role},
                    "utilities": {seat_to_role[s]: round(rewards.for_seat(s), 2) for s in seat_to_role},
                })

        diag = candidate.get_diagnostics_summary()

        return HonestDiagnosticResult(
            adapter_mode=prob_mode,
            total_games=total_games,
            table_wins=table_wins,
            table_win_rates={r: round(table_wins[r] / total_games, 3) for r in roles},
            table_top2=table_top2,
            table_top2_rates={r: round(table_top2[r] / total_games, 3) for r in roles},
            table_ranks=table_ranks,
            table_mean_utils={r: round(table_utils[r] / total_games, 2) for r in roles},
            cand_active_challenges_on_bot=cand_active_on_bot,
            cand_forced_challenges_on_bot=cand_forced_on_bot,
            bot_challenges_on_cand=bot_on_cand,
            cand_play_honesty=cand_play_honesty,
            cand_seat_wins=cand_seat_wins,
            diagnostics=diag,
            per_game_records=records,
        )


def print_honest_comparison(ctrl: HonestDiagnosticResult, trt: HonestDiagnosticResult) -> None:
    print("\n" + "=" * 76)
    print(f"      🎯 针对已暴露短板【诚实 Bot】原转换 vs 新转换对照 ({ctrl.total_games} 局) 🎯")
    print("=" * 76)
    print(f"{'角色':<14} | {'对照组 (原 Softmax)':<22} | {'实验组 (非负截断归一化)':<22} | {'净变化':<10}")
    print("-" * 76)

    for r in ["candidate", "honest_bot", "v14", "v15"]:
        w1, t1 = ctrl.table_win_rates[r] * 100, ctrl.table_top2_rates[r] * 100
        w2, t2 = trt.table_win_rates[r] * 100, trt.table_top2_rates[r] * 100
        u1, u2 = ctrl.table_mean_utils[r], trt.table_mean_utils[r]
        dw = w2 - w1
        print(
            f"{r:<14} | 胜率 {w1:4.1f}% (前二 {t1:4.1f}% 得分 {u1:+5.2f}) | "
            f"胜率 {w2:4.1f}% (前二 {t2:4.1f}% 得分 {u2:+5.2f}) | "
            f"{dw:+5.1f}%"
        )

    print("-" * 76)
    print("【考生 vs 诚实 Bot 定向交互拆解】:")
    c1 = ctrl.cand_active_challenges_on_bot
    c2 = trt.cand_active_challenges_on_bot
    acc1 = (c1["correct"] / c1["total"] * 100) if c1["total"] > 0 else 0
    acc2 = (c2["correct"] / c2["total"] * 100) if c2["total"] > 0 else 0
    print(
        f"  • 考生【主动】质疑 Bot: 对照组 {c1['total']} 次 (命中 {acc1:4.1f}%, 误抓真牌 {c1['wrong_honest']}) "
        f"-> 实验组 {c2['total']} 次 (命中 {acc2:4.1f}%, 误抓真牌 {c2['wrong_honest']})"
    )

    f1 = ctrl.cand_forced_challenges_on_bot
    f2 = trt.cand_forced_challenges_on_bot
    print(
        f"  • 考生【强制】质疑 Bot (手牌出空): 对照组 {f1['total']} 次 (识破 {f1['correct']}) "
        f"-> 实验组 {f2['total']} 次 (识破 {f2['correct']})"
    )

    b1 = ctrl.bot_challenges_on_cand
    b2 = trt.bot_challenges_on_cand
    print(
        f"  • 诚实 Bot 质疑考生: 对照组 {b1['total']} 次 (抓到诈唬 {b1['cand_lied']}, 误抓真牌 {b1['cand_honest']}, 被鬼反杀 {b1['cand_ghost']}) "
        f"-> 实验组 {b2['total']} 次 (抓到诈唬 {b2['cand_lied']}, 误抓真牌 {b2['cand_honest']}, 被鬼反杀 {b2['cand_ghost']})"
    )

    p1 = ctrl.cand_play_honesty
    p2 = trt.cand_play_honesty
    br1 = (p1["bluff_plays"] / p1["total_plays"] * 100) if p1["total_plays"] > 0 else 0
    br2 = (p2["bluff_plays"] / p2["total_plays"] * 100) if p2["total_plays"] > 0 else 0
    print(
        f"  • 考生实际出牌构成: 对照组诈唬率 {br1:4.1f}% ({p1['bluff_plays']}/{p1['total_plays']}) "
        f"-> 实验组诈唬率 {br2:4.1f}% ({p2['bluff_plays']}/{p2['total_plays']})"
    )

    print("-" * 76)
    print("【内部决策特征诊断】:")
    d1 = ctrl.diagnostics
    d2 = trt.diagnostics
    print(
        f"  • 平均最高动作概率 (Top-1 Prob): 对照组 {d1['avg_top1_prob']*100:.1f}% -> 实验组 {d2['avg_top1_prob']*100:.1f}% | "
        f"全负回退触发: 对照组 {d1['fallback_count']} 次 -> 实验组 {d2['fallback_count']} 次"
    )
    print("=" * 76)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="runs_deep_cfr/deep_cfr_v17_balanced/checkpoint_iter_01000.pt")
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--output", type=str, default="runs_deep_cfr/deep_cfr_v17_balanced/honest_bot_comparison_200games.json")
    args = parser.parse_args()

    bench = HonestBotDiagnosticBenchmark()
    res_ctrl = bench.run_benchmark(args.checkpoint, prob_mode="softmax", num_seeds=args.seeds)
    res_trt = bench.run_benchmark(args.checkpoint, prob_mode="linear_norm", num_seeds=args.seeds)

    print_honest_comparison(res_ctrl, res_trt)

    out = {
        "control_softmax": asdict(res_ctrl),
        "treatment_linear_norm": asdict(res_trt),
    }
    Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[Saved] 诚实 Bot 对照报告已保存至: {args.output}")
