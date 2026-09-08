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
from .rewards import TerminalRewardModel
from .types import ChallengeAction, GameAction, GameState, PlayAction, Policy


@dataclass
class AdapterComparisonResult:
    adapter_mode: str
    total_games: int
    wins: int
    win_rate: float
    top2_count: int
    top2_rate: float
    mean_utility: float
    finals_entered: int
    finals_entered_rate: float
    finals_won: int
    finals_conversion_rate: float
    active_challenges: int
    active_challenges_won: int
    active_challenge_acc: float
    table_rank_counts: dict[str, dict[int, int]]
    table_wins: dict[str, int]
    table_win_rates: dict[str, float]
    table_mean_utils: dict[str, float]
    seat_wins: dict[int, int]
    diagnostics: dict[str, float]
    per_game_records: list[dict] = field(default_factory=list)


class InferenceAdapterBenchmark:
    def __init__(
        self,
        v13_path: str = "runs_deep_cfr/deep_cfr_v13_extern_sampling/policy_best.pt",
        v14_path: str = "runs_deep_cfr/deep_cfr_v14_full_selfplay/policy_best.pt",
        v15_path: str = "runs_deep_cfr/deep_cfr_v15_league/policy_best.pt",
    ) -> None:
        self.engine = SurvivalGameEngine()
        self.reward_model = TerminalRewardModel()
        # 考官严格保留原始默认推理（隔离变量）
        self.opponents = {
            "v13": NeuralPolicy.load(v13_path),
            "v14": NeuralPolicy.load(v14_path),
            "v15": NeuralPolicy.load(v15_path),
        }

    def run_benchmark(
        self,
        checkpoint_path: str,
        prob_mode: str,
        num_seeds: int = 50,
    ) -> AdapterComparisonResult:
        # 加载同一权重，仅改变概率转换模式
        candidate = NeuralPolicy.load(checkpoint_path, prob_mode=prob_mode)
        candidate.reset_diagnostics()

        seeds = tuple(50000 + i * 37 for i in range(num_seeds))
        total_games = len(seeds) * 4
        opp_names = ["v13", "v14", "v15"]
        roles = ["candidate", "v13", "v14", "v15"]

        wins = 0
        top2 = 0
        total_utility = 0.0
        finals_entered = 0
        finals_won = 0
        active_challenges = 0
        active_challenges_won = 0
        seat_wins = {1: 0, 2: 0, 3: 0, 4: 0}

        table_wins = {r: 0 for r in roles}
        table_top2 = {r: 0 for r in roles}
        table_ranks = {r: {1: 0, 2: 0, 3: 0, 4: 0} for r in roles}
        table_utils = {r: 0.0 for r in roles}

        records: list[dict] = []

        print(f"\n[Inference Benchmark] 开始评测: prob_mode={prob_mode} | 总局数={total_games} (种子数={num_seeds} × 4轮)")

        for s_idx, seed in enumerate(seeds):
            for cand_seat in range(1, 5):
                other_seats = [s for s in range(1, 5) if s != cand_seat]
                seat_map: dict[int, Policy] = {cand_seat: candidate}
                seat_to_role: dict[int, str] = {cand_seat: "candidate"}

                for idx, os in enumerate(other_seats):
                    seat_map[os] = self.opponents[opp_names[idx]]
                    seat_to_role[os] = opp_names[idx]

                state = self.engine.new_game(seed=seed, starting_seat=1)
                step = 0
                saw_finals = False

                while not self.engine.is_terminal(state) and step < 200:
                    alive = [p for p in state.players if p.alive]
                    cand_p = state.player_by_seat(cand_seat)
                    if len(alive) == 2 and cand_p.alive:
                        saw_finals = True

                    actor = self.engine.current_actor(state)
                    legal = self.engine.legal_actions(state)
                    if not legal:
                        break

                    obs = self.engine.observe(state, actor)
                    act = seat_map[actor].choose_action(obs, legal)

                    if actor == cand_seat and isinstance(act, ChallengeAction):
                        if len(cand_p.hand) > 0:
                            active_challenges += 1
                            lp = state.round_state.latest_play
                            if lp:
                                ev_res = self.engine._evaluate_play(lp)
                                if ev_res.value in ("lie", "ghost"):
                                    active_challenges_won += 1

                    self.engine.apply_action(state, act)
                    step += 1

                finish_order = self.reward_model._successful_finish_order(state)
                rewards = self.reward_model.evaluate(state)

                cand_rank = finish_order.index(cand_seat)
                cand_win = (cand_rank == 0)
                cand_util = rewards.for_seat(cand_seat)
                total_utility += cand_util

                if cand_win:
                    wins += 1
                    top2 += 1
                    seat_wins[cand_seat] += 1
                    if saw_finals:
                        finals_won += 1
                elif cand_rank == 1:
                    top2 += 1

                if saw_finals:
                    finals_entered += 1

                for s, r in seat_to_role.items():
                    rk = finish_order.index(s) + 1
                    u = rewards.for_seat(s)
                    table_ranks[r][rk] += 1
                    table_utils[r] += u
                    if rk == 1:
                        table_wins[r] += 1

                records.append({
                    "seed": seed,
                    "cand_seat": cand_seat,
                    "cand_rank": cand_rank + 1,
                    "cand_win": cand_win,
                    "entered_finals": saw_finals,
                    "cand_util": round(cand_util, 2),
                    "finish_order": finish_order,
                })

        diag = candidate.get_diagnostics_summary()

        return AdapterComparisonResult(
            adapter_mode=prob_mode,
            total_games=total_games,
            wins=wins,
            win_rate=round(wins / total_games, 3),
            top2_count=top2,
            top2_rate=round(top2 / total_games, 3),
            mean_utility=round(total_utility / total_games, 2),
            finals_entered=finals_entered,
            finals_entered_rate=round(finals_entered / total_games, 3),
            finals_won=finals_won,
            finals_conversion_rate=round(finals_won / max(1, finals_entered), 3),
            active_challenges=active_challenges,
            active_challenges_won=active_challenges_won,
            active_challenge_acc=round(active_challenges_won / max(1, active_challenges), 3),
            table_rank_counts=table_ranks,
            table_wins=table_wins,
            table_win_rates={r: round(table_wins[r] / total_games, 3) for r in roles},
            table_mean_utils={r: round(table_utils[r] / total_games, 2) for r in roles},
            seat_wins=seat_wins,
            diagnostics=diag,
            per_game_records=records,
        )


def print_comparison(ctrl: AdapterComparisonResult, trt: AdapterComparisonResult) -> None:
    print("\n" + "=" * 76)
    print(f"      🎯 推理转换一致性单变量严格对照结果 ({ctrl.total_games} 局固定历史考官) 🎯")
    print("=" * 76)
    print(f"{'考核指标':<22} | {'对照组 (原 Softmax)':<22} | {'实验组 (非负截断归一化)':<22} | {'净变化':<10}")
    print("-" * 76)

    def row(label, v1, v2, fmt="{:.1f}%", diff_fmt="{:+.1f}%"):
        s1 = fmt.format(v1)
        s2 = fmt.format(v2)
        diff = v2 - v1
        sd = diff_fmt.format(diff)
        print(f"{label:<22} | {s1:<22} | {s2:<22} | {sd:<10}")

    row("考生吃鸡胜率", ctrl.win_rate * 100, trt.win_rate * 100)
    row("考生保底前二率", ctrl.top2_rate * 100, trt.top2_rate * 100)
    row("进入决赛圈率", ctrl.finals_entered_rate * 100, trt.finals_entered_rate * 100)
    row("单挑决胜转化率", ctrl.finals_conversion_rate * 100, trt.finals_conversion_rate * 100)
    row("考生场均得分", ctrl.mean_utility, trt.mean_utility, fmt="{:+.2f}", diff_fmt="{:+.2f}")
    row("主动质疑命中率", ctrl.active_challenge_acc * 100, trt.active_challenge_acc * 100)

    print("-" * 76)
    print("【内部决策特征诊断 (确认修正是否生效)】:")
    row("平均最高动作概率(Top1)", ctrl.diagnostics["avg_top1_prob"] * 100, trt.diagnostics["avg_top1_prob"] * 100)
    row("平均合法候选动作数", ctrl.diagnostics["avg_legal_actions"], trt.diagnostics["avg_legal_actions"], fmt="{:.2f}", diff_fmt="{:+.2f}")
    print(f"{'全负回退触发次数':<22} | {ctrl.diagnostics['fallback_count']:<22d} | {trt.diagnostics['fallback_count']:<22d} | (率: {trt.diagnostics['fallback_rate']:.2%})")

    print("-" * 76)
    print("【逐局同种子同座次配对差异分析 (Paired Comparison)】:")
    better_rank = 0
    worse_rank = 0
    equal_rank = 0
    both_win = 0
    gain_win = 0
    lose_win = 0
    both_lose = 0

    ctrl_dict = {(r["seed"], r["cand_seat"]): r for r in ctrl.per_game_records}
    trt_dict = {(r["seed"], r["cand_seat"]): r for r in trt.per_game_records}

    for key, c_rec in ctrl_dict.items():
        if key in trt_dict:
            t_rec = trt_dict[key]
            c_rk = c_rec["cand_rank"]
            t_rk = t_rec["cand_rank"]
            if t_rk < c_rk:
                better_rank += 1
            elif t_rk > c_rk:
                worse_rank += 1
            else:
                equal_rank += 1

            c_win = c_rec["cand_win"]
            t_win = t_rec["cand_win"]
            if c_win and t_win:
                both_win += 1
            elif not c_win and t_win:
                gain_win += 1
            elif c_win and not t_win:
                lose_win += 1
            else:
                both_lose += 1

    total_pairs = len(ctrl_dict)
    print(f"  • 名次变动: 实验组更优 {better_rank} 局 ({better_rank/total_pairs:.1%}) | 持平 {equal_rank} 局 ({equal_rank/total_pairs:.1%}) | 更差 {worse_rank} 局 ({worse_rank/total_pairs:.1%})")
    print(f"  • 胜负流向: 纯新增吃鸡 +{gain_win} 局 | 遗失吃鸡 -{lose_win} 局 (净胜增量: {gain_win - lose_win:+d} 局) | 双方皆赢 {both_win} 局 | 双方皆负 {both_lose} 局")

    print("-" * 76)
    print("【四家整桌胜率与得分流向】:")
    for r in ["candidate", "v13", "v14", "v15"]:
        w1, u1 = ctrl.table_win_rates[r] * 100, ctrl.table_mean_utils[r]
        w2, u2 = trt.table_win_rates[r] * 100, trt.table_mean_utils[r]
        print(f"  • {r:<10}: 对照组胜率 {w1:4.1f}% (得分 {u1:+5.2f}) -> 实验组胜率 {w2:4.1f}% (得分 {u2:+5.2f})")
    print("=" * 76)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="runs_deep_cfr/deep_cfr_v17_balanced/checkpoint_iter_01000.pt")
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--output", type=str, default="runs_deep_cfr/deep_cfr_v17_balanced/ab_eval_adapter_200games.json")
    args = parser.parse_args()

    bench = InferenceAdapterBenchmark()
    res_softmax = bench.run_benchmark(args.checkpoint, prob_mode="softmax", num_seeds=args.seeds)
    res_linear = bench.run_benchmark(args.checkpoint, prob_mode="linear_norm", num_seeds=args.seeds)

    print_comparison(res_softmax, res_linear)

    out = {
        "control_softmax": asdict(res_softmax),
        "treatment_linear_norm": asdict(res_linear),
    }
    Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[Saved] 对照实验数据已完整保存至: {args.output}")
