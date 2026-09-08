from __future__ import annotations

"""标准固定对照 A/B 测试套件 (Fixed-Opponent A/B Benchmark)。

评测原则：
1. 固定 3 个历史对手：[v13_sampling, v14_selfplay, v15_league]；
2. 固定 25 组测试随机种子，每组种子被测模型轮流坐 1, 2, 3, 4 号位，共 100 场绝对公平考卷；
3. 统计关键指标：吃鸡率、保底前二率、决赛圈进入率、决赛圈吃鸡转化率、按座位分布、排除 0 张牌的主动质疑准确率。
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from dataclasses import dataclass
from random import Random
from typing import Callable

from .engine import SurvivalGameEngine
from .neural_policy import NeuralPolicy
from .rewards import TerminalRewardModel
from .types import ChallengeAction, GameState, PlayAction, Policy

FIXED_TEST_SEEDS: tuple[int, ...] = tuple(50000 + i * 37 for i in range(25))


from dataclasses import dataclass, field

@dataclass
class ABBenchmarkResult:
    candidate_name: str
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
    seat_wins: dict[int, int]
    active_challenges: int
    active_challenges_won: int
    active_challenge_acc: float
    per_game_records: list[dict] = field(default_factory=list)



class FixedOpponentBenchmark:
    """固定历史对手的标准 A/B 对照评测套件。"""

    def __init__(
        self,
        opponent_models: dict[str, Policy] | None = None,
        engine: SurvivalGameEngine | None = None,
        reward_model: TerminalRewardModel | None = None,
    ) -> None:
        self.engine = engine or SurvivalGameEngine()
        self.reward_model = reward_model or TerminalRewardModel()

        if opponent_models is None:
            self.opponents = {
                "v13": NeuralPolicy.load("runs_deep_cfr/deep_cfr_v13_extern_sampling/policy_best.pt"),
                "v14": NeuralPolicy.load("runs_deep_cfr/deep_cfr_v14_full_selfplay/policy_best.pt"),
                "v15": NeuralPolicy.load("runs_deep_cfr/deep_cfr_v15_league/policy_best.pt"),
            }
        else:
            self.opponents = opponent_models

    def evaluate_candidate(
        self,
        candidate_policy: Policy,
        candidate_name: str = "candidate",
        games: int = 1000,
        seeds: tuple[int, ...] | None = None,
    ) -> ABBenchmarkResult:
        opp_names = ["v13", "v14", "v15"]

        if seeds is None:
            num_seeds = max(1, games // 4)
            seeds = tuple(50000 + i * 37 for i in range(num_seeds))

        wins = 0
        top2 = 0
        total_utility = 0.0
        finals_entered = 0
        finals_won = 0
        seat_wins = {1: 0, 2: 0, 3: 0, 4: 0}
        active_challenges = 0
        active_challenges_won = 0
        per_game_records: list[dict] = []

        total_games = len(seeds) * 4
        print(f"[A/B Benchmark] 启动固定对手 1k 标准考卷评估: {candidate_name} | 种子数={len(seeds)} | 总局数={total_games}")

        for s_idx, seed in enumerate(seeds):
            if (s_idx * 4) > 0 and (s_idx * 4) % 200 == 0:
                print(f"  ... 已推进 {s_idx * 4}/{total_games} 场 (当前吃鸡={wins} | 胜率={wins / (s_idx * 4):.1%})")
            for cand_seat in range(1, 5):
                # 安排座位：被测选手坐 cand_seat，其余位置按固定顺序由 v13, v14, v15 填补
                other_seats = [s for s in range(1, 5) if s != cand_seat]
                seat_map: dict[int, Policy] = {cand_seat: candidate_policy}
                for idx, os in enumerate(other_seats):
                    seat_map[os] = self.opponents[opp_names[idx]]

                state = self.engine.new_game(seed=seed, starting_seat=1)
                step = 0
                saw_finals_for_candidate = False

                while not self.engine.is_terminal(state) and step < 200:
                    alive_players = [p for p in state.players if p.alive]
                    cand_player = state.player_by_seat(cand_seat)

                    # 判定是否进入了 2 人决赛圈且候选者还活着
                    if len(alive_players) == 2 and cand_player.alive:
                        saw_finals_for_candidate = True

                    actor = self.engine.current_actor(state)
                    legal = self.engine.legal_actions(state)
                    if not legal:
                        break

                    obs = self.engine.observe(state, actor)
                    act = seat_map[actor].choose_action(obs, legal)

                    # 统计被测模型的主动质疑（必须手里有牌，排除 0 张牌强制质疑）
                    if actor == cand_seat and isinstance(act, ChallengeAction):
                        if len(cand_player.hand) > 0:
                            active_challenges += 1
                            last_p = state.round_state.latest_play
                            if last_p:
                                eval_res = self.engine._evaluate_play(last_p)
                                if eval_res.value in ("lie", "ghost"):
                                    active_challenges_won += 1

                    self.engine.apply_action(state, act)
                    step += 1

                # 终局判定
                alive = [p for p in state.players if p.alive]
                finish_order = self.reward_model._successful_finish_order(state)
                rewards = self.reward_model.evaluate(state)
                util = rewards.for_seat(cand_seat)
                total_utility += util

                cand_rank = finish_order.index(cand_seat)  # 0 为第一，1 为第二
                is_win = (cand_rank == 0)
                is_finals_win = (is_win and saw_finals_for_candidate)

                if is_win:
                    wins += 1
                    seat_wins[cand_seat] += 1
                    top2 += 1
                    if saw_finals_for_candidate:
                        finals_won += 1
                elif cand_rank == 1:
                    top2 += 1

                if saw_finals_for_candidate:
                    finals_entered += 1

                per_game_records.append({
                    "game_index": len(per_game_records),
                    "seed": seed,
                    "seat": cand_seat,
                    "win": is_win,
                    "rank": cand_rank + 1,
                    "entered_finals": saw_finals_for_candidate,
                    "finals_won": is_finals_win,
                    "utility": round(util, 2),
                })

        win_rate = round(wins / total_games, 3)
        top2_rate = round(top2 / total_games, 3)
        mean_u = round(total_utility / total_games, 2)
        finals_rate = round(finals_entered / total_games, 3)
        conv_rate = round(finals_won / max(1, finals_entered), 3)
        ch_acc = round(active_challenges_won / max(1, active_challenges), 3)

        return ABBenchmarkResult(
            candidate_name=candidate_name,
            total_games=total_games,
            wins=wins,
            win_rate=win_rate,
            top2_count=top2,
            top2_rate=top2_rate,
            mean_utility=mean_u,
            finals_entered=finals_entered,
            finals_entered_rate=finals_rate,
            finals_won=finals_won,
            finals_conversion_rate=conv_rate,
            seat_wins=seat_wins,
            active_challenges=active_challenges,
            active_challenges_won=active_challenges_won,
            active_challenge_acc=ch_acc,
            per_game_records=per_game_records,
        )


def print_ab_result(res: ABBenchmarkResult) -> None:
    seat_total = res.total_games // 4
    print("\n" + "=" * 62)
    print(f"       🎯 固定阵容 A/B 基准考卷报告: 【 {res.candidate_name} 】")
    print("=" * 62)
    print(f"测试局数 ({seat_total}种子×4座位):  {res.total_games} 场")
    print(f"🏆 终局吃鸡胜率 (主指标): {res.win_rate * 100:.1f}%  ({res.wins}/{res.total_games})")
    print(f"🛡️ 稳健保底前二率:       {res.top2_rate * 100:.1f}%  ({res.top2_count}/{res.total_games})")
    print(f"💰 场均综合效用得分:     {res.mean_utility:+.2f}")
    print("-" * 62)
    print(f"🚪 决赛圈进入率 (两人单挑): {res.finals_entered_rate * 100:.1f}%  ({res.finals_entered}/{res.total_games})")
    print(f"⚔️ 决赛圈转化吃鸡率:       {res.finals_conversion_rate * 100:.1f}%  ({res.finals_won}/{res.finals_entered})")
    print(f"🎯 主动质疑命中率 (排空手): {res.active_challenge_acc * 100:.1f}%  ({res.active_challenges_won}/{res.active_challenges})")
    print(f"🪑 各座位胜率 (1/2/3/4号位): [{res.seat_wins[1]}, {res.seat_wins[2]}, {res.seat_wins[3]}, {res.seat_wins[4]}] / {seat_total}")
    print("=" * 62 + "\n")


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Run fixed-opponent A/B benchmark")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt")
    parser.add_argument("--name", type=str, default="Candidate", help="Model candidate name")
    parser.add_argument("--games", type=int, default=1000, help="Total games (default: 1000)")
    parser.add_argument("--output-json", type=str, default=None, help="Path to save detailed per-game results")
    args = parser.parse_args()

    candidate = NeuralPolicy.load(args.checkpoint)
    bench = FixedOpponentBenchmark()
    res = bench.evaluate_candidate(candidate, candidate_name=args.name, games=args.games)
    print_ab_result(res)

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        from dataclasses import asdict
        out_path.write_text(json.dumps(asdict(res), indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[A/B Benchmark] 逐局明细数据已落盘至: {out_path}")



