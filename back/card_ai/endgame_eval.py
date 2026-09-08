from __future__ import annotations

"""残局与劣势博弈专项评估套件 (Endgame & Underdog Evaluation Benchmark)。

专门在 100 场极高压死斗残局（手牌 1~2 张、濒死劣势、单挑决死、魔牌陷阱）中
测试 AI 模型的逆风翻盘胜率、残局生存率与生死抉择准确率。
"""

from dataclasses import dataclass
from random import Random
from typing import Callable

from .endgame_generator import EndgameScenarioGenerator
from .engine import SurvivalGameEngine
from .opponents import build_policy_from_profile, default_opponent_profiles
from .rewards import TerminalRewardModel
from .types import ChallengeAction, GameState, PlayAction, Policy


@dataclass
class EndgameBenchmarkResult:
    total_games: int
    wins: int
    win_rate: float
    top2_count: int
    top2_rate: float
    mean_utility: float
    challenge_count: int
    challenge_success: int
    challenge_accuracy: float
    bluff_count: int
    bluff_survived: int
    bluff_survival_rate: float


class EndgameBenchmark:
    """100 场极端残局与劣势对战评测基准。"""

    def __init__(
        self,
        engine: SurvivalGameEngine | None = None,
        reward_model: TerminalRewardModel | None = None,
    ) -> None:
        self.engine = engine or SurvivalGameEngine()
        self.reward_model = reward_model or TerminalRewardModel()
        self.generator = EndgameScenarioGenerator(self.engine)

    def evaluate_policy(
        self,
        hero_policy: Policy,
        game_count: int = 100,
        seed_start: int = 42000,
        hero_seat: int = 1,
        max_steps: int = 64,
    ) -> EndgameBenchmarkResult:
        wins = 0
        top2 = 0
        total_utility = 0.0
        challenges = 0
        challenge_success = 0
        bluffs = 0
        bluff_survived = 0

        profiles = default_opponent_profiles()

        for g in range(game_count):
            seed = seed_start + g
            rng = Random(seed)

            # 采样极端残局
            state = self.generator.sample_endgame_state(
                seed=seed, scenario="random_mix", hero_seat=hero_seat
            )

            # 为其余座位构建对手
            opponents: dict[int, Policy] = {}
            for p in state.players:
                if p.seat != hero_seat:
                    prof = rng.choice(profiles)
                    opponents[p.seat] = build_policy_from_profile(prof, seed=seed * 10 + p.seat)

            step = 0
            while not self.engine.is_terminal(state) and step < max_steps:
                acting = self.engine.current_actor(state)
                legal = self.engine.legal_actions(state)
                if not legal:
                    break

                obs = self.engine.observe(state, acting)
                if acting == hero_seat:
                    action = hero_policy.choose_action(obs, legal)
                    # 统计 Hero 行为
                    if isinstance(action, ChallengeAction):
                        challenges += 1
                        last_p = state.round_state.latest_play
                        if last_p:
                            eval_res = self.engine._evaluate_play(last_p)
                            if eval_res.value in ("lie", "ghost"):
                                challenge_success += 1
                    elif isinstance(action, PlayAction):
                        # 判断是否为诈唬
                        hero_state = state.player_by_seat(hero_seat)
                        is_bluff = False
                        for cid in action.card_ids:
                            c = next((card for card in hero_state.hand if card.card_id == cid), None)
                            if c and c.kind.value != "wild" and c.kind.value != "ghost" and c.printed_rank != action.claim_rank:
                                is_bluff = True
                                break
                        if is_bluff:
                            bluffs += 1
                else:
                    opp = opponents.get(acting)
                    if opp:
                        action = opp.choose_action(obs, legal)
                    else:
                        action = rng.choice(legal)

                self.engine.apply_action(state, action)
                step += 1

            # 结算
            hero = state.player_by_seat(hero_seat)
            alive_players = [p for p in state.players if p.alive]
            if hero.alive:
                if len(alive_players) <= 1:
                    wins += 1
                    top2 += 1
                elif len(alive_players) == 2:
                    top2 += 1
            if self.engine.is_terminal(state):
                util = self.reward_model.evaluate(state).for_seat(hero_seat)
            else:
                alive_count = sum(1 for p in state.players if p.alive)
                hero = state.player_by_seat(hero_seat)
                util = 10.0 if hero.alive else -10.0
            total_utility += util

        return EndgameBenchmarkResult(
            total_games=game_count,
            wins=wins,
            win_rate=round(wins / max(1, game_count), 3),
            top2_count=top2,
            top2_rate=round(top2 / max(1, game_count), 3),
            mean_utility=round(total_utility / max(1, game_count), 2),
            challenge_count=challenges,
            challenge_success=challenge_success,
            challenge_accuracy=round(challenge_success / max(1, challenges), 3),
            bluff_count=bluffs,
            bluff_survived=bluff_survived,
            bluff_survival_rate=round(bluff_survived / max(1, bluffs), 3),
        )
