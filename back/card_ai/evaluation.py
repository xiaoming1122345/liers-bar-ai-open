from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Callable

from .engine import SurvivalGameEngine
from .identities import build_default_identities
from .modeling import build_player_model_summaries
from .opponents import OpponentProfile, build_policy_from_profile
from .rewards import TerminalRewardModel
from .self_play import SelfPlayRunner, StateSnapshot
from .types import MatchRecord, PerspectiveEvaluationSummary, PlayerIdentity, Policy


@dataclass(frozen=True)
class EvaluationRecordSet:
    identities: tuple[PlayerIdentity, ...]
    records: tuple[MatchRecord, ...]
    summary: PerspectiveEvaluationSummary


class MatchupEvaluator:
    def __init__(
        self,
        engine: SurvivalGameEngine | None = None,
        reward_model: TerminalRewardModel | None = None,
    ) -> None:
        self.engine = engine or SurvivalGameEngine()
        self.reward_model = reward_model or TerminalRewardModel()
        self.self_play = SelfPlayRunner(self.engine)

    def evaluate_hero_vs_pool(
        self,
        hero_policy_factory: Callable[[int, int], Policy],
        opponent_profiles: tuple[OpponentProfile, ...],
        game_count: int,
        hero_seat: int,
        seed_start: int = 0,
        max_steps: int = 512,
        custom_ids: dict[int, str] | None = None,
        hero_label: str = "我",
    ) -> EvaluationRecordSet:
        identities = build_default_identities(
            player_count=self.engine.player_count,
            hero_seat=hero_seat,
            custom_ids=custom_ids,
            hero_label=hero_label,
        )
        rng = Random(seed_start)
        records = []
        snapshots = []
        for game_index in range(game_count):
            policies: dict[int, Policy] = {}
            seat_profile_names: dict[int, str] = {}
            for seat in range(1, self.engine.player_count + 1):
                if seat == hero_seat:
                    policies[seat] = hero_policy_factory(game_index, seat)
                    seat_profile_names[seat] = "hero"
                    continue

                profile = opponent_profiles[rng.randrange(len(opponent_profiles))]
                policies[seat] = build_policy_from_profile(
                    profile=profile,
                    seed=seed_start + game_index * 101 + seat,
                )
                seat_profile_names[seat] = profile.name

            trace = self.self_play.run_game(
                policies=policies,
                seed=seed_start + game_index,
                max_steps=max_steps,
            )
            records.append(self._build_match_record(game_index, trace.final_state, seat_profile_names))
            snapshots.extend(trace.snapshots)

        summary = self._summarize_records(
            mode="hero_vs_pool",
            identities=identities,
            records=tuple(records),
            hero_seat=hero_seat,
            snapshots=tuple(snapshots),
        )
        return EvaluationRecordSet(identities=identities, records=tuple(records), summary=summary)

    def evaluate_pool_self_play(
        self,
        opponent_profiles: tuple[OpponentProfile, ...],
        game_count: int,
        seed_start: int = 0,
        max_steps: int = 512,
        custom_ids: dict[int, str] | None = None,
    ) -> EvaluationRecordSet:
        identities = build_default_identities(
            player_count=self.engine.player_count,
            hero_seat=None,
            custom_ids=custom_ids,
        )
        rng = Random(seed_start)
        records = []
        snapshots = []
        for game_index in range(game_count):
            policies: dict[int, Policy] = {}
            seat_profile_names: dict[int, str] = {}
            for seat in range(1, self.engine.player_count + 1):
                profile = opponent_profiles[rng.randrange(len(opponent_profiles))]
                policies[seat] = build_policy_from_profile(
                    profile=profile,
                    seed=seed_start + game_index * 131 + seat,
                )
                seat_profile_names[seat] = profile.name

            trace = self.self_play.run_game(
                policies=policies,
                seed=seed_start + game_index,
                max_steps=max_steps,
            )
            records.append(self._build_match_record(game_index, trace.final_state, seat_profile_names))
            snapshots.extend(trace.snapshots)

        summary = self._summarize_records(
            mode="pool_self_play",
            identities=identities,
            records=tuple(records),
            hero_seat=None,
            snapshots=tuple(snapshots),
        )
        return EvaluationRecordSet(identities=identities, records=tuple(records), summary=summary)

    def _build_match_record(self, game_index: int, final_state, seat_profile_names: dict[int, str]) -> MatchRecord:
        rewards = self.reward_model.evaluate(final_state)
        utilities = {seat: rewards.for_seat(seat) for seat in range(1, self.engine.player_count + 1)}
        sorted_seats = sorted(utilities, key=lambda seat: (-utilities[seat], seat))
        ranks = {seat: index + 1 for index, seat in enumerate(sorted_seats)}
        winner_seat = sorted_seats[0]
        return MatchRecord(
            game_index=game_index,
            winner_seat=winner_seat,
            utilities=utilities,
            ranks=ranks,
            seat_profile_names=seat_profile_names,
        )

    def _summarize_records(
        self,
        mode: str,
        identities: tuple[PlayerIdentity, ...],
        records: tuple[MatchRecord, ...],
        hero_seat: int | None,
        snapshots: tuple[StateSnapshot, ...],
    ) -> PerspectiveEvaluationSummary:
        games = len(records)
        seat_wins = {identity.seat: 0 for identity in identities}
        seat_utilities = {identity.seat: 0.0 for identity in identities}
        opponent_profile_counts: dict[str, int] = {}
        seat_models = build_player_model_summaries(identities=identities, snapshots=snapshots)
        opponent_inferred_profile_counts: dict[str, int] = {}

        hero_wins = 0
        hero_top2 = 0
        hero_rank_total = 0.0
        hero_utility_total = 0.0

        for record in records:
            seat_wins[record.winner_seat] = seat_wins.get(record.winner_seat, 0) + 1
            for seat, utility in record.utilities.items():
                seat_utilities[seat] = seat_utilities.get(seat, 0.0) + utility
            for seat, profile_name in record.seat_profile_names.items():
                if hero_seat is not None and seat == hero_seat:
                    continue
                opponent_profile_counts[profile_name] = opponent_profile_counts.get(profile_name, 0) + 1

            if hero_seat is not None:
                hero_rank = record.ranks[hero_seat]
                hero_utility = record.utilities[hero_seat]
                if hero_rank == 1:
                    hero_wins += 1
                if hero_rank <= 2:
                    hero_top2 += 1
                hero_rank_total += hero_rank
                hero_utility_total += hero_utility

        hero_identity = next((identity for identity in identities if identity.seat == hero_seat), None)
        for model in seat_models:
            if hero_seat is not None and model.seat == hero_seat:
                continue
            opponent_inferred_profile_counts[model.inferred_profile_name] = (
                opponent_inferred_profile_counts.get(model.inferred_profile_name, 0) + 1
            )
        return PerspectiveEvaluationSummary(
            mode=mode,
            games=games,
            hero_seat=hero_seat,
            hero_player_id=hero_identity.player_id if hero_identity is not None else None,
            hero_display_name=hero_identity.display_name if hero_identity is not None else None,
            win_rate=(hero_wins / games) if hero_seat is not None and games > 0 else None,
            top2_rate=(hero_top2 / games) if hero_seat is not None and games > 0 else None,
            mean_rank=(hero_rank_total / games) if hero_seat is not None and games > 0 else None,
            mean_utility=(hero_utility_total / games) if hero_seat is not None and games > 0 else None,
            seat_win_rate={
                seat: (seat_wins.get(seat, 0) / games) if games > 0 else 0.0
                for seat in sorted(seat_wins)
            },
            seat_mean_utility={
                seat: (seat_utilities.get(seat, 0.0) / games) if games > 0 else 0.0
                for seat in sorted(seat_utilities)
            },
            opponent_profile_counts=dict(sorted(opponent_profile_counts.items())),
            opponent_inferred_profile_counts=dict(sorted(opponent_inferred_profile_counts.items())),
            identities=identities,
            seat_models=seat_models,
        )
