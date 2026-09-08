from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .belief import BeliefTracker, rank_support_count
from .confidence import build_confidence_report
from .config import TrainingRunConfig
from .engine import SurvivalGameEngine
from .policies import SolverPolicy
from .rewards import RewardVector, TerminalRewardModel
from .self_play import RandomPolicy, SelfPlayRunner
from .solvers import Solver
from .types import (
    BeliefDiagnostics,
    BeliefTruth,
    CardKind,
    ConfidenceReport,
    Policy,
    PublicBeliefState,
    Rank,
    TrainingBatchReport,
)


@dataclass
class TrainingSample:
    seat: int
    step_index: int
    terminal_utility: float
    belief: PublicBeliefState
    confidence: ConfidenceReport
    truth: BeliefTruth
    diagnostics: BeliefDiagnostics


class TrainingBackend:
    def __init__(
        self,
        engine: SurvivalGameEngine | None = None,
        belief_tracker: BeliefTracker | None = None,
        reward_model: TerminalRewardModel | None = None,
    ) -> None:
        self.engine = engine or SurvivalGameEngine()
        self.self_play = SelfPlayRunner(self.engine)
        self.belief_tracker = belief_tracker or BeliefTracker()
        self.reward_model = reward_model or TerminalRewardModel()

    def collect_random_self_play_batch(
        self,
        game_count: int,
        seed_start: int = 0,
        include_all_snapshots: bool = True,
        max_steps: int = 512,
    ) -> list[TrainingSample]:
        config = TrainingRunConfig(
            game_count=game_count,
            seed_start=seed_start,
            include_all_snapshots=include_all_snapshots,
            max_steps=max_steps,
        )
        return self.collect_self_play_batch(
            policy_factory=lambda game_offset: {
                seat: RandomPolicy(seed_start + game_offset * 17 + seat)
                for seat in range(1, self.engine.player_count + 1)
            },
            config=config,
        )

    def collect_solver_self_play_batch(
        self,
        solver: Solver,
        config: TrainingRunConfig,
        policy_seed_base: int = 0,
    ) -> list[TrainingSample]:
        return self.collect_self_play_batch(
            policy_factory=lambda game_offset: {
                seat: SolverPolicy(
                    solver=solver,
                    seed=policy_seed_base + game_offset * 31 + seat,
                )
                for seat in range(1, self.engine.player_count + 1)
            },
            config=config,
        )

    def collect_self_play_batch(
        self,
        policy_factory: Callable[[int], dict[int, Policy]],
        config: TrainingRunConfig,
    ) -> list[TrainingSample]:
        samples: list[TrainingSample] = []

        for game_offset in range(config.game_count):
            policies = policy_factory(game_offset)
            trace = self.self_play.run_game(
                policies=policies,
                seed=config.seed_start + game_offset,
                max_steps=config.max_steps,
            )
            rewards = self.reward_model.evaluate(trace.final_state)

            states = [snapshot.state_before for snapshot in trace.snapshots] if config.include_all_snapshots else [trace.final_state]

            for step_index, state in enumerate(states):
                samples.extend(self._build_samples_for_state(state, step_index, rewards))

        return samples

    def _build_samples_for_state(
        self,
        state,
        step_index: int,
        rewards: RewardVector,
    ) -> list[TrainingSample]:
        truth = self._build_truth(state)
        samples = []
        for seat in range(1, self.engine.player_count + 1):
            observation = self.engine.observe(state, seat)
            belief = self.belief_tracker.build(observation)
            confidence = build_confidence_report(belief)
            diagnostics = self._build_diagnostics(belief, confidence, truth)
            samples.append(
                TrainingSample(
                    seat=seat,
                    step_index=step_index,
                    terminal_utility=rewards.for_seat(seat),
                    belief=belief,
                    confidence=confidence,
                    truth=truth,
                    diagnostics=diagnostics,
                )
            )
        return samples

    def summarize_batch(self, samples: list[TrainingSample]) -> TrainingBatchReport:
        if not samples:
            return TrainingBatchReport(
                sample_count=0,
                mean_modeling_degree=0.0,
                mean_calibration_score=0.0,
                mean_confidence_gap=0.0,
                mean_ghost_location_probability=0.0,
                mean_ghost_rank_probability=0.0,
                mean_support_alignment=0.0,
                mean_support_error=0.0,
                mean_terminal_utility=0.0,
                seat_mean_terminal_utility={},
                seat_sample_count={},
            )

        sample_count = len(samples)
        seat_totals: dict[int, float] = {}
        seat_counts: dict[int, int] = {}
        for sample in samples:
            seat_totals[sample.seat] = seat_totals.get(sample.seat, 0.0) + sample.terminal_utility
            seat_counts[sample.seat] = seat_counts.get(sample.seat, 0) + 1

        return TrainingBatchReport(
            sample_count=sample_count,
            mean_modeling_degree=sum(sample.confidence.modeling_degree for sample in samples) / sample_count,
            mean_calibration_score=sum(sample.diagnostics.calibration_score for sample in samples) / sample_count,
            mean_confidence_gap=sum(sample.diagnostics.confidence_gap for sample in samples) / sample_count,
            mean_ghost_location_probability=sum(sample.diagnostics.ghost_location_probability for sample in samples) / sample_count,
            mean_ghost_rank_probability=sum(sample.diagnostics.ghost_rank_probability for sample in samples) / sample_count,
            mean_support_alignment=sum(sample.diagnostics.support_alignment for sample in samples) / sample_count,
            mean_support_error=sum(sample.diagnostics.mean_support_error for sample in samples) / sample_count,
            mean_terminal_utility=sum(sample.terminal_utility for sample in samples) / sample_count,
            seat_mean_terminal_utility={
                seat: seat_totals[seat] / seat_counts[seat]
                for seat in sorted(seat_totals)
            },
            seat_sample_count={seat: seat_counts[seat] for seat in sorted(seat_counts)},
        )

    def _build_truth(self, state) -> BeliefTruth:
        ghost_holder: int | None = None
        ghost_rank: Rank | None = None
        ghost_revealed = False
        seat_rank_support: dict[int, dict[Rank, int]] = {}

        for player in state.players:
            seat_rank_support[player.seat] = {
                rank: rank_support_count(player.hand, rank) for rank in Rank
            }
            for card in player.hand:
                if card.kind == CardKind.GHOST:
                    ghost_holder = player.seat
                    ghost_rank = card.printed_rank

        if ghost_rank is None:
            for played_set in state.round_state.plays:
                for card in played_set.cards:
                    if card.kind == CardKind.GHOST:
                        ghost_holder = played_set.seat
                        ghost_rank = card.printed_rank
                        break
                if ghost_rank is not None:
                    break

        if ghost_rank is None:
            for card in state.discard_pile:
                if card.kind == CardKind.GHOST:
                    ghost_rank = card.printed_rank
                    ghost_revealed = True
                    break

        if ghost_rank is None:
            raise ValueError("ghost rank could not be recovered from the state")

        return BeliefTruth(
            ghost_hidden_holder_seat=ghost_holder,
            ghost_rank=ghost_rank,
            ghost_revealed=ghost_revealed,
            seat_rank_support=seat_rank_support,
        )

    def _build_diagnostics(
        self,
        belief: PublicBeliefState,
        confidence: ConfidenceReport,
        truth: BeliefTruth,
    ) -> BeliefDiagnostics:
        if truth.ghost_revealed:
            ghost_location_probability = belief.ghost_out_of_play_prob
        elif truth.ghost_hidden_holder_seat is None:
            ghost_location_probability = 0.0
        else:
            seat_lookup = {seat.seat: seat for seat in belief.seat_beliefs}
            ghost_location_probability = seat_lookup[truth.ghost_hidden_holder_seat].ghost_hidden_prob

        ghost_rank_probability = belief.ghost_rank_probs[truth.ghost_rank]

        support_errors = []
        seat_lookup = {seat.seat: seat for seat in belief.seat_beliefs}
        for seat, truth_support in truth.seat_rank_support.items():
            predicted = seat_lookup[seat].rank_support_expectation
            for rank, true_count in truth_support.items():
                support_errors.append(abs(predicted[rank] - true_count))

        mean_support_error = sum(support_errors) / len(support_errors) if support_errors else 0.0
        support_alignment = max(0.0, 1.0 - mean_support_error / 3.0)
        calibration_score = (
            0.40 * ghost_location_probability
            + 0.30 * ghost_rank_probability
            + 0.30 * support_alignment
        )
        confidence_gap = confidence.modeling_degree - calibration_score

        return BeliefDiagnostics(
            ghost_location_probability=ghost_location_probability,
            ghost_rank_probability=ghost_rank_probability,
            mean_support_error=mean_support_error,
            support_alignment=support_alignment,
            calibration_score=max(0.0, min(1.0, calibration_score)),
            confidence_gap=confidence_gap,
        )
