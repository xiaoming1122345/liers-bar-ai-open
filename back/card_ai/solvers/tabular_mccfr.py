from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from random import Random

from ..abstractions import AbstractAction, ActionAbstractor, InformationSetEncoder, InformationSetKey
from ..artifacts import SolverArtifact
from ..config import SolverConfig
from ..engine import SurvivalGameEngine
from ..rewards import TerminalRewardModel
from ..types import GameAction, PrivateObservation, Rank
from .base import Solver, StrategyResult


@dataclass
class TabularInfoSetStats:
    regrets: dict[str, float] = field(default_factory=dict)
    strategy_sum: dict[str, float] = field(default_factory=dict)
    visit_count: int = 0


@dataclass(frozen=True)
class TabularMCCFRTrainingSummary:
    iteration_count: int
    traverser_updates: int
    info_set_count: int
    mean_traverser_utility: float


@dataclass(frozen=True)
class TabularMCCFRTrainingResult:
    solver: "TabularBlueprintSolver"
    artifact: SolverArtifact
    summary: TabularMCCFRTrainingSummary


class TabularBlueprintSolver(Solver):
    def __init__(
        self,
        info_sets: dict[InformationSetKey, TabularInfoSetStats],
        encoder: InformationSetEncoder | None = None,
        abstractor: ActionAbstractor | None = None,
    ) -> None:
        self.info_sets = info_sets
        self.encoder = encoder or InformationSetEncoder()
        self.abstractor = abstractor or ActionAbstractor()

    def strategy_for(
        self,
        observation: PrivateObservation,
        legal_actions: tuple[GameAction, ...],
    ) -> StrategyResult:
        if not legal_actions:
            return StrategyResult(action_probs={})

        grouped = self.abstractor.abstract_legal_actions(legal_actions, observation)
        key = self.encoder.encode(observation)
        label_probs = self._average_strategy_for(key, tuple(grouped.keys()))
        concrete_probs: dict[GameAction, float] = {}

        for abstract_action, concrete_group in grouped.items():
            abstract_prob = label_probs.get(abstract_action.label, 0.0)
            if not concrete_group:
                continue
            per_action = abstract_prob / len(concrete_group)
            for action in concrete_group:
                concrete_probs[action] = per_action

        return StrategyResult(
            action_probs=concrete_probs,
            info={
                "solver": "tabular_mccfr_blueprint",
                "info_set_seen": int(key in self.info_sets),
            },
        ).normalized()

    def _average_strategy_for(
        self,
        key: InformationSetKey,
        available_actions: tuple[AbstractAction, ...],
    ) -> dict[str, float]:
        stats = self.info_sets.get(key)
        labels = [action.label for action in available_actions]
        if stats is None:
            return self._uniform_labels(labels)

        total = sum(max(0.0, stats.strategy_sum.get(label, 0.0)) for label in labels)
        if total <= 0:
            return self._regret_matching(stats, labels)

        return {
            label: max(0.0, stats.strategy_sum.get(label, 0.0)) / total
            for label in labels
        }

    def _regret_matching(
        self,
        stats: TabularInfoSetStats,
        labels: list[str],
    ) -> dict[str, float]:
        positives = {label: max(0.0, stats.regrets.get(label, 0.0)) for label in labels}
        total = sum(positives.values())
        if total <= 0:
            return self._uniform_labels(labels)
        return {label: positives[label] / total for label in labels}

    def _uniform_labels(self, labels: list[str]) -> dict[str, float]:
        if not labels:
            return {}
        probability = 1.0 / len(labels)
        return {label: probability for label in labels}

    def to_dict(self) -> dict:
        return {
            "info_sets": [
                {
                    "key": self._serialize_info_key(key),
                    "stats": {
                        "regrets": stats.regrets,
                        "strategy_sum": stats.strategy_sum,
                        "visit_count": stats.visit_count,
                    },
                }
                for key, stats in self.info_sets.items()
            ]
        }

    @classmethod
    def from_dict(
        cls,
        payload: dict,
        encoder: InformationSetEncoder | None = None,
        abstractor: ActionAbstractor | None = None,
    ) -> "TabularBlueprintSolver":
        info_sets: dict[InformationSetKey, TabularInfoSetStats] = {}
        for item in payload.get("info_sets", []):
            key = cls._deserialize_info_key(item["key"])
            stats_payload = item["stats"]
            info_sets[key] = TabularInfoSetStats(
                regrets={str(k): float(v) for k, v in stats_payload.get("regrets", {}).items()},
                strategy_sum={str(k): float(v) for k, v in stats_payload.get("strategy_sum", {}).items()},
                visit_count=int(stats_payload.get("visit_count", 0)),
            )
        return cls(info_sets=info_sets, encoder=encoder, abstractor=abstractor)

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return output

    @classmethod
    def load(
        cls,
        path: str | Path,
        encoder: InformationSetEncoder | None = None,
        abstractor: ActionAbstractor | None = None,
    ) -> "TabularBlueprintSolver":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload, encoder=encoder, abstractor=abstractor)

    @staticmethod
    def _serialize_info_key(key: InformationSetKey) -> dict:
        return {
            "seat": key.seat,
            "current_seat": key.current_seat,
            "claim_rank": key.claim_rank.value if key.claim_rank is not None else None,
            "latest_play_seat": key.latest_play_seat,
            "latest_play_count": key.latest_play_count,
            "private_hand_signature": [list(item) for item in key.private_hand_signature],
            "public_player_signature": [list(item) for item in key.public_player_signature],
            "public_history_signature": [
                [
                    turn_index,
                    event_type,
                    seat,
                    [list(pair) for pair in detail_pairs],
                ]
                for turn_index, event_type, seat, detail_pairs in key.public_history_signature
            ],
        }

    @staticmethod
    def _deserialize_info_key(payload: dict) -> InformationSetKey:
        return InformationSetKey(
            seat=int(payload["seat"]),
            current_seat=int(payload["current_seat"]),
            claim_rank=None if payload["claim_rank"] is None else Rank(payload["claim_rank"]),
            latest_play_seat=payload["latest_play_seat"],
            latest_play_count=int(payload["latest_play_count"]),
            private_hand_signature=tuple(
                (str(label), int(count))
                for label, count in payload.get("private_hand_signature", [])
            ),
            public_player_signature=tuple(
                (
                    int(seat),
                    bool(alive),
                    bool(escaped),
                    bool(pending_escape),
                    int(hand_count),
                    int(shots_taken),
                )
                for seat, alive, escaped, pending_escape, hand_count, shots_taken in payload.get("public_player_signature", [])
            ),
            public_history_signature=tuple(
                (
                    int(turn_index),
                    str(event_type),
                    None if seat is None else int(seat),
                    tuple((str(key), str(value)) for key, value in detail_pairs),
                )
                for turn_index, event_type, seat, detail_pairs in payload.get("public_history_signature", [])
            ),
        )


class TabularMCCFRTrainer:
    """
    External-sampling style tabular regret trainer.

    This is intentionally a scaffold for the current 4-player general-sum game:
    each traverser updates its own regrets against sampled opponents, which is
    useful as an engineering blueprint even though it does not provide the clean
    convergence guarantees of 2-player zero-sum CFR.
    """

    def __init__(
        self,
        engine: SurvivalGameEngine | None = None,
        reward_model: TerminalRewardModel | None = None,
        encoder: InformationSetEncoder | None = None,
        abstractor: ActionAbstractor | None = None,
    ) -> None:
        self.engine = engine or SurvivalGameEngine()
        self.reward_model = reward_model or TerminalRewardModel()
        self.encoder = encoder or InformationSetEncoder()
        self.abstractor = abstractor or ActionAbstractor()
        self.info_sets: dict[InformationSetKey, TabularInfoSetStats] = {}
        self._rng = Random()

    def train(self, config: SolverConfig) -> TabularMCCFRTrainingResult:
        self._rng.seed(config.seed)
        traverser_utilities: list[float] = []
        traverser_updates = 0
        seed_base = config.seed or 0

        for iteration in range(config.iteration_count):
            for traverser_seat in range(1, self.engine.player_count + 1):
                state = self.engine.new_game(
                    seed=seed_base + iteration * self.engine.player_count + traverser_seat,
                    starting_seat=1,
                )
                utility = self._traverse(
                    state,
                    traverser_seat,
                    depth=0,
                    max_depth=config.max_depth,
                )
                traverser_utilities.append(utility)
                traverser_updates += 1

        solver = self.build_solver()
        summary = TabularMCCFRTrainingSummary(
            iteration_count=config.iteration_count,
            traverser_updates=traverser_updates,
            info_set_count=len(self.info_sets),
            mean_traverser_utility=(
                sum(traverser_utilities) / len(traverser_utilities)
                if traverser_utilities
                else 0.0
            ),
        )
        artifact = SolverArtifact.create(
            artifact_id=f"{config.name}:{config.iteration_count}:{len(self.info_sets)}",
            solver_config=config,
            metadata={
                "trainer": "tabular_mccfr",
                "info_set_count": len(self.info_sets),
                "traverser_updates": traverser_updates,
                "mean_traverser_utility": summary.mean_traverser_utility,
                "max_depth": config.max_depth if config.max_depth is not None else -1,
            },
        )
        return TabularMCCFRTrainingResult(
            solver=solver,
            artifact=artifact,
            summary=summary,
        )

    def build_solver(self) -> TabularBlueprintSolver:
        return TabularBlueprintSolver(
            info_sets=self.info_sets,
            encoder=self.encoder,
            abstractor=self.abstractor,
        )

    def _traverse(
        self,
        state,
        traverser_seat: int,
        depth: int,
        max_depth: int | None,
    ) -> float:
        if self.engine.is_terminal(state):
            rewards = self.reward_model.evaluate(state)
            return rewards.for_seat(traverser_seat)
        if max_depth is not None and depth >= max_depth:
            return self._leaf_utility(state, traverser_seat)

        acting_seat = self.engine.current_actor(state)
        observation = self.engine.observe(state, acting_seat)
        legal_actions = self.engine.legal_actions(state)
        grouped = self.abstractor.abstract_legal_actions(legal_actions, observation)
        info_key = self.encoder.encode(observation)
        available_actions = tuple(grouped.keys())
        stats = self._ensure_info_set(info_key, available_actions)
        strategy = self._current_strategy(stats, available_actions)
        self._accumulate_average_strategy(stats, strategy)
        stats.visit_count += 1

        if acting_seat == traverser_seat:
            action_utilities: dict[str, float] = {}
            node_utility = 0.0
            for abstract_action, concrete_group in grouped.items():
                # We use one representative concrete action per abstract bucket.
                # With the current abstraction, duplicated rank copies are exchangeable.
                representative_action = concrete_group[0]
                next_state = self.engine.clone_state(state)
                self.engine.apply_action(next_state, representative_action)
                utility = self._traverse(
                    next_state,
                    traverser_seat,
                    depth=depth + 1,
                    max_depth=max_depth,
                )
                action_utilities[abstract_action.label] = utility
                node_utility += strategy[abstract_action.label] * utility

            for abstract_action in available_actions:
                label = abstract_action.label
                regret_delta = action_utilities[label] - node_utility
                stats.regrets[label] = stats.regrets.get(label, 0.0) + regret_delta

            return node_utility

        sampled_abstract = self._sample_abstract_action(available_actions, strategy)
        sampled_concrete = self._rng.choice(grouped[sampled_abstract])
        next_state = self.engine.clone_state(state)
        self.engine.apply_action(next_state, sampled_concrete)
        return self._traverse(
            next_state,
            traverser_seat,
            depth=depth + 1,
            max_depth=max_depth,
        )

    def _ensure_info_set(
        self,
        info_key: InformationSetKey,
        available_actions: tuple[AbstractAction, ...],
    ) -> TabularInfoSetStats:
        stats = self.info_sets.setdefault(info_key, TabularInfoSetStats())
        for action in available_actions:
            stats.regrets.setdefault(action.label, 0.0)
            stats.strategy_sum.setdefault(action.label, 0.0)
        return stats

    def _current_strategy(
        self,
        stats: TabularInfoSetStats,
        available_actions: tuple[AbstractAction, ...],
    ) -> dict[str, float]:
        labels = [action.label for action in available_actions]
        positives = {label: max(0.0, stats.regrets.get(label, 0.0)) for label in labels}
        total = sum(positives.values())
        if total <= 0:
            probability = 1.0 / len(labels)
            return {label: probability for label in labels}
        return {label: positives[label] / total for label in labels}

    def _accumulate_average_strategy(
        self,
        stats: TabularInfoSetStats,
        strategy: dict[str, float],
    ) -> None:
        for label, probability in strategy.items():
            stats.strategy_sum[label] = stats.strategy_sum.get(label, 0.0) + probability

    def _sample_abstract_action(
        self,
        available_actions: tuple[AbstractAction, ...],
        strategy: dict[str, float],
    ) -> AbstractAction:
        threshold = self._rng.random()
        cumulative = 0.0
        last_action = available_actions[-1]
        for abstract_action in available_actions:
            cumulative += strategy[abstract_action.label]
            last_action = abstract_action
            if threshold <= cumulative:
                return abstract_action
        return last_action

    def _leaf_utility(self, state, traverser_seat: int) -> float:
        player = state.player_by_seat(traverser_seat)
        if not player.alive:
            return self.reward_model.score_rules.eliminated_or_last
        if player.escaped:
            return self.reward_model.score_rules.third
        return 0.0
