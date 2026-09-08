from __future__ import annotations

from dataclasses import dataclass
from random import Random

from .engine import SurvivalGameEngine
from .types import GameAction, GameState, Policy


@dataclass
class StateSnapshot:
    step_index: int
    acting_seat: int
    chosen_action: GameAction
    state_before: GameState
    state_after: GameState


@dataclass
class SelfPlayTrace:
    final_state: GameState
    seat_action_counts: dict[int, int]
    snapshots: tuple[StateSnapshot, ...]


class RandomPolicy:
    def __init__(self, seed: int | None = None) -> None:
        self._rng = Random(seed)

    def choose_action(self, observation, legal_actions):
        if not legal_actions:
            raise ValueError("policy was asked to act without legal actions")
        return self._rng.choice(legal_actions)


class SelfPlayRunner:
    def __init__(self, engine: SurvivalGameEngine) -> None:
        self.engine = engine

    def run_game(
        self,
        policies: dict[int, Policy],
        seed: int | None = None,
        starting_seat: int = 1,
        max_steps: int = 512,
    ) -> SelfPlayTrace:
        state = self.engine.new_game(seed=seed, starting_seat=starting_seat)
        seat_action_counts = {seat: 0 for seat in policies}
        snapshots: list[StateSnapshot] = []

        for step_index in range(max_steps):
            if self.engine.is_terminal(state):
                return SelfPlayTrace(
                    final_state=state,
                    seat_action_counts=seat_action_counts,
                    snapshots=tuple(snapshots),
                )

            current_seat = self.engine.current_actor(state)
            policy = policies[current_seat]
            observation = self.engine.observe(state, current_seat)
            legal_actions = self.engine.legal_actions(state)
            action = policy.choose_action(observation, legal_actions)
            seat_action_counts[current_seat] = seat_action_counts.get(current_seat, 0) + 1
            state_before = self.engine.clone_state(state)
            state = self.engine.apply_action(state, action)
            state_after = self.engine.clone_state(state)
            snapshots.append(
                StateSnapshot(
                    step_index=step_index,
                    acting_seat=current_seat,
                    chosen_action=action,
                    state_before=state_before,
                    state_after=state_after,
                )
            )

        raise RuntimeError("self-play exceeded max_steps; likely a rules bug or cyclic policy")
