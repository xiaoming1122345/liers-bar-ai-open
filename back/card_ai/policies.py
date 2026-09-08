from __future__ import annotations

from random import Random

from .solvers import Solver
from .types import GameAction, PrivateObservation


class SolverPolicy:
    def __init__(self, solver: Solver, seed: int | None = None) -> None:
        self.solver = solver
        self._rng = Random(seed)

    def choose_action(
        self,
        observation: PrivateObservation,
        legal_actions: tuple[GameAction, ...],
    ) -> GameAction:
        result = self.solver.strategy_for(observation, legal_actions).normalized()
        if not result.action_probs:
            raise ValueError("solver policy received no legal actions")

        threshold = self._rng.random()
        cumulative = 0.0
        last_action = None
        for action, probability in result.action_probs.items():
            cumulative += probability
            last_action = action
            if threshold <= cumulative:
                return action

        if last_action is None:
            raise ValueError("solver policy could not sample an action")
        return last_action

