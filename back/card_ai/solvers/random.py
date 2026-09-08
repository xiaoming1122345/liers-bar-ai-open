from __future__ import annotations

from .base import StrategyResult
from ..types import GameAction, PrivateObservation


class UniformRandomSolver:
    def strategy_for(
        self,
        observation: PrivateObservation,
        legal_actions: tuple[GameAction, ...],
    ) -> StrategyResult:
        if not legal_actions:
            return StrategyResult(action_probs={})
        probability = 1.0 / len(legal_actions)
        return StrategyResult(
            action_probs={action: probability for action in legal_actions},
            info={"solver": "uniform_random"},
        )

