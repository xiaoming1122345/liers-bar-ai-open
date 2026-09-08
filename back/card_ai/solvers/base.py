from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ..types import GameAction, PrivateObservation


@dataclass(frozen=True)
class StrategyResult:
    action_probs: dict[GameAction, float]
    action_values: dict[str, float] = field(default_factory=dict)
    info: dict[str, float | int | str] = field(default_factory=dict)

    def normalized(self) -> "StrategyResult":
        total = sum(max(0.0, value) for value in self.action_probs.values())
        if total <= 0:
            count = len(self.action_probs)
            if count == 0:
                return self
            return StrategyResult(
                action_probs={action: 1.0 / count for action in self.action_probs},
                action_values=self.action_values,
                info=self.info,
            )

        return StrategyResult(
            action_probs={
                action: max(0.0, value) / total
                for action, value in self.action_probs.items()
            },
            action_values=self.action_values,
            info=self.info,
        )


class Solver(Protocol):
    def strategy_for(
        self,
        observation: PrivateObservation,
        legal_actions: tuple[GameAction, ...],
    ) -> StrategyResult:
        ...

