from __future__ import annotations

import math

from ..abstractions import ActionAbstractor
from ..opponent_state import OpponentStateInferer
from ..response_model import StyleResponseModel
from ..types import ChallengeAction, GameAction, PlayAction, PrivateObservation
from .base import Solver, StrategyResult


class OpponentAwareAdaptiveSolver(Solver):
    def __init__(
        self,
        base_solver: Solver,
        abstractor: ActionAbstractor | None = None,
        state_inferer: OpponentStateInferer | None = None,
        response_model: StyleResponseModel | None = None,
    ) -> None:
        self.base_solver = base_solver
        self.abstractor = abstractor or ActionAbstractor()
        self.state_inferer = state_inferer or OpponentStateInferer(response_model=response_model)

    def strategy_for(
        self,
        observation: PrivateObservation,
        legal_actions: tuple[GameAction, ...],
    ) -> StrategyResult:
        base = self.base_solver.strategy_for(observation, legal_actions).normalized()
        if not legal_actions or not base.action_probs:
            return base

        runtime_states = self.state_inferer.build(observation)
        hero_state = runtime_states.get(observation.hero_seat)
        logits: dict[GameAction, float] = {}

        for action in legal_actions:
            base_prob = max(1e-6, base.action_probs.get(action, 1e-6))
            logit = math.log(base_prob)
            logit += self._action_adjustment(
                observation=observation,
                action=action,
                runtime_states=runtime_states,
                hero_state=hero_state,
            )
            logits[action] = logit

        max_logit = max(logits.values())
        exp_probs = {action: math.exp(value - max_logit) for action, value in logits.items()}
        total = sum(exp_probs.values())
        action_probs = {action: value / total for action, value in exp_probs.items()}

        return StrategyResult(
            action_probs=action_probs,
            action_values=base.action_values,
            info={
                **base.info,
                "solver": "opponent_aware_adaptive",
            },
        )

    def _action_adjustment(
        self,
        observation: PrivateObservation,
        action: GameAction,
        runtime_states,
        hero_state,
    ) -> float:
        hero_pressure = hero_state.pressure_score if hero_state is not None else 0.0

        if isinstance(action, ChallengeAction):
            target_state = runtime_states.get(action.challenged_seat)
            if target_state is None:
                return 0.0
            adjusted_challenge_tendency = min(
                1.0,
                max(0.0, target_state.challenge_tendency + target_state.predicted_post_shot_challenge_delta),
            )
            adjustment = 0.0
            adjustment += 0.60 * target_state.desperation_score
            adjustment -= 0.55 * target_state.caution_score
            adjustment += 0.35 * adjusted_challenge_tendency
            adjustment += 0.25 * max(0.0, target_state.predicted_post_shot_bluff_delta)
            adjustment -= 0.20 * max(0.0, -target_state.predicted_post_shot_challenge_delta)
            adjustment -= 0.40 * hero_pressure
            return adjustment

        abstract_action = self.abstractor.abstract(action, observation)
        next_state = runtime_states.get(self._next_responder_seat(observation))
        adjustment = 0.0
        bluff_ratio = 0.0
        honest_ratio = 0.0
        if abstract_action.count > 0:
            bluff_ratio = sum(1 for item in abstract_action.composition if item == "non_target") / abstract_action.count
            honest_ratio = sum(1 for item in abstract_action.composition if item in {"target", "wild", "ghost"}) / abstract_action.count

        if next_state is not None:
            adjusted_next_challenge = min(
                1.0,
                max(0.0, next_state.challenge_tendency + next_state.predicted_post_shot_challenge_delta),
            )
            adjustment += 0.65 * next_state.caution_score * bluff_ratio
            adjustment -= 0.70 * adjusted_next_challenge * bluff_ratio
            adjustment += 0.25 * next_state.desperation_score * honest_ratio
            adjustment += 0.30 * max(0.0, -next_state.predicted_post_shot_challenge_delta) * bluff_ratio
            adjustment -= 0.25 * max(0.0, next_state.predicted_post_shot_challenge_delta) * bluff_ratio
            if next_state.predicted_pressure_response_label == "tightens_up":
                adjustment += 0.15 * bluff_ratio
            elif next_state.predicted_pressure_response_label == "tilts_aggressive":
                adjustment -= 0.10 * bluff_ratio
                adjustment += 0.08 * honest_ratio

        adjustment -= 0.70 * hero_pressure * bluff_ratio
        adjustment += 0.45 * hero_pressure * honest_ratio
        if len(observation.hero_hand) - abstract_action.count == 0:
            adjustment += 0.50
        return adjustment

    def _next_responder_seat(self, observation: PrivateObservation) -> int | None:
        ordered = list(observation.players)
        seats = [player.seat for player in ordered]
        if observation.hero_seat not in seats:
            return None
        start_index = seats.index(observation.hero_seat)
        for offset in range(1, len(ordered) + 1):
            candidate = ordered[(start_index + offset) % len(ordered)]
            if candidate.alive and not candidate.escaped and not candidate.pending_escape:
                return candidate.seat
        return None
