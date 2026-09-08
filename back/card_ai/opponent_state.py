from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import log

from .response_model import StyleResponseModel
from .style import infer_public_style_cluster
from .types import PrivateObservation


@dataclass(frozen=True)
class OpponentRuntimeState:
    seat: int
    shots_taken: int
    hand_count: int
    recent_shot_age: int | None
    recent_shot_flag: bool
    recent_punish_flag: bool
    recent_challenge_count: int
    recent_play_count: int
    recent_claim_count: int
    mean_recent_play_count: float
    recent_variability_score: float
    challenge_tendency: float
    public_style_cluster: str
    predicted_post_shot_bluff_delta: float
    predicted_post_shot_challenge_delta: float
    predicted_pressure_response_label: str
    pressure_score: float
    caution_score: float
    desperation_score: float


class OpponentStateInferer:
    def __init__(
        self,
        response_model: StyleResponseModel | None = None,
    ) -> None:
        self.response_model = response_model or StyleResponseModel.default()

    def build(self, observation: PrivateObservation) -> dict[int, OpponentRuntimeState]:
        states: dict[int, OpponentRuntimeState] = {}
        for player in observation.players:
            recent_shot_age = self._recent_shot_age(observation, player.seat)
            recent_shot_flag = recent_shot_age is not None and recent_shot_age <= 4
            recent_punish_flag = self._recent_punish_flag(observation, player.seat)
            recent_challenge_count = self._recent_event_count(observation, player.seat, "challenge", 8)
            recent_play_count = self._recent_event_count(observation, player.seat, "play", 8)
            recent_claim_count = self._recent_claim_count(observation, player.seat, 8)
            recent_play_sizes = self._recent_play_sizes(observation, player.seat, 8)
            mean_recent_play_count = (
                sum(recent_play_sizes) / len(recent_play_sizes)
                if recent_play_sizes
                else 1.0
            )
            recent_variability_score = self._recent_variability_score(observation, player.seat, 8)
            challenge_tendency = recent_challenge_count / max(1, recent_challenge_count + recent_play_count)
            public_style_cluster = infer_public_style_cluster(
                challenge_rate=challenge_tendency,
                mean_play_count=mean_recent_play_count,
                variability_score=recent_variability_score,
            )
            predicted_bluff_delta, predicted_challenge_delta, predicted_response_label = (
                self.response_model.predict(public_style_cluster, recent_shot_age)
            )
            response_shift = predicted_bluff_delta + predicted_challenge_delta
            pressure_score = min(
                1.0,
                0.45 * min(1.0, player.shots_taken / 4.0)
                + 0.30 * (1.0 if recent_shot_flag else 0.0)
                + 0.25 * (1.0 if player.hand_count <= 2 else 0.0),
            )
            caution_score = min(
                1.0,
                (
                    0.55 * pressure_score
                    + 0.20 * (1.0 if recent_punish_flag else 0.0)
                    + 0.25 * max(0.0, 1.0 - challenge_tendency)
                    + 0.50 * max(0.0, -response_shift)
                ),
            )
            desperation_score = min(
                1.0,
                (
                    0.45 * (1.0 if player.hand_count <= 2 else 0.0)
                    + 0.30 * min(1.0, player.shots_taken / 4.0)
                    + 0.25 * challenge_tendency
                    + 0.50 * max(0.0, response_shift)
                ),
            )

            states[player.seat] = OpponentRuntimeState(
                seat=player.seat,
                shots_taken=player.shots_taken,
                hand_count=player.hand_count,
                recent_shot_age=recent_shot_age,
                recent_shot_flag=recent_shot_flag,
                recent_punish_flag=recent_punish_flag,
                recent_challenge_count=recent_challenge_count,
                recent_play_count=recent_play_count,
                recent_claim_count=recent_claim_count,
                mean_recent_play_count=mean_recent_play_count,
                recent_variability_score=recent_variability_score,
                challenge_tendency=challenge_tendency,
                public_style_cluster=public_style_cluster,
                predicted_post_shot_bluff_delta=predicted_bluff_delta,
                predicted_post_shot_challenge_delta=predicted_challenge_delta,
                predicted_pressure_response_label=predicted_response_label,
                pressure_score=pressure_score,
                caution_score=caution_score,
                desperation_score=desperation_score,
            )
        return states

    def _recent_shot_age(self, observation: PrivateObservation, seat: int) -> int | None:
        age = 0
        for event in reversed(observation.public_history):
            if event.event_type == "shot" and event.seat == seat:
                return age
            age += 1
        return None

    def _recent_punish_flag(self, observation: PrivateObservation, seat: int) -> bool:
        for event in reversed(observation.public_history[-8:]):
            if event.event_type == "shot" and event.seat == seat:
                return True
            if event.event_type == "challenge":
                challenged_seat = event.detail.get("challenged_seat")
                outcome = event.detail.get("outcome")
                if outcome == "lie" and challenged_seat == seat:
                    return True
                if outcome == "honest" and event.seat == seat:
                    return True
        return False

    def _recent_event_count(
        self,
        observation: PrivateObservation,
        seat: int,
        event_type: str,
        window: int,
    ) -> int:
        return sum(
            1
            for event in observation.public_history[-window:]
            if event.event_type == event_type and event.seat == seat
        )

    def _recent_claim_count(
        self,
        observation: PrivateObservation,
        seat: int,
        window: int,
    ) -> int:
        return sum(
            1
            for event in observation.public_history[-window:]
            if event.event_type == "play" and event.seat == seat
        )

    def _recent_play_sizes(
        self,
        observation: PrivateObservation,
        seat: int,
        window: int,
    ) -> tuple[int, ...]:
        sizes = []
        for event in observation.public_history[-window:]:
            if event.event_type != "play" or event.seat != seat:
                continue
            sizes.append(int(event.detail.get("count", 1)))
        return tuple(sizes)

    def _recent_variability_score(
        self,
        observation: PrivateObservation,
        seat: int,
        window: int,
    ) -> float:
        action_mass = []
        challenge_count = 0
        play_size_counter: Counter[int] = Counter()
        claim_counter: Counter[str] = Counter()
        for event in observation.public_history[-window:]:
            if event.seat != seat:
                continue
            if event.event_type == "challenge":
                challenge_count += 1
                continue
            if event.event_type == "play":
                play_size_counter[int(event.detail.get("count", 1))] += 1
                claim_counter[str(event.detail.get("claim_rank", "?"))] += 1

        if challenge_count > 0:
            action_mass.append(challenge_count)
        action_mass.extend(play_size_counter.values())
        action_mass.extend(claim_counter.values())
        total = sum(action_mass)
        if total <= 1:
            return 0.0
        probabilities = [value / total for value in action_mass if value > 0]
        entropy = -sum(prob * log(prob) for prob in probabilities)
        max_entropy = log(len(probabilities)) if len(probabilities) > 1 else 1.0
        if max_entropy <= 0.0:
            return 0.0
        return max(0.0, min(1.0, entropy / max_entropy))
