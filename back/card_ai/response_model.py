from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from random import Random

from .engine import SurvivalGameEngine
from .identities import build_default_identities
from .modeling import build_player_model_summaries
from .opponents import OpponentProfile, build_policy_from_profile
from .self_play import SelfPlayRunner
from .style import pressure_response_label_from_deltas


@dataclass(frozen=True)
class StyleResponsePrior:
    public_style_cluster: str
    seat_model_count: int
    observed_actions: int
    post_shot_case_count: int
    mean_play_rate: float
    mean_challenge_rate: float
    mean_bluff_rate: float
    mean_mean_play_count: float
    mean_variability_score: float
    mean_aggression_score: float
    mean_post_shot_bluff_delta: float
    mean_post_shot_challenge_delta: float
    pressure_response_label: str
    source_profile_counts: dict[str, int]


@dataclass(frozen=True)
class StyleResponseTrainingResult:
    games: int
    priors: tuple[StyleResponsePrior, ...]
    profile_counts: dict[str, int]

    def to_dict(self) -> dict:
        return {
            "games": self.games,
            "priors": [asdict(prior) for prior in self.priors],
            "profile_counts": self.profile_counts,
        }


@dataclass
class _StyleAccumulator:
    seat_model_count: int = 0
    observed_actions: int = 0
    post_shot_case_count: int = 0
    play_rate_sum: float = 0.0
    challenge_rate_sum: float = 0.0
    bluff_rate_sum: float = 0.0
    mean_play_count_sum: float = 0.0
    variability_score_sum: float = 0.0
    aggression_score_sum: float = 0.0
    post_shot_bluff_delta_sum: float = 0.0
    post_shot_challenge_delta_sum: float = 0.0
    pressure_response_counts: Counter[str] = field(default_factory=Counter)
    source_profile_counts: Counter[str] = field(default_factory=Counter)


class StyleResponseModel:
    def __init__(self, priors: tuple[StyleResponsePrior, ...] | None = None) -> None:
        by_style = {
            prior.public_style_cluster: prior
            for prior in self.default_priors()
        }
        if priors is not None:
            by_style.update({prior.public_style_cluster: prior for prior in priors})
        self._by_style = by_style
        self.priors = tuple(self._by_style[style] for style in sorted(self._by_style))

    @classmethod
    def default(cls) -> "StyleResponseModel":
        return cls(priors=cls.default_priors())

    @staticmethod
    def default_priors() -> tuple[StyleResponsePrior, ...]:
        return (
            StyleResponsePrior(
                public_style_cluster="aggressive",
                seat_model_count=0,
                observed_actions=0,
                post_shot_case_count=0,
                mean_play_rate=0.0,
                mean_challenge_rate=0.0,
                mean_bluff_rate=0.0,
                mean_mean_play_count=0.0,
                mean_variability_score=0.0,
                mean_aggression_score=0.0,
                mean_post_shot_bluff_delta=0.03,
                mean_post_shot_challenge_delta=0.10,
                pressure_response_label="tilts_aggressive",
                source_profile_counts={},
            ),
            StyleResponsePrior(
                public_style_cluster="volatile",
                seat_model_count=0,
                observed_actions=0,
                post_shot_case_count=0,
                mean_play_rate=0.0,
                mean_challenge_rate=0.0,
                mean_bluff_rate=0.0,
                mean_mean_play_count=0.0,
                mean_variability_score=0.0,
                mean_aggression_score=0.0,
                mean_post_shot_bluff_delta=0.08,
                mean_post_shot_challenge_delta=0.03,
                pressure_response_label="tilts_aggressive",
                source_profile_counts={},
            ),
            StyleResponsePrior(
                public_style_cluster="tempo",
                seat_model_count=0,
                observed_actions=0,
                post_shot_case_count=0,
                mean_play_rate=0.0,
                mean_challenge_rate=0.0,
                mean_bluff_rate=0.0,
                mean_mean_play_count=0.0,
                mean_variability_score=0.0,
                mean_aggression_score=0.0,
                mean_post_shot_bluff_delta=-0.02,
                mean_post_shot_challenge_delta=-0.03,
                pressure_response_label="stable",
                source_profile_counts={},
            ),
            StyleResponsePrior(
                public_style_cluster="reserved",
                seat_model_count=0,
                observed_actions=0,
                post_shot_case_count=0,
                mean_play_rate=0.0,
                mean_challenge_rate=0.0,
                mean_bluff_rate=0.0,
                mean_mean_play_count=0.0,
                mean_variability_score=0.0,
                mean_aggression_score=0.0,
                mean_post_shot_bluff_delta=-0.08,
                mean_post_shot_challenge_delta=-0.08,
                pressure_response_label="tightens_up",
                source_profile_counts={},
            ),
            StyleResponsePrior(
                public_style_cluster="randomish",
                seat_model_count=0,
                observed_actions=0,
                post_shot_case_count=0,
                mean_play_rate=0.0,
                mean_challenge_rate=0.0,
                mean_bluff_rate=0.0,
                mean_mean_play_count=0.0,
                mean_variability_score=0.0,
                mean_aggression_score=0.0,
                mean_post_shot_bluff_delta=0.04,
                mean_post_shot_challenge_delta=0.01,
                pressure_response_label="stable",
                source_profile_counts={},
            ),
            StyleResponsePrior(
                public_style_cluster="balanced",
                seat_model_count=0,
                observed_actions=0,
                post_shot_case_count=0,
                mean_play_rate=0.0,
                mean_challenge_rate=0.0,
                mean_bluff_rate=0.0,
                mean_mean_play_count=0.0,
                mean_variability_score=0.0,
                mean_aggression_score=0.0,
                mean_post_shot_bluff_delta=0.0,
                mean_post_shot_challenge_delta=0.0,
                pressure_response_label="stable",
                source_profile_counts={},
            ),
        )

    def prior_for(self, public_style_cluster: str) -> StyleResponsePrior:
        return self._by_style.get(public_style_cluster, self._by_style["balanced"])

    def predict(
        self,
        public_style_cluster: str,
        recent_shot_age: int | None,
    ) -> tuple[float, float, str]:
        prior = self.prior_for(public_style_cluster)
        if recent_shot_age is None:
            return 0.0, 0.0, "stable"
        decay = max(0.20, 1.0 - 0.18 * recent_shot_age)
        return (
            prior.mean_post_shot_bluff_delta * decay,
            prior.mean_post_shot_challenge_delta * decay,
            prior.pressure_response_label,
        )

    def to_dict(self) -> dict:
        return {"priors": [asdict(prior) for prior in self.priors]}

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return output

    @classmethod
    def load(cls, path: str | Path) -> "StyleResponseModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        priors = tuple(
            StyleResponsePrior(
                public_style_cluster=str(item["public_style_cluster"]),
                seat_model_count=int(item["seat_model_count"]),
                observed_actions=int(item["observed_actions"]),
                post_shot_case_count=int(item["post_shot_case_count"]),
                mean_play_rate=float(item["mean_play_rate"]),
                mean_challenge_rate=float(item["mean_challenge_rate"]),
                mean_bluff_rate=float(item["mean_bluff_rate"]),
                mean_mean_play_count=float(item["mean_mean_play_count"]),
                mean_variability_score=float(item["mean_variability_score"]),
                mean_aggression_score=float(item["mean_aggression_score"]),
                mean_post_shot_bluff_delta=float(item["mean_post_shot_bluff_delta"]),
                mean_post_shot_challenge_delta=float(item["mean_post_shot_challenge_delta"]),
                pressure_response_label=str(item["pressure_response_label"]),
                source_profile_counts={
                    str(name): int(count)
                    for name, count in item.get("source_profile_counts", {}).items()
                },
            )
            for item in payload.get("priors", [])
        )
        return cls(priors=priors)


class StyleResponseTrainer:
    def __init__(
        self,
        engine: SurvivalGameEngine | None = None,
    ) -> None:
        self.engine = engine or SurvivalGameEngine()
        self.self_play = SelfPlayRunner(self.engine)

    def train_pool_priors(
        self,
        *,
        opponent_profiles: tuple[OpponentProfile, ...],
        game_count: int,
        seed_start: int = 0,
        max_steps: int = 512,
    ) -> StyleResponseTrainingResult:
        identities = build_default_identities(
            player_count=self.engine.player_count,
            hero_seat=None,
        )
        rng = Random(seed_start)
        accumulators: dict[str, _StyleAccumulator] = {}
        profile_counts: Counter[str] = Counter()

        for game_index in range(game_count):
            policies = {}
            seat_profile_names: dict[int, str] = {}
            for seat in range(1, self.engine.player_count + 1):
                profile = opponent_profiles[rng.randrange(len(opponent_profiles))]
                policies[seat] = build_policy_from_profile(
                    profile=profile,
                    seed=seed_start + game_index * 151 + seat,
                )
                seat_profile_names[seat] = profile.name
                profile_counts[profile.name] += 1

            trace = self.self_play.run_game(
                policies=policies,
                seed=seed_start + game_index,
                max_steps=max_steps,
            )
            seat_models = build_player_model_summaries(identities=identities, snapshots=trace.snapshots)
            for model in seat_models:
                accumulator = accumulators.setdefault(model.public_style_cluster, _StyleAccumulator())
                accumulator.seat_model_count += 1
                accumulator.observed_actions += model.observed_actions
                accumulator.play_rate_sum += model.play_rate
                accumulator.challenge_rate_sum += model.challenge_rate
                accumulator.bluff_rate_sum += model.bluff_rate
                accumulator.mean_play_count_sum += model.mean_play_count
                accumulator.variability_score_sum += model.variability_score
                accumulator.aggression_score_sum += model.aggression_score
                profile_name = seat_profile_names.get(model.seat, "unknown")
                accumulator.source_profile_counts[profile_name] += 1
                if model.post_shot_action_count > 0:
                    accumulator.post_shot_case_count += 1
                    accumulator.post_shot_bluff_delta_sum += model.post_shot_bluff_delta
                    accumulator.post_shot_challenge_delta_sum += model.post_shot_challenge_delta
                    accumulator.pressure_response_counts[model.pressure_response_label] += 1

        priors = tuple(
            self._build_prior(style, accumulators[style])
            for style in sorted(accumulators)
        )
        return StyleResponseTrainingResult(
            games=game_count,
            priors=priors,
            profile_counts=dict(sorted(profile_counts.items())),
        )

    def _build_prior(
        self,
        public_style_cluster: str,
        accumulator: _StyleAccumulator,
    ) -> StyleResponsePrior:
        seat_models = max(1, accumulator.seat_model_count)
        post_shot_cases = accumulator.post_shot_case_count
        bluff_delta = (
            accumulator.post_shot_bluff_delta_sum / post_shot_cases
            if post_shot_cases > 0
            else 0.0
        )
        challenge_delta = (
            accumulator.post_shot_challenge_delta_sum / post_shot_cases
            if post_shot_cases > 0
            else 0.0
        )
        if accumulator.pressure_response_counts:
            response_label = accumulator.pressure_response_counts.most_common(1)[0][0]
        else:
            response_label = pressure_response_label_from_deltas(
                bluff_delta=bluff_delta,
                challenge_delta=challenge_delta,
                sample_count=post_shot_cases,
            )
        return StyleResponsePrior(
            public_style_cluster=public_style_cluster,
            seat_model_count=accumulator.seat_model_count,
            observed_actions=accumulator.observed_actions,
            post_shot_case_count=post_shot_cases,
            mean_play_rate=accumulator.play_rate_sum / seat_models,
            mean_challenge_rate=accumulator.challenge_rate_sum / seat_models,
            mean_bluff_rate=accumulator.bluff_rate_sum / seat_models,
            mean_mean_play_count=accumulator.mean_play_count_sum / seat_models,
            mean_variability_score=accumulator.variability_score_sum / seat_models,
            mean_aggression_score=accumulator.aggression_score_sum / seat_models,
            mean_post_shot_bluff_delta=bluff_delta,
            mean_post_shot_challenge_delta=challenge_delta,
            pressure_response_label=response_label,
            source_profile_counts=dict(sorted(accumulator.source_profile_counts.items())),
        )
