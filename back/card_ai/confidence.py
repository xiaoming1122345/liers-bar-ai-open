from __future__ import annotations

import math

from .types import ConfidenceReport, PublicBeliefState


def _normalized_entropy(probabilities: list[float]) -> float:
    positive = [value for value in probabilities if value > 0]
    if len(positive) <= 1:
        return 0.0

    entropy = -sum(value * math.log(value) for value in positive)
    max_entropy = math.log(len(probabilities))
    return entropy / max_entropy if max_entropy > 0 else 0.0


def _distribution_confidence(probabilities: list[float]) -> float:
    if not probabilities:
        return 0.0
    return 1.0 - _normalized_entropy(probabilities)


def _action_margin_confidence(action_values: dict[str, float] | None) -> float:
    if not action_values:
        return 0.0

    ordered = sorted(action_values.values(), reverse=True)
    if len(ordered) == 1:
        return 1.0

    margin = max(0.0, ordered[0] - ordered[1])
    return max(0.0, min(1.0, margin))


def build_confidence_report(
    belief: PublicBeliefState,
    action_values: dict[str, float] | None = None,
) -> ConfidenceReport:
    ghost_holder_probs = [seat.ghost_holder_prob for seat in belief.seat_beliefs]
    ghost_holder_probs.append(belief.ghost_out_of_play_prob)
    ghost_rank_probs = list(belief.ghost_rank_probs.values())

    ghost_holder_confidence = _distribution_confidence(ghost_holder_probs)
    ghost_rank_confidence = _distribution_confidence(ghost_rank_probs)

    uncertainty_values = [seat.target_card_uncertainty for seat in belief.seat_beliefs]
    if uncertainty_values:
        belief_sharpness = 1.0 - sum(uncertainty_values) / len(uncertainty_values)
    else:
        belief_sharpness = 0.0
    belief_sharpness = max(0.0, min(1.0, belief_sharpness))

    evidence_scale = max(20.0, float(belief.particle_count or 20))
    evidence_coverage = max(0.0, min(1.0, belief.evidence_count / evidence_scale + belief.effective_sample_size / max(1.0, evidence_scale * 4.0)))
    margin_conf = _action_margin_confidence(action_values)

    modeling_degree = (
        0.30 * ghost_holder_confidence
        + 0.20 * ghost_rank_confidence
        + 0.25 * belief_sharpness
        + 0.15 * evidence_coverage
        + 0.10 * margin_conf
    )

    return ConfidenceReport(
        ghost_holder_confidence=ghost_holder_confidence,
        ghost_rank_confidence=ghost_rank_confidence,
        belief_sharpness=belief_sharpness,
        evidence_coverage=evidence_coverage,
        action_margin_confidence=margin_conf,
        modeling_degree=max(0.0, min(1.0, modeling_degree)),
    )
