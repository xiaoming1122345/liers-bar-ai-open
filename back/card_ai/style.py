from __future__ import annotations


def infer_public_style_cluster(
    *,
    challenge_rate: float,
    mean_play_count: float,
    variability_score: float,
) -> str:
    if challenge_rate >= 0.38:
        return "aggressive"
    if variability_score >= 0.82 and 0.15 <= challenge_rate <= 0.35:
        return "volatile"
    if mean_play_count >= 2.15 and challenge_rate <= 0.22:
        return "tempo"
    if challenge_rate <= 0.12 and mean_play_count <= 1.45 and variability_score <= 0.45:
        return "reserved"
    if variability_score >= 0.88:
        return "randomish"
    return "balanced"


def pressure_response_label_from_deltas(
    *,
    bluff_delta: float,
    challenge_delta: float,
    sample_count: int,
) -> str:
    if sample_count <= 0:
        return "unknown"
    delta = bluff_delta + challenge_delta
    if delta >= 0.20:
        return "tilts_aggressive"
    if delta <= -0.20:
        return "tightens_up"
    return "stable"
