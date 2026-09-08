from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from math import log, sqrt

from .self_play import StateSnapshot
from .style import infer_public_style_cluster, pressure_response_label_from_deltas
from .types import CardKind, ChallengeAction, PlayerIdentity, PlayAction, PlayerModelSummary


@dataclass
class _SeatBehaviorAccumulator:
    action_count: int = 0
    play_count: int = 0
    challenge_count: int = 0
    bluff_play_count: int = 0
    honest_play_count: int = 0
    ghost_play_count: int = 0
    play_size_sum: int = 0
    play_size_counter: Counter[int] = field(default_factory=Counter)
    claim_rank_counter: Counter[str] = field(default_factory=Counter)
    post_shot_action_count: int = 0
    post_shot_play_count: int = 0
    post_shot_challenge_count: int = 0
    post_shot_bluff_play_count: int = 0


def build_player_model_summaries(
    identities: tuple[PlayerIdentity, ...],
    snapshots: tuple[StateSnapshot, ...],
) -> tuple[PlayerModelSummary, ...]:
    accumulators = {identity.seat: _SeatBehaviorAccumulator() for identity in identities}

    for snapshot in snapshots:
        seat = snapshot.acting_seat
        accumulator = accumulators[seat]
        action = snapshot.chosen_action
        accumulator.action_count += 1
        post_shot_flag = _post_shot_flag(snapshot, seat)
        if post_shot_flag:
            accumulator.post_shot_action_count += 1

        if isinstance(action, ChallengeAction):
            accumulator.challenge_count += 1
            if post_shot_flag:
                accumulator.post_shot_challenge_count += 1
            continue

        accumulator.play_count += 1
        if post_shot_flag:
            accumulator.post_shot_play_count += 1
        accumulator.play_size_sum += len(action.card_ids)
        accumulator.play_size_counter[len(action.card_ids)] += 1
        accumulator.claim_rank_counter[action.claim_rank.value] += 1

        cards = _cards_for_play(snapshot)
        if len(cards) == 1 and cards[0].kind == CardKind.GHOST:
            accumulator.ghost_play_count += 1
            continue

        lied = False
        for card in cards:
            if card.kind == CardKind.WILD:
                continue
            if card.kind == CardKind.GHOST:
                continue
            if card.printed_rank == action.claim_rank:
                continue
            lied = True
            break

        if lied:
            accumulator.bluff_play_count += 1
            if post_shot_flag:
                accumulator.post_shot_bluff_play_count += 1
        else:
            accumulator.honest_play_count += 1

    summaries = []
    for identity in identities:
        accumulator = accumulators[identity.seat]
        summary = _summarize_identity(identity, accumulator)
        summaries.append(summary)
    return tuple(summaries)


def _cards_for_play(snapshot: StateSnapshot):
    action = snapshot.chosen_action
    if not isinstance(action, PlayAction):
        return ()
    player = snapshot.state_before.player_by_seat(action.seat)
    by_id = {card.card_id: card for card in player.hand}
    return tuple(by_id[card_id] for card_id in action.card_ids if card_id in by_id)


def _summarize_identity(
    identity: PlayerIdentity,
    accumulator: _SeatBehaviorAccumulator,
) -> PlayerModelSummary:
    actions = max(1, accumulator.action_count)
    plays = max(1, accumulator.play_count)
    play_rate = accumulator.play_count / actions
    challenge_rate = accumulator.challenge_count / actions
    bluff_rate = accumulator.bluff_play_count / plays
    honest_rate = accumulator.honest_play_count / plays
    ghost_play_rate = accumulator.ghost_play_count / plays
    mean_play_count = accumulator.play_size_sum / plays if accumulator.play_count > 0 else 0.0
    variability_score = _variability_score(accumulator)
    aggression_score = max(
        0.0,
        min(
            1.0,
            0.45 * challenge_rate
            + 0.25 * bluff_rate
            + 0.20 * min(1.0, mean_play_count / 3.0)
            + 0.10 * variability_score,
        ),
    )
    profile_name, style = _infer_profile(
        bluff_rate=bluff_rate,
        honest_rate=honest_rate,
        challenge_rate=challenge_rate,
        mean_play_count=mean_play_count,
        variability_score=variability_score,
        aggression_score=aggression_score,
    )
    post_shot_plays = max(1, accumulator.post_shot_play_count)
    post_shot_bluff_rate = (
        accumulator.post_shot_bluff_play_count / post_shot_plays
        if accumulator.post_shot_play_count > 0
        else bluff_rate
    )
    post_shot_challenge_rate = (
        accumulator.post_shot_challenge_count / accumulator.post_shot_action_count
        if accumulator.post_shot_action_count > 0
        else challenge_rate
    )
    post_shot_bluff_delta = post_shot_bluff_rate - bluff_rate
    post_shot_challenge_delta = post_shot_challenge_rate - challenge_rate

    return PlayerModelSummary(
        seat=identity.seat,
        player_id=identity.player_id,
        display_name=identity.display_name,
        is_hero=identity.is_hero,
        observed_actions=accumulator.action_count,
        play_rate=play_rate,
        challenge_rate=challenge_rate,
        bluff_rate=bluff_rate,
        honest_rate=honest_rate,
        ghost_play_rate=ghost_play_rate,
        mean_play_count=mean_play_count,
        variability_score=variability_score,
        variability_label=_variability_label(variability_score),
        public_style_cluster=infer_public_style_cluster(
            challenge_rate=challenge_rate,
            mean_play_count=mean_play_count,
            variability_score=variability_score,
        ),
        aggression_score=aggression_score,
        post_shot_action_count=accumulator.post_shot_action_count,
        post_shot_bluff_rate=post_shot_bluff_rate,
        post_shot_challenge_rate=post_shot_challenge_rate,
        post_shot_bluff_delta=post_shot_bluff_delta,
        post_shot_challenge_delta=post_shot_challenge_delta,
        pressure_response_label=pressure_response_label_from_deltas(
            bluff_delta=post_shot_bluff_delta,
            challenge_delta=post_shot_challenge_delta,
            sample_count=accumulator.post_shot_action_count,
        ),
        inferred_profile_name=profile_name,
        inferred_style=style,
    )


def _variability_score(accumulator: _SeatBehaviorAccumulator) -> float:
    action_mass = []
    if accumulator.challenge_count > 0:
        action_mass.append(accumulator.challenge_count)
    for value in accumulator.play_size_counter.values():
        action_mass.append(value)
    for value in accumulator.claim_rank_counter.values():
        action_mass.append(value)

    total = sum(action_mass)
    if total <= 1:
        return 0.0

    probabilities = [value / total for value in action_mass if value > 0]
    entropy = -sum(prob * log(prob) for prob in probabilities)
    max_entropy = log(len(probabilities)) if len(probabilities) > 1 else 1.0
    if max_entropy <= 0:
        return 0.0
    return max(0.0, min(1.0, entropy / max_entropy))


def _variability_label(score: float) -> str:
    if score >= 0.67:
        return "high"
    if score >= 0.34:
        return "medium"
    return "low"


def _infer_profile(
    *,
    bluff_rate: float,
    honest_rate: float,
    challenge_rate: float,
    mean_play_count: float,
    variability_score: float,
    aggression_score: float,
) -> tuple[str, str]:
    targets = [
        ("no_brain_random", "random", (0.33, 0.33, 0.25, 2.0 / 3.0, 0.95, 0.45)),
        ("traditional_cautious", "traditional", (0.10, 0.80, 0.10, 1.4 / 3.0, 0.20, 0.20)),
        ("chaotic_bluffer", "volatile", (0.55, 0.30, 0.25, 1.9 / 3.0, 0.90, 0.65)),
        ("pressure_challenger", "aggressive", (0.20, 0.45, 0.45, 1.8 / 3.0, 0.55, 0.85)),
        ("greedy_local_ev", "greedy", (0.20, 0.65, 0.10, 2.2 / 3.0, 0.25, 0.55)),
    ]
    observed = (
        bluff_rate,
        honest_rate,
        challenge_rate,
        min(1.0, mean_play_count / 3.0),
        variability_score,
        aggression_score,
    )
    best_name = targets[0][0]
    best_style = targets[0][1]
    best_distance = float("inf")
    for name, style, target in targets:
        distance = sqrt(sum((a - b) ** 2 for a, b in zip(observed, target)))
        if distance < best_distance:
            best_distance = distance
            best_name = name
            best_style = style
    return best_name, best_style


def _post_shot_flag(snapshot: StateSnapshot, seat: int) -> bool:
    age = 0
    for event in reversed(snapshot.state_before.public_history):
        if event.event_type == "shot" and event.seat == seat:
            return age <= 4
        age += 1
    return False
