from __future__ import annotations

"""
POMDP Particle Filter for imperfect information survival card game.

Tracks the posterior belief over hidden cards (opponent hands and ghost card location),
and computes the posterior probability of the previous play being honest, bluff, or ghost.
"""

from collections import Counter
from dataclasses import dataclass
from math import sqrt
from random import Random
from typing import Any

from .types import Card, CardKind, PrivateObservation, Rank


@dataclass(frozen=True)
class HandCardEstimate:
    seat: int
    target_count_mean: float
    target_count_stdev: float
    has_ghost_prob: float
    wild_count_mean: float
    hand_count: int


@dataclass(frozen=True)
class LastPlayBelief:
    seat: int
    claim_rank: Rank | None
    claim_count: int
    honest_prob: float
    bluff_prob: float
    ghost_trap_prob: float
    reason: str


@dataclass(frozen=True)
class POMDPInferenceResult:
    claim_rank: Rank | None
    ghost_rank_probs: dict[str, float]
    ghost_revealed: bool
    seat_estimates: dict[int, HandCardEstimate]
    last_play: LastPlayBelief | None
    effective_particles: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_rank": self.claim_rank.value if self.claim_rank else None,
            "ghost_rank_probs": self.ghost_rank_probs,
            "ghost_revealed": self.ghost_revealed,
            "seat_estimates": {
                str(seat): {
                    "seat": est.seat,
                    "target_count_mean": round(est.target_count_mean, 2),
                    "target_count_stdev": round(est.target_count_stdev, 2),
                    "has_ghost_prob": round(est.has_ghost_prob, 3),
                    "wild_count_mean": round(est.wild_count_mean, 2),
                    "hand_count": est.hand_count,
                }
                for seat, est in self.seat_estimates.items()
            },
            "last_play": {
                "seat": self.last_play.seat,
                "claim_rank": self.last_play.claim_rank.value if self.last_play.claim_rank else None,
                "claim_count": self.last_play.claim_count,
                "honest_prob": round(self.last_play.honest_prob, 3),
                "bluff_prob": round(self.last_play.bluff_prob, 3),
                "ghost_trap_prob": round(self.last_play.ghost_trap_prob, 3),
                "reason": self.last_play.reason,
            } if self.last_play else None,
            "effective_particles": round(self.effective_particles, 1),
        }


@dataclass
class Particle:
    ghost_rank: Rank
    # Mapping from seat to list of Card objects
    seat_hands: dict[int, list[Card]]
    weight: float = 1.0


class POMDPParticleFilter:
    """
    Maintains and updates a posterior belief over hidden states using particle filtering.
    """

    def __init__(self, num_particles: int = 512, seed: int | None = 42) -> None:
        self.num_particles = num_particles
        self._rng = Random(seed)

    def infer(self, observation: PrivateObservation) -> POMDPInferenceResult:
        revealed_cards, revealed_ghost_rank = self._extract_revealed_cards(observation)
        claim_rank = observation.round_claim_rank or Rank.A

        # In this game, each round has a declared rank, and exactly that rank contains the ghost card!
        if observation.round_claim_rank is not None:
            possible_ghost_ranks = [observation.round_claim_rank]
        elif revealed_ghost_rank is not None:
            possible_ghost_ranks = [revealed_ghost_rank]
        else:
            possible_ghost_ranks = self._possible_ghost_ranks(observation, revealed_cards)
            if not possible_ghost_ranks:
                possible_ghost_ranks = list(Rank)

        particles = self._sample_particles(
            observation=observation,
            revealed_cards=revealed_cards,
            possible_ghost_ranks=possible_ghost_ranks,
        )


        if not particles:
            return self._fallback_inference(observation, claim_rank)

        # Likelihood weighting based on observation history
        total_weight = 0.0
        total_weight_sq = 0.0

        for p in particles:
            weight = self._compute_likelihood(observation, p)
            p.weight = weight
            total_weight += weight
            total_weight_sq += weight * weight

        if total_weight <= 1e-12:
            return self._fallback_inference(observation, claim_rank)

        # Normalize weights
        for p in particles:
            p.weight /= total_weight

        ess = 1.0 / total_weight_sq if total_weight_sq > 0 else 0.0

        # Compute posterior statistics
        ghost_rank_probs = {r.value: 0.0 for r in Rank}
        seat_targets = {p.seat: 0.0 for p in observation.players}
        seat_targets_sq = {p.seat: 0.0 for p in observation.players}
        seat_ghost_prob = {p.seat: 0.0 for p in observation.players}
        seat_wilds = {p.seat: 0.0 for p in observation.players}

        for p in particles:
            ghost_rank_probs[p.ghost_rank.value] += p.weight
            for player in observation.players:
                hand = p.seat_hands.get(player.seat, [])
                targets = sum(
                    1 for c in hand if c.printed_rank == claim_rank or c.kind == CardKind.WILD
                )
                has_ghost = 1.0 if any(c.kind == CardKind.GHOST for c in hand) else 0.0
                wilds = sum(1 for c in hand if c.kind == CardKind.WILD)

                seat_targets[player.seat] += p.weight * targets
                seat_targets_sq[player.seat] += p.weight * (targets ** 2)
                seat_ghost_prob[player.seat] += p.weight * has_ghost
                seat_wilds[player.seat] += p.weight * wilds

        seat_estimates = {}
        for player in observation.players:
            mean_target = seat_targets[player.seat]
            mean_target_sq = seat_targets_sq[player.seat]
            var = max(0.0, mean_target_sq - (mean_target ** 2))
            stdev = sqrt(var)

            seat_estimates[player.seat] = HandCardEstimate(
                seat=player.seat,
                target_count_mean=mean_target,
                target_count_stdev=stdev,
                has_ghost_prob=seat_ghost_prob[player.seat],
                wild_count_mean=seat_wilds[player.seat],
                hand_count=player.hand_count,
            )

        # Infer last play if available
        last_play_belief = self._infer_last_play(observation, particles, claim_rank)

        return POMDPInferenceResult(
            claim_rank=observation.round_claim_rank,
            ghost_rank_probs=ghost_rank_probs,
            ghost_revealed=revealed_ghost_rank is not None,
            seat_estimates=seat_estimates,
            last_play=last_play_belief,
            effective_particles=ess,
        )

    def _sample_particles(
        self,
        observation: PrivateObservation,
        revealed_cards: list[Card],
        possible_ghost_ranks: list[Rank],
    ) -> list[Particle]:
        particles: list[Particle] = []
        known_cards = list(observation.hero_hand) + revealed_cards

        for _ in range(self.num_particles):
            ghost_rank = self._rng.choice(possible_ghost_ranks)
            pool = self._build_card_pool(ghost_rank)

            try:
                pool = self._remove_known_cards(pool, known_cards)
            except ValueError:
                continue

            self._rng.shuffle(pool)

            seat_hands: dict[int, list[Card]] = {
                observation.hero_seat: list(observation.hero_hand)
            }
            cursor = 0
            possible = True

            for player in observation.players:
                if player.seat == observation.hero_seat:
                    continue
                count = player.hand_count
                if cursor + count > len(pool):
                    possible = False
                    break
                seat_hands[player.seat] = pool[cursor : cursor + count]
                cursor += count

            if possible:
                particles.append(Particle(ghost_rank=ghost_rank, seat_hands=seat_hands))

        return particles

    def _build_card_pool(self, ghost_rank: Rank) -> list[Card]:
        cards: list[Card] = []
        for rank in Rank:
            count = 5 if rank == ghost_rank else 6
            for i in range(count):
                cards.append(Card(card_id=f"{rank.value}_{i}", printed_rank=rank, kind=CardKind.NORMAL))
        cards.append(Card(card_id=f"G_{ghost_rank.value}", printed_rank=ghost_rank, kind=CardKind.GHOST))
        cards.append(Card(card_id="W_0", printed_rank=None, kind=CardKind.WILD))
        cards.append(Card(card_id="W_1", printed_rank=None, kind=CardKind.WILD))
        return cards

    def _remove_known_cards(self, pool: list[Card], known_cards: list[Card]) -> list[Card]:
        remaining = list(pool)
        for known in known_cards:
            found_idx = None
            for idx, c in enumerate(remaining):
                if c.kind == known.kind:
                    if c.kind == CardKind.WILD:
                        found_idx = idx
                        break
                    elif c.printed_rank == known.printed_rank:
                        found_idx = idx
                        break
            if found_idx is None:
                raise ValueError("Known card not present in pool")
            remaining.pop(found_idx)
        return remaining

    def _compute_likelihood(self, observation: PrivateObservation, particle: Particle) -> float:
        """Computes likelihood of the game history given this particle state."""
        weight = 1.0

        for event in observation.public_history:
            seat = event.seat
            if seat is None:
                continue

            if event.event_type == "play":
                claim_str = event.detail.get("claim_rank") or event.detail.get("claimRank")
                if not claim_str:
                    continue
                claim_rank = Rank(claim_str)
                count = int(event.detail.get("count", 1))

                hand = particle.seat_hands.get(seat, [])
                support = sum(
                    1 for c in hand if c.printed_rank == claim_rank or c.kind == CardKind.WILD
                )
                has_ghost = any(c.kind == CardKind.GHOST for c in hand)

                if support >= count:
                    # Honest play has high likelihood
                    prob = 0.65 + 0.15 * (support / max(1, len(hand)))
                else:
                    # Bluff play
                    prob = 0.20 + 0.10 * (1.0 - support / max(1, count))

                # Ghost single-play bonus
                if count == 1 and has_ghost:
                    prob = max(prob, 0.40)

                weight *= max(1e-3, prob)

            elif event.event_type == "challenge":
                # Challenger's action: likelihood increases if challenger has few support cards
                hand = particle.seat_hands.get(seat, [])
                claim_str = event.detail.get("claim_rank")
                if claim_str:
                    claim_rank = Rank(claim_str)
                    support = sum(
                        1 for c in hand if c.printed_rank == claim_rank or c.kind == CardKind.WILD
                    )
                    prob = 0.40 + 0.40 * (1.0 - min(1.0, support / 3.0))
                    weight *= max(1e-3, prob)

        return weight

    def _infer_last_play(
        self,
        observation: PrivateObservation,
        particles: list[Particle],
        claim_rank: Rank,
    ) -> LastPlayBelief | None:
        if observation.latest_play_seat is None or observation.latest_play_seat == observation.hero_seat:
            return None

        seat = observation.latest_play_seat
        count = max(1, observation.latest_play_count)

        honest_weight = 0.0
        ghost_weight = 0.0
        bluff_weight = 0.0

        for p in particles:
            hand = p.seat_hands.get(seat, [])
            support = sum(
                1 for c in hand if c.printed_rank == claim_rank or c.kind == CardKind.WILD
            )
            has_ghost = any(c.kind == CardKind.GHOST for c in hand)

            if has_ghost:
                # 握有鬼牌时，单出或组合出牌均可作为鬼牌诱捕陷阱（对手质疑将遭反噬爆头）
                ghost_weight += p.weight * (0.50 if count == 1 else 0.35)
                if support >= count:
                    honest_weight += p.weight * 0.50
                else:
                    bluff_weight += p.weight * 0.50
            elif support >= count:
                honest_weight += p.weight
            else:
                bluff_weight += p.weight

        total = honest_weight + ghost_weight + bluff_weight
        if total <= 0:
            return None

        h_prob = honest_weight / total
        g_prob = ghost_weight / total
        b_prob = bluff_weight / total

        if b_prob > 0.55:
            reason = f"推测上家手里目标牌不足 {count} 张，诈唬嫌疑较高"
        elif g_prob > 0.30:
            reason = f"上家大概率握有鬼牌并设下反噬陷阱（出牌 {count} 张），谨防遭鬼牌反噬爆头！"
        elif h_prob > 0.60:
            reason = f"推测上家真实持有至少 {count} 张匹配牌，建议保守"
        else:
            reason = "局面扑朔迷离，真假参半"

        return LastPlayBelief(
            seat=seat,
            claim_rank=claim_rank,
            claim_count=count,
            honest_prob=h_prob,
            bluff_prob=b_prob,
            ghost_trap_prob=g_prob,
            reason=reason,
        )

    def _extract_revealed_cards(
        self, observation: PrivateObservation
    ) -> tuple[list[Card], Rank | None]:
        revealed: list[Card] = []
        ghost_rank: Rank | None = None

        for event in observation.public_history:
            if event.event_type != "challenge":
                continue
            for idx, label in enumerate(event.detail.get("revealed_cards", [])):
                if label == "W":
                    revealed.append(Card(f"rev_w_{idx}", None, CardKind.WILD))
                elif label.startswith("G(") and label.endswith(")"):
                    r = Rank(label[2:-1])
                    ghost_rank = r
                    revealed.append(Card(f"rev_g_{idx}", r, CardKind.GHOST))
                elif label in ("A", "K", "Q"):
                    r = Rank(label)
                    revealed.append(Card(f"rev_{label}_{idx}", r, CardKind.NORMAL))

        return revealed, ghost_rank

    def _possible_ghost_ranks(
        self, observation: PrivateObservation, revealed_cards: list[Card]
    ) -> list[Rank]:
        visible = Counter[Rank]()
        for c in list(observation.hero_hand) + revealed_cards:
            if c.kind == CardKind.NORMAL and c.printed_rank is not None:
                visible[c.printed_rank] += 1

        possible = []
        for candidate in Rank:
            # If visible normal cards of that rank exceed 5, it cannot be the ghost rank
            if visible[candidate] <= 5:
                possible.append(candidate)
        return possible

    def _fallback_inference(
        self, observation: PrivateObservation, claim_rank: Rank
    ) -> POMDPInferenceResult:
        estimates = {}
        for player in observation.players:
            estimates[player.seat] = HandCardEstimate(
                seat=player.seat,
                target_count_mean=player.hand_count / 3.0,
                target_count_stdev=0.8,
                has_ghost_prob=0.25,
                wild_count_mean=player.hand_count * (2.0 / 20.0),
                hand_count=player.hand_count,
            )
        return POMDPInferenceResult(
            claim_rank=claim_rank,
            ghost_rank_probs={r.value: 1.0 / 3.0 for r in Rank},
            ghost_revealed=False,
            seat_estimates=estimates,
            last_play=None,
            effective_particles=0.0,
        )
