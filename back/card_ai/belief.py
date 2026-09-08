from __future__ import annotations

from collections import Counter
from math import sqrt
from random import Random

from .types import Card, CardKind, PrivateObservation, PublicBeliefState, Rank, SeatBelief


def rank_support_count(cards: list[Card] | tuple[Card, ...], rank: Rank) -> int:
    support = 0
    for card in cards:
        if card.kind == CardKind.WILD:
            support += 1
        elif card.kind == CardKind.GHOST:
            support += 1
        elif card.printed_rank == rank:
            support += 1
    return support


class BeliefTracker:
    """
    A lightweight public-belief scaffold.

    This is intentionally conservative:
    - it conditions on hero private cards and all publicly revealed cards
    - it samples hidden allocations for the remaining hands
    - it does not yet replay the full hidden discard history or action likelihoods

    It is designed to be replaced later by a particle filter or a ReBeL-style PBS tracker.
    """

    def __init__(self, particle_count: int = 512, seed: int | None = None) -> None:
        self.particle_count = particle_count
        self._rng = Random(seed)

    def build(self, observation: PrivateObservation) -> PublicBeliefState:
        revealed_cards, revealed_ghost_rank = self._extract_revealed_cards(observation)
        evidence_count = self._count_evidence(observation)
        seat_order = [player.seat for player in observation.players]

        if revealed_ghost_rank is not None:
            ghost_rank_probs = {
                rank: 1.0 if rank == revealed_ghost_rank else 0.0 for rank in Rank
            }
            seat_beliefs = tuple(
                self._deterministic_seat_belief(
                    observation=observation,
                    seat=seat,
                    ghost_hidden_prob=0.0,
                )
                for seat in seat_order
            )
            return PublicBeliefState(
                claim_rank=observation.round_claim_rank,
                ghost_rank_probs=ghost_rank_probs,
                ghost_out_of_play_prob=1.0,
                ghost_revealed=True,
                seat_beliefs=seat_beliefs,
                revealed_card_count=len(revealed_cards),
                evidence_count=evidence_count,
                particle_count=0,
                effective_sample_size=0.0,
            )

        possible_ranks = self._possible_ghost_ranks(observation, revealed_cards)
        if not possible_ranks:
            possible_ranks = list(Rank)

        rank_particles: Counter[Rank] = Counter()
        ghost_holder_weight_sum: Counter[int] = Counter()
        support_sum: dict[int, dict[Rank, float]] = {
            player.seat: {rank: 0.0 for rank in Rank} for player in observation.players
        }
        support_sq_sum: dict[int, dict[Rank, float]] = {
            player.seat: {rank: 0.0 for rank in Rank} for player in observation.players
        }

        accepted_particles = 0
        total_weight = 0.0
        total_weight_sq = 0.0
        hero_seat = observation.hero_seat
        hero_ghost = any(card.kind == CardKind.GHOST for card in observation.hero_hand)

        for _ in range(self.particle_count):
            sampled_rank = self._rng.choice(possible_ranks)
            hands = self._sample_hidden_hands(
                observation=observation,
                ghost_rank=sampled_rank,
                revealed_cards=revealed_cards,
            )
            if hands is None:
                continue

            accepted_particles += 1
            seat_cards = {
                player.seat: list(observation.hero_hand) if player.seat == hero_seat else hands.get(player.seat, [])
                for player in observation.players
            }
            weight = self._particle_weight(observation, seat_cards)
            if weight <= 0.0:
                continue

            total_weight += weight
            total_weight_sq += weight * weight
            rank_particles[sampled_rank] += weight

            holder_seat = None
            if hero_ghost:
                holder_seat = hero_seat
            else:
                for seat, cards in hands.items():
                    if any(card.kind == CardKind.GHOST for card in cards):
                        holder_seat = seat
                        break

            if holder_seat is not None:
                ghost_holder_weight_sum[holder_seat] += weight

            for player in observation.players:
                cards = seat_cards[player.seat]

                for rank in Rank:
                    support = rank_support_count(cards, rank)
                    support_sum[player.seat][rank] += weight * support
                    support_sq_sum[player.seat][rank] += weight * support * support

        if accepted_particles == 0 or total_weight <= 0.0:
            return self._fallback_build(
                observation=observation,
                evidence_count=evidence_count,
                revealed_card_count=len(revealed_cards),
            )

        ghost_rank_probs = {
            rank: rank_particles[rank] / total_weight for rank in Rank
        }

        seat_beliefs = []
        for player in observation.players:
            support_expectation = {}
            support_stdev = {}
            for rank in Rank:
                mean = support_sum[player.seat][rank] / total_weight
                mean_sq = support_sq_sum[player.seat][rank] / total_weight
                variance = max(0.0, mean_sq - mean * mean)
                support_expectation[rank] = mean
                support_stdev[rank] = sqrt(variance)

            current_rank = observation.round_claim_rank
            if current_rank is None:
                expected_target = max(support_expectation.values()) if support_expectation else 0.0
                uncertainty = max(support_stdev.values()) if support_stdev else 0.0
            else:
                expected_target = support_expectation[current_rank]
                uncertainty = support_stdev[current_rank]

            hidden_prob = ghost_holder_weight_sum[player.seat] / total_weight
            seat_beliefs.append(
                SeatBelief(
                    seat=player.seat,
                    ghost_holder_prob=hidden_prob,
                    ghost_hidden_prob=hidden_prob,
                    expected_target_cards=expected_target,
                    target_card_uncertainty=min(1.0, uncertainty / 3.0),
                    rank_support_expectation=support_expectation,
                    rank_support_stdev=support_stdev,
                )
            )

        return PublicBeliefState(
            claim_rank=observation.round_claim_rank,
            ghost_rank_probs=ghost_rank_probs,
            ghost_out_of_play_prob=0.0,
            ghost_revealed=False,
            seat_beliefs=tuple(seat_beliefs),
            revealed_card_count=len(revealed_cards),
            evidence_count=evidence_count,
            particle_count=accepted_particles,
            effective_sample_size=(total_weight * total_weight / total_weight_sq) if total_weight_sq > 0 else 0.0,
        )

    def _deterministic_seat_belief(
        self,
        observation: PrivateObservation,
        seat: int,
        ghost_hidden_prob: float,
    ) -> SeatBelief:
        if seat == observation.hero_seat:
            cards = list(observation.hero_hand)
        else:
            cards = []

        rank_support_expectation = {
            rank: float(rank_support_count(cards, rank)) for rank in Rank
        }
        rank_support_stdev = {rank: 0.0 for rank in Rank}
        current_rank = observation.round_claim_rank
        if current_rank is None:
            expected_target = max(rank_support_expectation.values()) if rank_support_expectation else 0.0
        else:
            expected_target = rank_support_expectation[current_rank]

        return SeatBelief(
            seat=seat,
            ghost_holder_prob=ghost_hidden_prob,
            ghost_hidden_prob=ghost_hidden_prob,
            expected_target_cards=expected_target,
            target_card_uncertainty=0.0,
            rank_support_expectation=rank_support_expectation,
            rank_support_stdev=rank_support_stdev,
        )

    def _fallback_build(
        self,
        observation: PrivateObservation,
        evidence_count: int,
        revealed_card_count: int,
    ) -> PublicBeliefState:
        claim_rank = observation.round_claim_rank
        live_slots = sum(player.hand_count for player in observation.players if player.seat != observation.hero_seat)
        live_slots = max(1, live_slots)

        seat_beliefs = []
        for player in observation.players:
            if player.seat == observation.hero_seat:
                ghost_prob = 1.0 if any(card.kind == CardKind.GHOST for card in observation.hero_hand) else 0.0
                cards = list(observation.hero_hand)
                support_expectation = {rank: float(rank_support_count(cards, rank)) for rank in Rank}
                support_stdev = {rank: 0.0 for rank in Rank}
            else:
                ghost_prob = player.hand_count / live_slots
                support_expectation = {rank: player.hand_count / 3.0 for rank in Rank}
                support_stdev = {rank: min(3.0, player.hand_count / 2.0) for rank in Rank}

            if claim_rank is None:
                expected_target = max(support_expectation.values()) if support_expectation else 0.0
                uncertainty = max(support_stdev.values()) if support_stdev else 0.0
            else:
                expected_target = support_expectation[claim_rank]
                uncertainty = support_stdev[claim_rank]

            seat_beliefs.append(
                SeatBelief(
                    seat=player.seat,
                    ghost_holder_prob=ghost_prob,
                    ghost_hidden_prob=ghost_prob,
                    expected_target_cards=expected_target,
                    target_card_uncertainty=min(1.0, uncertainty / 3.0),
                    rank_support_expectation=support_expectation,
                    rank_support_stdev=support_stdev,
                )
            )

        return PublicBeliefState(
            claim_rank=claim_rank,
            ghost_rank_probs={rank: 1 / 3 for rank in Rank},
            ghost_out_of_play_prob=0.0,
            ghost_revealed=False,
            seat_beliefs=tuple(seat_beliefs),
            revealed_card_count=revealed_card_count,
            evidence_count=evidence_count,
            particle_count=0,
            effective_sample_size=0.0,
        )

    def _sample_hidden_hands(
        self,
        observation: PrivateObservation,
        ghost_rank: Rank,
        revealed_cards: list[Card],
    ) -> dict[int, list[Card]] | None:
        pool = self._build_pool_for_rank(ghost_rank)

        known_cards = list(observation.hero_hand) + revealed_cards
        try:
            pool = self._remove_known_cards(pool, known_cards)
        except ValueError:
            return None

        self._rng.shuffle(pool)
        hands: dict[int, list[Card]] = {}
        cursor = 0

        for player in observation.players:
            if player.seat == observation.hero_seat:
                continue
            count = player.hand_count
            if cursor + count > len(pool):
                return None
            hands[player.seat] = pool[cursor : cursor + count]
            cursor += count

        return hands

    def _build_pool_for_rank(self, ghost_rank: Rank) -> list[Card]:
        cards: list[Card] = []
        for rank in Rank:
            normal_count = 5 if rank == ghost_rank else 6
            for index in range(normal_count):
                cards.append(Card(card_id=f"{rank.value}-pool-{index}", printed_rank=rank, kind=CardKind.NORMAL))
        cards.append(Card(card_id=f"G-{ghost_rank.value}", printed_rank=ghost_rank, kind=CardKind.GHOST))
        cards.append(Card(card_id="W-pool-0", printed_rank=None, kind=CardKind.WILD))
        cards.append(Card(card_id="W-pool-1", printed_rank=None, kind=CardKind.WILD))
        return cards

    def _remove_known_cards(self, pool: list[Card], known_cards: list[Card]) -> list[Card]:
        remaining = list(pool)
        for known in known_cards:
            match_index = self._find_matching_card_index(remaining, known)
            if match_index is None:
                raise ValueError("known card is inconsistent with the sampled pool")
            remaining.pop(match_index)
        return remaining

    def _find_matching_card_index(self, pool: list[Card], target: Card) -> int | None:
        for index, card in enumerate(pool):
            if card.kind != target.kind:
                continue
            if card.kind == CardKind.WILD:
                return index
            if card.printed_rank == target.printed_rank:
                return index
        return None

    def _extract_revealed_cards(
        self,
        observation: PrivateObservation,
    ) -> tuple[list[Card], Rank | None]:
        revealed_cards: list[Card] = []
        ghost_rank: Rank | None = None
        for event in observation.public_history:
            if event.event_type != "challenge":
                continue
            for index, label in enumerate(event.detail.get("revealed_cards", [])):
                if label == "W":
                    revealed_cards.append(
                        Card(card_id=f"revealed-W-{event.turn_index}-{index}", printed_rank=None, kind=CardKind.WILD)
                    )
                elif label.startswith("G("):
                    rank = Rank(label[2:-1])
                    ghost_rank = rank
                    revealed_cards.append(
                        Card(
                            card_id=f"revealed-G-{rank.value}-{event.turn_index}-{index}",
                            printed_rank=rank,
                            kind=CardKind.GHOST,
                        )
                    )
                else:
                    rank = Rank(label)
                    revealed_cards.append(
                        Card(
                            card_id=f"revealed-{rank.value}-{event.turn_index}-{index}",
                            printed_rank=rank,
                            kind=CardKind.NORMAL,
                        )
                    )
        return revealed_cards, ghost_rank

    def _count_evidence(self, observation: PrivateObservation) -> int:
        evidence_count = 0
        for event in observation.public_history:
            if event.event_type in {"play", "challenge", "shot", "escape"}:
                evidence_count += 1
        return evidence_count

    def _possible_ghost_ranks(
        self,
        observation: PrivateObservation,
        revealed_cards: list[Card],
    ) -> list[Rank]:
        visible_normals = Counter[Rank]()
        visible_ghosts = Counter[Rank]()
        for card in list(observation.hero_hand) + revealed_cards:
            if card.kind == CardKind.NORMAL and card.printed_rank is not None:
                visible_normals[card.printed_rank] += 1
            elif card.kind == CardKind.GHOST and card.printed_rank is not None:
                visible_ghosts[card.printed_rank] += 1

        possible = []
        for candidate in Rank:
            valid = True
            for rank in Rank:
                max_normal = 5 if rank == candidate else 6
                if visible_normals[rank] > max_normal:
                    valid = False
                    break
            ghost_total = sum(visible_ghosts.values())
            if ghost_total > 1:
                valid = False
            if ghost_total == 1 and visible_ghosts[candidate] != 1:
                valid = False
            if valid:
                possible.append(candidate)
        return possible

    def _particle_weight(
        self,
        observation: PrivateObservation,
        seat_cards: dict[int, list[Card]],
    ) -> float:
        weight = 1.0
        current_claim_rank: Rank | None = None
        current_claim_seat: int | None = None
        current_claim_count = 0

        for event in observation.public_history:
            if event.event_type == "play":
                seat = event.seat
                if seat is None:
                    continue
                claim_rank = Rank(event.detail["claim_rank"])
                count = int(event.detail["count"])
                support = rank_support_count(seat_cards.get(seat, []), claim_rank)
                non_support = max(0.0, len(seat_cards.get(seat, [])) - support)
                honest_signal = min(1.0, support / max(1, count))
                bluff_signal = min(1.0, non_support / max(1, count))
                ghost_signal = 1.0 if any(card.kind == CardKind.GHOST for card in seat_cards.get(seat, [])) else 0.0

                likelihood = 0.15
                likelihood += 0.55 * honest_signal
                likelihood += 0.20 * bluff_signal
                if count == 1:
                    likelihood += 0.10 * ghost_signal
                if count >= 2:
                    likelihood += 0.05 * honest_signal

                weight *= max(1e-4, min(1.0, likelihood))
                current_claim_rank = claim_rank
                current_claim_seat = seat
                current_claim_count = count
                continue

            if event.event_type == "challenge":
                seat = event.seat
                if seat is None or current_claim_rank is None or current_claim_seat is None:
                    current_claim_rank = None
                    current_claim_seat = None
                    current_claim_count = 0
                    continue

                challenger_support = rank_support_count(seat_cards.get(seat, []), current_claim_rank)
                challenged_support = rank_support_count(seat_cards.get(current_claim_seat, []), current_claim_rank)
                challenger_pressure = 1.0 - min(1.0, challenger_support / max(1, current_claim_count))
                challenged_suspicion = 1.0 - min(1.0, challenged_support / max(1, current_claim_count))
                likelihood = (
                    0.20
                    + 0.40 * challenger_pressure
                    + 0.25 * challenged_suspicion
                    + 0.15 * min(1.0, current_claim_count / 3.0)
                )
                weight *= max(1e-4, min(1.0, likelihood))
                current_claim_rank = None
                current_claim_seat = None
                current_claim_count = 0

        return weight
