from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .types import Card, CardKind, ChallengeAction, GameAction, PlayAction, PrivateObservation, Rank


@dataclass(frozen=True)
class AbstractAction:
    label: str
    kind: str
    claim_rank: Rank | None = None
    count: int = 0
    composition: tuple[str, ...] = ()
    challenged_seat: int | None = None


@dataclass(frozen=True)
class InformationSetKey:
    seat: int
    current_seat: int
    claim_rank: Rank | None
    latest_play_seat: int | None
    latest_play_count: int
    private_hand_signature: tuple[tuple[str, int], ...]
    public_player_signature: tuple[tuple[int, bool, bool, bool, int, int], ...]
    public_history_signature: tuple[tuple[int, str, int | None, tuple[tuple[str, str], ...]], ...]


class InformationSetEncoder:
    def encode(self, observation: PrivateObservation) -> InformationSetKey:
        return InformationSetKey(
            seat=observation.hero_seat,
            current_seat=observation.current_seat,
            claim_rank=observation.round_claim_rank,
            latest_play_seat=observation.latest_play_seat,
            latest_play_count=observation.latest_play_count,
            private_hand_signature=self._hand_signature(observation.hero_hand),
            public_player_signature=tuple(
                (
                    player.seat,
                    player.alive,
                    player.escaped,
                    player.pending_escape,
                    player.hand_count,
                    player.shots_taken,
                )
                for player in observation.players
            ),
            public_history_signature=tuple(
                (
                    event.turn_index,
                    event.event_type,
                    event.seat,
                    tuple(sorted((str(key), str(value)) for key, value in event.detail.items())),
                )
                for event in observation.public_history
            ),
        )

    def _hand_signature(self, cards: tuple[Card, ...]) -> tuple[tuple[str, int], ...]:
        counts: Counter[str] = Counter()
        for card in cards:
            if card.kind == CardKind.WILD:
                counts["wild"] += 1
            elif card.kind == CardKind.GHOST:
                rank = card.printed_rank.value if card.printed_rank else "unknown"
                counts[f"ghost:{rank}"] += 1
            elif card.printed_rank is not None:
                counts[f"normal:{card.printed_rank.value}"] += 1
        return tuple(sorted(counts.items()))


class ActionAbstractor:
    def abstract(self, action: GameAction, observation: PrivateObservation) -> AbstractAction:
        if isinstance(action, ChallengeAction):
            label = f"challenge:{action.challenged_seat}"
            return AbstractAction(
                label=label,
                kind="challenge",
                challenged_seat=action.challenged_seat,
            )

        card_lookup = {card.card_id: card for card in observation.hero_hand}
        cards = tuple(card_lookup[card_id] for card_id in action.card_ids)
        composition = tuple(sorted(self._card_role(card, action.claim_rank) for card in cards))
        composition_text = ",".join(composition)
        label = f"play:{action.claim_rank.value}:{len(cards)}:{composition_text}"
        return AbstractAction(
            label=label,
            kind="play",
            claim_rank=action.claim_rank,
            count=len(cards),
            composition=composition,
        )

    def abstract_legal_actions(
        self,
        legal_actions: tuple[GameAction, ...],
        observation: PrivateObservation,
    ) -> dict[AbstractAction, tuple[GameAction, ...]]:
        grouped: dict[AbstractAction, list[GameAction]] = {}
        for action in legal_actions:
            abstract_action = self.abstract(action, observation)
            grouped.setdefault(abstract_action, []).append(action)
        return {key: tuple(value) for key, value in grouped.items()}

    def _card_role(self, card: Card, claim_rank: Rank) -> str:
        if card.kind == CardKind.WILD:
            return "wild"
        if card.kind == CardKind.GHOST:
            return "ghost"
        if card.printed_rank == claim_rank:
            return "target"
        return "non_target"


# ---------------------------------------------------------------------------
# Fixed abstract action space for neural network output indexing
# ---------------------------------------------------------------------------

_COMPOSITIONS_1: list[tuple[str, ...]] = [
    ("ghost",),
    ("non_target",),
    ("target",),
    ("wild",),
]

_COMPOSITIONS_2: list[tuple[str, ...]] = [
    ("non_target", "non_target"),
    ("non_target", "target"),
    ("non_target", "wild"),
    ("target", "target"),
    ("target", "wild"),
    ("wild", "wild"),
]

_COMPOSITIONS_3: list[tuple[str, ...]] = [
    ("non_target", "non_target", "non_target"),
    ("non_target", "non_target", "target"),
    ("non_target", "non_target", "wild"),
    ("non_target", "target", "target"),
    ("non_target", "target", "wild"),
    ("non_target", "wild", "wild"),
    ("target", "target", "target"),
    ("target", "target", "wild"),
    ("target", "wild", "wild"),
]

_ALL_COMPOSITIONS = _COMPOSITIONS_1 + _COMPOSITIONS_2 + _COMPOSITIONS_3

ABSTRACT_ACTION_LABELS: tuple[str, ...] = tuple(
    f"play:{rank.value}:{len(comp)}:{','.join(comp)}"
    for rank in Rank
    for comp in _ALL_COMPOSITIONS
) + ("challenge",)

ACTION_SPACE_SIZE: int = len(ABSTRACT_ACTION_LABELS)

_LABEL_TO_INDEX: dict[str, int] = {
    label: idx for idx, label in enumerate(ABSTRACT_ACTION_LABELS)
}


def action_to_index(label: str) -> int:
    """Map an abstract action label to its fixed index.

    Challenge actions from ActionAbstractor carry a seat suffix (e.g.
    'challenge:2').  For the neural network we collapse all challenge
    variants to the single 'challenge' bucket.
    """
    if label.startswith("challenge"):
        return _LABEL_TO_INDEX["challenge"]
    return _LABEL_TO_INDEX[label]


def index_to_label(index: int) -> str:
    """Map a fixed index back to its abstract action label."""
    return ABSTRACT_ACTION_LABELS[index]


def to_network_label(label: str) -> str:
    """Normalise an ActionAbstractor label to the fixed action-space label."""
    if label.startswith("challenge"):
        return "challenge"
    return label

