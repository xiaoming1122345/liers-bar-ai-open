from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol


class Rank(str, Enum):
    A = "A"
    K = "K"
    Q = "Q"


class CardKind(str, Enum):
    NORMAL = "normal"
    GHOST = "ghost"
    WILD = "wild"


class ChallengeOutcome(str, Enum):
    HONEST = "honest"
    LIE = "lie"
    GHOST = "ghost"


@dataclass(frozen=True)
class Card:
    card_id: str
    printed_rank: Rank | None
    kind: CardKind

    def short_label(self) -> str:
        if self.kind == CardKind.WILD:
            return "W"
        if self.kind == CardKind.GHOST and self.printed_rank is not None:
            return f"G({self.printed_rank.value})"
        return self.printed_rank.value if self.printed_rank is not None else "?"


@dataclass(frozen=True)
class PlayAction:
    seat: int
    card_ids: tuple[str, ...]
    claim_rank: Rank


@dataclass(frozen=True)
class ChallengeAction:
    seat: int
    challenged_seat: int


GameAction = PlayAction | ChallengeAction


@dataclass
class PlayedSet:
    seat: int
    claim_rank: Rank
    cards: tuple[Card, ...]


@dataclass
class PlayerState:
    seat: int
    hand: list[Card]
    live_round_index: int
    shots_taken: int = 0
    alive: bool = True
    escaped: bool = False
    pending_escape: bool = False

    @property
    def hand_count(self) -> int:
        return len(self.hand)


@dataclass
class RoundState:
    claim_rank: Rank | None = None
    leader_seat: int | None = None
    latest_play: PlayedSet | None = None
    plays: list[PlayedSet] = field(default_factory=list)


@dataclass
class PublicEvent:
    turn_index: int
    event_type: str
    seat: int | None
    detail: dict


@dataclass
class GameState:
    players: list[PlayerState]
    current_seat: int
    round_state: RoundState = field(default_factory=RoundState)
    discard_pile: list[Card] = field(default_factory=list)
    public_history: list[PublicEvent] = field(default_factory=list)
    finish_order: list[int] = field(default_factory=list)
    turn_index: int = 0
    round_index: int = 1

    def player_by_seat(self, seat: int) -> PlayerState:
        for player in self.players:
            if player.seat == seat:
                return player
        raise ValueError(f"unknown seat: {seat}")


@dataclass
class PublicPlayerView:
    seat: int
    alive: bool
    escaped: bool
    pending_escape: bool
    hand_count: int
    shots_taken: int


@dataclass
class PrivateObservation:
    hero_seat: int
    hero_hand: tuple[Card, ...]
    current_seat: int
    round_claim_rank: Rank | None
    latest_play_seat: int | None
    latest_play_count: int
    players: tuple[PublicPlayerView, ...]
    public_history: tuple[PublicEvent, ...]


@dataclass
class SeatBelief:
    seat: int
    ghost_holder_prob: float
    ghost_hidden_prob: float
    expected_target_cards: float
    target_card_uncertainty: float
    rank_support_expectation: dict[Rank, float]
    rank_support_stdev: dict[Rank, float]


@dataclass
class PublicBeliefState:
    claim_rank: Rank | None
    ghost_rank_probs: dict[Rank, float]
    ghost_out_of_play_prob: float
    ghost_revealed: bool
    seat_beliefs: tuple[SeatBelief, ...]
    revealed_card_count: int
    evidence_count: int
    particle_count: int
    effective_sample_size: float


@dataclass
class ConfidenceReport:
    ghost_holder_confidence: float
    ghost_rank_confidence: float
    belief_sharpness: float
    evidence_coverage: float
    action_margin_confidence: float
    modeling_degree: float


@dataclass
class BeliefTruth:
    ghost_hidden_holder_seat: int | None
    ghost_rank: Rank
    ghost_revealed: bool
    seat_rank_support: dict[int, dict[Rank, int]]


@dataclass
class BeliefDiagnostics:
    ghost_location_probability: float
    ghost_rank_probability: float
    mean_support_error: float
    support_alignment: float
    calibration_score: float
    confidence_gap: float


@dataclass
class TrainingBatchReport:
    sample_count: int
    mean_modeling_degree: float
    mean_calibration_score: float
    mean_confidence_gap: float
    mean_ghost_location_probability: float
    mean_ghost_rank_probability: float
    mean_support_alignment: float
    mean_support_error: float
    mean_terminal_utility: float
    seat_mean_terminal_utility: dict[int, float]
    seat_sample_count: dict[int, int]


@dataclass(frozen=True)
class PlayerIdentity:
    seat: int
    player_id: str
    display_name: str
    is_hero: bool = False


@dataclass(frozen=True)
class MatchRecord:
    game_index: int
    winner_seat: int
    utilities: dict[int, float]
    ranks: dict[int, int]
    seat_profile_names: dict[int, str]


@dataclass(frozen=True)
class PerspectiveEvaluationSummary:
    mode: str
    games: int
    hero_seat: int | None
    hero_player_id: str | None
    hero_display_name: str | None
    win_rate: float | None
    top2_rate: float | None
    mean_rank: float | None
    mean_utility: float | None
    seat_win_rate: dict[int, float]
    seat_mean_utility: dict[int, float]
    opponent_profile_counts: dict[str, int]
    opponent_inferred_profile_counts: dict[str, int]
    identities: tuple[PlayerIdentity, ...]
    seat_models: tuple["PlayerModelSummary", ...]


@dataclass(frozen=True)
class PlayerModelSummary:
    seat: int
    player_id: str
    display_name: str
    is_hero: bool
    observed_actions: int
    play_rate: float
    challenge_rate: float
    bluff_rate: float
    honest_rate: float
    ghost_play_rate: float
    mean_play_count: float
    variability_score: float
    variability_label: str
    public_style_cluster: str
    aggression_score: float
    post_shot_action_count: int
    post_shot_bluff_rate: float
    post_shot_challenge_rate: float
    post_shot_bluff_delta: float
    post_shot_challenge_delta: float
    pressure_response_label: str
    inferred_profile_name: str
    inferred_style: str


class Policy(Protocol):
    def choose_action(
        self,
        observation: PrivateObservation,
        legal_actions: tuple[GameAction, ...],
    ) -> GameAction:
        ...
