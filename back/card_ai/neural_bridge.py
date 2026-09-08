from __future__ import annotations

"""Bridge between the frontend overlay's JSON payload and the game engine's
PrivateObservation, enabling NeuralPolicy to provide action advice.

The frontend sends a hand summary (targetCount / nonTargetCount / ghostCount /
wildCount), not actual Card objects.  We reconstruct a synthetic PrivateObservation
that is close enough for the feature encoder to work with.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .types import (
    Card,
    CardKind,
    PrivateObservation,
    PublicEvent,
    PublicPlayerView,
    Rank,
)


# ---------------------------------------------------------------------------
# Payload helpers
# ---------------------------------------------------------------------------

def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _rank_from_str(s: str | None) -> Rank | None:
    if s == "A":
        return Rank.A
    if s == "K":
        return Rank.K
    if s == "Q":
        return Rank.Q
    return None


# ---------------------------------------------------------------------------
# Synthetic Card construction
# ---------------------------------------------------------------------------

def _build_synthetic_hand(
    *,
    target_count: int,
    non_target_count: int,
    ghost_count: int,
    wild_count: int,
    claim_rank: Rank | None,
) -> tuple[Card, ...]:
    """Build a plausible hand of Card objects from counts.

    We assign printed_rank to each card so the FeatureEncoder's hand features
    (target fraction, non-target fraction, etc.) compute correctly.
    """
    cards: list[Card] = []
    card_id = 0

    # Target cards (normal, matching claim_rank)
    for _ in range(target_count):
        cards.append(Card(
            card_id=f"syn_{card_id}",
            printed_rank=claim_rank,
            kind=CardKind.NORMAL,
        ))
        card_id += 1

    # Non-target cards (normal, use a different rank)
    filler_rank = next(
        (r for r in [Rank.A, Rank.K, Rank.Q] if r != claim_rank),
        Rank.A,
    )
    for _ in range(non_target_count):
        cards.append(Card(
            card_id=f"syn_{card_id}",
            printed_rank=filler_rank,
            kind=CardKind.NORMAL,
        ))
        card_id += 1

    # Ghost card(s)
    for _ in range(ghost_count):
        cards.append(Card(
            card_id=f"syn_{card_id}",
            printed_rank=claim_rank,  # ghost takes claim_rank as its face rank
            kind=CardKind.GHOST,
        ))
        card_id += 1

    # Wild card(s)
    for _ in range(wild_count):
        cards.append(Card(
            card_id=f"syn_{card_id}",
            printed_rank=None,
            kind=CardKind.WILD,
        ))
        card_id += 1

    return tuple(cards)


# ---------------------------------------------------------------------------
# Main reconstruction function
# ---------------------------------------------------------------------------

def payload_to_observation(payload: dict[str, Any]) -> PrivateObservation | None:
    """Convert a frontend overlay JSON payload into a PrivateObservation.

    Returns None if essential information is missing (e.g., hero hand not set).
    """
    session = payload.get("session", payload)

    my_seat = _safe_int(session.get("mySeat"), 1)
    current_turn_seat = _safe_int(session.get("currentTurnSeat"), 1)
    target_rank = _rank_from_str(session.get("targetRank"))
    stack_count = _safe_int(session.get("stackCount"), 0)

    # Hero hand counts
    hero_hand_raw = session.get("heroHandView") or {}
    target_count = _safe_int(hero_hand_raw.get("targetCount"), 0)
    non_target_count = _safe_int(hero_hand_raw.get("nonTargetCount"), 0)
    ghost_count = _safe_int(hero_hand_raw.get("ghostCount"), 0)
    wild_count = _safe_int(hero_hand_raw.get("wildCount"), 0)
    total_hand = target_count + non_target_count + ghost_count + wild_count

    if total_hand == 0:
        # Hand not populated yet — cannot build meaningful observation
        return None

    hero_hand = _build_synthetic_hand(
        target_count=target_count,
        non_target_count=non_target_count,
        ghost_count=ghost_count,
        wild_count=wild_count,
        claim_rank=target_rank,
    )

    # Build PublicPlayerView for each seat
    seats_raw = session.get("seats", [])
    player_views: list[PublicPlayerView] = []
    latest_play_seat: int | None = None
    latest_play_count: int = 0

    for seat_data in seats_raw:
        seat_num = _safe_int(seat_data.get("seat"), 0)
        if seat_num == 0:
            continue
        status = seat_data.get("status", "alive")
        hand_count = _safe_int(seat_data.get("handCount"), 0)
        shots_taken = _safe_int(seat_data.get("shotsTaken"), 0)
        last_action = seat_data.get("lastAction", "")

        player_views.append(PublicPlayerView(
            seat=seat_num,
            alive=(status == "alive"),
            escaped=(status == "escaped"),
            pending_escape=False,
            hand_count=hand_count,
            shots_taken=shots_taken,
        ))

        if last_action == "play":
            latest_play_seat = seat_num
            latest_play_count = stack_count  # approximate

    # Build public history from event log
    history_raw = session.get("history", [])
    public_history: list[PublicEvent] = []
    for i, event in enumerate(reversed(history_raw)):  # oldest first
        etype = event.get("type", "")
        seat_val = event.get("seat") or event.get("targetSeat")
        detail: dict[str, Any] = {}

        if etype == "play":
            detail = {
                "claim_rank": event.get("claimRank", ""),
                "count": event.get("count", 0),
            }
        elif etype == "challenge":
            detail = {
                "outcome": event.get("outcome", ""),
                "revealed_cards": event.get("revealedCards", []),
            }
        elif etype == "shot":
            detail = {"chamber": event.get("chamber", "")}

        public_history.append(PublicEvent(
            turn_index=i,
            event_type=etype,
            seat=_safe_int(seat_val) if seat_val is not None else None,
            detail=detail,
        ))

    # Determine latest play from history (more reliable than lastAction)
    for event in reversed(public_history):
        if event.event_type == "play":
            latest_play_seat = event.seat
            latest_play_count = _safe_int(
                next((e.detail.get("count", 0) for e in public_history
                      if e.event_type == "play"), 0)
            )
            break

    return PrivateObservation(
        hero_seat=my_seat,
        hero_hand=hero_hand,
        current_seat=current_turn_seat,
        round_claim_rank=target_rank,
        latest_play_seat=latest_play_seat,
        latest_play_count=latest_play_count,
        players=tuple(player_views),
        public_history=tuple(public_history),
    )


# ---------------------------------------------------------------------------
# Neural advisor wrapper with manifest validation, masking & idempotent sampling
# ---------------------------------------------------------------------------

@dataclass
class NeuralAdviceResult:
    """Action probability distribution from the neural policy."""
    action_probs: dict[str, float]   # legal abstract label -> linear_norm probability
    top_action_label: str
    top_action_prob: float
    sampled_action_label: str         # 单次采样选中的推荐动作 (与离线评测采样机制完全一致)
    sampled_action_prob: float
    model_manifest: dict[str, Any]    # 包含完整64位哈希、张量哈希与版本
    state_fingerprint: str            # 局面指纹 (用于保证刷新幂等性)
    prob_semantics: str = "动作选择概率 (模型根据当前策略分布的行动倾向，非胜率或正确率)"
    is_fallback: bool = False


class NeuralAdvisorBridge:
    """Wraps a NeuralPolicy so it can be called from RuntimeAdvisor.

    Validates model manifest, lazily loads checkpoint, and provides legal action masking.
    """

    def __init__(self, checkpoint_path: str | Path | None = None) -> None:
        from .manifest import verify_production_model, PRODUCTION_MODEL_MANIFEST
        self._manifest = PRODUCTION_MODEL_MANIFEST
        if checkpoint_path is None:
            checkpoint_path = Path(__file__).resolve().parent.parent / PRODUCTION_MODEL_MANIFEST["relative_path"]
        self._checkpoint_path = Path(checkpoint_path)
        self._policy = None
        self._last_mtime: float = 0.0
        self.is_valid_model = False
        self.validation_message = ""
        self._validate_and_load()

    def _validate_and_load(self) -> None:
        from .manifest import verify_production_model
        ok, msg, meta = verify_production_model()
        self.is_valid_model = ok
        self.validation_message = msg
        self._meta = meta

        if not ok:
            print(f"[NeuralAdvisorBridge WARNING] {msg}", file=sys.stderr)
            return

        if self._checkpoint_path.exists():
            from .neural_policy import NeuralPolicy
            self._policy = NeuralPolicy.load(self._checkpoint_path, prob_mode="linear_norm", greedy=False)
            self._last_mtime = self._checkpoint_path.stat().st_mtime

    def advise(self, observation: PrivateObservation, state_fingerprint: str = "") -> NeuralAdviceResult | None:
        """Return action probability distribution for the given observation.

        Returns None if checkpoint not available or failed integrity check.
        """
        if not self.is_valid_model or self._policy is None:
            return None

        try:
            import hashlib
            import torch
            from itertools import combinations
            from .abstractions import ActionAbstractor, action_to_index, ABSTRACT_ACTION_LABELS
            from .types import PlayAction, ChallengeAction, CardKind
            from .features import FeatureEncoder

            # 1. 严格构建当前局面下的候选合法动作列表
            hero_hand = observation.hero_hand
            claim_rank = observation.round_claim_rank
            legal_actions = []

            # (1) 出牌合法动作构建: 选 1~3 张手牌
            hand_cards = list(hero_hand)
            max_play = min(3, len(hand_cards))
            for k in range(1, max_play + 1):
                for combo in combinations(hand_cards, k):
                    # 鬼牌只能单出
                    ghost_cards = [c for c in combo if c.kind == CardKind.GHOST]
                    if ghost_cards and len(combo) > 1:
                        continue
                    legal_actions.append(PlayAction(
                        seat=observation.hero_seat,
                        claim_rank=claim_rank,
                        card_ids=tuple(c.card_id for c in combo),
                    ))

            # (2) 质疑合法动作构建: 上一家有出牌且非我方
            if observation.latest_play_seat is not None and observation.latest_play_seat != observation.hero_seat:
                legal_actions.append(ChallengeAction(seat=observation.hero_seat))

            if not legal_actions:
                return None

            # 2. 提取特征并在合法抽象动作上计算 linear_norm 概率
            abstractor = ActionAbstractor()
            grouped = abstractor.abstract_legal_actions(tuple(legal_actions), observation)
            legal_indices = [action_to_index(aa.label) for aa in grouped]

            encoder = FeatureEncoder()
            features = encoder.encode(observation)
            net = self._policy.strategy_net
            net.eval()
            device = next(net.parameters()).device

            with torch.no_grad():
                feat_dev = features.unsqueeze(0).to(device)
                logits = net(feat_dev).squeeze(0).cpu()

            # 严格按照评测一致的 linear_norm 逻辑
            raw_scores = logits[legal_indices]
            clamped = torch.clamp(raw_scores, min=0.0)
            sum_val = clamped.sum()

            if sum_val > 1e-8:
                legal_probs = clamped / sum_val
            else:
                legal_probs = torch.full_like(clamped, 1.0 / len(legal_indices))

            # 建立动作标签到选择概率的映射
            grouped_keys = list(grouped.keys())
            label_probs = {}
            for i, aa in enumerate(grouped_keys):
                label_probs[aa.label] = float(legal_probs[i].item())

            top_label = max(label_probs, key=label_probs.__getitem__)
            top_prob = label_probs[top_label]

            # 3. 幂等采样推荐动作 (Sampled Choice):
            # 若传入局面指纹，根据指纹种子确定性抽样，保证同一状态刷新不跳变！
            if state_fingerprint:
                seed_int = int(hashlib.md5(state_fingerprint.encode()).hexdigest()[:8], 16)
                gen = torch.Generator().manual_seed(seed_int)
                sampled_local_idx = torch.multinomial(legal_probs, 1, generator=gen).item()
            else:
                sampled_local_idx = torch.multinomial(legal_probs, 1).item()

            sampled_label = grouped_keys[sampled_local_idx].label
            sampled_prob = label_probs[sampled_label]

            return NeuralAdviceResult(
                action_probs=label_probs,
                top_action_label=top_label,
                top_action_prob=top_prob,
                sampled_action_label=sampled_label,
                sampled_action_prob=sampled_prob,
                model_manifest=self._meta,
                state_fingerprint=state_fingerprint,
                is_fallback=False,
            )

        except Exception as e:
            print(f"[NeuralAdvisorBridge ERROR] 推理异常: {e}", file=sys.stderr)
            return None
