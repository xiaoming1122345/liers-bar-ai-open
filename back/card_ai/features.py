from __future__ import annotations

import torch
from collections import Counter

from .types import Card, CardKind, PrivateObservation, PublicEvent, Rank

FEATURE_DIM: int = 80


class FeatureEncoder:
    """Encodes a PrivateObservation into a fixed-size feature tensor (80-dim or legacy 64-dim)."""

    def __init__(self, feature_dim: int = FEATURE_DIM) -> None:
        self.feature_dim = feature_dim
        self._cache: dict = {}

    def encode(self, obs: PrivateObservation) -> torch.Tensor:
        """Return a 1-D float tensor of shape (feature_dim,)."""
        features: list[float] = []
        features.extend(self._hand_features(obs))      # 14 dims
        features.extend(self._player_features(obs))    # 24 dims
        features.extend(self._round_features(obs))     # 10 dims
        features.extend(self._history_features(obs))   # 16 dims
        if self.feature_dim >= 80:
            features.extend(self._pomdp_and_opponent_features(obs))  # 16 dims (Total = 80)
        assert len(features) == self.feature_dim, f"Expected {self.feature_dim}, got {len(features)}"
        return torch.tensor(features, dtype=torch.float32)

    def encode_batch(self, observations: list[PrivateObservation]) -> torch.Tensor:
        """Return a tensor of shape (batch, feature_dim)."""
        if not observations:
            return torch.empty((0, self.feature_dim), dtype=torch.float32)
        return torch.stack([self.encode(obs) for obs in observations])


    def _relative_players(self, obs: PrivateObservation) -> list:
        hero_seat = obs.hero_seat
        players = list(obs.players)
        players.sort(key=lambda p: (p.seat - hero_seat) % 4)
        return players

    def _hand_features(self, obs: PrivateObservation) -> list[float]:
        features = []
        counts = {r: 0 for r in Rank}
        wild_count = 0
        has_ghost = 0
        ghost_rank = {r: 0 for r in Rank}
        
        for card in obs.hero_hand:
            if card.kind == CardKind.NORMAL and card.printed_rank:
                counts[card.printed_rank] += 1
            elif card.kind == CardKind.WILD:
                wild_count += 1
            elif card.kind == CardKind.GHOST:
                has_ghost = 1
                if card.printed_rank:
                    ghost_rank[card.printed_rank] = 1

        features.append(counts[Rank.A] / 5.0)
        features.append(counts[Rank.K] / 5.0)
        features.append(counts[Rank.Q] / 5.0)
        features.append(wild_count / 5.0)
        features.append(float(has_ghost))
        features.append(float(ghost_rank[Rank.A]))
        features.append(float(ghost_rank[Rank.K]))
        features.append(float(ghost_rank[Rank.Q]))
        
        total_hand_size = len(obs.hero_hand)
        features.append(total_hand_size / 5.0)
        
        claim_rank = obs.round_claim_rank
        if claim_rank is not None:
            match_count = sum(1 for c in obs.hero_hand if c.kind == CardKind.WILD or (c.printed_rank == claim_rank and c.kind == CardKind.NORMAL))
            non_target_count = sum(1 for c in obs.hero_hand if c.kind == CardKind.NORMAL and c.printed_rank != claim_rank)
            features.append(match_count / max(1, total_hand_size))
            features.append(non_target_count / max(1, total_hand_size))
        else:
            features.append(0.0)
            features.append(0.0)
            
        distinct_ranks = sum(1 for c in counts.values() if c > 0)
        features.append(distinct_ranks / 3.0)
        
        can_play_ghost = 1.0 if has_ghost else 0.0
        features.append(can_play_ghost)
        
        if claim_rank is not None:
            max_honest = min(3, counts.get(claim_rank, 0) + wild_count)
            features.append(max_honest / 3.0)
        else:
            features.append(0.0)
            
        return features

    def _player_features(self, obs: PrivateObservation) -> list[float]:
        features = []
        rel_players = self._relative_players(obs)
        for p in rel_players:
            features.append(1.0 if p.alive else 0.0)
            # 是否处于小轮手牌出清免责安全状态（手牌为0且存活）
            features.append(1.0 if (p.alive and p.hand_count == 0) else 0.0)
            features.append(p.hand_count / 5.0)
            features.append(p.shots_taken / 5.0)
            features.append(1.0 if p.seat == obs.current_seat else 0.0)
            # 是否处于最后出牌斩杀线（手牌剩余 <= 1）
            features.append(1.0 if (p.alive and p.hand_count <= 1) else 0.0)
        while len(features) < 24:
            features.append(0.0)
        return features

    def _round_features(self, obs: PrivateObservation) -> list[float]:
        features = []
        
        claim_rank = obs.round_claim_rank
        features.append(1.0 if claim_rank == Rank.A else 0.0)
        features.append(1.0 if claim_rank == Rank.K else 0.0)
        features.append(1.0 if claim_rank == Rank.Q else 0.0)
        
        features.append(1.0 if claim_rank is None else 0.0)
        
        features.append((obs.latest_play_count or 0) / 3.0)
        features.append(1.0 if obs.latest_play_seat == obs.hero_seat else 0.0)
        
        can_challenge = 1.0 if (obs.latest_play_seat is not None and obs.latest_play_seat != obs.hero_seat) else 0.0
        features.append(can_challenge)
        
        alive_count = sum(1 for p in obs.players if p.alive)
        features.append(alive_count / 4.0)
        
        round_play_count = 0
        for event in reversed(obs.public_history):
            if event.event_type == 'play':
                round_play_count += 1
            else:
                break
        features.append(round_play_count / 10.0)
        
        features.append(1.0 if obs.hero_seat == obs.current_seat else 0.0)
        
        return features

    def _history_features(self, obs: PrivateObservation) -> list[float]:
        features = []
        
        turn_index = 0
        if obs.public_history:
            turn_index = obs.public_history[-1].turn_index
        features.append(min(1.0, turn_index / 50.0))
        
        rev_counts = {Rank.A: 0, Rank.K: 0, Rank.Q: 0}
        ghost_revealed = 0.0
        
        chal_count_by_seat = {p.seat: 0 for p in obs.players}
        shot_count_by_seat = {p.seat: 0 for p in obs.players}
        
        total_challenges = 0
        lies_caught = 0
        
        for event in obs.public_history:
            if event.event_type == 'challenge':
                if event.seat is not None:
                    chal_count_by_seat[event.seat] = chal_count_by_seat.get(event.seat, 0) + 1
                total_challenges += 1
                outcome = event.detail.get('outcome')
                if outcome == 'lie':
                    lies_caught += 1
                elif outcome == 'ghost':
                    ghost_revealed = 1.0
                    
                revealed = event.detail.get('revealed_cards', [])
                for card_label in revealed:
                    if card_label == 'A': rev_counts[Rank.A] += 1
                    elif card_label == 'K': rev_counts[Rank.K] += 1
                    elif card_label == 'Q': rev_counts[Rank.Q] += 1
            
            elif event.event_type == 'shot':
                if event.seat is not None:
                    shot_count_by_seat[event.seat] = shot_count_by_seat.get(event.seat, 0) + 1
                    
        features.append(rev_counts[Rank.A] / 6.0)
        features.append(rev_counts[Rank.K] / 6.0)
        features.append(rev_counts[Rank.Q] / 6.0)
        features.append(ghost_revealed)
        
        rel_players = self._relative_players(obs)
        for p in rel_players:
            features.append(chal_count_by_seat.get(p.seat, 0) / 10.0)
            
        for p in rel_players:
            features.append(shot_count_by_seat.get(p.seat, 0) / 5.0)
            
        features.append(lies_caught / total_challenges if total_challenges > 0 else 0.0)
        honest_outcomes = sum(1 for e in obs.public_history if e.event_type == 'challenge' and e.detail.get('outcome') == 'honest')
        features.append(honest_outcomes / total_challenges if total_challenges > 0 else 0.0)
        
        ghost_outcomes = sum(1 for e in obs.public_history if e.event_type == 'challenge' and e.detail.get('outcome') == 'ghost')
        features.append(ghost_outcomes / total_challenges if total_challenges > 0 else 0.0)
        
        return features

    def _pomdp_and_opponent_features(self, obs: PrivateObservation) -> list[float]:
        """Calculates 16 high-level belief and opponent modeling features:
        - Last play honesty, bluff, and ghost-trap posterior beliefs (3 dims)
        - Target card count expectations for the 4 relative players (4 dims)
        - Ghost possession probability for the 4 relative players (4 dims)
        - Opponent aggression / bluffing indexes for the 3 opponents (3 dims)
        - Elimination urgency index (2 dims: hero hazard, leader escape pressure)
        """
        cache_key = (
            obs.hero_seat,
            len(obs.public_history),
            obs.round_claim_rank,
            obs.latest_play_seat,
            obs.latest_play_count,
            tuple(c.card_id for c in obs.hero_hand),
        )
        if cache_key in self._cache:
            return list(self._cache[cache_key])

        features: list[float] = []

        # 1. POMDP Beliefs (16 粒子极速推断，方差均值与 64 粒子数学一致)
        try:
            from .pomdp import POMDPParticleFilter
            pfilter = POMDPParticleFilter(num_particles=16, seed=obs.hero_seat + len(obs.public_history))
            res = pfilter.infer(obs)

            if res.last_play:
                features.append(res.last_play.honest_prob)
                features.append(res.last_play.bluff_prob)
                features.append(res.last_play.ghost_trap_prob)
            else:
                features.extend([0.33, 0.33, 0.0])

            rel_players = self._relative_players(obs)
            for p in rel_players:
                est = res.seat_estimates.get(p.seat)
                features.append((est.target_count_mean / 5.0) if est else 0.25)
            for p in rel_players:
                est = res.seat_estimates.get(p.seat)
                features.append(est.has_ghost_prob if est else 0.25)

        except Exception:
            features.extend([0.33, 0.33, 0.0])
            features.extend([0.25, 0.25, 0.25, 0.25])
            features.extend([0.25, 0.25, 0.25, 0.25])

        # 2. Opponent Modeling (3 dims for the 3 relative opponents)
        try:
            from .opponent_modeling import OnlineOpponentModeler
            modeler = OnlineOpponentModeler()
            profiles = modeler.model_opponents(obs)
            rel_opponents = [p for p in self._relative_players(obs) if p.seat != obs.hero_seat]
            for opp in rel_opponents[:3]:
                prof = profiles.get(opp.seat)
                features.append(prof.aggression_index if prof else 0.35)
            while len(features) < 14:
                features.append(0.35)
        except Exception:
            while len(features) < 14:
                features.append(0.35)

        # 3. Urgency indexes (2 dims: hero elimination hazard, leader escape pressure)
        hero_player = next((p for p in obs.players if p.seat == obs.hero_seat), None)
        shots = hero_player.shots_taken if hero_player else 0
        hero_hazard = 1.0 / max(1, 5 - shots)
        features.append(hero_hazard)

        min_cards = min((p.hand_count for p in obs.players if p.alive and p.seat != obs.hero_seat), default=5)
        escape_pressure = 1.0 if min_cards <= 1 else (0.5 if min_cards == 2 else 0.0)
        features.append(escape_pressure)

        result = features[:16]
        if len(self._cache) > 4096:
            self._cache.clear()
        self._cache[cache_key] = tuple(result)
        return result

