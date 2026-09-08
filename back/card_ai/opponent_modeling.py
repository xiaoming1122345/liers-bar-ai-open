from __future__ import annotations

"""
Online Opponent Modeling and Style Profiling.

Dynamically tracks each player's behavioral statistics from public game events:
- Bluff tendencies, honesty rates, ghost trapping frequencies
- Aggressiveness, challenge frequency, and tempo
- Post-shot pressure response (whether they tighten up or tilt aggressive after being shot)
"""

from collections import Counter
from dataclasses import dataclass
from typing import Any

from .style import infer_public_style_cluster, pressure_response_label_from_deltas
from .types import PrivateObservation, PublicEvent, Rank


@dataclass(frozen=True)
class OpponentStyleProfile:
    seat: int
    play_count: int
    challenge_count: int
    challenge_rate: float
    mean_play_count: float
    bluff_tendency: float
    honesty_tendency: float
    ghost_play_frequency: float
    aggression_index: float
    style_cluster: str
    pressure_response: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "seat": self.seat,
            "play_count": self.play_count,
            "challenge_count": self.challenge_count,
            "challenge_rate": round(self.challenge_rate, 3),
            "mean_play_count": round(self.mean_play_count, 2),
            "bluff_tendency": round(self.bluff_tendency, 3),
            "honesty_tendency": round(self.honesty_tendency, 3),
            "ghost_play_frequency": round(self.ghost_play_frequency, 3),
            "aggression_index": round(self.aggression_index, 3),
            "style_cluster": self.style_cluster,
            "pressure_response": self.pressure_response,
        }


class OnlineOpponentModeler:
    """
    Builds dynamic behavioral profiles for all opponents directly from observation history.
    """

    def __init__(self) -> None:
        pass

    def model_opponents(self, obs: PrivateObservation) -> dict[int, OpponentStyleProfile]:
        profiles: dict[int, OpponentStyleProfile] = {}

        play_counts = {p.seat: 0 for p in obs.players}
        play_cards_sum = {p.seat: 0 for p in obs.players}
        challenge_counts = {p.seat: 0 for p in obs.players}
        bluffs_detected = {p.seat: 0 for p in obs.players}
        honests_detected = {p.seat: 0 for p in obs.players}
        ghosts_detected = {p.seat: 0 for p in obs.players}

        last_play_seat: int | None = None

        for event in obs.public_history:
            if event.event_type == "play":
                seat = event.seat
                if seat is not None:
                    count = int(event.detail.get("count", 1))
                    play_counts[seat] = play_counts.get(seat, 0) + 1
                    play_cards_sum[seat] = play_cards_sum.get(seat, 0) + count
                    last_play_seat = seat

            elif event.event_type == "challenge":
                challenger = event.seat
                if challenger is not None:
                    challenge_counts[challenger] = challenge_counts.get(challenger, 0) + 1

                outcome = event.detail.get("outcome")
                if last_play_seat is not None:
                    if outcome == "lie":
                        bluffs_detected[last_play_seat] = bluffs_detected.get(last_play_seat, 0) + 1
                    elif outcome == "honest":
                        honests_detected[last_play_seat] = honests_detected.get(last_play_seat, 0) + 1
                    elif outcome == "ghost":
                        ghosts_detected[last_play_seat] = ghosts_detected.get(last_play_seat, 0) + 1
                last_play_seat = None

        for player in obs.players:
            seat = player.seat
            p_cnt = play_counts[seat]
            c_cnt = challenge_counts[seat]
            total_actions = p_cnt + c_cnt
            chal_rate = c_cnt / max(1, total_actions)
            mean_play = play_cards_sum[seat] / max(1, p_cnt) if p_cnt > 0 else 1.5

            revealed_trials = bluffs_detected[seat] + honests_detected[seat] + ghosts_detected[seat]
            if revealed_trials > 0:
                bluff_tend = bluffs_detected[seat] / revealed_trials
                honest_tend = honests_detected[seat] / revealed_trials
                ghost_freq = ghosts_detected[seat] / revealed_trials
            else:
                bluff_tend = 0.30
                honest_tend = 0.60
                ghost_freq = 0.10

            aggression = 0.5 * chal_rate + 0.3 * (mean_play / 3.0) + 0.2 * bluff_tend
            cluster = infer_public_style_cluster(
                challenge_rate=chal_rate,
                mean_play_count=mean_play,
                variability_score=0.45,
            )

            # Pressure response after being shot
            pressure_resp = "stable"
            if player.shots_taken >= 2:
                pressure_resp = "tilts_aggressive" if chal_rate > 0.30 else "tightens_up"

            profiles[seat] = OpponentStyleProfile(
                seat=seat,
                play_count=p_cnt,
                challenge_count=c_cnt,
                challenge_rate=chal_rate,
                mean_play_count=mean_play,
                bluff_tendency=bluff_tend,
                honesty_tendency=honest_tend,
                ghost_play_frequency=ghost_freq,
                aggression_index=aggression,
                style_cluster=cluster,
                pressure_response=pressure_resp,
            )

        return profiles
