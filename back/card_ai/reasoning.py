from __future__ import annotations

"""
Multi-Level Game Theory Reasoning (Level-k) and 1-Step Lookahead Rollout.

Evaluates candidate actions by simulating opponent reactions and predicting
expected outcomes based on POMDP beliefs.
"""

from dataclasses import dataclass
from typing import Any

from .pomdp import POMDPInferenceResult
from .types import PrivateObservation, Rank


@dataclass(frozen=True)
class OpponentReactionPrediction:
    responder_seat: int
    challenge_prob: float
    pass_prob: float
    counter_play_prob: float
    most_likely_response: str


@dataclass(frozen=True)
class ActionLookaheadEvaluation:
    action_type: str
    action_label: str
    immediate_risk: float        # Risk of being challenged or eliminated immediately
    projected_survival_rate: float # Chance of surviving the round
    expected_value: float        # Estimated game EV score
    predicted_reaction: OpponentReactionPrediction | None
    tactical_summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type,
            "action_label": self.action_label,
            "immediate_risk": round(self.immediate_risk, 3),
            "projected_survival_rate": round(self.projected_survival_rate, 3),
            "expected_value": round(self.expected_value, 3),
            "predicted_reaction": {
                "responder_seat": self.predicted_reaction.responder_seat,
                "challenge_prob": round(self.predicted_reaction.challenge_prob, 3),
                "pass_prob": round(self.predicted_reaction.pass_prob, 3),
                "counter_play_prob": round(self.predicted_reaction.counter_play_prob, 3),
                "most_likely_response": self.predicted_reaction.most_likely_response,
            } if self.predicted_reaction else None,
            "tactical_summary": self.tactical_summary,
        }


class LookaheadEngine:
    """
    Simulates Level-1 / Level-2 game tree lookahead for candidate hero actions.
    """

    def __init__(self) -> None:
        pass

    def evaluate_lookahead(
        self,
        observation: PrivateObservation,
        pomdp_result: POMDPInferenceResult,
        candidate_options: list[dict[str, Any]],
        next_responder_seat: int | None,
        opponent_profiles: dict[int, Any] | None = None,
    ) -> list[ActionLookaheadEvaluation]:
        evaluations: list[ActionLookaheadEvaluation] = []
        hero_player = next((p for p in observation.players if p.seat == observation.hero_seat), None)
        shots_taken = hero_player.shots_taken if hero_player else 0
        base_elim_hazard = 1.0 / max(1, 5 - shots_taken)

        responder_estimate = (
            pomdp_result.seat_estimates.get(next_responder_seat)
            if next_responder_seat is not None
            else None
        )
        responder_profile = (
            opponent_profiles.get(next_responder_seat)
            if opponent_profiles and next_responder_seat is not None
            else None
        )

        for opt in candidate_options:
            action_type = opt.get("action_type", "")
            action_label = opt.get("label", "")

            if action_type == "challenge":
                eval_item = self._eval_challenge(
                    opt=opt,
                    pomdp_result=pomdp_result,
                    base_elim_hazard=base_elim_hazard,
                )
            elif "play" in action_type:
                eval_item = self._eval_play(
                    opt=opt,
                    responder_estimate=responder_estimate,
                    responder_profile=responder_profile,
                    next_responder_seat=next_responder_seat,
                    base_elim_hazard=base_elim_hazard,
                )
            else:
                eval_item = ActionLookaheadEvaluation(
                    action_type=action_type,
                    action_label=action_label,
                    immediate_risk=0.0,
                    projected_survival_rate=1.0 - base_elim_hazard * 0.5,
                    expected_value=0.0,
                    predicted_reaction=None,
                    tactical_summary="观望等待局面变化",
                )

            evaluations.append(eval_item)

        # Sort by expected_value descending
        evaluations.sort(key=lambda x: x.expected_value, reverse=True)
        return evaluations

    def _eval_challenge(
        self,
        opt: dict[str, Any],
        pomdp_result: POMDPInferenceResult,
        base_elim_hazard: float,
    ) -> ActionLookaheadEvaluation:
        last_play = pomdp_result.last_play
        if last_play is None:
            bluff_p = 0.35
            ghost_p = 0.10
        else:
            bluff_p = last_play.bluff_prob
            ghost_p = last_play.ghost_trap_prob

        honest_p = max(0.0, 1.0 - bluff_p - ghost_p)

        immediate_risk = honest_p + ghost_p
        survival_rate = max(0.05, 1.0 - immediate_risk * base_elim_hazard)
        ev = bluff_p * 15.0 - (honest_p + ghost_p) * (15.0 + 30.0 * base_elim_hazard)

        if bluff_p > 0.60:
            summary = f"高把握质疑：抓诈唬概率 {round(bluff_p*100)}%，值得出击"
        elif ghost_p > 0.30:
            summary = f"极高风险：怀疑上家出鬼牌伏击（{round(ghost_p*100)}%），慎防中招"
        else:
            summary = f"平稳对局：胜算约 {round(bluff_p*100)}%，失败将面临下一枪淘汰风险"

        return ActionLookaheadEvaluation(
            action_type="challenge",
            action_label=opt.get("label", "质疑上一手"),
            immediate_risk=immediate_risk,
            projected_survival_rate=survival_rate,
            expected_value=ev,
            predicted_reaction=None,
            tactical_summary=summary,
        )

    def _eval_play(
        self,
        opt: dict[str, Any],
        responder_estimate: Any | None,
        responder_profile: Any | None,
        next_responder_seat: int | None,
        base_elim_hazard: float,
    ) -> ActionLookaheadEvaluation:
        action_type = opt.get("action_type", "")
        label = opt.get("label", "")

        is_bluff = "bluff" in action_type
        is_mix = "mix" in action_type
        is_ghost = "ghost" in action_type
        is_honest = "honest" in action_type

        # Next responder challenge probability estimation
        # Modulated by opponent's empirical aggression and POMDP card density
        responder_targets = responder_estimate.target_count_mean if responder_estimate else 1.2
        target_card_density_factor = min(1.0, responder_targets / 2.5)

        # Base style tendency modifier
        aggression_bonus = 0.0
        if responder_profile:
            aggression_bonus = (responder_profile.aggression_index - 0.35) * 0.3

        if is_honest:
            challenge_p = 0.15 + 0.15 * target_card_density_factor + aggression_bonus
            immediate_risk = 0.0  # Even if challenged, hero is honest -> survives!
            summary = "稳健诚实牌：若遭质疑对方中枪，我方零风险"
            ev = 8.0 + 5.0 * challenge_p  # Extra value if opponent mistakenly challenges
        elif is_ghost:
            challenge_p = 0.25 + 0.20 * target_card_density_factor + aggression_bonus
            immediate_risk = 0.0  # Ghost trap triggers against challenger
            summary = "鬼牌诱捕：一旦下家质疑将引发鬼牌惩罚"
            ev = 12.0 + 10.0 * challenge_p
        elif is_bluff:
            challenge_p = 0.35 + 0.35 * target_card_density_factor + aggression_bonus
            immediate_risk = challenge_p  # If challenged, hero takes shot
            summary = f"虚张声势：下家识破并质疑概率约 {round(challenge_p*100)}%"
            ev = (1.0 - challenge_p) * 10.0 - challenge_p * (15.0 + 35.0 * base_elim_hazard)
        else:  # Mix
            challenge_p = 0.25 + 0.25 * target_card_density_factor + aggression_bonus
            immediate_risk = challenge_p * 0.5
            summary = f"半虚半实：部分卡牌匹配，下家质疑率约 {round(challenge_p*100)}%"
            ev = (1.0 - challenge_p) * 6.0 - challenge_p * 8.0

        challenge_p = max(0.05, min(0.95, challenge_p))
        pass_p = (1.0 - challenge_p) * 0.7
        counter_p = (1.0 - challenge_p) * 0.3

        if challenge_p > max(pass_p, counter_p):
            most_likely = "倾向发起质疑"
        elif counter_p > pass_p:
            most_likely = "倾向继续跟出"
        else:
            most_likely = "倾向弃权/保守过牌"

        reaction = (
            OpponentReactionPrediction(
                responder_seat=next_responder_seat,
                challenge_prob=challenge_p,
                pass_prob=pass_p,
                counter_play_prob=counter_p,
                most_likely_response=most_likely,
            )
            if next_responder_seat is not None
            else None
        )

        survival_rate = max(0.1, 1.0 - immediate_risk * base_elim_hazard)

        return ActionLookaheadEvaluation(
            action_type=action_type,
            action_label=label,
            immediate_risk=immediate_risk,
            projected_survival_rate=survival_rate,
            expected_value=ev,
            predicted_reaction=reaction,
            tactical_summary=summary,
        )
