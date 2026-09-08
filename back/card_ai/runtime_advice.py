from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Any

from .response_model import StyleResponseModel


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _string(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + exp(-value))


@dataclass(frozen=True)
class SeatRuntimeView:
    seat: int
    player_id: str
    display_name: str
    status: str
    shots_taken: int
    hand_count: int
    last_action: str
    target_prob: float
    non_target_prob: float
    ghost_prob: float
    wild_prob: float
    public_style_cluster: str
    inferred_profile_name: str
    pressure_response_label: str
    bluff_rate: float
    challenge_rate: float
    aggression_score: float
    variability_score: float
    post_shot_bluff_delta: float
    post_shot_challenge_delta: float
    pressure_score: float
    caution_score: float
    desperation_score: float
    challenge_pressure: float
    bluff_pressure: float
    elimination_hazard: float
    threat_score: float
    target_card_estimate: float
    likely_bucket: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "seat": self.seat,
            "player_id": self.player_id,
            "display_name": self.display_name,
            "status": self.status,
            "shots_taken": self.shots_taken,
            "hand_count": self.hand_count,
            "last_action": self.last_action,
            "target_prob": self.target_prob,
            "non_target_prob": self.non_target_prob,
            "ghost_prob": self.ghost_prob,
            "wild_prob": self.wild_prob,
            "public_style_cluster": self.public_style_cluster,
            "inferred_profile_name": self.inferred_profile_name,
            "pressure_response_label": self.pressure_response_label,
            "bluff_rate": self.bluff_rate,
            "challenge_rate": self.challenge_rate,
            "aggression_score": self.aggression_score,
            "variability_score": self.variability_score,
            "post_shot_bluff_delta": self.post_shot_bluff_delta,
            "post_shot_challenge_delta": self.post_shot_challenge_delta,
            "pressure_score": self.pressure_score,
            "caution_score": self.caution_score,
            "desperation_score": self.desperation_score,
            "challenge_pressure": self.challenge_pressure,
            "bluff_pressure": self.bluff_pressure,
            "elimination_hazard": self.elimination_hazard,
            "threat_score": self.threat_score,
            "target_card_estimate": self.target_card_estimate,
            "likely_bucket": self.likely_bucket,
        }


class RuntimeAdvisor:
    def __init__(
        self,
        response_model: StyleResponseModel | None = None,
        neural_checkpoint: str | None = None,
    ) -> None:
        self.response_model = response_model or StyleResponseModel.default()
        self._neural_bridge = None
        from .neural_bridge import NeuralAdvisorBridge
        self._neural_bridge = NeuralAdvisorBridge(neural_checkpoint)

    def analyze_snapshot(self, payload: dict[str, Any]) -> dict[str, Any]:
        session = payload.get("session", payload)
        seats = session.get("seats", [])
        history = session.get("history", [])
        my_seat = _safe_int(session.get("mySeat"), 1)
        current_turn_seat = _safe_int(session.get("currentTurnSeat"), 1)
        hero_hand = session.get("heroHandView", {})
        hero_view = self._hero_hand_view(hero_hand)

        seat_views = [
            self._seat_view(
                seat_payload=seat,
                history=history,
            )
            for seat in seats
        ]
        seat_by_number = {seat.seat: seat for seat in seat_views}

        latest_play = next((event for event in history if event.get("type") == "play"), None)
        latest_challenge = next((event for event in history if event.get("type") == "challenge"), None)
        hero_state = seat_by_number.get(my_seat)
        next_live_seat = self._next_active_seat(seat_views, my_seat)
        next_state = seat_by_number.get(next_live_seat)

        action_options = self._action_options(
            my_seat=my_seat,
            current_turn_seat=current_turn_seat,
            hero_state=hero_state,
            next_state=next_state,
            latest_play=latest_play,
            latest_challenge=latest_challenge,
            target_rank=_string(session.get("targetRank"), "A"),
            stack_count=_safe_int(session.get("stackCount"), 0),
            hero_hand=hero_view,
            seat_by_number=seat_by_number,
        )
        recommended = action_options[0] if action_options else {
            "label": "等待",
            "action_type": "observe",
            "score": 0.0,
            "confidence": 0.5,
            "reasons": ["当前不是我方操作，先继续录入桌面信息。"],
        }

        modeling_degree = self._modeling_degree(seat_views, my_seat)
        result = {
            "analysis_version": 2,
            "my_seat": my_seat,
            "current_turn_seat": current_turn_seat,
            "target_rank": _string(session.get("targetRank"), "A"),
            "stack_count": _safe_int(session.get("stackCount"), 0),
            "hero_hand_view": hero_view,
            "recommended_action": recommended,
            "action_options": action_options[:5],
            "seat_analysis": [seat.to_dict() for seat in seat_views],
            "table_factors": {
                "modeling_degree": modeling_degree,
                "next_responder_seat": next_live_seat,
                "latest_play_seat": _safe_int(latest_play.get("seat"), 0) if latest_play else None,
                "latest_play_claim_rank": latest_play.get("claimRank") if latest_play else None,
                "latest_play_count": _safe_int(latest_play.get("count"), 0) if latest_play else 0,
            },
        }

        # Build PrivateObservation for advanced POMDP & Lookahead reasoning
        obs = None
        try:
            from .neural_bridge import payload_to_observation
            obs = payload_to_observation(payload)
        except Exception:
            pass

        # 1. POMDP Bayesian Particle Filter Inference & Opponent Modeling
        if obs is not None:
            try:
                from .pomdp import POMDPParticleFilter
                from .opponent_modeling import OnlineOpponentModeler
                
                pomdp_filter = POMDPParticleFilter(num_particles=512, seed=42)
                pomdp_res = pomdp_filter.infer(obs)
                result["pomdp_inference"] = pomdp_res.to_dict()

                modeler = OnlineOpponentModeler()
                opp_profiles = modeler.model_opponents(obs)
                result["opponent_profiles"] = {
                    str(seat): prof.to_dict() for seat, prof in opp_profiles.items()
                }

                # 2. Level-k & 1-Step Lookahead Rollout
                from .reasoning import LookaheadEngine
                lookahead_engine = LookaheadEngine()
                lookahead_evals = lookahead_engine.evaluate_lookahead(
                    observation=obs,
                    pomdp_result=pomdp_res,
                    candidate_options=action_options,
                    next_responder_seat=next_live_seat,
                    opponent_profiles=opp_profiles,
                )
                result["lookahead_tree"] = [item.to_dict() for item in lookahead_evals[:5]]
            except Exception:
                pass


        # 生成确定性局面指纹 (用于保证界面多次刷新采样动作严格幂等)
        import hashlib
        hand_key = f"{hero_view.get('targetCount')}_{hero_view.get('nonTargetCount')}_{hero_view.get('ghostCount')}_{hero_view.get('wildCount')}"
        seats_key = "_".join(f"{s.seat}:{s.status}:{s.shots_taken}:{s.hand_count}" for s in seat_views)
        last_play_key = f"{latest_play.get('seat')}:{latest_play.get('count')}" if latest_play else "none"
        raw_fp = f"s{my_seat}_turn{current_turn_seat}_tgt{session.get('targetRank')}_hand{hand_key}_seats{seats_key}_last{last_play_key}"
        state_fingerprint = hashlib.md5(raw_fp.encode()).hexdigest()[:16]

        neural_result = None
        # 3. Neural network advice with manifest & idempotent sampling
        if self._neural_bridge is not None:
            if not self._neural_bridge.is_valid_model:
                result["neural_advice"] = {
                    "is_valid": False,
                    "validation_message": self._neural_bridge.validation_message,
                    "manifest": getattr(self._neural_bridge, "_manifest", {}),
                }
            elif obs is not None:
                try:
                    neural_result = self._neural_bridge.advise(obs, state_fingerprint=state_fingerprint)
                    if neural_result is not None:
                        top_actions = sorted(
                            neural_result.action_probs.items(),
                            key=lambda x: x[1],
                            reverse=True,
                        )[:5]
                        result["neural_advice"] = {
                            "is_valid": True,
                            "model_id": neural_result.model_manifest.get("model_id"),
                            "iteration": neural_result.model_manifest.get("iteration"),
                            "short_sha256": neural_result.model_manifest.get("short_sha256"),
                            "full_file_sha256": neural_result.model_manifest.get("expected_file_sha256"),
                            "tensor_sha256": neural_result.model_manifest.get("expected_tensor_sha256"),
                            "prob_semantics": neural_result.prob_semantics,
                            "sampled_action": neural_result.sampled_action_label,
                            "sampled_prob": round(neural_result.sampled_action_prob, 4),
                            "top_action": neural_result.top_action_label,
                            "top_prob": round(neural_result.top_action_prob, 4),
                            "action_selection_probs": [
                                {
                                    "label": label,
                                    "prob": round(prob, 4),
                                    "prob_percent": f"{prob*100:.1f}%",
                                    "is_sampled": (label == neural_result.sampled_action_label),
                                }
                                for label, prob in top_actions
                            ],
                            "state_fingerprint": state_fingerprint,
                        }
                except Exception as e:
                    pass

        # 4. 判定动作交互对象 (实际接牌者 vs 质疑对象)
        chosen_action_label = ""
        if neural_result is not None:
            chosen_action_label = neural_result.sampled_action_label
        elif recommended:
            chosen_action_label = recommended.get("label", "")

        interaction_target = self._compute_interaction_target(
            seat_views=seat_views,
            my_seat=my_seat,
            action_label=chosen_action_label,
            latest_play_seat=_safe_int(latest_play.get("seat"), 0) if latest_play else None,
        )
        result["interaction_target"] = interaction_target
        result["state_fingerprint"] = state_fingerprint

        return result

    def _hero_hand_view(self, payload: dict[str, Any]) -> dict[str, int]:
        target_count = max(0, _safe_int(payload.get("targetCount"), 0))
        non_target_count = max(0, _safe_int(payload.get("nonTargetCount"), 0))
        ghost_count = max(0, _safe_int(payload.get("ghostCount"), 0))
        wild_count = max(0, _safe_int(payload.get("wildCount"), 0))
        return {
            "targetCount": target_count,
            "nonTargetCount": non_target_count,
            "ghostCount": ghost_count,
            "wildCount": wild_count,
            "supportCount": target_count + ghost_count + wild_count,
        }

    def _seat_view(
        self,
        seat_payload: dict[str, Any],
        history: list[dict[str, Any]],
    ) -> SeatRuntimeView:
        seat = _safe_int(seat_payload.get("seat"), 1)
        metrics = seat_payload.get("metrics") or {}
        beliefs = seat_payload.get("manualBeliefs") or {}
        shots_taken = _safe_int(seat_payload.get("shotsTaken"), 0)
        hand_count = _safe_int(seat_payload.get("handCount"), 0)
        status = _string(seat_payload.get("status"), "alive")

        sample_count = _safe_int(metrics.get("sampleCount"), 0)
        prior_style = _string(metrics.get("publicStyleCluster"), "")
        if not prior_style:
            prior_style = self._derive_style_from_beliefs(
                challenge_rate=_safe_float(metrics.get("challengeRate"), 0.18),
                variability_score=_safe_float(metrics.get("variabilityScore"), 0.45),
                target_prob=_safe_float(beliefs.get("target"), 50.0) / 100.0,
                hand_count=hand_count,
            )
        recent_shot_age = self._recent_event_age(history, "shot", seat)
        prior_bluff_delta, prior_challenge_delta, prior_response = self.response_model.predict(
            prior_style,
            recent_shot_age,
        )

        blend = _clamp(sample_count / 3.0)
        metric_bluff_delta = _safe_float(metrics.get("postShotBluffDelta"), 0.0)
        metric_challenge_delta = _safe_float(metrics.get("postShotChallengeDelta"), 0.0)
        bluff_delta = blend * metric_bluff_delta + (1.0 - blend) * prior_bluff_delta
        challenge_delta = blend * metric_challenge_delta + (1.0 - blend) * prior_challenge_delta
        pressure_response_label = _string(metrics.get("pressureResponseLabel"), prior_response) or prior_response

        target_prob = _clamp(_safe_float(beliefs.get("target"), 50.0) / 100.0)
        non_target_prob = _clamp(_safe_float(beliefs.get("nonTarget"), 50.0) / 100.0)
        ghost_prob = _clamp(_safe_float(beliefs.get("ghost"), 10.0) / 100.0)
        wild_prob = _clamp(_safe_float(beliefs.get("wild"), 10.0) / 100.0)

        pressure_score = _clamp(
            0.48 * (shots_taken / 4.0 if shots_taken < 5 else 1.0)
            + 0.28 * (1.0 if hand_count <= 2 else 0.0)
            + 0.24 * (1.0 if recent_shot_age is not None and recent_shot_age <= 4 else 0.0)
        )
        bluff_rate = _clamp(_safe_float(metrics.get("bluffRate"), 0.30) + bluff_delta)
        challenge_rate = _clamp(_safe_float(metrics.get("challengeRate"), 0.18) + challenge_delta)
        aggression_score = _clamp(_safe_float(metrics.get("aggressionScore"), 0.35))
        variability_score = _clamp(_safe_float(metrics.get("variabilityScore"), 0.45))
        caution_score = _clamp(
            0.50 * pressure_score
            + 0.30 * (1.0 - challenge_rate)
            + 0.20 * max(0.0, -(bluff_delta + challenge_delta))
        )
        desperation_score = _clamp(
            0.42 * pressure_score
            + 0.28 * challenge_rate
            + 0.18 * (1.0 if hand_count <= 2 else 0.0)
            + 0.12 * max(0.0, bluff_delta + challenge_delta)
        )
        challenge_pressure = _clamp(challenge_rate * (0.70 + 0.30 * pressure_score))
        bluff_pressure = _clamp(bluff_rate * (0.55 + 0.45 * desperation_score))
        elimination_hazard = self._elimination_hazard(shots_taken, status)
        threat_score = _clamp(
            0.34 * challenge_pressure
            + 0.26 * bluff_pressure
            + 0.20 * aggression_score
            + 0.20 * elimination_hazard
        )
        target_card_estimate = target_prob * hand_count
        likely_bucket = self._likely_bucket(
            status=status,
            hand_count=hand_count,
            target_prob=target_prob,
            non_target_prob=non_target_prob,
            ghost_prob=ghost_prob,
        )
        return SeatRuntimeView(
            seat=seat,
            player_id=_string(seat_payload.get("playerId")),
            display_name=_string(seat_payload.get("displayName"), f"玩家{seat}"),
            status=status,
            shots_taken=shots_taken,
            hand_count=hand_count,
            last_action=_string(seat_payload.get("lastAction")),
            target_prob=target_prob,
            non_target_prob=non_target_prob,
            ghost_prob=ghost_prob,
            wild_prob=wild_prob,
            public_style_cluster=prior_style,
            inferred_profile_name=_string(metrics.get("inferredProfileName"), "unknown"),
            pressure_response_label=pressure_response_label,
            bluff_rate=bluff_rate,
            challenge_rate=challenge_rate,
            aggression_score=aggression_score,
            variability_score=variability_score,
            post_shot_bluff_delta=bluff_delta,
            post_shot_challenge_delta=challenge_delta,
            pressure_score=pressure_score,
            caution_score=caution_score,
            desperation_score=desperation_score,
            challenge_pressure=challenge_pressure,
            bluff_pressure=bluff_pressure,
            elimination_hazard=elimination_hazard,
            threat_score=threat_score,
            target_card_estimate=target_card_estimate,
            likely_bucket=likely_bucket,
        )

    def _compute_interaction_target(
        self,
        seat_views: list[SeatRuntimeView],
        my_seat: int,
        action_label: str,
        latest_play_seat: int | None,
    ) -> dict[str, Any]:
        """严格依据游戏引擎规则判定动作交互目标：
        - 若动作为质疑 (challenge)：目标为上一出牌者（质疑对象），明确顺时针相对位置；
        - 若动作为出牌 (play)：从我方顺时针寻找下一个存活且未脱身玩家（实际接牌者）；
          严禁将手牌为0但处于出空待回应（pending_escape）的玩家误判为脱身！
        - 若信息不足：返回状态 "待确认"。
        """
        seat_by_number = {s.seat: s for s in seat_views}
        ordered = sorted(seat_views, key=lambda item: item.seat)
        seats = [item.seat for item in ordered]

        def get_relative_pos(from_s: int, to_s: int) -> str:
            diff = (to_s - from_s) % 4
            if diff == 1:
                return "顺时针直接下家"
            elif diff == 2:
                return "对家"
            elif diff == 3:
                return "直接上家"
            return "自身"

        if action_label.startswith("challenge") or action_label == "质疑":
            if latest_play_seat is not None and latest_play_seat != my_seat:
                target_view = seat_by_number.get(latest_play_seat)
                return {
                    "role": "质疑对象",
                    "seat": latest_play_seat,
                    "display_name": target_view.display_name if target_view else f"{latest_play_seat}号位",
                    "relative_pos": get_relative_pos(my_seat, latest_play_seat),
                    "status_description": f"正在审判其打出的上一手牌",
                    "is_confirmed": True,
                }
            return {
                "role": "质疑对象",
                "seat": None,
                "display_name": "待确认",
                "relative_pos": "未知",
                "status_description": "当前暂无其他玩家打出的合法牌面供质疑",
                "is_confirmed": False,
            }

        # 出牌动作：顺时针寻找真正接牌者
        if my_seat in seats:
            start_index = seats.index(my_seat)
            skipped_seats = []
            for offset in range(1, len(seats)):
                candidate = ordered[(start_index + offset) % len(seats)]
                # 只有真正阵亡淘汰的玩家才被过滤
                is_eliminated = candidate.status in ("dead", "eliminated") or (candidate.shots_taken >= 5)
                if is_eliminated:
                    continue
                # 已经安全脱身判定：
                # 只有明确标记为 "escaped" 的玩家才是安全脱身！
                # 若玩家处于 "pending_escape" (手牌刚出空，正待下家回应)，仍有被质疑中弹风险，绝不能当作安全脱身跳过！
                if candidate.status == "escaped":
                    skipped_seats.append(candidate.seat)
                    continue
                # 找到真正的实际接牌者
                rel = get_relative_pos(my_seat, candidate.seat)
                return {
                    "role": "实际接牌者",
                    "seat": candidate.seat,
                    "display_name": candidate.display_name,
                    "relative_pos": rel,
                    "hand_count": candidate.hand_count,
                    "shots_taken": candidate.shots_taken,
                    "skipped_escaped_seats": skipped_seats,
                    "status_description": f"{rel}接牌审判" + (f" (已跳过脱身者: {skipped_seats})" if skipped_seats else ""),
                    "is_confirmed": True,
                }

        return {
            "role": "实际接牌者",
            "seat": None,
            "display_name": "待确认",
            "relative_pos": "未知",
            "status_description": "桌面玩家存活与脱身信息不全",
            "is_confirmed": False,
        }

    def _derive_style_from_beliefs(
        self,
        *,
        challenge_rate: float,
        variability_score: float,
        target_prob: float,
        hand_count: int,
    ) -> str:
        if challenge_rate >= 0.35:
            return "aggressive"
        if variability_score >= 0.80:
            return "volatile"
        if hand_count <= 2 and target_prob >= 0.55:
            return "tempo"
        if target_prob >= 0.60 and variability_score <= 0.30:
            return "reserved"
        return "balanced"

    def _recent_event_age(
        self,
        history: list[dict[str, Any]],
        event_type: str,
        seat: int,
    ) -> int | None:
        for age, event in enumerate(history):
            if event.get("type") == event_type and _safe_int(event.get("targetSeat") or event.get("seat")) == seat:
                return age
        return None

    def _elimination_hazard(self, shots_taken: int, status: str) -> float:
        if status != "alive":
            return 1.0
        if shots_taken >= 5:
            return 1.0
        return 1.0 / max(1, 5 - shots_taken)

    def _likely_bucket(
        self,
        *,
        status: str,
        hand_count: int,
        target_prob: float,
        non_target_prob: float,
        ghost_prob: float,
    ) -> str:
        if status != "alive":
            return status
        if hand_count == 0:
            return "empty"
        if ghost_prob >= 0.35:
            return "ghost-live"
        if target_prob >= 0.60:
            return "target-heavy"
        if non_target_prob >= 0.60:
            return "bluff-ready"
        return "mixed"

    def _next_active_seat(self, seat_views: list[SeatRuntimeView], seat: int) -> int | None:
        ordered = sorted(seat_views, key=lambda item: item.seat)
        seats = [item.seat for item in ordered]
        if seat not in seats:
            return None
        start_index = seats.index(seat)
        for offset in range(1, len(seats) + 1):
            candidate = ordered[(start_index + offset) % len(seats)]
            if candidate.status == "alive":
                return candidate.seat
        return None

    def _modeling_degree(self, seat_views: list[SeatRuntimeView], my_seat: int) -> float:
        scores = []
        for seat in seat_views:
            if seat.seat == my_seat:
                continue
            score = 0.0
            if seat.player_id:
                score += 0.25
            if seat.public_style_cluster != "balanced":
                score += 0.25
            score += 0.20 * seat.variability_score
            score += 0.15 * max(seat.target_prob, seat.non_target_prob)
            score += 0.15 * max(seat.ghost_prob, seat.wild_prob)
            scores.append(_clamp(score))
        if not scores:
            return 0.0
        return sum(scores) / len(scores)

    def _challenge_option(
        self,
        *,
        hero_state: SeatRuntimeView,
        actor_state: SeatRuntimeView,
        latest_play: dict[str, Any],
        hero_hand: dict[str, int],
        target_rank: str,
    ) -> dict[str, Any]:
        claim_count = _safe_int(latest_play.get("count"), 1)
        support_pressure = 1.0 if hero_hand["supportCount"] == 0 else _clamp(1.0 - hero_hand["supportCount"] / max(1, claim_count))
        suspicion = _clamp(
            0.38 * actor_state.bluff_rate
            + 0.22 * (1.0 - _clamp(actor_state.target_card_estimate / max(1.0, actor_state.hand_count)))
            + 0.16 * actor_state.desperation_score
            + 0.14 * (claim_count / 3.0)
            + 0.10 * max(0.0, actor_state.post_shot_bluff_delta + actor_state.post_shot_challenge_delta)
        )
        score = (
            1.15 * suspicion
            + 0.28 * support_pressure
            - 0.48 * hero_state.elimination_hazard
            - 0.18 * hero_state.pressure_score
        )
        reasons = [
            f"{actor_state.display_name} 当前诈牌压力约 {round(actor_state.bluff_pressure * 100)}%",
            f"其该点数支持估计约 {actor_state.target_card_estimate:.1f} 张",
            f"你自己的下一枪危险率约 {round(hero_state.elimination_hazard * 100)}%",
            f"当前声明 {target_rank} × {claim_count}",
        ]
        return {
            "label": "质疑上一手",
            "action_type": "challenge",
            "score": round(score, 4),
            "confidence": 0.0,
            "reasons": reasons,
        }

    def _play_option(
        self,
        *,
        label: str,
        action_type: str,
        honest_ratio: float,
        bluff_ratio: float,
        count: int,
        remaining_cards: int,
        hero_state: SeatRuntimeView,
        next_state: SeatRuntimeView | None,
        ghost_bonus: float = 0.0,
    ) -> dict[str, Any]:
        next_challenge = next_state.challenge_pressure if next_state is not None else 0.20
        next_caution = next_state.caution_score if next_state is not None else 0.30
        next_desperation = next_state.desperation_score if next_state is not None else 0.25
        score = (
            0.18
            + 0.32 * honest_ratio
            + 0.16 * (count / 3.0)
            + 0.20 * next_caution * bluff_ratio
            - 0.48 * next_challenge * bluff_ratio
            + 0.18 * next_desperation * honest_ratio
            - 0.28 * hero_state.pressure_score * bluff_ratio
            + 0.16 * hero_state.pressure_score * honest_ratio
            + ghost_bonus
            + (0.55 if remaining_cards == 0 else 0.0)
        )
        reasons = [
            f"下家质疑压力约 {round(next_challenge * 100)}%",
            f"下家谨慎度约 {round(next_caution * 100)}%",
            f"你自身危险率约 {round(hero_state.elimination_hazard * 100)}%",
        ]
        return {
            "label": label,
            "action_type": action_type,
            "score": round(score, 4),
            "confidence": 0.0,
            "reasons": reasons,
        }

    def _action_options(
        self,
        *,
        my_seat: int,
        current_turn_seat: int,
        hero_state: SeatRuntimeView | None,
        next_state: SeatRuntimeView | None,
        latest_play: dict[str, Any] | None,
        latest_challenge: dict[str, Any] | None,
        target_rank: str,
        stack_count: int,
        hero_hand: dict[str, int],
        seat_by_number: dict[int, SeatRuntimeView],
    ) -> list[dict[str, Any]]:
        if hero_state is None:
            return []

        if current_turn_seat != my_seat:
            acting_state = seat_by_number.get(current_turn_seat)
            if acting_state is None:
                return []
            watch_score = 0.15 + 0.45 * acting_state.threat_score + 0.20 * acting_state.bluff_pressure
            return [
                {
                    "label": f"观察 {acting_state.display_name}",
                    "action_type": "observe",
                    "score": round(watch_score, 4),
                    "confidence": 0.62,
                    "reasons": [
                        f"当前轮到 {acting_state.display_name}",
                        f"其桌面威胁约 {round(acting_state.threat_score * 100)}%",
                        f"先继续录入动作，等轮到你再做出牌/质疑决策。",
                    ],
                }
            ]

        options: list[dict[str, Any]] = []
        support_count = hero_hand["supportCount"]
        target_count = hero_hand["targetCount"]
        non_target_count = hero_hand["nonTargetCount"]
        ghost_count = hero_hand["ghostCount"]
        total_cards = support_count + non_target_count

        if latest_play is not None and _safe_int(latest_play.get("seat"), 0) != my_seat:
            actor_state = seat_by_number.get(_safe_int(latest_play.get("seat"), 0))
            if actor_state is not None:
                options.append(
                    self._challenge_option(
                        hero_state=hero_state,
                        actor_state=actor_state,
                        latest_play=latest_play,
                        hero_hand=hero_hand,
                        target_rank=target_rank,
                    )
                )

        max_honest = min(3, support_count)
        for count in range(1, max_honest + 1):
            options.append(
                self._play_option(
                    label=f"出{count}张目标",
                    action_type=f"play_honest_{count}",
                    honest_ratio=1.0,
                    bluff_ratio=0.0,
                    count=count,
                    remaining_cards=max(0, total_cards - count),
                    hero_state=hero_state,
                    next_state=next_state,
                )
            )

        max_bluff = min(3, non_target_count)
        for count in range(1, max_bluff + 1):
            options.append(
                self._play_option(
                    label=f"出{count}张非目标",
                    action_type=f"play_bluff_{count}",
                    honest_ratio=0.0,
                    bluff_ratio=1.0,
                    count=count,
                    remaining_cards=max(0, total_cards - count),
                    hero_state=hero_state,
                    next_state=next_state,
                )
            )

        max_mix = min(3, support_count + non_target_count)
        for count in range(2, max_mix + 1):
            if support_count <= 0 or non_target_count <= 0:
                break
            honest_cards = min(count - 1, support_count)
            bluff_cards = count - honest_cards
            if bluff_cards > non_target_count:
                continue
            honest_ratio = honest_cards / count
            bluff_ratio = bluff_cards / count
            options.append(
                self._play_option(
                    label=f"混出{count}张",
                    action_type=f"play_mix_{count}",
                    honest_ratio=honest_ratio,
                    bluff_ratio=bluff_ratio,
                    count=count,
                    remaining_cards=max(0, total_cards - count),
                    hero_state=hero_state,
                    next_state=next_state,
                )
            )

        if ghost_count > 0:
            options.append(
                self._play_option(
                    label="鬼牌单出",
                    action_type="play_ghost_single",
                    honest_ratio=0.9,
                    bluff_ratio=0.1,
                    count=1,
                    remaining_cards=max(0, total_cards - 1),
                    hero_state=hero_state,
                    next_state=next_state,
                    ghost_bonus=0.18 + 0.12 * (next_state.challenge_pressure if next_state else 0.2),
                )
            )

        if not options:
            options.append(
                {
                    "label": "信息不足",
                    "action_type": "observe",
                    "score": 0.0,
                    "confidence": 0.45,
                    "reasons": ["还没录入自己的目标/非目标/鬼/万能数量。"],
                }
            )

        options.sort(key=lambda item: item["score"], reverse=True)
        if len(options) == 1:
            options[0]["confidence"] = 0.68
            return options

        top_score = options[0]["score"]
        second_score = options[1]["score"]
        confidence = _clamp(0.5 + (top_score - second_score) / 1.8, 0.1, 0.95)
        options[0]["confidence"] = round(confidence, 4)
        for index in range(1, len(options)):
            gap = top_score - options[index]["score"]
            options[index]["confidence"] = round(_clamp(0.5 - gap / 2.2, 0.05, 0.80), 4)
        return options
