from __future__ import annotations

"""
骗子酒馆 / 逆水寒大逃杀真实多轮规则人机策略池（Opponent Profiles）。

全面重构适配新规则：
1. 彻底废弃跑得快 escape 逻辑，实装【手牌打空斩杀与自杀惩罚】；
2. 实装【硬算牌铁证机制 (Card Counting)】：手牌+公开翻开+宣称超限 100% 抓；
3. 实装【死人底牌轮空期望 (Burned Cards Probability)】：随存活人数动态修正真牌期望；
4. 实装【俄罗斯轮盘弹仓危险度 (Chamber Hazard)】：濒死极度慎重、健康主动施压；
5. 实装【魔牌/鬼牌诱捕陷阱 (Ghost Trap Strategy)】：手握鬼牌故意装诈反杀；
6. 扩展为 8 种立体性格风格池，赋予局内动态扰动（Variability & Jitter）。
"""

from dataclasses import dataclass
from math import exp
from random import Random

from .abstractions import ActionAbstractor
from .self_play import RandomPolicy
from .types import Card, CardKind, GameAction, PlayAction, Policy, PrivateObservation, Rank


@dataclass(frozen=True)
class OpponentProfile:
    name: str
    description: str
    variability: str
    style: str
    honesty_bias: float
    bluff_bias: float
    challenge_bias: float
    tempo_bias: float
    escape_bias: float
    single_bias: float
    pair_bias: float
    triple_bias: float
    temperature: float
    random_only: bool = False
    ghost_trap_bias: float = 0.5
    last_card_trap_bias: float = 1.0
    card_counting_weight: float = 1.0
    hazard_sensitivity: float = 1.0
    residual_hand_weight: float = 1.0
    counter_play_weight: float = 1.0


def _support_count(cards: tuple[Card, ...], rank: Rank | None) -> int:
    """计算手中与目标点数匹配的真牌张数（包括普通目标牌、Wild 牌和魔牌/鬼牌）。"""
    if rank is None:
        return 0
    support = 0
    for card in cards:
        if card.kind == CardKind.WILD:
            support += 1
        elif card.kind == CardKind.GHOST:
            support += 1
        elif card.printed_rank == rank:
            support += 1
    return support


def _revealed_target_count(observation: PrivateObservation, rank: Rank | None) -> int:
    """统计历史公开事件中已被翻开并确认的目标点数卡牌数量。"""
    if rank is None:
        return 0
    count = 0
    for event in observation.public_history:
        if event.event_type == "challenge":
            revealed = event.detail.get("revealed_cards", [])
            for r_str in revealed:
                if r_str == rank.value:
                    count += 1
    return count


class HeuristicProfilePolicy:
    """基于骗子酒馆多轮大逃杀轮盘生存规则的高拟真人机策略。"""

    def __init__(self, profile: OpponentProfile, seed: int | None = None) -> None:
        self.profile = profile
        self._rng = Random(seed)
        self._abstractor = ActionAbstractor()

        # 局内个性浮动扰动（消除死板常数，模拟真实玩家心态起伏）
        jitter = 0.0
        if profile.variability == "high":
            jitter = self._rng.uniform(-0.35, 0.35)
        elif profile.variability == "medium":
            jitter = self._rng.uniform(-0.15, 0.15)
        elif profile.variability == "low":
            jitter = self._rng.uniform(-0.05, 0.05)

        self._runtime_challenge_bias = profile.challenge_bias + jitter
        self._runtime_bluff_bias = profile.bluff_bias + jitter * 0.5
        self._runtime_temperature = max(0.05, profile.temperature * (1.0 + jitter * 0.3))

    def choose_action(
        self,
        observation: PrivateObservation,
        legal_actions: tuple[GameAction, ...],
    ) -> GameAction:
        if not legal_actions:
            raise ValueError("heuristic policy received no legal actions")
        if self.profile.random_only:
            return self._rng.choice(legal_actions)

        # 单一合法动作直接返回（如残局强制质疑）
        if len(legal_actions) == 1:
            return legal_actions[0]

        scored = [(action, self._score_action(observation, action)) for action in legal_actions]

        # 低温贪心决策
        if self._runtime_temperature <= 0.08:
            return max(scored, key=lambda item: item[1])[0]

        # 玻尔兹曼 Softmax 概率分布采样
        max_score = max(score for _, score in scored)
        weights = [
            exp((score - max_score) / self._runtime_temperature)
            for _, score in scored
        ]
        total = sum(weights)
        if total <= 1e-9:
            return self._rng.choice(legal_actions)

        threshold = self._rng.random() * total
        cumulative = 0.0
        chosen_action = scored[-1][0]
        for (action, _), weight in zip(scored, weights):
            cumulative += weight
            chosen_action = action
            if threshold <= cumulative:
                break
        return chosen_action

    def _score_action(self, observation: PrivateObservation, action: GameAction) -> float:
        # 获取自身状态与生存数据
        my_state = next(
            (p for p in observation.players if p.seat == observation.hero_seat),
            None,
        )
        my_shots = my_state.shots_taken if my_state else 0
        alive_players = sum(1 for p in observation.players if p.alive)

        # 俄罗斯轮盘弹仓危险度（中枪致死率）：已开 shots_taken 枪，下一枪实弹概率
        # 弹仓按标准 5 发或 6 发（取 5 发），分母保底 1
        shots_remaining = max(1, 5 - my_shots)
        chamber_hazard = 1.0 / shots_remaining  # 0.2 ~ 1.0

        claim_rank = observation.round_claim_rank

        # ──────────────────────────────────────────────────────────────────
        # ──────────────────────────────────────────────────────────────────
        # 1. 质疑动作启发值估算（Challenge Heuristic Scoring）
        # ──────────────────────────────────────────────────────────────────
        if not isinstance(action, PlayAction):
            if observation.latest_play_count <= 0 or observation.latest_play_seat is None:
                return -100.0

            claim_support = _support_count(observation.hero_hand, claim_rank)
            revealed_support = _revealed_target_count(observation, claim_rank)
            latest_count = observation.latest_play_count

            # ── 启发值 A: 硬算牌铁证（绝对上限 8 张） ──
            known_in_play = claim_support + revealed_support + latest_count
            if known_in_play > 8:
                return 85.0  # 绝对铁证，坚决暴扣抓死

            # ── 启发值 B: 贝叶斯浓度与超几何推断启发值（Bayesian Concentration） ──
            # 全场剩余该点数真牌理论上限（包括普通目标牌与 Wild）
            remain_targets = max(0, 8 - claim_support - revealed_support)
            # 己方以外全场未翻开的总未知牌数（含死人底牌与对手暗手牌）
            total_revealed = sum(1 for e in observation.public_history if e.event_type == "challenge" for _ in e.detail.get("revealed_cards", []))
            unseen_total = max(1, 20 - len(observation.hero_hand) - total_revealed)
            target_density = min(1.0, remain_targets / unseen_total)

            # 若剩余真牌总数连上家宣称的张数都不够：必然夹带杂牌，几乎必抓
            if remain_targets < latest_count:
                bayes_suspicion = 8.5 * getattr(self.profile, "card_counting_weight", 1.0)
            else:
                # 浓度越低、宣称张数越多，全部是真牌的概率指数暴跌
                honest_prob_estimate = target_density ** latest_count
                bayes_suspicion = (1.0 - honest_prob_estimate) * 4.8 * getattr(self.profile, "card_counting_weight", 1.0)

            # ── 启发值 C: 轮空底牌与存活人数修正 ──
            burned_factor = (4 - alive_players) * 0.9

            # ── 启发值 D: 轮盘死线不对称优势（Lethal Asymmetry Advantage） ──
            last_actor = next(
                (p for p in observation.players if p.seat == observation.latest_play_seat),
                None,
            )
            opp_shots = last_actor.shots_taken if last_actor else 0
            opp_hazard = 1.0 / max(1, 5 - opp_shots)

            # 若对手已濒死（下一枪致死率 >= 50%）：抓死对手直接吃鸡/淘汰一人，赋予重大斩杀激励
            lethal_opp_incentive = 3.2 if opp_hazard >= 0.5 else 0.0

            # 若自己濒死（致死率 >= 50%）：抓错自己必死，抑制盲目质疑
            my_death_risk_penalty = chamber_hazard * 4.2 * getattr(self.profile, "hazard_sensitivity", 1.0)

            # ── 启发值 E: 上家打空手牌斩杀免责防守（Lethal Gatekeeper） ──
            actor_empty_hand_bonus = 0.0
            if last_actor and last_actor.hand_count == 0:
                actor_empty_hand_bonus = 3.8  # 放过他就免责过关，必须严防死守

            # ── 启发值 F: 对手历史惯犯针对启发值（Counter-Play Adaptation） ──
            counter_bonus = 0.0
            if last_actor:
                opp_lies = sum(
                    1 for e in observation.public_history
                    if e.event_type == "challenge" and e.detail.get("challenged_seat") == last_actor.seat and e.detail.get("outcome") == "lie"
                )
                if opp_lies > 0:
                    counter_bonus += min(3.5, opp_lies * 1.2) * getattr(self.profile, "counter_play_weight", 1.0)

            score = (
                self._runtime_challenge_bias
                + bayes_suspicion
                + burned_factor
                + lethal_opp_incentive
                + actor_empty_hand_bonus
                + counter_bonus
                - my_death_risk_penalty
            )
            return score

        # ──────────────────────────────────────────────────────────────────
        # 2. 出牌动作启发值估算（Play Action Heuristic Scoring）
        # ──────────────────────────────────────────────────────────────────
        abstract_action = self._abstractor.abstract(action, observation)
        composition = list(abstract_action.composition)
        count = abstract_action.count

        honest_count = sum(1 for item in composition if item in {"target", "wild", "ghost"})
        bluff_count = sum(1 for item in composition if item == "non_target")
        has_ghost = any(item == "ghost" for item in composition)
        cards_left_after = len(observation.hero_hand) - count

        # ── 启发值 A: 诚实牌 vs 诈唬牌收益基准 ──
        score = 0.0
        score += self.profile.honesty_bias * honest_count
        score += self._runtime_bluff_bias * bluff_count
        score += self.profile.tempo_bias * count

        # ── 启发值 B: 终极打空手牌绝杀陷阱与自杀严惩 ──
        if cards_left_after == 0:
            if bluff_count > 0:
                # 空手出假牌等于将死线送给下家，自杀暴毙重罚
                score -= 16.0
            else:
                # 空手全真牌或魔牌：下家被逼必须质疑，下家必死，终极斩杀大奖！
                score += 13.5 * getattr(self.profile, "last_card_trap_bias", 1.0)

        # ── 启发值 C: 魔牌/鬼牌诱捕反杀策略 ──
        if has_ghost:
            score += 3.5 * getattr(self.profile, "ghost_trap_bias", 0.5)
            if count >= 2:
                # 拿魔牌混杂牌故意引诱下家开枪，反弹对手全场中枪
                score += 2.0 * getattr(self.profile, "ghost_trap_bias", 0.5)

        # ── 启发值 D: 轮盘濒死状态下的生存本能 ──
        if chamber_hazard >= 0.5:
            score -= bluff_count * 5.5 * getattr(self.profile, "hazard_sensitivity", 1.0)
            score += honest_count * 2.2

        # ── 启发值 E: 残余手牌结构健康度（Residual Hand Potential） ──
        played_card_ids = set(action.card_ids)
        hand_left = [c for c in observation.hero_hand if c.card_id not in played_card_ids]

        residual_heuristic = 0.0
        if hand_left:
            # 1. 留存 Wild 万能牌：应对任何后续点数的免死金牌
            wild_left = sum(1 for c in hand_left if c.kind == CardKind.WILD)
            residual_heuristic += wild_left * 2.2

            # 2. 留存 Ghost 魔牌：护身终极反杀底牌
            ghost_left = sum(1 for c in hand_left if c.kind == CardKind.GHOST)
            residual_heuristic += ghost_left * 3.2

            # 3. 留存同点数成对/三张（利于后续换点数后成套快速出清）
            rank_counts = {}
            for c in hand_left:
                if c.printed_rank:
                    rank_counts[c.printed_rank] = rank_counts.get(c.printed_rank, 0) + 1
            pairs = sum(1 for cnt in rank_counts.values() if cnt >= 2)
            residual_heuristic += pairs * 1.5

            # 4. 杂牌孤儿惩罚：手牌剩下 2 张以上且全是互不相同的杂牌单张（无 Wild/Ghost/对子）
            if wild_left == 0 and ghost_left == 0 and pairs == 0 and len(hand_left) >= 2:
                residual_heuristic -= 2.0

        score += residual_heuristic * getattr(self.profile, "residual_hand_weight", 1.0)

        # ── 启发值 F: 出牌张数偏好 ──
        size_bonus = {
            1: self.profile.single_bias,
            2: self.profile.pair_bias,
            3: self.profile.triple_bias,
        }.get(count, 0.0)
        score += size_bonus

        return score


# ---------------------------------------------------------------------------
# 8 款立体风格人机生态（覆盖保守、激进、诈唬、算牌、魔牌反杀、变色龙等打法）
# ---------------------------------------------------------------------------

def default_opponent_profiles() -> tuple[OpponentProfile, ...]:
    return (
        # 1. 传统稳健型：严守牌理，极度重视轮盘弹仓风险，很少瞎诈唬
        OpponentProfile(
            name="traditional_cautious",
            description="偏传统稳健，极少盲目诈牌，濒死时极端求生，非高概率不质疑。",
            variability="low",
            style="traditional",
            honesty_bias=1.4,
            bluff_bias=-2.2,
            challenge_bias=-0.6,
            tempo_bias=0.3,
            escape_bias=0.0,
            single_bias=0.6,
            pair_bias=0.1,
            triple_bias=-0.8,
            temperature=0.12,
            ghost_trap_bias=0.3,
            card_counting_weight=1.2,
            hazard_sensitivity=1.5,
        ),
        # 2. 铁面质疑型：怀疑心强，节奏紧凑，善于抓大牌并给下家窒息压迫
        OpponentProfile(
            name="pressure_challenger",
            description="质疑频繁，紧咬出牌节奏，对连续出多张牌怀疑度极高，喜欢施压。",
            variability="medium",
            style="aggressive",
            honesty_bias=0.5,
            bluff_bias=0.1,
            challenge_bias=1.3,
            tempo_bias=0.8,
            escape_bias=0.0,
            single_bias=-0.2,
            pair_bias=0.3,
            triple_bias=0.6,
            temperature=0.35,
            ghost_trap_bias=0.5,
            card_counting_weight=1.1,
            hazard_sensitivity=0.6,
        ),
        # 3. 亡命狂徒型：高波动，敢诈唬，在濒死局也敢殊死一搏，打法不可预测
        OpponentProfile(
            name="chaotic_bluffer",
            description="高波动狂徒，经常多张诈唬，即使残局也敢冒险博弈，节奏凶悍。",
            variability="high",
            style="volatile",
            honesty_bias=0.2,
            bluff_bias=1.2,
            challenge_bias=0.4,
            tempo_bias=1.0,
            escape_bias=0.0,
            single_bias=-0.3,
            pair_bias=0.4,
            triple_bias=0.9,
            temperature=0.85,
            ghost_trap_bias=0.8,
            card_counting_weight=0.5,
            hazard_sensitivity=0.3,
        ),
        # 4. 精明算牌客：基于公开信息与底牌轮空严格计算数学期望，专抓数学不可能
        OpponentProfile(
            name="card_counter_pro",
            description="精通算牌与底牌轮空期望，一旦对手宣称牌数超限坚决抓死，极高智商。",
            variability="low",
            style="analytical",
            honesty_bias=0.9,
            bluff_bias=-0.5,
            challenge_bias=0.2,
            tempo_bias=0.5,
            escape_bias=0.0,
            single_bias=0.2,
            pair_bias=0.3,
            triple_bias=0.1,
            temperature=0.08,
            ghost_trap_bias=0.9,
            card_counting_weight=2.0,
            hazard_sensitivity=1.1,
        ),
        # 5. 魔牌诱杀专家：深谙魔牌反噬规则，手握魔牌必设陷阱，诱导质疑借刀杀人
        OpponentProfile(
            name="ghost_trapper",
            description="魔牌诱杀专家，擅长手握魔牌时虚张声势诱惑对手质疑，反噬杀人。",
            variability="medium",
            style="deceptive",
            honesty_bias=0.6,
            bluff_bias=0.4,
            challenge_bias=-0.2,
            tempo_bias=0.7,
            escape_bias=0.0,
            single_bias=-0.1,
            pair_bias=0.5,
            triple_bias=0.7,
            temperature=0.25,
            ghost_trap_bias=2.2,
            card_counting_weight=1.0,
            hazard_sensitivity=0.9,
        ),
        # 6. 求生变色龙：健康时主动欺压，濒死时极端谨慎，生存本能极强
        OpponentProfile(
            name="adaptive_survivalist",
            description="变色龙型求生者，无伤时凶狠施压，中枪后策略急剧收缩转为绝对保命。",
            variability="medium",
            style="adaptive",
            honesty_bias=0.8,
            bluff_bias=0.0,
            challenge_bias=0.3,
            tempo_bias=0.6,
            escape_bias=0.0,
            single_bias=0.1,
            pair_bias=0.2,
            triple_bias=0.0,
            temperature=0.20,
            ghost_trap_bias=1.0,
            card_counting_weight=1.3,
            hazard_sensitivity=1.8,
        ),
        # 7. 局部贪心速攻流：追求单次出手大量泄牌，快速压缩手牌，给下家转嫁难题
        OpponentProfile(
            name="greedy_tempo",
            description="追求节奏压制，喜欢出对子和三张快速出清，强行把难题甩给下家。",
            variability="low",
            style="greedy",
            honesty_bias=0.7,
            bluff_bias=0.2,
            challenge_bias=0.1,
            tempo_bias=1.3,
            escape_bias=0.0,
            single_bias=-0.4,
            pair_bias=0.6,
            triple_bias=1.0,
            temperature=0.15,
            ghost_trap_bias=0.6,
            card_counting_weight=0.8,
            hazard_sensitivity=0.8,
        ),
        # 8. 混沌探索型：保持对罕见边缘状态与无序行为的覆盖，增强模型泛化
        OpponentProfile(
            name="pure_random",
            description="完全随机探索，动作分布均匀，供神经网络训练对罕见局面的鲁棒性。",
            variability="high",
            style="random",
            honesty_bias=0.0,
            bluff_bias=0.0,
            challenge_bias=0.0,
            tempo_bias=0.0,
            escape_bias=0.0,
            single_bias=0.0,
            pair_bias=0.0,
            triple_bias=0.0,
            temperature=1.0,
            random_only=True,
        ),
        # 9. 贝叶斯神算客：极致发挥超几何剩余真牌浓度推断，识破诈唬率极高，专治假牌狂徒
        OpponentProfile(
            name="bayesian_master",
            description="顶尖数学算牌客，严密推算牌池浓度与上家真牌后验概率，抓诈极其毒辣精确。",
            variability="low",
            style="analytical",
            honesty_bias=1.2,
            bluff_bias=-1.5,
            challenge_bias=0.4,
            tempo_bias=0.4,
            escape_bias=0.0,
            single_bias=0.3,
            pair_bias=0.3,
            triple_bias=-0.2,
            temperature=0.10,
            ghost_trap_bias=0.8,
            card_counting_weight=2.0,
            hazard_sensitivity=1.2,
            residual_hand_weight=1.2,
            counter_play_weight=1.6,
        ),
        # 10. 残局诱杀者：深谙手牌残余价值与魔牌陷阱，不见兔子不撒鹰，残局胜率奇高
        OpponentProfile(
            name="endgame_trapper",
            description="终局布局大师，极其珍惜万能牌与魔牌，擅长手牌结构优化与绝杀免责陷阱。",
            variability="medium",
            style="trapper",
            honesty_bias=1.5,
            bluff_bias=-0.8,
            challenge_bias=0.2,
            tempo_bias=0.8,
            escape_bias=0.0,
            single_bias=-0.2,
            pair_bias=0.5,
            triple_bias=0.6,
            temperature=0.14,
            ghost_trap_bias=2.6,
            last_card_trap_bias=2.2,
            card_counting_weight=1.4,
            hazard_sensitivity=1.4,
            residual_hand_weight=2.0,
            counter_play_weight=1.2,
        ),
    )


def build_policy_from_profile(profile: OpponentProfile, seed: int | None = None) -> Policy:
    if profile.random_only:
        return RandomPolicy(seed=seed)
    return HeuristicProfilePolicy(profile=profile, seed=seed)
