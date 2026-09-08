"""
PPO 训练环境随机对手池与对局抽样模块 (v27 纯 RL 路线)。

设计规范：
1. 真实四人满手牌开局，候选席位与先手均衡随机；
2. 训练侧对手每局为 3 个对手独立抽样，抽中后整局固定：
   - 40%: 冻结历史模型池 (均匀抽取自 D50, v21, v14, v15, v18);
   - 50%: 多维参数化规则对手 (独立正交变化诚实倾向、质疑倾向与张数偏好，不绑定单一特征，不复制评测专用的 HonestBot);
   - 10%: 合法动作随机对手 (RandomPolicy);
3. 对手模型预加载并全程 eval() 冻结，不作为输入，不增加自博弈动态更新。
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from .types import Policy
from .neural_policy import NeuralPolicy
from .opponents import OpponentProfile, HeuristicProfilePolicy
from .self_play import RandomPolicy

# 历史名宿权重路径
HISTORICAL_OPPONENT_PATHS: dict[str, str] = {
    "D50": "runs_deep_cfr/v22_D_top2_penalty_512/checkpoint_iter_00050.pt",
    "v21": "runs_deep_cfr/deep_cfr_v21_fixed_rule_rollout_512/policy_iter_00025_baseline.pt",
    "v14": "runs_deep_cfr/deep_cfr_v14_full_selfplay/policy_best.pt",
    "v15": "runs_deep_cfr/deep_cfr_v15_league/policy_best.pt",
    "v18": "runs_deep_cfr/deep_cfr_v18_honest_adaptive/policy_best.pt",
}

# 预设参数化规则对手库 (6种不同诚实、质疑与张数组合)
PARAMETRIC_RULE_PROFILES: list[OpponentProfile] = [
    # 1. 诚实且保守 (高诚实，低质疑，偏单张)
    OpponentProfile(
        name="rule_honest_passive",
        description="出牌极其诚实，很少主动质疑，偏好单张稳扎稳打",
        variability="medium",
        style="conservative",
        honesty_bias=0.8,
        bluff_bias=-1.0,
        challenge_bias=-0.4,
        tempo_bias=0.0,
        escape_bias=0.0,
        single_bias=0.5,
        pair_bias=0.0,
        triple_bias=-0.3,
        temperature=0.2,
    ),
    # 2. 诚实且警觉 (高诚实，高质疑，偏对子)
    OpponentProfile(
        name="rule_honest_suspicious",
        description="出牌诚实，但防备心极重，热衷抓假",
        variability="medium",
        style="tight_aggressive",
        honesty_bias=0.7,
        bluff_bias=-0.8,
        challenge_bias=0.5,
        tempo_bias=0.2,
        escape_bias=0.0,
        single_bias=0.0,
        pair_bias=0.4,
        triple_bias=0.1,
        temperature=0.2,
    ),
    # 3. 诈唬型潜行客 (爱出假牌，很少质疑，偏好批量倾倒手牌)
    OpponentProfile(
        name="rule_bluffer_passive",
        description="经常偷鸡出假牌，但很少抓别人，喜欢出对子三张速攻",
        variability="high",
        style="loose_passive",
        honesty_bias=-0.7,
        bluff_bias=0.9,
        challenge_bias=-0.5,
        tempo_bias=0.8,
        escape_bias=0.0,
        single_bias=-0.3,
        pair_bias=0.4,
        triple_bias=0.6,
        temperature=0.3,
    ),
    # 4. 激进狂徒 (既爱诈唬，又极度热衷质疑抓别人)
    OpponentProfile(
        name="rule_wild_aggressive",
        description="赌徒风格，无论真假都敢出，见疑即抓，节奏狂躁",
        variability="high",
        style="wild",
        honesty_bias=-0.5,
        bluff_bias=0.7,
        challenge_bias=0.6,
        tempo_bias=0.5,
        escape_bias=0.0,
        single_bias=0.0,
        pair_bias=0.3,
        triple_bias=0.3,
        temperature=0.35,
    ),
    # 5. 平衡求生者 (诚实与质疑均衡，注重生存生命值)
    OpponentProfile(
        name="rule_balanced_survival",
        description="标准中庸策略，无明显偏好，依局势动态调整",
        variability="low",
        style="balanced",
        honesty_bias=0.2,
        bluff_bias=0.0,
        challenge_bias=0.1,
        tempo_bias=0.2,
        escape_bias=0.0,
        single_bias=0.2,
        pair_bias=0.2,
        triple_bias=0.0,
        temperature=0.15,
    ),
    # 6. 速攻倾泄者 (中度诚实，极度偏好出对子和三张压缩手牌)
    OpponentProfile(
        name="rule_tempo_dumper",
        description="追求迅速打空手牌，优先大牌量出牌，施压下家",
        variability="medium",
        style="tempo",
        honesty_bias=0.3,
        bluff_bias=0.2,
        challenge_bias=0.0,
        tempo_bias=1.2,
        escape_bias=0.0,
        single_bias=-0.5,
        pair_bias=0.5,
        triple_bias=0.8,
        temperature=0.2,
    ),
]


class PPOTrainingOpponentPool:
    """管理 PPO 训练期间抽样的对手池管理器。"""

    def __init__(self) -> None:
        self.historical_models: dict[str, NeuralPolicy] = {}
        self.historical_names: list[str] = list(HISTORICAL_OPPONENT_PATHS.keys())

        # 预加载历史名宿模型
        for name, p_str in HISTORICAL_OPPONENT_PATHS.items():
            if Path(p_str).is_file():
                pol = NeuralPolicy.load(p_str, prob_mode="linear_norm")
                pol.strategy_net.eval()
                self.historical_models[name] = pol
            else:
                raise FileNotFoundError(f"[PPOOpponentPool] 找不到历史名宿权重: {p_str}")

    def sample_opponent_for_seat(self, rng: random.Random) -> Policy:
        """按 40% 历史名宿 / 50% 规则对手 / 10% 随机对手独立抽样单个对手。"""
        prob = rng.random()
        if prob < 0.40:
            # 40% 历史名宿
            h_name = rng.choice(self.historical_names)
            base_pol = self.historical_models[h_name]
            # 为该局赋予独立种子副本
            pol_copy = NeuralPolicy(
                strategy_net=base_pol.strategy_net,
                encoder=base_pol.encoder,
                abstractor=base_pol.abstractor,
                temperature=base_pol.temperature,
                greedy=base_pol.greedy,
                seed=rng.randint(1, 10000000),
                prob_mode="linear_norm",
            )
            return pol_copy
        elif prob < 0.90:
            # 50% 参数化规则对手
            prof = rng.choice(PARAMETRIC_RULE_PROFILES)
            return HeuristicProfilePolicy(profile=prof, seed=rng.randint(1, 10000000))
        else:
            # 10% 随机动作对手
            return RandomPolicy(seed=rng.randint(1, 10000000))

    def sample_table_opponents(
        self,
        candidate_seat: int,
        rng: random.Random,
    ) -> dict[int, Policy]:
        """为除候选人以外的其余 3 个席位各自独立抽样对手策略。"""
        table_opponents: dict[int, Policy] = {}
        for seat in range(1, 5):
            if seat != candidate_seat:
                table_opponents[seat] = self.sample_opponent_for_seat(rng)
        return table_opponents
