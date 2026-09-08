from __future__ import annotations

from pathlib import Path
from random import Random

import torch

from .abstractions import ACTION_SPACE_SIZE, ActionAbstractor, action_to_index
from .features import FeatureEncoder
from .networks import StrategyNetwork, build_strategy_net
from .features import FEATURE_DIM
from .types import GameAction, PrivateObservation


class NeuralPolicy:
    """Policy that uses a trained StrategyNetwork for action selection."""

    # 显式语义标识：当前网络训练目标为概率回归，推理输出解释为非负概率分布而非logits
    output_semantics: str = "linear_probability"

    def __init__(
        self,
        strategy_net: StrategyNetwork,
        encoder: FeatureEncoder | None = None,
        abstractor: ActionAbstractor | None = None,
        temperature: float = 1.0,
        greedy: bool = False,
        seed: int | None = None,
        prob_mode: str = "linear_norm",
    ) -> None:
        self.strategy_net = strategy_net
        self.encoder = encoder or FeatureEncoder()
        self.abstractor = abstractor or ActionAbstractor()
        self.temperature = temperature
        self.greedy = greedy
        self._rng = Random(seed)
        self.prob_mode = prob_mode
        self.output_semantics = "linear_probability"
        self.diagnostics = {
            "total_decisions": 0,
            "fallback_count": 0,
            "sum_top1_prob": 0.0,
            "sum_legal_actions": 0,
        }

    def reset_diagnostics(self) -> None:
        self.diagnostics = {
            "total_decisions": 0,
            "fallback_count": 0,
            "sum_top1_prob": 0.0,
            "sum_legal_actions": 0,
        }

    def get_diagnostics_summary(self) -> dict[str, float]:
        n = max(1, self.diagnostics["total_decisions"])
        return {
            "total_decisions": self.diagnostics["total_decisions"],
            "fallback_count": self.diagnostics["fallback_count"],
            "fallback_rate": round(self.diagnostics["fallback_count"] / n, 4),
            "avg_top1_prob": round(self.diagnostics["sum_top1_prob"] / n, 4),
            "avg_legal_actions": round(self.diagnostics["sum_legal_actions"] / n, 2),
        }

    @torch.no_grad()
    def choose_action(
        self,
        observation: PrivateObservation,
        legal_actions: tuple[GameAction, ...],
    ) -> GameAction:
        """Choose an action using the strategy network."""
        if not legal_actions:
            raise ValueError("no legal actions")
        if len(legal_actions) == 1:
            return legal_actions[0]

        # 1. Encode features
        features = self.encoder.encode(observation)
        grouped = self.abstractor.abstract_legal_actions(legal_actions, observation)
        return self.choose_action_from_features(
            observation=observation,
            legal_actions=legal_actions,
            features=features,
            grouped=grouped,
        )

    @torch.no_grad()
    def choose_action_from_features(
        self,
        observation: PrivateObservation,
        legal_actions: tuple[GameAction, ...],
        features: torch.Tensor,
        grouped: dict | None = None,
    ) -> GameAction:
        """高性能快速路径：直接使用已提取特征和动作分组，消除重复计算。"""
        if not legal_actions:
            raise ValueError("no legal actions")

        device = next(self.strategy_net.parameters()).device
        if features.dim() == 1:
            feat_dev = features.unsqueeze(0).to(device)
        else:
            feat_dev = features.to(device)

        logits = self.strategy_net(feat_dev).squeeze(0)  # (ACTION_SPACE_SIZE,)

        if grouped is None:
            grouped = self.abstractor.abstract_legal_actions(legal_actions, observation)
        legal_indices = [action_to_index(aa.label) for aa in grouped]

        # 概率转换与采样
        if self.prob_mode == "linear_norm":
            raw_scores = logits[legal_indices]
            clamped = torch.clamp(raw_scores, min=0.0)
            sum_val = clamped.sum()

            if sum_val > 1e-8:
                legal_probs = clamped / sum_val
            else:
                self.diagnostics["fallback_count"] += 1
                legal_probs = torch.full_like(clamped, 1.0 / len(legal_indices))

            top1_p = legal_probs.max().item()
            self.diagnostics["total_decisions"] += 1
            self.diagnostics["sum_top1_prob"] += top1_p
            self.diagnostics["sum_legal_actions"] += len(legal_indices)

            if self.greedy:
                chosen_local_idx = legal_probs.argmax().item()
            else:
                chosen_local_idx = torch.multinomial(legal_probs, 1).item()
            chosen_idx = legal_indices[chosen_local_idx]

        else:
            mask = torch.full((ACTION_SPACE_SIZE,), float("-inf"), device=device)
            for idx in legal_indices:
                mask[idx] = 0.0

            masked_logits = (logits + mask) / max(self.temperature, 1e-8)
            probs = torch.softmax(masked_logits, dim=0)

            top1_p = probs[legal_indices].max().item()
            self.diagnostics["total_decisions"] += 1
            self.diagnostics["sum_top1_prob"] += top1_p
            self.diagnostics["sum_legal_actions"] += len(legal_indices)

            if self.greedy:
                chosen_idx = probs.argmax().item()
            else:
                chosen_idx = torch.multinomial(probs, 1).item()

        for abstract_action, concrete_group in grouped.items():
            if action_to_index(abstract_action.label) == chosen_idx:
                return self._rng.choice(concrete_group)

        return self._rng.choice(legal_actions)

    @classmethod
    def load(cls, checkpoint_path: str | Path, **kwargs) -> "NeuralPolicy":
        """Load a NeuralPolicy from a saved checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        strategy_weights = checkpoint["strategy_net"]
        # Detect input dimension directly from the first Linear layer weight shape
        first_weight = strategy_weights.get("net.0.weight")
        if first_weight is not None:
            feature_dim = first_weight.shape[1]
        else:
            feature_dim = checkpoint.get("feature_dim", FEATURE_DIM)

        net = build_strategy_net(
            input_dim=feature_dim,
            action_space_size=ACTION_SPACE_SIZE,
            hidden_dim=checkpoint.get("hidden_dim", 256),
            num_layers=checkpoint.get("num_layers", 3),
        )
        net.load_state_dict(strategy_weights)
        net.eval()
        encoder = kwargs.pop("encoder", None) or FeatureEncoder(feature_dim=feature_dim)
        return cls(strategy_net=net, encoder=encoder, **kwargs)


