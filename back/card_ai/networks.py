from __future__ import annotations

import torch
import torch.nn as nn


class CardNet(nn.Module):
    """Shared MLP backbone with configurable depth."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
            
        layers: list[nn.Module] = []
        current_in_dim = input_dim
        
        for _ in range(num_layers):
            layers.append(nn.Linear(current_in_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_in_dim = hidden_dim
            
        layers.append(nn.Linear(current_in_dim, output_dim))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FP8Linear(nn.Module):
    """基于 Blackwell 架构原生 Float8_e4m3fn 硬件缩放矩阵乘法的极速全连接层。"""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features) * (2.0 / in_features) ** 0.5
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)
        self.scale_a = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.scale_b = nn.Parameter(torch.tensor(1.0), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            x.is_cuda
            and hasattr(torch, "float8_e4m3fn")
            and hasattr(torch, "_scaled_mm")
            and x.numel() >= 4096
            and not torch.is_grad_enabled()
        ):
            try:
                x_f8 = x.to(torch.float8_e4m3fn)
                w_f8 = self.weight.to(torch.float8_e4m3fn)
                out = torch._scaled_mm(
                    x_f8,
                    w_f8.t(),
                    scale_a=self.scale_a,
                    scale_b=self.scale_b,
                    out_dtype=x.dtype if x.dtype in (torch.bfloat16, torch.float16) else torch.bfloat16,
                )
                if self.bias is not None:
                    out = out + self.bias
                return out
            except Exception:
                pass
        return nn.functional.linear(x, self.weight, self.bias)


class ResCardBlock(nn.Module):
    """残差特征块：保持深层梯度稳定与高密度张量乘法吞吐。"""

    def __init__(self, dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class ResCardNet(nn.Module):
    """基于残差网络的高容量骨干网（专为大显存与 GPU 满载计算设计）。"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 768,
        num_blocks: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList(
            [ResCardBlock(hidden_dim, dropout=dropout) for _ in range(num_blocks)]
        )
        self.out_head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        for b in self.blocks:
            h = b(h)
        return self.out_head(h)


class AdvantageNetwork(CardNet):
    """Predicts cumulative regret/advantage values for each abstract action.
    
    Output: raw advantage values (no activation). Shape: (batch, action_space_size)
    Loss: weighted MSE between predicted advantages and sampled advantages.
    """
    pass


class StrategyNetwork(CardNet):
    """Predicts the average strategy (action probabilities) for each abstract action.
    
    Output: raw logits. Apply masked softmax externally during inference.
    Loss: weighted cross-entropy with target strategy distribution.
    """
    pass


def build_advantage_net(
    input_dim: int,
    action_space_size: int,
    hidden_dim: int = 256,
    num_layers: int = 3,
    dropout: float = 0.1,
) -> AdvantageNetwork:
    return AdvantageNetwork(input_dim, action_space_size, hidden_dim, num_layers, dropout)


def build_strategy_net(
    input_dim: int,
    action_space_size: int,
    hidden_dim: int = 256,
    num_layers: int = 3,
    dropout: float = 0.1,
) -> StrategyNetwork:
    return StrategyNetwork(input_dim, action_space_size, hidden_dim, num_layers, dropout)

