from __future__ import annotations

from pathlib import Path
import torch
import torch.nn as nn

from .networks import AdvantageNetwork, StrategyNetwork, build_advantage_net, build_strategy_net
from .profiling_features import PROFILING_FEATURE_DIM
from .abstractions import ACTION_SPACE_SIZE


def expand_checkpoint_to_104dim(source_ckpt_path: str | Path, target_ckpt_path: str | Path) -> dict:
    source_p = Path(source_ckpt_path)
    target_p = Path(target_ckpt_path)
    target_p.parent.mkdir(parents=True, exist_ok=True)

    print(f'[Network Expansion] 正在从 {source_p} 读取原网络权重...')
    old_ckpt = torch.load(source_p, map_location='cpu', weights_only=False)

    hidden_dim = old_ckpt.get('hidden_dim', 512)
    num_layers = old_ckpt.get('num_layers', 4)

    # 1. 实例化 104 维新网络
    new_adv_net = build_advantage_net(
        input_dim=PROFILING_FEATURE_DIM,
        action_space_size=ACTION_SPACE_SIZE,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    )
    new_strat_net = build_strategy_net(
        input_dim=PROFILING_FEATURE_DIM,
        action_space_size=ACTION_SPACE_SIZE,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    )

    # 2. 迁移并安全扩维
    for net_name, new_net in [('advantage_net', new_adv_net), ('strategy_net', new_strat_net)]:
        old_sd = old_ckpt[net_name]
        new_sd = new_net.state_dict()

        for k, v in old_sd.items():
            if k == 'net.0.weight':
                # 输入层权重: shape [512, 104]
                # 前 80 列完全复制原权重，后 24 列严格初始化为 0.0
                expanded_w = torch.zeros((hidden_dim, PROFILING_FEATURE_DIM), dtype=v.dtype)
                expanded_w[:, :80] = v
                expanded_w[:, 80:] = 0.0
                new_sd[k] = expanded_w
            else:
                new_sd[k] = v.clone()

        new_net.load_state_dict(new_sd)

    # 3. 构造受控新检查点
    new_payload = {
        'objective_version': old_ckpt.get('objective_version', 'rank_order_survival_v1'),
        'rollout_mode': old_ckpt.get('rollout_mode', 'fixed_rule'),
        'score_rules': old_ckpt.get('score_rules', {'first': 20.0, 'second': 15.0, 'third': -5.0, 'fourth': -30.0}),
        'iteration': 0, # 受控初始快照标为 iter 0
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,
        'feature_dim': PROFILING_FEATURE_DIM,
        'action_space_size': ACTION_SPACE_SIZE,
        'best_rank_score': -999.0,
        'training_log': [],
        'advantage_net': new_adv_net.state_dict(),
        'strategy_net': new_strat_net.state_dict(),
    }

    torch.save(new_payload, target_p)
    print(f'[Network Expansion] 成功生成 104 维受控初始化检查点: {target_p}')
    return new_payload


if __name__ == '__main__':
    src = 'runs_deep_cfr/v22_D_top2_penalty_512/checkpoint_iter_00050.pt'
    tgt = 'runs_deep_cfr/v26_shared_init_104dim.pt'
    expand_checkpoint_to_104dim(src, tgt)
