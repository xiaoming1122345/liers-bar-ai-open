"""
PPO Actor 起点蒸馏迁移与门槛验收模块 (v27 纯 RL 路线)。

设计规范：
1. 教师模型: runs_deep_cfr/v26_shared_init_104dim.pt 中的 strategy_net (104 维 control 模式);
2. 收集数据: 在混合训练对手池中运行完整对局，收集 ~5,000 个可见决策观测及合法动作掩码;
   - 目标概率: 教师在 linear_norm 下的合法动作概率分布;
   - 数据切分: 按来源【整局】划分为 80% 训练集与 20% 验证集，杜绝局内单步泄漏;
3. 迁移拟合: 复制教师权重热启动 PPOActor (无 Dropout)，在训练集上通过梯度下降微调使得 masked softmax 逼近 linear_norm 分布;
4. 验收门槛: 验证集平均 TVD <= 0.03 (预算上限 10 分钟);
5. 固化产物: runs_ppo/v27_ppo_init_actor.pt 和 runs_ppo/v27_ppo_init_critic.pt。
"""

from __future__ import annotations

import os
import sys
import time
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

cwd = Path(os.getcwd()).resolve()
if str(cwd) not in sys.path:
    sys.path.insert(0, str(cwd))

from card_ai.engine import SurvivalGameEngine
from card_ai.types import PrivateObservation, GameAction, PlayAction
from card_ai.abstractions import ActionAbstractor, action_to_index, ACTION_SPACE_SIZE
from card_ai.profiling_features import ProfilingFeatureEncoder
from card_ai.neural_policy import NeuralPolicy
from card_ai.ppo_networks import PPOActor, PPOCritic
from card_ai.ppo_env_pool import PPOTrainingOpponentPool

TEACHER_CKPT_PATH = "runs_deep_cfr/v26_shared_init_104dim.pt"
OUTPUT_DIR = "runs_ppo"
ACTOR_INIT_PATH = os.path.join(OUTPUT_DIR, "v27_ppo_init_actor.pt")
CRITIC_INIT_PATH = os.path.join(OUTPUT_DIR, "v27_ppo_init_critic.pt")
REPORT_PATH = os.path.join(OUTPUT_DIR, "v27_distill_acceptance_report.json")


class DistillSampleDataset(Dataset):
    def __init__(self, samples: list[dict[str, Any]]) -> None:
        self.features = torch.stack([s["features"] for s in samples])
        self.legal_masks = torch.stack([s["legal_mask"] for s in samples])
        self.target_probs = torch.stack([s["target_probs"] for s in samples])
        self.is_challenge = torch.tensor([s["has_challenge"] for s in samples], dtype=torch.bool)
        self.teacher_ch_prob = torch.tensor([s["teacher_challenge_prob"] for s in samples], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "features": self.features[idx],
            "legal_mask": self.legal_masks[idx],
            "target_probs": self.target_probs[idx],
            "has_challenge": self.is_challenge[idx],
            "teacher_ch_prob": self.teacher_ch_prob[idx],
        }


def collect_distill_samples(
    target_samples: int = 5000,
    seed_base: int = 12345,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """使用教师 Init0 与混合对手池运行完整对局采集样本，按整局 8:2 切分训练/验证集。"""
    print(f"[*] 开始采集蒸馏样本，目标有效决策数: ~{target_samples}...")
    engine = SurvivalGameEngine()
    encoder = ProfilingFeatureEncoder(mode="control")
    abstractor = ActionAbstractor()
    opponent_pool = PPOTrainingOpponentPool()

    # 加载教师模型
    teacher_policy = NeuralPolicy.load(TEACHER_CKPT_PATH, encoder=encoder, prob_mode="linear_norm")
    teacher_policy.strategy_net.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_policy.strategy_net.to(device)

    all_games_samples: list[list[dict[str, Any]]] = []
    total_collected = 0
    game_idx = 0
    rng = random.Random(seed_base)
    ch_idx = action_to_index("challenge")

    t0 = time.time()
    while total_collected < target_samples:
        game_idx += 1
        game_seed = seed_base + game_idx * 1009
        hero_seat = (game_idx % 4) + 1
        starting_seat = ((game_idx // 4) % 4) + 1

        state = engine.new_game(seed=game_seed, starting_seat=starting_seat)
        opponents = opponent_pool.sample_table_opponents(candidate_seat=hero_seat, rng=rng)
        policies = {hero_seat: teacher_policy, **opponents}

        game_samples: list[dict[str, Any]] = []
        step_count = 0

        while not engine.is_terminal(state) and step_count < 250:
            step_count += 1
            actor = engine.current_actor(state)
            legal = engine.legal_actions(state)
            if not legal:
                break

            if actor == hero_seat:
                # 记录该决策观测
                obs = engine.observe(state, hero_seat)
                feat = encoder.encode(obs)  # (104,)
                grouped = abstractor.abstract_legal_actions(legal, obs)
                legal_indices = [action_to_index(aa.label) for aa in grouped]

                # 构建掩码
                mask = torch.full((ACTION_SPACE_SIZE,), -1e9, dtype=torch.float32)
                for idx in legal_indices:
                    mask[idx] = 0.0

                # 教师模型在 linear_norm 下的概率
                feat_dev = feat.unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = teacher_policy.strategy_net(feat_dev).squeeze(0).cpu()

                raw_scores = logits[legal_indices]
                clamped = torch.clamp(raw_scores, min=0.0)
                sum_val = clamped.sum()
                if sum_val > 1e-8:
                    target_sub = clamped / sum_val
                else:
                    target_sub = torch.full_like(clamped, 1.0 / len(legal_indices))

                target_probs = torch.zeros(ACTION_SPACE_SIZE, dtype=torch.float32)
                for l_idx, sub_p in zip(legal_indices, target_sub):
                    target_probs[l_idx] = float(sub_p)

                has_ch = (ch_idx in legal_indices)
                ch_p = float(target_probs[ch_idx]) if has_ch else 0.0

                # 采样并执行动作
                chosen_local = torch.multinomial(target_sub, 1).item()
                chosen_idx = legal_indices[chosen_local]
                for aa, concrete_actions in grouped.items():
                    if action_to_index(aa.label) == chosen_idx:
                        action = rng.choice(concrete_actions)
                        break
                else:
                    action = rng.choice(legal)

                # 只保留合法动作数 > 1 的决策步进入蒸馏池
                if len(legal) > 1:
                    game_samples.append({
                        "features": feat.cpu(),
                        "legal_mask": mask.cpu(),
                        "target_probs": target_probs.cpu(),
                        "has_challenge": has_ch,
                        "teacher_challenge_prob": ch_p,
                    })
            else:
                obs = engine.observe(state, actor)
                pol = policies[actor]
                with torch.no_grad():
                    action = pol.choose_action(obs, legal)

            engine.apply_action(state, action)

        if game_samples:
            all_games_samples.append(game_samples)
            total_collected += len(game_samples)

    elapsed = time.time() - t0
    print(f"[OK] 样本采集完成: 共 {len(all_games_samples)} 局, 累积有效决策步: {total_collected} 条, 耗时: {elapsed:.1f}s")

    # 按整局 8:2 切分
    rng.shuffle(all_games_samples)
    split_point = int(len(all_games_samples) * 0.8)
    train_games = all_games_samples[:split_point]
    val_games = all_games_samples[split_point:]

    train_samples = [s for g in train_games for s in g]
    val_samples = [s for g in val_games for s in g]
    print(f"    - 训练集: {len(train_games)} 局, {len(train_samples)} 条样本")
    print(f"    - 验证集: {len(val_games)} 局, {len(val_samples)} 条样本")

    return train_samples, val_samples


def evaluate_actor_fit(
    actor: PPOActor,
    val_loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """评估 Actor 在验证集上的概率拟合程度：TVD, 概率 MSE, 质疑概率差。"""
    actor.eval()
    ch_idx = action_to_index("challenge")

    total_tvd = 0.0
    total_prob_mse = 0.0
    total_ch_diff = 0.0
    total_ch_count = 0
    total_count = 0

    with torch.no_grad():
        for batch in val_loader:
            feat = batch["features"].to(device)
            mask = batch["legal_mask"].to(device)
            target = batch["target_probs"].to(device)

            # Actor masked softmax 输出
            logits = actor(feat)
            masked_logits = logits + mask
            probs = torch.softmax(masked_logits, dim=-1)  # (B, 58)

            # 1. Total Variation Distance: 0.5 * sum(|p - q|)
            diff = torch.abs(probs - target).sum(dim=-1) * 0.5  # (B,)
            total_tvd += float(diff.sum().item())

            # 2. 概率 MSE: mean((p - q)^2)
            mse = torch.mean((probs - target) ** 2, dim=-1)
            total_prob_mse += float(mse.sum().item())

            # 3. 质疑概率差
            has_ch = batch["has_challenge"].to(device)
            if has_ch.any():
                pred_ch = probs[has_ch, ch_idx]
                targ_ch = batch["teacher_ch_prob"].to(device)[has_ch]
                ch_diff = torch.abs(pred_ch - targ_ch)
                total_ch_diff += float(ch_diff.sum().item())
                total_ch_count += int(has_ch.sum().item())

            total_count += len(feat)

    avg_tvd = total_tvd / total_count
    avg_mse = total_prob_mse / total_count
    avg_ch_diff = total_ch_diff / max(total_ch_count, 1)

    return {
        "mean_tvd": avg_tvd,
        "mean_prob_mse": avg_mse,
        "mean_challenge_diff": avg_ch_diff,
        "val_samples": total_count,
    }


def run_distillation_and_acceptance(
    max_minutes: float = 10.0,
    target_tvd: float = 0.03,
) -> dict[str, Any]:
    print("=" * 75)
    print("  启动 PPO Actor 起点蒸馏迁移与门槛验收流程 (Stage 1)")
    print(f"  验收门槛: 验证集平均 TVD <= {target_tvd} | 时间预算: {max_minutes} 分钟")
    print("=" * 75)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] 使用设备: {device}")

    # 1. 采集数据并切分
    train_samples, val_samples = collect_distill_samples(target_samples=5000)
    train_dataset = DistillSampleDataset(train_samples)
    val_dataset = DistillSampleDataset(val_samples)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    # 2. 初始化 Actor 并加载 Init0 隐藏层与输出层权重
    actor = PPOActor(input_dim=104, action_space_size=58, hidden_dim=512, num_layers=4)
    teacher_ckpt = torch.load(TEACHER_CKPT_PATH, map_location="cpu", weights_only=False)
    actor.load_state_dict(teacher_ckpt["strategy_net"])
    actor.to(device)

    # 先评估未微调直接做 masked softmax 时的初始差距
    init_metrics = evaluate_actor_fit(actor, val_loader, device)
    print(f"\n[*] 权重直接复制后（未经蒸馏梯度更新）的初始验证集指标:")
    print(f"    - 平均 TVD:             {init_metrics['mean_tvd']:.5f}")
    print(f"    - 概率均方误差 (MSE):    {init_metrics['mean_prob_mse']:.6f}")
    print(f"    - 质疑动作绝对误差:      {init_metrics['mean_challenge_diff']:.5f}")

    if init_metrics["mean_tvd"] <= target_tvd:
        print(f"\n[!] 初始 TVD 已达到门槛 ({init_metrics['mean_tvd']:.5f} <= {target_tvd})！")

    print(f"\n[*] 开始梯度下降拟合概率分布 (优化器: Adam, lr=2e-4, 带学习率衰减)...")
    optimizer = torch.optim.Adam(actor.parameters(), lr=2e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=15, min_lr=1e-5)

    t_start = time.time()
    max_seconds = max_minutes * 60.0
    best_tvd = init_metrics["mean_tvd"]
    best_metrics = init_metrics
    best_state_dict = actor.state_dict()

    epoch = 0
    patience_counter = 0
    while time.time() - t_start < max_seconds:
        epoch += 1
        actor.train()
        total_loss = 0.0

        for batch in train_loader:
            feat = batch["features"].to(device)
            mask = batch["legal_mask"].to(device)
            target = batch["target_probs"].to(device)

            logits = actor(feat)
            masked_logits = logits + mask
            log_probs = torch.log_softmax(masked_logits, dim=-1)

            # 掩码交叉熵: - sum(target * log_p)
            loss = -(target * log_probs).sum(dim=-1).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())

        # 每轮评估验证集
        cur_metrics = evaluate_actor_fit(actor, val_loader, device)
        cur_tvd = cur_metrics["mean_tvd"]
        scheduler.step(cur_tvd)

        if cur_tvd < best_tvd:
            best_tvd = cur_tvd
            best_metrics = cur_metrics
            best_state_dict = {k: v.cpu() for k, v in actor.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 5 == 0 or cur_tvd <= target_tvd:
            elapsed = time.time() - t_start
            cur_lr = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:3d} | 耗时: {elapsed:5.1f}s | lr: {cur_lr:.2e} | Train Loss: {total_loss/len(train_loader):.4f} | Val TVD: {cur_tvd:.5f} (Best: {best_tvd:.5f}) | Val MSE: {cur_metrics['mean_prob_mse']:.6f} | Ch Diff: {cur_metrics['mean_challenge_diff']:.5f}")

        # 如果已经稳定低于门槛，且已经训练至少 10 轮达到充分拟合，即可优雅完成
        if cur_tvd <= target_tvd and epoch >= 10:
            print(f"\n[OK] 验证集平均 TVD 成功达到并稳固在门槛以下 ({cur_tvd:.5f} <= {target_tvd})！")
            break

        # 如果连续 40 轮无改善且学习率已衰减到最小，已充分收敛
        if patience_counter >= 40 and optimizer.param_groups[0]["lr"] <= 1.5e-5:
            print(f"\n[*] 验证集指标已充分收敛（连续 40 轮无显著下降且 lr 已达下限），停止搜索。")
            break

    total_fit_time = time.time() - t_start
    print(f"\n[*] 蒸馏拟合阶段结束！总耗时: {total_fit_time:.2f}s, 完成 {epoch} 个 Epochs")
    print(f"    - 最优验证集平均 TVD:       {best_metrics['mean_tvd']:.5f} (门槛: <= {target_tvd})")
    print(f"    - 最优概率均方误差 (MSE):  {best_metrics['mean_prob_mse']:.6f}")
    print(f"    - 最优质疑绝对差:          {best_metrics['mean_challenge_diff']:.5f}")

    is_passed = (best_metrics["mean_tvd"] <= target_tvd)
    verdict = "PASS (门槛通过)" if is_passed else "FAIL (未达门槛)"

    # 4. 保存迁移 Actor
    actor.load_state_dict(best_state_dict)
    actor_ckpt = {
        "model_type": "ppo_actor",
        "input_dim": 104,
        "action_space_size": 58,
        "hidden_dim": 512,
        "num_layers": 4,
        "dropout": 0.0,
        "prob_mode": "masked_softmax",
        "actor_net": actor.state_dict(),
        "source_teacher": TEACHER_CKPT_PATH,
        "distill_metrics": best_metrics,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    torch.save(actor_ckpt, ACTOR_INIT_PATH)
    print(f"[OK] 迁移 Actor 检查点已保存至: {ACTOR_INIT_PATH}")

    # 5. 新建初始 Critic 并保存
    critic = PPOCritic(input_dim=104, hidden_dim=256, num_layers=2)
    critic_ckpt = {
        "model_type": "ppo_critic",
        "input_dim": 104,
        "hidden_dim": 256,
        "num_layers": 2,
        "critic_net": critic.state_dict(),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    torch.save(critic_ckpt, CRITIC_INIT_PATH)
    print(f"[OK] 初始 Critic 检查点已保存至: {CRITIC_INIT_PATH}")

    # 6. 保存报告
    report = {
        "verdict": verdict,
        "is_passed": is_passed,
        "target_tvd_threshold": target_tvd,
        "time_budget_minutes": max_minutes,
        "actual_time_seconds": total_fit_time,
        "epochs_run": epoch,
        "initial_metrics_before_distill": init_metrics,
        "best_metrics_after_distill": best_metrics,
        "dataset_summary": {
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "total_samples": len(train_samples) + len(val_samples),
        },
        "saved_checkpoints": {
            "actor": ACTOR_INIT_PATH,
            "critic": CRITIC_INIT_PATH,
        }
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[OK] 验收报告已保存至: {REPORT_PATH}")

    return report


if __name__ == "__main__":
    run_distillation_and_acceptance()
