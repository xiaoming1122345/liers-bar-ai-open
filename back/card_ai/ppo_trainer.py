"""
v27 完整对局 PPO-Clip 独立双网络训练器 (PPOTrainer)。

核心设计规范：
1. 双网络独立：PPOActor (512x4, 无 Dropout, masked softmax) 与 PPOCritic (256x2, 独立标量输出);
2. 终局奖励与缩放：
   - A组 (v27_A_standard): [20, 15, -5, -30] / 50 -> [+0.4, +0.3, -0.1, -0.6]
   - B组 (v27_B_top2_priority): [20, 15, -15, -50] / 50 -> [+0.4, +0.3, -0.3, -1.0]
3. 轨迹采样：
   - 每批至少收集 1,024 个有效自主决策步 (len(legal) > 1)，整局采集完成后进入更新;
   - 候选席位与先手均衡随机，对手从混合对手池随机抽样并整局固定;
   - 记录耗时：采样对局耗时、特征编码耗时、网络批推理耗时、PPO 反向更新耗时;
4. Monte Carlo 回报回传：
   - gamma = 1.0，终局名次标量回报回传给候选本局所有决策;
   - 只有单一合法动作的非自主选择步保留终局回报供 Critic 拟合，但不贡献 Actor 策略梯度;
   - advantage = G - V_sampled，进行批标准化;
5. PPO 更新参数：
   - Actor lr: 1e-4, Critic lr: 3e-4;
   - clip ratio: 0.2, epochs: 4, minibatch: 256;
   - entropy coeff: 0.001, target KL: 0.01 (超阈值早停 Actor 后续 epochs);
   - 严格 on-policy，更新后丢弃当前批轨迹;
6. 保存第 0、10、20 次更新完整检查点。
"""

from __future__ import annotations

import os
import sys
import time
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import json
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

cwd = Path(os.getcwd()).resolve()
if str(cwd) not in sys.path:
    sys.path.insert(0, str(cwd))

from card_ai.engine import SurvivalGameEngine
from card_ai.rewards import RankRewardModel, RankScoreRules
from card_ai.types import PrivateObservation, GameAction, PlayAction
from card_ai.abstractions import ActionAbstractor, action_to_index, ACTION_SPACE_SIZE
from card_ai.profiling_features import ProfilingFeatureEncoder
from card_ai.ppo_networks import PPOActor, PPOCritic
from card_ai.ppo_env_pool import PPOTrainingOpponentPool


@dataclass
class PPOTrajectoryBatch:
    features: torch.Tensor          # (N, 104)
    legal_masks: torch.Tensor       # (N, 58)
    actions: torch.Tensor           # (N,)
    old_log_probs: torch.Tensor     # (N,)
    returns: torch.Tensor           # (N,)
    advantages: torch.Tensor        # (N,)
    is_valid_decision: torch.Tensor # (N,) bool
    num_games: int
    num_valid_steps: int
    timing_metrics: dict[str, float]


class PPOTrainer:
    """v27 纯 RL 路线：PPO-Clip 双网络训练器。"""

    def __init__(
        self,
        actor: PPOActor,
        critic: PPOCritic,
        score_rules: list[float],
        run_name: str,
        output_dir: str = "runs_ppo",
        actor_lr: float = 1e-4,
        critic_lr: float = 3e-4,
        clip_ratio: float = 0.2,
        epochs_per_batch: int = 4,
        minibatch_size: int = 256,
        target_kl: float = 0.01,
        entropy_coeff: float = 0.001,
        min_steps_per_batch: int = 1024,
        seed: int = 42,
        device: str = "auto",
    ) -> None:
        self.actor = actor
        self.critic = critic
        self.score_rules = list(score_rules)
        self.scaled_rules = [r / 50.0 for r in score_rules]
        self.run_name = run_name
        self.output_dir = output_dir
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.clip_ratio = clip_ratio
        self.epochs_per_batch = epochs_per_batch
        self.minibatch_size = minibatch_size
        self.target_kl = target_kl
        self.entropy_coeff = entropy_coeff
        self.min_steps_per_batch = min_steps_per_batch

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.actor.to(self.device)
        self.critic.to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.engine = SurvivalGameEngine()
        self.encoder = ProfilingFeatureEncoder(mode="control")
        self.abstractor = ActionAbstractor()
        self.opponent_pool = PPOTrainingOpponentPool()
        self.rank_model = RankRewardModel(
            score_rules=RankScoreRules(
                first=self.score_rules[0],
                second=self.score_rules[1],
                third=self.score_rules[2],
                fourth=self.score_rules[3],
            )
        )

        self.rng = random.Random(seed)
        self.game_counter = 0
        self.iteration = 0
        self.history_logs: list[dict[str, Any]] = []

        os.makedirs(os.path.join(self.output_dir, self.run_name), exist_ok=True)

    def evaluate_game_terminal_payoff(self, state: Any, candidate_seat: int) -> float:
        """根据实际终局名次与并列槽位平均结算名次分，并做 1/50 缩放。"""
        payoffs = self.rank_model.evaluate(state)
        raw_score = payoffs.for_seat(candidate_seat)
        return float(raw_score / 50.0)

    def collect_on_policy_batch(self) -> PPOTrajectoryBatch:
        """执行完整对局轨迹采样，直到有效自主决策数 >= min_steps_per_batch 且整局结束。"""
        self.actor.eval()
        self.critic.eval()

        t_start = time.time()
        time_sampling = 0.0
        time_encoding = 0.0
        time_inference = 0.0

        batch_features: list[torch.Tensor] = []
        batch_legal_masks: list[torch.Tensor] = []
        batch_actions: list[int] = []
        batch_old_log_probs: list[float] = []
        batch_returns: list[float] = []
        batch_advantages: list[float] = []
        batch_is_valid: list[bool] = []

        valid_steps_count = 0
        games_collected = 0

        while valid_steps_count < self.min_steps_per_batch:
            games_collected += 1
            self.game_counter += 1
            game_seed = self.rng.randint(1, 2000000000)

            # 候选席位与先手均衡随机
            hero_seat = ((self.game_counter - 1) % 4) + 1
            starting_seat = (((self.game_counter - 1) // 4) % 4) + 1

            t_g0 = time.time()
            state = self.engine.new_game(seed=game_seed, starting_seat=starting_seat)
            opponents = self.opponent_pool.sample_table_opponents(candidate_seat=hero_seat, rng=self.rng)
            time_sampling += (time.time() - t_g0)

            game_history_steps: list[dict[str, Any]] = []
            step_count = 0

            while not self.engine.is_terminal(state) and step_count < 250:
                step_count += 1
                actor = self.engine.current_actor(state)
                legal = self.engine.legal_actions(state)
                if not legal:
                    break

                if actor == hero_seat:
                    t_enc0 = time.time()
                    obs = self.engine.observe(state, hero_seat)
                    feat = self.encoder.encode(obs)  # (104,)
                    grouped = self.abstractor.abstract_legal_actions(legal, obs)
                    legal_indices = [action_to_index(aa.label) for aa in grouped]
                    time_encoding += (time.time() - t_enc0)

                    # 构建动作掩码
                    mask = torch.full((ACTION_SPACE_SIZE,), -1e9, dtype=torch.float32)
                    for idx in legal_indices:
                        mask[idx] = 0.0

                    if len(legal) == 1:
                        # 唯一强制动作
                        action = legal[0]
                        chosen_idx = legal_indices[0]
                        old_log_p = 0.0

                        t_inf0 = time.time()
                        with torch.no_grad():
                            feat_dev = feat.unsqueeze(0).to(self.device)
                            v_pred = float(self.critic(feat_dev).item())
                        time_inference += (time.time() - t_inf0)

                        game_history_steps.append({
                            "features": feat,
                            "legal_mask": mask,
                            "action": chosen_idx,
                            "old_log_prob": old_log_p,
                            "value": v_pred,
                            "is_valid": False,  # 唯一动作不参与策略梯度
                        })
                    else:
                        # 有效自主决策
                        t_inf0 = time.time()
                        feat_dev = feat.unsqueeze(0).to(self.device)
                        mask_dev = mask.unsqueeze(0).to(self.device)
                        with torch.no_grad():
                            logits = self.actor(feat_dev)
                            masked_logits = logits + mask_dev
                            probs = torch.softmax(masked_logits, dim=-1).squeeze(0)
                            v_pred = float(self.critic(feat_dev).item())

                        probs_legal = probs[legal_indices]
                        sum_p = probs_legal.sum()
                        if sum_p > 1e-8:
                            norm_p = probs_legal / sum_p
                        else:
                            norm_p = torch.full_like(probs_legal, 1.0 / len(legal_indices))

                        chosen_local = torch.multinomial(norm_p, 1).item()
                        chosen_idx = legal_indices[chosen_local]
                        old_log_p = float(torch.log(torch.clamp(probs[chosen_idx], min=1e-12)).item())
                        time_inference += (time.time() - t_inf0)

                        # 具体手牌映射
                        for aa, concrete_group in grouped.items():
                            if action_to_index(aa.label) == chosen_idx:
                                action = self.rng.choice(concrete_group)
                                break
                        else:
                            action = self.rng.choice(legal)

                        game_history_steps.append({
                            "features": feat,
                            "legal_mask": mask,
                            "action": chosen_idx,
                            "old_log_prob": old_log_p,
                            "value": v_pred,
                            "is_valid": True,
                        })
                        valid_steps_count += 1
                else:
                    t_g1 = time.time()
                    obs = self.engine.observe(state, actor)
                    pol = opponents[actor]
                    with torch.no_grad():
                        action = pol.choose_action(obs, legal)
                    time_sampling += (time.time() - t_g1)

                t_g2 = time.time()
                self.engine.apply_action(state, action)
                time_sampling += (time.time() - t_g2)

            # 终局回报结算
            if self.engine.is_terminal(state):
                payoff = self.evaluate_game_terminal_payoff(state, hero_seat)
                for step_data in game_history_steps:
                    batch_features.append(step_data["features"])
                    batch_legal_masks.append(step_data["legal_mask"])
                    batch_actions.append(step_data["action"])
                    batch_old_log_probs.append(step_data["old_log_prob"])
                    batch_returns.append(payoff)
                    batch_advantages.append(payoff - step_data["value"])
                    batch_is_valid.append(step_data["is_valid"])
            else:
                # 触发步数限制未正常终局，丢弃该未完成局并回滚计数
                num_dropped_valid = sum(1 for s in game_history_steps if s["is_valid"])
                valid_steps_count -= num_dropped_valid

        total_sample_time = time.time() - t_start
        timing_metrics = {
            "total_sampling_time": total_sample_time,
            "engine_play_time": time_sampling,
            "feature_encoding_time": time_encoding,
            "network_inference_time": time_inference,
        }

        # 转换为张量
        return PPOTrajectoryBatch(
            features=torch.stack(batch_features).to(self.device),
            legal_masks=torch.stack(batch_legal_masks).to(self.device),
            actions=torch.tensor(batch_actions, dtype=torch.long, device=self.device),
            old_log_probs=torch.tensor(batch_old_log_probs, dtype=torch.float32, device=self.device),
            returns=torch.tensor(batch_returns, dtype=torch.float32, device=self.device),
            advantages=torch.tensor(batch_advantages, dtype=torch.float32, device=self.device),
            is_valid_decision=torch.tensor(batch_is_valid, dtype=torch.bool, device=self.device),
            num_games=games_collected,
            num_valid_steps=valid_steps_count,
            timing_metrics=timing_metrics,
        )

    def update_ppo(self, batch: PPOTrajectoryBatch) -> dict[str, float]:
        """使用当前批数据执行 PPO-Clip 优化 (4 epochs, minibatch 256)。"""
        self.actor.train()
        self.critic.train()

        t_update0 = time.time()

        # 优势标准化 (仅在 valid 步上计算均值与标准差)
        valid_mask = batch.is_valid_decision
        adv = batch.advantages.clone()
        if valid_mask.sum() > 1:
            valid_adv = adv[valid_mask]
            adv_mean = valid_adv.mean()
            adv_std = valid_adv.std() + 1e-8
            adv = (adv - adv_mean) / adv_std

        dataset = TensorDataset(
            batch.features,
            batch.legal_masks,
            batch.actions,
            batch.old_log_probs,
            batch.returns,
            adv,
            valid_mask,
        )
        loader = DataLoader(dataset, batch_size=self.minibatch_size, shuffle=True)

        actor_early_stopped = False
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        total_approx_kl = 0.0
        total_actor_batches = 0
        total_critic_batches = 0

        for epoch in range(self.epochs_per_batch):
            epoch_kls = []
            for b_feat, b_mask, b_act, b_old_logp, b_ret, b_adv, b_valid in loader:
                # 1. Critic 更新 (不受 KL 影响，拟合实际终局回报)
                self.critic_optimizer.zero_grad()
                v_pred = self.critic(b_feat)
                c_loss = nn.functional.mse_loss(v_pred, b_ret)
                c_loss.backward()
                self.critic_optimizer.step()

                total_critic_loss += float(c_loss.item())
                total_critic_batches += 1

                # 2. Actor 更新
                if not actor_early_stopped and b_valid.any():
                    self.actor_optimizer.zero_grad()
                    logp, entropy = self.actor.evaluate_actions(b_feat, b_mask, b_act)

                    ratio = torch.exp(logp - b_old_logp)
                    surr1 = ratio * b_adv
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * b_adv

                    # 仅在自主决策步上反传
                    actor_loss_raw = -torch.min(surr1, surr2)
                    valid_actor_loss = (actor_loss_raw * b_valid.float()).sum() / (b_valid.float().sum() + 1e-8)
                    valid_entropy = (entropy * b_valid.float()).sum() / (b_valid.float().sum() + 1e-8)

                    loss = valid_actor_loss - self.entropy_coeff * valid_entropy
                    loss.backward()
                    self.actor_optimizer.step()

                    # 近似 KL 监控
                    with torch.no_grad():
                        approx_kl = ((b_old_logp - logp) * b_valid.float()).sum() / (b_valid.float().sum() + 1e-8)
                        epoch_kls.append(float(approx_kl.item()))

                    total_actor_loss += float(valid_actor_loss.item())
                    total_entropy += float(valid_entropy.item())
                    total_approx_kl += float(approx_kl.item())
                    total_actor_batches += 1

            if epoch_kls and np.mean(epoch_kls) > self.target_kl:
                actor_early_stopped = True

        time_update = time.time() - t_update0
        return {
            "actor_loss": total_actor_loss / max(total_actor_batches, 1),
            "critic_loss": total_critic_loss / max(total_critic_batches, 1),
            "entropy": total_entropy / max(total_actor_batches, 1),
            "approx_kl": total_approx_kl / max(total_actor_batches, 1),
            "actor_early_stopped": float(actor_early_stopped),
            "update_time": time_update,
        }

    def save_checkpoint(self, iter_idx: int) -> str:
        """保存指定轮次的完整检查点。"""
        ckpt_path = os.path.join(
            self.output_dir,
            self.run_name,
            f"checkpoint_iter_{iter_idx:05d}.pt",
        )
        payload = {
            "iteration": iter_idx,
            "run_name": self.run_name,
            "score_rules": self.score_rules,
            "scaled_rules": self.scaled_rules,
            "actor_net": self.actor.state_dict(),
            "critic_net": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "game_counter": self.game_counter,
            "history_logs": self.history_logs,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        torch.save(payload, ckpt_path)
        return ckpt_path

    def train_iterations(self, num_iterations: int = 20) -> list[dict[str, Any]]:
        """主训练流水线：执行 num_iterations 次 PPO 更新。"""
        print("=" * 75)
        print(f"  启动 PPO 训练分支: {self.run_name}")
        print(f"  终局名次积分: {self.score_rules} (除以50缩放为 {self.scaled_rules})")
        print(f"  目标更新次数: {num_iterations} 次 | 批最小有效决策步: {self.min_steps_per_batch}")
        print("=" * 75)

        # 保存第 0 轮
        p0 = self.save_checkpoint(0)
        print(f"[OK] 第 0 轮初始检查点已保存至: {p0}")

        wall_start = time.time()
        for i in range(1, num_iterations + 1):
            t_iter0 = time.time()

            # 1. 采集批轨迹
            batch = self.collect_on_policy_batch()

            # 2. PPO 优化更新
            update_metrics = self.update_ppo(batch)

            iter_wall_time = time.time() - t_iter0

            log_entry = {
                "iteration": i,
                "iter_time_seconds": iter_wall_time,
                "cumulative_wall_time": time.time() - wall_start,
                "num_games": batch.num_games,
                "num_valid_steps": batch.num_valid_steps,
                "mean_return_scaled": float(batch.returns.mean().item()),
                "mean_return_raw": float(batch.returns.mean().item() * 50.0),
                "actor_loss": update_metrics["actor_loss"],
                "critic_loss": update_metrics["critic_loss"],
                "entropy": update_metrics["entropy"],
                "approx_kl": update_metrics["approx_kl"],
                "actor_early_stopped": bool(update_metrics["actor_early_stopped"]),
                "timing": {
                    "iter_total": iter_wall_time,
                    "sampling": batch.timing_metrics["total_sampling_time"],
                    "encoding": batch.timing_metrics["feature_encoding_time"],
                    "inference": batch.timing_metrics["network_inference_time"],
                    "engine_play": batch.timing_metrics["engine_play_time"],
                    "update": update_metrics["update_time"],
                }
            }
            self.history_logs.append(log_entry)

            print(
                f"  Iter [{i:2d}/{num_iterations}] | 耗时: {iter_wall_time:4.1f}s "
                f"(采:{batch.timing_metrics['total_sampling_time']:3.1f}s, 推:{batch.timing_metrics['network_inference_time']:3.1f}s, 优:{update_metrics['update_time']:3.1f}s) | "
                f"局数: {batch.num_games:2d} | 有效步: {batch.num_valid_steps:4d} | "
                f"回报: {log_entry['mean_return_raw']:+5.1f} | "
                f"A-Loss: {update_metrics['actor_loss']:+.3f} | C-Loss: {update_metrics['critic_loss']:.4f} | "
                f"KL: {update_metrics['approx_kl']:.4f}"
            )

            # 前两次更新耗时统计与预算熔断检查
            if i == 2:
                avg_iter_time = (self.history_logs[0]["iter_time_seconds"] + self.history_logs[1]["iter_time_seconds"]) / 2.0
                estimated_total_time = avg_iter_time * num_iterations
                print("-" * 75)
                print(f"  [预算测速报告] 前 2 次平均单轮耗时: {avg_iter_time:.2f}s | 预计完成 {num_iterations} 轮总耗时: {estimated_total_time:.1f}s ({estimated_total_time/60.0:.2f} min)")
                if estimated_total_time > 20 * 60:
                    print(f"  [!] 预计耗时超过 20 分钟熔断预算！保存检查点并报告，不擅自改变采样量。")
                    self.save_checkpoint(i)
                    break
                print("-" * 75)

            # 保存特定检查点 (10, 20)
            if i in (10, 20) or i == num_iterations:
                cp = self.save_checkpoint(i)
                print(f"  [OK] 保存检查点: {cp}")

        # 保存训练完整日志 JSON
        log_path = os.path.join(self.output_dir, self.run_name, "training_log.json")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(self.history_logs, f, indent=2, ensure_ascii=False)
        print(f"\n[✓] 训练日志已成功落盘: {log_path}")

        return self.history_logs
