from __future__ import annotations

import random
import threading
from dataclasses import dataclass, field

import torch


class ReservoirBuffer:
    """高性能连续张量蓄水池缓冲区（线程安全版）。
    
    使用预分配连续张量存储样本，消除 Python list 的遍历和低效逐元素 stack，
    使得 sample_batch 操作纯在底层的 C++/张量索引中以微秒级完成，
    让 GPU 能够零等待以大批次持续跑满 Tensor Core。
    """

    def __init__(
        self,
        capacity: int = 500_000,
        feature_dim: int = 80,
        action_space_size: int = 58,
        device: torch.device | str = "cpu",
    ) -> None:
        self.capacity = capacity
        self.feature_dim = feature_dim
        self.action_space_size = action_space_size
        self.device = torch.device(device)
        self._lock = threading.Lock()
        # 预分配在 GPU 显存或目标设备，享受数百 GB/s 显存带宽，消除采样 PCIe 瓶颈
        self._features = torch.zeros((capacity, feature_dim), dtype=torch.float32, device=self.device)
        self._targets = torch.zeros((capacity, action_space_size), dtype=torch.float32, device=self.device)
        self._weights = torch.zeros((capacity,), dtype=torch.float32, device=self.device)
        self._size = 0
        self._count = 0

    def __len__(self) -> int:
        return self._size

    @property
    def total_seen(self) -> int:
        return self._count

    def add(self, features: torch.Tensor, targets: torch.Tensor, iteration: int) -> None:
        """Add a single sample using reservoir sampling."""
        f_dev = features.to(self.device)
        t_dev = targets.to(self.device)
        with self._lock:
            self._count += 1
            if self._size < self.capacity:
                idx = self._size
                self._size += 1
                self._features[idx] = f_dev
                self._targets[idx] = t_dev
                self._weights[idx] = float(iteration)
            else:
                j = random.randrange(self._count)
                if j < self.capacity:
                    self._features[j] = f_dev
                    self._targets[j] = t_dev
                    self._weights[j] = float(iteration)

    def extend(
        self,
        samples: list[tuple[torch.Tensor, torch.Tensor, int]] | tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        """批量写入样本，一次性传输到显存，极大减少频繁拷贝。"""
        if not samples:
            return
        if isinstance(samples, tuple) and len(samples) == 3 and isinstance(samples[0], torch.Tensor):
            batch_f = samples[0].to(self.device)
            batch_t = samples[1].to(self.device)
            batch_w = samples[2].to(self.device)
            n = batch_f.size(0)
            if n == 0:
                return
        else:
            n = len(samples)
            batch_f = torch.stack([s[0] for s in samples]).to(self.device)
            batch_t = torch.stack([s[1] for s in samples]).to(self.device)
            batch_w = torch.tensor([float(s[2]) for s in samples], dtype=torch.float32, device=self.device)

        with self._lock:
            available = self.capacity - self._size
            if available >= n:
                self._features[self._size : self._size + n] = batch_f
                self._targets[self._size : self._size + n] = batch_t
                self._weights[self._size : self._size + n] = batch_w
                self._size += n
                self._count += n
            else:
                if available > 0:
                    self._features[self._size : self._size + available] = batch_f[:available]
                    self._targets[self._size : self._size + available] = batch_t[:available]
                    self._weights[self._size : self._size + available] = batch_w[:available]
                    self._size = self.capacity
                    self._count += available

                rem_f = batch_f[available:]
                rem_t = batch_t[available:]
                rem_w = batch_w[available:]
                rem_len = len(rem_f)
                if rem_len > 0:
                    start_count = self._count
                    i_indices = torch.arange(
                        start_count + 1,
                        start_count + rem_len + 1,
                        dtype=torch.float32,
                        device=self.device,
                    )
                    accept_probs = torch.clamp(float(self.capacity) / i_indices, max=1.0)
                    rand_uniform = torch.rand(rem_len, device=self.device)
                    accepted_mask = rand_uniform < accept_probs

                    self._count += rem_len

                    num_accepted = int(accepted_mask.sum().item())
                    if num_accepted > 0:
                        acc_f = rem_f[accepted_mask]
                        acc_t = rem_t[accepted_mask]
                        acc_w = rem_w[accepted_mask]
                        replace_idx = torch.randint(0, self.capacity, (num_accepted,), device=self.device)
                        self._features[replace_idx] = acc_f
                        self._targets[replace_idx] = acc_t
                        self._weights[replace_idx] = acc_w

    def sample_batch(
        self, batch_size: int, device: torch.device | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """纯 GPU 显存内极速切片采样 Mini-batch（完全零 PCIe 传输开销）。"""
        with self._lock:
            if batch_size > self._size:
                raise ValueError(
                    f"Cannot sample batch of size {batch_size} from buffer of size {self._size}"
                )

            indices = torch.randint(0, self._size, (batch_size,), device=self.device)
            features_batch = self._features[indices]
            targets_batch = self._targets[indices]
            weights_batch = self._weights[indices]

            max_weight = float(weights_batch.max().item())
            if max_weight > 0:
                weights_batch = weights_batch / max_weight

        return features_batch, targets_batch, weights_batch

    def clear(self) -> None:
        """Reset the buffer."""
        with self._lock:
            self._size = 0
            self._count = 0

    def state_dict(self) -> dict:
        """Serialize buffer state (only clones active slice to CPU to minimize disk size)."""
        with self._lock:
            return {
                "capacity": self.capacity,
                "feature_dim": self.feature_dim,
                "action_space_size": self.action_space_size,
                "size": self._size,
                "count": self._count,
                "features": self._features[:self._size].detach().cpu().clone(),
                "targets": self._targets[:self._size].detach().cpu().clone(),
                "weights": self._weights[:self._size].detach().cpu().clone(),
            }

    def load_state_dict(self, state: dict) -> None:
        """Restore buffer state from state_dict."""
        with self._lock:
            self.capacity = state["capacity"]
            self.feature_dim = state["feature_dim"]
            self.action_space_size = state["action_space_size"]
            self._size = state["size"]
            self._count = state["count"]

            if self._features.shape[0] != self.capacity or self._features.shape[1] != self.feature_dim:
                self._features = torch.zeros((self.capacity, self.feature_dim), dtype=torch.float32, device=self.device)
                self._targets = torch.zeros((self.capacity, self.action_space_size), dtype=torch.float32, device=self.device)
                self._weights = torch.zeros((self.capacity,), dtype=torch.float32, device=self.device)

            if self._size > 0:
                self._features[:self._size] = state["features"].to(self.device)
                self._targets[:self._size] = state["targets"].to(self.device)
                self._weights[:self._size] = state["weights"].to(self.device)

