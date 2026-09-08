from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from random import Random

import torch
import torch.nn as nn
import torch.optim as optim

from .abstractions import (
    ACTION_SPACE_SIZE,
    ActionAbstractor,
    action_to_index,
)
from .buffers import ReservoirBuffer
from .config import DeepCFRConfig
from .engine import SurvivalGameEngine
from .evaluation import MatchupEvaluator
from .features import FEATURE_DIM, FeatureEncoder
from .networks import (
    AdvantageNetwork,
    StrategyNetwork,
    build_advantage_net,
    build_strategy_net,
)
from .neural_policy import NeuralPolicy
from .opponents import (
    OpponentProfile,
    build_policy_from_profile,
    default_opponent_profiles,
)
from .rewards import TerminalRewardModel
from .self_play import RandomPolicy
from .types import CardKind, ChallengeAction, PlayAction, Policy


# ---------------------------------------------------------------------------
# Policy Space Response League 对手池（解决纯 self-play 循环博弈问题）
# ---------------------------------------------------------------------------

class _LeagueOpponentPool:
    """历史快照联盟对手池（Policy Space Response / League Training）。

    维护历史各时间点的神经网络策略快照，对手从 league 中随机采样，
    从根本上打破纯 self-play 的循环博弈陷阱（A->B->C->A 的循环）。

    采样分布：
    - 70% 从历史快照 league 随机均匀采样
    - 15% 当前最新神经网络（维持一定 self-play 压力）
    -  8% 启发式人机（贝叶斯算牌、残局陷阱等，保持博弈多样性）
    -  7% 纯随机（覆盖罕见边缘状态，增强鲁棒性）
    """

    MAX_LEAGUE_SIZE = 20

    def __init__(
        self,
        profiles: tuple[OpponentProfile, ...],
        rng: Random | None = None,
    ) -> None:
        self.profiles = profiles
        self._rng = rng or Random()
        self._league: list[NeuralPolicy] = []   # 历史快照池
        self._current_policy: NeuralPolicy | None = None

    # ── League 管理 ────────────────────────────────────────────────────────

    def add_snapshot(self, policy: NeuralPolicy) -> None:
        """将当前策略快照加入 league（满时随机替换中间位，永久保留首尾）。"""
        if len(self._league) < self.MAX_LEAGUE_SIZE:
            self._league.append(policy)
        else:
            # 保留第 0（最早均衡）和最后一个（最近），随机替换中间
            idx = self._rng.randint(1, self.MAX_LEAGUE_SIZE - 2)
            self._league[idx] = policy

    def update_current_policy(self, policy: NeuralPolicy) -> None:
        """更新当前最新策略（每轮训练后同步）。"""
        self._current_policy = policy

    def league_size(self) -> int:
        return len(self._league)

    # ── 对手采样 ───────────────────────────────────────────────────────────

    def sample_opponent(self, seed: int) -> Policy:
        """按权重分布采样一个对手策略。"""
        r = self._rng.random()

        if r < 0.70 and self._league:
            # 从历史快照 league 均匀采样（核心机制）
            return self._rng.choice(self._league)

        if r < 0.85 and self._current_policy is not None:
            # 当前最新网络（维持少量 self-play 压力）
            return self._current_policy

        if r < 0.93 and self.profiles:
            # 启发式人机（贝叶斯算牌客、残局诱杀者等）
            profile = self._rng.choice(self.profiles)
            return build_policy_from_profile(profile, seed=seed)

        # 纯随机兜底（覆盖边缘状态）
        return RandomPolicy(seed=seed)


# ---------------------------------------------------------------------------
# 训练结果数据结构
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DeepCFRTrainingResult:
    strategy_net: StrategyNetwork
    advantage_net: AdvantageNetwork
    output_dir: Path
    iterations_completed: int
    total_traversals: int
    final_advantage_loss: float
    final_strategy_loss: float
    eval_results: dict


# ---------------------------------------------------------------------------
# Deep CFR 高性能异步双缓冲训练系统
# ---------------------------------------------------------------------------

class DeepCFRTrainer:
    """逆水寒 / 骗子酒馆真实大逃杀 Deep CFR 高吞吐训练系统。

    核心性能架构：
    1. 生产者-消费者双缓冲重叠流水线（Async Overlap Pipeline）：
       - CPU 端：利用 12~16 个并发 Worker 线程全天候不间断采样博弈树轨迹；
       - GPU 端：RTX 5070 Ti 显存预分配连续张量蓄水池，大批次（Batch=8192）持续反向传播；
       - 彻底消除 CPU 与 GPU 之间交替串行死等的波谷问题。
    2. CPU 无锁本地推理副本：
       - 多线程 Worker 树遍历时的 regret matching 纯在 CPU 内存中运行，零 PCIe 往返延迟，
         彻底解放 GPU 专注于密集梯度反传。
    3. 100% 数据与维度兼容：
       - 特征维度恒定为 80，动作空间恒定为 58，无缝兼容并复用历史 Checkpoint 权重。
    """

    def __init__(
        self,
        config: DeepCFRConfig | None = None,
        engine: SurvivalGameEngine | None = None,
        reward_model: TerminalRewardModel | None = None,
        encoder: FeatureEncoder | None = None,
        abstractor: ActionAbstractor | None = None,
    ) -> None:
        self.config = config or DeepCFRConfig()
        self.engine = engine or SurvivalGameEngine()
        self.reward_model = reward_model or TerminalRewardModel()
        self.encoder = encoder or FeatureEncoder()
        self.abstractor = abstractor or ActionAbstractor()
        self._rng = Random(self.config.seed)

        # 硬件设备判定
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)

        # GPU 主训练神经网络
        self.advantage_net: AdvantageNetwork = build_advantage_net(
            input_dim=FEATURE_DIM,
            action_space_size=ACTION_SPACE_SIZE,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
        ).to(self.device)

        self.strategy_net: StrategyNetwork = build_strategy_net(
            input_dim=FEATURE_DIM,
            action_space_size=ACTION_SPACE_SIZE,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
        ).to(self.device)

        # CPU 轻量推理副本（供多线程 Worker 无锁并发调用，避免冲击 GPU 上下文）
        self._eval_advantage_net: AdvantageNetwork = build_advantage_net(
            input_dim=FEATURE_DIM,
            action_space_size=ACTION_SPACE_SIZE,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            dropout=0.0,
        ).to(torch.device("cpu"))
        self._eval_advantage_net.load_state_dict(self.advantage_net.state_dict())
        self._eval_advantage_net.eval()

        self._eval_strategy_net: StrategyNetwork = build_strategy_net(
            input_dim=FEATURE_DIM,
            action_space_size=ACTION_SPACE_SIZE,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            dropout=0.0,
        ).to(torch.device("cpu"))
        self._eval_strategy_net.load_state_dict(self.strategy_net.state_dict())
        self._eval_strategy_net.eval()

        # 显存常驻连续张量蓄水池
        capacity = getattr(self.config, "advantage_buffer_size", 500_000)
        self.advantage_buffer = ReservoirBuffer(
            capacity=capacity,
            feature_dim=FEATURE_DIM,
            action_space_size=ACTION_SPACE_SIZE,
            device=self.device,
        )
        self.strategy_buffer = ReservoirBuffer(
            capacity=capacity,
            feature_dim=FEATURE_DIM,
            action_space_size=ACTION_SPACE_SIZE,
            device=self.device,
        )

        # 优化器
        self.advantage_optimizer = optim.Adam(
            self.advantage_net.parameters(), lr=self.config.learning_rate
        )
        self.strategy_optimizer = optim.Adam(
            self.strategy_net.parameters(), lr=self.config.learning_rate
        )

        # Policy Space Response League 对手池
        self._opponent_pool = _LeagueOpponentPool(
            profiles=default_opponent_profiles(),
            rng=Random(self.config.seed),
        )

        # 历史最佳评估指标跟踪与存储管理
        self.best_win_rate = -1.0

        # 断点续训支持
        self.start_iteration = 1
        if self.config.resume_from:
            resume_path = Path(self.config.resume_from)
            if not resume_path.is_file():
                raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
            ckpt = torch.load(resume_path, map_location=self.device, weights_only=False)
            self.advantage_net.load_state_dict(ckpt["advantage_net"])
            self.strategy_net.load_state_dict(ckpt["strategy_net"])
            if "advantage_optimizer" in ckpt:
                self.advantage_optimizer.load_state_dict(ckpt["advantage_optimizer"])
            if "strategy_optimizer" in ckpt:
                self.strategy_optimizer.load_state_dict(ckpt["strategy_optimizer"])

            # 权重同步至 CPU 副本
            self._eval_advantage_net.load_state_dict(
                {k: v.cpu() for k, v in self.advantage_net.state_dict().items()}
            )
            self._eval_advantage_net.eval()
            self._eval_strategy_net.load_state_dict(
                {k: v.cpu() for k, v in self.strategy_net.state_dict().items()}
            )
            self._eval_strategy_net.eval()

            # 当前网络作为 league 最新策略
            cpu_neural_policy = NeuralPolicy(
                strategy_net=self._eval_strategy_net,
                encoder=self.encoder,
                abstractor=self.abstractor,
                seed=self.config.seed or 42,
            )
            self._opponent_pool.update_current_policy(cpu_neural_policy)

            # 从断点目录（以及同级历史 checkpoint 目录）预加载历史快照初始化 league
            self._preload_league_snapshots(resume_path)

            self.start_iteration = ckpt.get("iteration", 0) + 1
            print(
                f"[Deep CFR] 成功从断点 {resume_path} 续训，"
                f"起始轮次: {self.start_iteration} / 目标轮次: {self.config.cfr_iterations}"
            )
            print(
                f"[Deep CFR] League 历史快照池已预载: {self._opponent_pool.league_size()} 个"
            )


    def _preload_league_snapshots(self, resume_path: Path) -> None:
        """从断点所在目录（及 v13 历史目录）加载历史策略快照初始化 league。"""
        candidate_dirs = [resume_path.parent]
        base_runs = Path("runs_deep_cfr")
        if base_runs.is_dir():
            for d in base_runs.iterdir():
                if d.is_dir() and d not in candidate_dirs:
                    candidate_dirs.append(d)

        loaded = 0
        for ckpt_dir in candidate_dirs:
            ckpt_files = sorted(ckpt_dir.glob("checkpoint_iter_*.pt"))
            for ckpt_file in ckpt_files:
                try:
                    c = torch.load(ckpt_file, map_location="cpu", weights_only=False)
                    snap_net = build_strategy_net(
                        input_dim=FEATURE_DIM,
                        action_space_size=ACTION_SPACE_SIZE,
                        hidden_dim=c.get("hidden_dim", self.config.hidden_dim),
                        num_layers=c.get("num_layers", self.config.num_layers),
                        dropout=0.0,
                    )
                    snap_net.load_state_dict(c["strategy_net"])
                    snap_net.eval()
                    snap_policy = NeuralPolicy(
                        strategy_net=snap_net,
                        encoder=self.encoder,
                        abstractor=self.abstractor,
                        seed=c.get("iteration", 0),
                    )
                    self._opponent_pool.add_snapshot(snap_policy)
                    loaded += 1
                except Exception:
                    pass
        print(f"[Deep CFR] 预加载历史快照完成，共载入 {loaded} 个（League 当前大小: {self._opponent_pool.league_size()}）")

    def train(self, output_dir: str | Path) -> DeepCFRTrainingResult:
        """执行异步生产者-消费者流水线主循环。"""

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        total_traversals = 0
        last_adv_loss = 0.0
        last_strat_loss = 0.0
        eval_results: dict = {}
        log_entries: list[dict] = []

        log_file = output_path / "training_log.json"
        if log_file.is_file():
            try:
                log_entries = json.loads(log_file.read_text(encoding="utf-8"))
                for entry in log_entries:
                    ev = entry.get("eval")
                    if isinstance(ev, dict) and "win_rate" in ev:
                        self.best_win_rate = max(self.best_win_rate, ev["win_rate"])
                print(f"[Deep CFR] 已读取历史日志 {len(log_entries)} 条，历史最高吃鸡胜率锁定为: {self.best_win_rate:.3f}")
            except Exception as e:
                print(f"[Deep CFR] 读取历史日志警告: {e}")

        # 限制 PyTorch 内部 CPU 单张量线程数为 1，防止多线程 Worker 派生 OpenMP 线程爆炸（避免数百线程自旋互锁）
        torch.set_num_threads(1)

        num_workers = max(1, getattr(self.config, "num_workers", 16))

        print(f"[Deep CFR] 计算设备: {self.device}")
        print(f"[Deep CFR] 特征空间维度: {FEATURE_DIM}, 动作空间维度: {ACTION_SPACE_SIZE}")
        print(f"[Deep CFR] 异步并发流水线启动: {num_workers} 个后台 CPU 采样 Worker 持续供数")
        print(f"[Deep CFR] 目标: {self.config.cfr_iterations} 轮，单轮 GPU 梯度反传步数: {self.config.train_epochs} 大批次")
        print()

        train_start = time.time()
        sample_queue: queue.Queue = queue.Queue(maxsize=100000)
        stop_event = threading.Event()
        current_iteration = [self.start_iteration]

        # ── 后台 CPU 多线程持续采样 Worker ────────────────────────────────
        def _sampler_worker(worker_id: int):
            # 每个子线程绑定独立的单线程 PyTorch 环境与随机种子
            torch.set_num_threads(1)
            worker_seed_base = (
                self.config.seed + worker_id * 10007
                if self.config.seed is not None
                else worker_id * 10007 + int(time.time())
            )
            local_rng = Random(worker_seed_base)
            local_abstractor = ActionAbstractor()
            local_encoder = FeatureEncoder()
            t_count = 0

            while not stop_event.is_set():
                # 队列过载背压调控（防止无限堆积占用系统内存）
                if sample_queue.qsize() > 30000:
                    time.sleep(0.02)
                    continue

                cur_iter = current_iteration[0]
                game_seed = local_rng.randint(0, 2**31 - 1)
                seat = (t_count % self.engine.player_count) + 1
                init_state = self.engine.new_game(seed=game_seed, starting_seat=1)
                adv_list: list = []
                strat_list: list = []

                try:
                    self._traverse(
                        state=init_state,
                        traverser_seat=seat,
                        iteration=cur_iter,
                        depth=0,
                        seed_base=game_seed,
                        adv_samples=adv_list,
                        strat_samples=strat_list,
                        worker_rng=local_rng,
                        worker_abstractor=local_abstractor,
                        worker_encoder=local_encoder,
                    )
                    if adv_list or strat_list:
                        sample_queue.put((adv_list, strat_list), timeout=1.0)
                    t_count += 1
                except Exception:
                    pass

        # 启动后台常驻线程集群
        threads: list[threading.Thread] = []
        for i in range(num_workers):
            t = threading.Thread(target=_sampler_worker, args=(i,), daemon=True)
            t.start()
            threads.append(t)

        try:
            # ── 启动前预热：若显存蓄水池为空，稍候收集首批博弈轨迹 ──
            if len(self.advantage_buffer) < 512:
                print("[Deep CFR] 正在进行初始点火采集（收集首批轨迹）...")
                while len(self.advantage_buffer) < 512 and not stop_event.is_set():
                    try:
                        adv_l, strat_l = sample_queue.get(timeout=0.1)
                        if adv_l:
                            self.advantage_buffer.extend(adv_l)
                        if strat_l:
                            self.strategy_buffer.extend(strat_l)
                        total_traversals += 1
                    except queue.Empty:
                        pass
                print(f"[Deep CFR] 点火完成！初始显存样本量: {len(self.advantage_buffer)}，GPU/CPU 满血重叠流水线全速开跑！\n")

            # ── 正式流水线主循环（GPU 持续全速矩阵更新，CPU 持续全速注水）──
            for iteration in range(self.start_iteration, self.config.cfr_iterations + 1):
                iter_start = time.time()
                current_iteration[0] = iteration

                # ── 1. 零等待 Drain：把后台 Worker 在上一轮训练期间产出的所有新样本瞬间灌入显存 ──
                drained_adv: list = []
                drained_strat: list = []
                while not sample_queue.empty():
                    try:
                        adv_l, strat_l = sample_queue.get_nowait()
                        drained_adv.extend(adv_l)
                        drained_strat.extend(strat_l)
                        total_traversals += 1
                    except queue.Empty:
                        break

                if drained_adv:
                    self.advantage_buffer.extend(drained_adv)
                if drained_strat:
                    self.strategy_buffer.extend(drained_strat)

                # ── 2. GPU 显存直通大张量反向传播 (与后台 CPU 采样完全无缝重叠) ──
                if len(self.advantage_buffer) >= 512:
                    adv_bs = min(self.config.batch_size, len(self.advantage_buffer))
                    last_adv_loss = self._train_network(
                        net=self.advantage_net,
                        optimizer=self.advantage_optimizer,
                        buffer=self.advantage_buffer,
                        epochs=self.config.train_epochs,
                        batch_size=adv_bs,
                    )
                    # 快速同步权重至 CPU 推理副本（耗时约 1ms）
                    self._eval_advantage_net.load_state_dict(
                        {k: v.cpu() for k, v in self.advantage_net.state_dict().items()}
                    )
                    self._eval_advantage_net.eval()

                if len(self.strategy_buffer) >= 512:
                    strat_bs = min(self.config.batch_size, len(self.strategy_buffer))
                    last_strat_loss = self._train_network(
                        net=self.strategy_net,
                        optimizer=self.strategy_optimizer,
                        buffer=self.strategy_buffer,
                        epochs=self.config.train_epochs,
                        batch_size=strat_bs,
                    )
                    self._eval_strategy_net.load_state_dict(
                        {k: v.cpu() for k, v in self.strategy_net.state_dict().items()}
                    )
                    self._eval_strategy_net.eval()
                    cpu_neural_policy = NeuralPolicy(
                        strategy_net=self._eval_strategy_net,
                        encoder=self.encoder,
                        abstractor=self.abstractor,
                        seed=iteration,
                    )
                    # League：更新当前最新策略
                    self._opponent_pool.update_current_policy(cpu_neural_policy)
                    # League：每 25 轮将当前网络快照存入历史池
                    if iteration % 25 == 0:
                        import copy
                        snap_net = copy.deepcopy(self._eval_strategy_net)
                        snap_policy = NeuralPolicy(
                            strategy_net=snap_net,
                            encoder=self.encoder,
                            abstractor=self.abstractor,
                            seed=iteration,
                        )
                        self._opponent_pool.add_snapshot(snap_policy)
                        print(f"  📸 [League] iter {iteration} 快照已入库 (League 大小: {self._opponent_pool.league_size()})")

                iter_time = time.time() - iter_start

                # ── 3. 统计与日志 ──────────────────────────────────────────
                entry = {
                    "iteration": iteration,
                    "traversals": total_traversals,
                    "adv_buffer": len(self.advantage_buffer),
                    "strat_buffer": len(self.strategy_buffer),
                    "adv_loss": round(last_adv_loss, 6),
                    "strat_loss": round(last_strat_loss, 6),
                    "league_size": self._opponent_pool.league_size(),
                    "iter_time": round(iter_time, 2),
                }

                total_elapsed = time.time() - train_start
                print(
                    f"[iter {iteration:4d}/{self.config.cfr_iterations}] "
                    f"adv_loss={last_adv_loss:.5f}  "
                    f"strat_loss={last_strat_loss:.5f}  "
                    f"league={self._opponent_pool.league_size()}  "

                    f"buf=({len(self.advantage_buffer)}/{len(self.strategy_buffer)})  "
                    f"{iter_time:.2f}s/iter  "
                    f"({total_elapsed:.0f}s total)"
                )

                # 定期评估与 Checkpoint 保存
                if iteration % self.config.eval_interval == 0:
                    eval_results = self._evaluate(iteration)
                    entry["eval"] = eval_results
                    win_rate = eval_results.get("win_rate", 0.0)
                    print(
                        f"  >> [eval@{iteration}] "
                        f"吃鸡胜率={win_rate:.3f}  "
                        f"前二率={eval_results.get('top2_rate', 0.0):.3f}  "
                        f"平均效用={eval_results.get('mean_utility', 0.0):.2f}"
                    )
                    if win_rate > self.best_win_rate:
                        self.best_win_rate = win_rate
                        self._save_checkpoint(output_path, iteration, tag="best")
                        print(f"  🏆 [黄金模型] 刷新历史最高吃鸡胜率 ({win_rate:.3f})，已持久化 policy_best.pt")

                if iteration % self.config.checkpoint_interval == 0:
                    self._save_checkpoint(output_path, iteration)
                    try:
                        (output_path / "training_log.json").write_text(
                            json.dumps(log_entries, indent=2, ensure_ascii=False),
                            encoding="utf-8",
                        )
                    except Exception:
                        pass

                log_entries.append(entry)

        finally:
            stop_event.set()

        # 终局保存与报告
        self._save_checkpoint(output_path, self.config.cfr_iterations, tag="final")
        latest_path = output_path / "policy_latest.pt"
        torch.save(
            {
                "iteration": self.config.cfr_iterations,
                "strategy_net": self.strategy_net.state_dict(),
                "hidden_dim": self.config.hidden_dim,
                "num_layers": self.config.num_layers,
                "feature_dim": FEATURE_DIM,
                "action_space_size": ACTION_SPACE_SIZE,
            },
            latest_path,
        )

        final_eval = self._evaluate(self.config.cfr_iterations)
        report = {
            "mode": "deep_cfr_train",
            "iterations": self.config.cfr_iterations,
            "total_traversals": total_traversals,
            "final_adv_loss": round(last_adv_loss, 6),
            "final_strat_loss": round(last_strat_loss, 6),
            "adv_buffer_size": len(self.advantage_buffer),
            "strat_buffer_size": len(self.strategy_buffer),
            "device": str(self.device),
            "eval": final_eval,
            "elapsed_seconds": round(time.time() - train_start, 1),
            "config": asdict(self.config),
        }
        (output_path / "report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (output_path / "training_log.json").write_text(
            json.dumps(log_entries, indent=2), encoding="utf-8"
        )

        elapsed_total = time.time() - train_start
        print(f"\n[Deep CFR] 训练完成！总耗时: {elapsed_total:.0f}s，完成遍历: {total_traversals}")
        print(f"[Deep CFR] 终局评估结果: {final_eval}")
        print(f"[Deep CFR] 权重已持久化到: {output_path}")

        return DeepCFRTrainingResult(
            strategy_net=self.strategy_net,
            advantage_net=self.advantage_net,
            output_dir=output_path,
            iterations_completed=self.config.cfr_iterations,
            total_traversals=total_traversals,
            final_advantage_loss=last_adv_loss,
            final_strategy_loss=last_strat_loss,
            eval_results=final_eval,
        )

    # ──────────────────────────────────────────────────────────────────────
    # 博弈树外采样遍历核心 (MCCFR Traversal)
    # ──────────────────────────────────────────────────────────────────────

    def _traverse(
        self,
        state,
        traverser_seat: int,
        iteration: int,
        depth: int,
        seed_base: int = 0,
        adv_samples: list | None = None,
        strat_samples: list | None = None,
        worker_rng: Random | None = None,
        worker_abstractor: ActionAbstractor | None = None,
        worker_encoder: FeatureEncoder | None = None,
    ) -> float:
        """外采样 MCCFR 递归遍历。

        - traverser 节点：遍历全部抽象动作，收集后悔值样本 -> 存入 advantage_buffer；
        - 对手节点：由策略采样单个动作，收集策略样本 -> 存入 strategy_buffer。
        """
        rng = worker_rng or self._rng
        abstractor = worker_abstractor or self.abstractor
        encoder = worker_encoder or self.encoder

        if adv_samples is None:
            adv_samples = []
        if strat_samples is None:
            strat_samples = []

        if self.engine.is_terminal(state):
            return self.reward_model.evaluate(state).for_seat(traverser_seat)

        if (
            self.config.max_traverse_depth is not None
            and depth >= self.config.max_traverse_depth
        ):
            return self._leaf_utility(state, traverser_seat)

        acting_seat = self.engine.current_actor(state)
        legal_actions = self.engine.legal_actions(state)
        if not legal_actions:
            return 0.0

        observation = self.engine.observe(state, acting_seat)
        grouped = abstractor.abstract_legal_actions(legal_actions, observation)
        abstract_actions = list(grouped.keys())

        features = encoder.encode(observation)
        strategy = self._get_strategy(features, abstract_actions)

        if acting_seat == traverser_seat:
            action_utilities: dict[str, float] = {}
            for aa in abstract_actions:
                concrete = rng.choice(grouped[aa])
                next_state = self.engine.clone_state(state)

                p_before = state.player_by_seat(traverser_seat)
                shots_before = p_before.shots_taken

                self.engine.apply_action(next_state, concrete)

                tactical_bonus = 0.0

                # 1. 质疑决策即时奖惩（鼓励抓诈，严惩盲目抓真牌）
                if isinstance(concrete, ChallengeAction):
                    last_ev = next_state.public_history[-1] if next_state.public_history else None
                    if last_ev and last_ev.event_type == "challenge":
                        outcome = last_ev.detail.get("outcome")
                        if outcome == "lie":
                            tactical_bonus += 5.0  # 成功识破谎言大奖！
                        elif outcome == "honest":
                            tactical_bonus -= 5.0  # 盲目抓真牌挨枪受惩！
                        elif outcome == "ghost":
                            tactical_bonus -= 3.0  # 中对手魔牌陷阱惩罚

                # 2. 递归获取子树价值
                sub_util = self._traverse(
                    next_state,
                    traverser_seat,
                    iteration,
                    depth + 1,
                    seed_base,
                    adv_samples,
                    strat_samples,
                    worker_rng=rng,
                    worker_abstractor=abstractor,
                    worker_encoder=encoder,
                )

                # 3. 出牌决策的延迟后果评估（严惩诈唬被抓暴毙，奖励老实出牌与魔牌陷阱）
                if isinstance(concrete, PlayAction):
                    is_bluff = False
                    for cid in concrete.card_ids:
                        c = next((card for card in p_before.hand if card.card_id == cid), None)
                        if c and c.kind != CardKind.WILD and c.kind != CardKind.GHOST and c.printed_rank != concrete.claim_rank:
                            is_bluff = True
                            break

                    p_end = next_state.player_by_seat(traverser_seat)
                    if p_end.shots_taken > shots_before:
                        if is_bluff:
                            tactical_bonus -= 6.0  # 诈唬被抓重惩
                            if not p_end.alive:
                                tactical_bonus -= 10.0  # 诈唬自杀暴毙极刑
                    else:
                        if not is_bluff:
                            tactical_bonus += 2.0  # 真牌安全脱身奖励

                total_action_util = max(-25.0, min(25.0, sub_util + tactical_bonus))
                action_utilities[aa.label] = total_action_util

            # 当前节点的无偏期望价值 V(I) = sum_a sigma(a) * u(a)
            expected_value = sum(
                strategy.get(a.label, 0.0) * action_utilities[a.label]
                for a in abstract_actions
            )

            # 正统 External Sampling 后悔值：Advantage = u(a) - V(I)（完全零除法，零方差爆炸！）
            advantage_target = torch.zeros(ACTION_SPACE_SIZE)
            for a in abstract_actions:
                idx = action_to_index(a.label)
                advantage_target[idx] = action_utilities[a.label] - expected_value

            adv_samples.append((features.detach().cpu(), advantage_target, iteration))

            # 记录策略网络训练样本
            strategy_target = torch.zeros(ACTION_SPACE_SIZE)
            for aa in abstract_actions:
                strategy_target[action_to_index(aa.label)] = strategy.get(aa.label, 0.0)
            strat_samples.append((features.detach().cpu(), strategy_target, iteration))

            return expected_value

        else:
            strategy_target = torch.zeros(ACTION_SPACE_SIZE)
            for abstract_action in abstract_actions:
                idx = action_to_index(abstract_action.label)
                strategy_target[idx] = strategy.get(abstract_action.label, 0.0)
            strat_samples.append((features.detach().cpu(), strategy_target, iteration))

            # ── 对手节点决策：Policy Space Response League 架构（v15）──
            # 从历史快照联盟中采样对手（70% 历史快照 + 15% 最新网络 + 8% 启发式 + 7% 随机），
            # 彻底消除纯 self-play 的循环博弈陷阱。
            concrete = None
            opp_policy = self._opponent_pool.sample_opponent(
                seed=seed_base + depth * 37 + acting_seat
            )
            try:
                concrete = opp_policy.choose_action(observation, legal_actions)
            except Exception:
                concrete = None

            if concrete is None:
                sampled = self._sample_from_strategy(abstract_actions, strategy, rng)
                concrete = rng.choice(grouped[sampled])


            next_state = self.engine.clone_state(state)
            self.engine.apply_action(next_state, concrete)
            return self._traverse(
                next_state,
                traverser_seat,
                iteration,
                depth + 1,
                seed_base,
                adv_samples,
                strat_samples,
                worker_rng=rng,
                worker_abstractor=abstractor,
                worker_encoder=encoder,
            )

    @torch.no_grad()
    def _get_strategy(
        self, features: torch.Tensor, abstract_actions: list
    ) -> dict[str, float]:
        """纯 CPU 极速前向推理计算当前后悔值与策略分布（零 PCIe 延迟，无锁并发）。"""
        x = features.unsqueeze(0)  # CPU tensor
        advantages = self._eval_advantage_net(x).squeeze(0)

        positive_advantages: dict[str, float] = {}
        for a in abstract_actions:
            idx = action_to_index(a.label)
            val = max(0.0, advantages[idx].item())
            existing = positive_advantages.get(a.label, 0.0)
            positive_advantages[a.label] = max(existing, val)

        total = sum(positive_advantages.values())
        if total <= 0:
            prob = 1.0 / len(abstract_actions)
            return {a.label: prob for a in abstract_actions}
        return {label: val / total for label, val in positive_advantages.items()}

    def _sample_from_strategy(
        self, abstract_actions: list, strategy: dict[str, float], rng: Random
    ):
        """按概率分布采样抽象动作。"""
        threshold = rng.random()
        cumulative = 0.0
        for action in abstract_actions:
            cumulative += strategy.get(action.label, 0.0)
            if threshold <= cumulative:
                return action
        return abstract_actions[-1]

    def _leaf_utility(self, state, traverser_seat: int) -> float:
        player = state.player_by_seat(traverser_seat)
        if not player.alive:
            return self.reward_model.score_rules.eliminated_or_last

        alive_count = sum(1 for p in state.players if p.alive)
        survival_bonus = (4 - alive_count) * 4.0
        shot_disadv = -player.shots_taken * 3.5
        hand_adv = (5 - len(player.hand)) * 1.0

        score = survival_bonus + shot_disadv + hand_adv
        return float(
            max(
                self.reward_model.score_rules.eliminated_or_last,
                min(self.reward_model.score_rules.first, score),
            )
        )

    def _train_network(
        self,
        net: nn.Module,
        optimizer: optim.Optimizer,
        buffer: ReservoirBuffer,
        epochs: int,
        batch_size: int | None = None,
    ) -> float:
        """GPU 显存直通大张量高吞吐梯度反传。"""
        net.train()
        bs = batch_size or self.config.batch_size
        total_loss = 0.0
        num_batches = 0

        for _ in range(epochs):
            features, targets, weights = buffer.sample_batch(bs, device=self.device)

            predictions = net(features)
            per_sample = ((predictions - targets) ** 2).mean(dim=1)
            loss = (per_sample * weights).mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        net.eval()
        return total_loss / max(1, num_batches)

    # ──────────────────────────────────────────────────────────────────────
    # 周期评估与 Checkpoint
    # ──────────────────────────────────────────────────────────────────────

    def _evaluate(self, iteration: int) -> dict:
        """在真实对局中评估当前神经网络策略。"""
        try:
            self.strategy_net.eval()
            neural_policy = NeuralPolicy(
                strategy_net=self.strategy_net,
                encoder=self.encoder,
                abstractor=self.abstractor,
                greedy=True,
                seed=iteration,
            )

            evaluator = MatchupEvaluator(
                engine=self.engine,
                reward_model=self.reward_model,
            )

            def hero_factory(game_index: int, seat: int) -> Policy:
                return neural_policy

            result = evaluator.evaluate_hero_vs_pool(
                hero_policy_factory=hero_factory,
                opponent_profiles=default_opponent_profiles(),
                game_count=self.config.eval_games,
                hero_seat=1,
                seed_start=iteration * 10000,
            )
            summary = result.summary
            return {
                "iteration": iteration,
                "win_rate": summary.win_rate,
                "top2_rate": summary.top2_rate,
                "mean_rank": summary.mean_rank,
                "mean_utility": summary.mean_utility,
            }
        except Exception as e:
            return {"iteration": iteration, "error": str(e)}

    def _save_checkpoint(
        self, output_dir: Path, iteration: int, tag: str | None = None
    ) -> None:
        suffix = tag or f"iter_{iteration:05d}"
        ckpt = {
            "iteration": iteration,
            "hidden_dim": self.config.hidden_dim,
            "num_layers": self.config.num_layers,
            "feature_dim": FEATURE_DIM,
            "action_space_size": ACTION_SPACE_SIZE,
            "advantage_net": self.advantage_net.state_dict(),
            "strategy_net": self.strategy_net.state_dict(),
            "advantage_optimizer": self.advantage_optimizer.state_dict(),
            "strategy_optimizer": self.strategy_optimizer.state_dict(),
        }
        if tag == "best":
            torch.save(ckpt, output_dir / "policy_best.pt")
            return

        # 完整保留每一个轮次的 Checkpoint 与最新策略，供后续数据分析与复盘
        torch.save(ckpt, output_dir / f"checkpoint_{suffix}.pt")
        torch.save(ckpt, output_dir / "policy_latest.pt")
