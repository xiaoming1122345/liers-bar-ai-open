from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GameRules:
    player_count: int = 4
    hand_size: int = 5
    chamber_count: int = 5
    target_rank_count: int = 6
    wild_count: int = 2
    max_play_count: int = 3


@dataclass(frozen=True)
class ScoreRules:
    first: float = 20.0
    second: float = 15.0
    third: float = 10.0
    eliminated_or_last: float = -15.0


@dataclass(frozen=True)
class BeliefConfig:
    particle_count: int = 512
    seed: int | None = None


@dataclass(frozen=True)
class TrainingRunConfig:
    game_count: int = 1
    seed_start: int = 0
    include_all_snapshots: bool = True
    max_steps: int = 512


@dataclass(frozen=True)
class SolverConfig:
    name: str
    iteration_count: int = 1_000
    seed: int | None = None
    max_depth: int | None = None


@dataclass(frozen=True)
class DeepCFRConfig:
    name: str = "deep_cfr"
    cfr_iterations: int = 1_000
    traversals_per_iteration: int = 150
    advantage_buffer_size: int = 2_000_000
    strategy_buffer_size: int = 2_000_000
    batch_size: int = 32768
    learning_rate: float = 1e-3
    train_epochs: int = 150
    hidden_dim: int = 768
    num_layers: int = 4
    dropout: float = 0.1
    eval_interval: int = 50
    eval_games: int = 200
    checkpoint_interval: int = 50
    max_traverse_depth: int | None = 8
    seed: int | None = 42
    device: str = "auto"
    resume_from: str | None = None
    num_workers: int = 16
    precision: str = "bf16"
