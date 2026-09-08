from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)

# 屏蔽 Windows Python 3.12 容易在 WMI RPC 调用中死锁的 _wmi 扩展，保障极端并发下秒级加载
sys.modules["_wmi"] = None

# 当多进程作为 Worker 子进程被 spawn 时 (__name__ == "__mp_main__")：
# 彻底屏蔽 GPU（设为 -1），使 16 个 Worker 纯走 CPU 独立多核，绝不向 NVIDIA 驱动注册任何 CUDA 句柄或争用 context
if __name__ == "__mp_main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 仅在作为主入口执行时导入业务全量符号；子进程 spawn (__mp_main__) 时直接跳过，彻底杜绝并发 import 内存爆炸
if __name__ == "__main__":
    from card_ai import (
        ArtifactStore,
        DeepCFRConfig,
        DeepCFRTrainer,
        MatchupEvaluator,
        OpponentAwareAdaptiveSolver,
        RandomPolicy,
        SolverConfig,
        SolverPolicy,
        StyleResponseModel,
        StyleResponseTrainer,
        TabularMCCFRTrainer,
        TrainingBackend,
        TrainingRunConfig,
        UniformRandomSolver,
        default_opponent_profiles,
    )


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _parse_custom_ids(raw: str | None) -> dict[int, str] | None:
    if not raw:
        return None
    payload = json.loads(raw)
    return {int(seat): str(value) for seat, value in payload.items()}


def _make_random_policy_factory(seed_base: int):
    def factory(game_index: int, seat: int):
        return RandomPolicy(seed=seed_base + game_index * 101 + seat)

    return factory, None


def _make_tabular_policy_factory(
    *,
    name: str,
    iterations: int,
    seed: int,
    max_depth: int,
    policy_seed_base: int,
):
    trainer = TabularMCCFRTrainer()
    training_result = trainer.train(
        SolverConfig(
            name=name,
            iteration_count=iterations,
            seed=seed,
            max_depth=max_depth,
        )
    )

    def factory(game_index: int, seat: int):
        return SolverPolicy(
            solver=training_result.solver,
            seed=policy_seed_base + game_index * 101 + seat,
        )

    return factory, training_result


def _make_hero_policy_factory(args: argparse.Namespace):
    response_model = None
    if getattr(args, "response_prior_path", None):
        response_model = StyleResponseModel.load(args.response_prior_path)
    if args.hero_mode == "random":
        return _make_random_policy_factory(seed_base=args.policy_seed_base)
    if args.hero_mode == "uniform_solver":
        solver = UniformRandomSolver()

        def factory(game_index: int, seat: int):
            return SolverPolicy(
                solver=solver,
                seed=args.policy_seed_base + game_index * 101 + seat,
            )

        return factory, None
    policy_factory, training_result = _make_tabular_policy_factory(
        name=args.name,
        iterations=args.iterations,
        seed=args.seed,
        max_depth=args.max_depth,
        policy_seed_base=args.policy_seed_base,
    )
    if args.hero_mode == "adaptive_tabular":
        adaptive_solver = OpponentAwareAdaptiveSolver(
            training_result.solver,
            response_model=response_model,
        )

        def adaptive_factory(game_index: int, seat: int):
            return SolverPolicy(
                solver=adaptive_solver,
                seed=args.policy_seed_base + game_index * 101 + seat,
            )

        return adaptive_factory, training_result
    return policy_factory, training_result


def _run_random_eval(
    *,
    games: int,
    seed_start: int,
    max_steps: int,
    output_dir: str,
    include_all_snapshots: bool,
) -> dict:
    backend = TrainingBackend()
    config = TrainingRunConfig(
        game_count=games,
        seed_start=seed_start,
        include_all_snapshots=include_all_snapshots,
        max_steps=max_steps,
    )
    batch = backend.collect_random_self_play_batch(
        game_count=config.game_count,
        seed_start=config.seed_start,
        include_all_snapshots=config.include_all_snapshots,
        max_steps=config.max_steps,
    )
    summary = backend.summarize_batch(batch)

    output_dir_path = Path(output_dir)
    run_name = f"random_eval_g{games}_s{seed_start}"
    report_path = _write_json(
        output_dir_path / run_name / "report.json",
        {
            "mode": "random_eval",
            "config": asdict(config),
            "summary": asdict(summary),
        },
    )
    return {
        "mode": "random_eval",
        "report_path": str(report_path),
        "config": asdict(config),
        "summary": asdict(summary),
        "sample_count": summary.sample_count,
        "mean_terminal_utility": round(summary.mean_terminal_utility, 4),
        "seat_mean_terminal_utility": summary.seat_mean_terminal_utility,
        "mean_modeling_degree": round(summary.mean_modeling_degree, 4),
    }


def _run_style_response_train(
    *,
    games: int,
    seed_start: int,
    max_steps: int,
    output_dir: str,
    run_name: str,
) -> dict:
    trainer = StyleResponseTrainer()
    result = trainer.train_pool_priors(
        opponent_profiles=default_opponent_profiles(),
        game_count=games,
        seed_start=seed_start,
        max_steps=max_steps,
    )
    run_dir = Path(output_dir) / run_name
    model = StyleResponseModel(priors=result.priors)
    prior_path = model.save(run_dir / "style_response_priors.json")
    report_path = _write_json(
        run_dir / "report.json",
        {
            "mode": "style_response_train",
            "summary": result.to_dict(),
        },
    )
    return {
        "mode": "style_response_train",
        "report_path": str(report_path),
        "prior_path": str(prior_path),
        "profile_counts": result.profile_counts,
        "style_clusters": [prior.public_style_cluster for prior in result.priors],
    }


def _random_eval(args: argparse.Namespace) -> None:
    result = _run_random_eval(
        games=args.games,
        seed_start=args.seed_start,
        max_steps=args.max_steps,
        output_dir=args.output_dir,
        include_all_snapshots=args.include_all_snapshots,
    )
    print(json.dumps(result, ensure_ascii=True))


def _style_response_train(args: argparse.Namespace) -> None:
    result = _run_style_response_train(
        games=args.games,
        seed_start=args.seed_start,
        max_steps=args.max_steps,
        output_dir=args.output_dir,
        run_name=args.run_name,
    )
    print(json.dumps(result, ensure_ascii=True))


def _run_tabular_mccfr_train(
    *,
    name: str,
    iterations: int,
    seed: int,
    max_depth: int,
    eval_games: int,
    eval_seed_start: int,
    policy_seed_base: int,
    max_steps: int,
    output_dir: str,
    include_all_snapshots: bool,
) -> dict:
    backend = TrainingBackend()
    trainer = TabularMCCFRTrainer()
    solver_config = SolverConfig(
        name=name,
        iteration_count=iterations,
        seed=seed,
        max_depth=max_depth,
    )
    training_result = trainer.train(solver_config)

    eval_config = TrainingRunConfig(
        game_count=eval_games,
        seed_start=eval_seed_start,
        include_all_snapshots=include_all_snapshots,
        max_steps=max_steps,
    )
    batch = backend.collect_solver_self_play_batch(
        solver=training_result.solver,
        config=eval_config,
        policy_seed_base=policy_seed_base,
    )
    summary = backend.summarize_batch(batch)

    artifact = replace(training_result.artifact, metrics=summary)
    store = ArtifactStore(output_dir)
    run_dir = store.run_dir(artifact.artifact_id)
    metadata_path = store.save_metadata(artifact)
    solver_path = training_result.solver.save(run_dir / "solver.json")
    report_path = _write_json(
        run_dir / "report.json",
        {
            "mode": "tabular_mccfr_train",
            "solver_config": asdict(solver_config),
            "eval_config": asdict(eval_config),
            "training_summary": asdict(training_result.summary),
            "evaluation_summary": asdict(summary),
            "artifact": artifact.to_dict(),
            "paths": {
                "metadata": str(metadata_path),
                "solver": str(solver_path),
            },
        },
    )
    return {
        "mode": "tabular_mccfr_train",
        "artifact_id": artifact.artifact_id,
        "report_path": str(report_path),
        "solver_path": str(solver_path),
        "metadata_path": str(metadata_path),
        "solver_config": asdict(solver_config),
        "eval_config": asdict(eval_config),
        "training_summary": asdict(training_result.summary),
        "evaluation_summary": asdict(summary),
        "info_set_count": training_result.summary.info_set_count,
        "mean_terminal_utility": round(summary.mean_terminal_utility, 4),
        "seat_mean_terminal_utility": summary.seat_mean_terminal_utility,
        "mean_modeling_degree": round(summary.mean_modeling_degree, 4),
    }


def _tabular_mccfr_train(args: argparse.Namespace) -> None:
    result = _run_tabular_mccfr_train(
        name=args.name,
        iterations=args.iterations,
        seed=args.seed,
        max_depth=args.max_depth,
        eval_games=args.eval_games,
        eval_seed_start=args.eval_seed_start,
        policy_seed_base=args.policy_seed_base,
        max_steps=args.max_steps,
        output_dir=args.output_dir,
        include_all_snapshots=args.include_all_snapshots,
    )
    print(json.dumps(result, ensure_ascii=True))


def _pool_self_play_eval(args: argparse.Namespace) -> None:
    evaluator = MatchupEvaluator()
    result = evaluator.evaluate_pool_self_play(
        opponent_profiles=default_opponent_profiles(),
        game_count=args.games,
        seed_start=args.seed_start,
        max_steps=args.max_steps,
        custom_ids=_parse_custom_ids(args.player_ids_json),
    )
    output_dir = Path(args.output_dir)
    report_path = _write_json(
        output_dir / args.run_name / "report.json",
        {
            "mode": "pool_self_play_eval",
            "summary": asdict(result.summary),
            "records": [asdict(record) for record in result.records],
        },
    )
    print(
        json.dumps(
            {
                "mode": "pool_self_play_eval",
                "report_path": str(report_path),
                "seat_win_rate": result.summary.seat_win_rate,
                "seat_mean_utility": result.summary.seat_mean_utility,
                "opponent_profile_counts": result.summary.opponent_profile_counts,
                "opponent_inferred_profile_counts": result.summary.opponent_inferred_profile_counts,
            },
            ensure_ascii=True,
        )
    )


def _hero_matchup_eval(args: argparse.Namespace) -> None:
    hero_policy_factory, training_result = _make_hero_policy_factory(args)
    evaluator = MatchupEvaluator()
    result = evaluator.evaluate_hero_vs_pool(
        hero_policy_factory=hero_policy_factory,
        opponent_profiles=default_opponent_profiles(),
        game_count=args.games,
        hero_seat=args.hero_seat,
        seed_start=args.seed_start,
        max_steps=args.max_steps,
        custom_ids=_parse_custom_ids(args.player_ids_json),
        hero_label=args.hero_label,
    )
    output_dir = Path(args.output_dir)
    payload = {
        "mode": "hero_matchup_eval",
        "summary": asdict(result.summary),
        "records": [asdict(record) for record in result.records],
    }
    if training_result is not None:
        payload["hero_training_summary"] = asdict(training_result.summary)

    report_path = _write_json(output_dir / args.run_name / "report.json", payload)
    print(
        json.dumps(
            {
                "mode": "hero_matchup_eval",
                "report_path": str(report_path),
                "hero_seat": result.summary.hero_seat,
                "hero_player_id": result.summary.hero_player_id,
                "win_rate": result.summary.win_rate,
                "top2_rate": result.summary.top2_rate,
                "mean_rank": result.summary.mean_rank,
                "mean_utility": result.summary.mean_utility,
                "opponent_profile_counts": result.summary.opponent_profile_counts,
                "opponent_inferred_profile_counts": result.summary.opponent_inferred_profile_counts,
            },
            ensure_ascii=True,
        )
    )


def _hero_seat_sweep(args: argparse.Namespace) -> None:
    hero_policy_factory, training_result = _make_hero_policy_factory(args)
    evaluator = MatchupEvaluator()
    seat_results = []
    for hero_seat in range(1, 5):
        result = evaluator.evaluate_hero_vs_pool(
            hero_policy_factory=hero_policy_factory,
            opponent_profiles=default_opponent_profiles(),
            game_count=args.games,
            hero_seat=hero_seat,
            seed_start=args.seed_start + hero_seat * 1000,
            max_steps=args.max_steps,
            custom_ids=_parse_custom_ids(args.player_ids_json),
            hero_label=args.hero_label,
        )
        seat_results.append(result)

    output_dir = Path(args.output_dir)
    sweep_dir = output_dir / args.run_name
    sweep_dir.mkdir(parents=True, exist_ok=True)
    scoreboard = []
    for result in seat_results:
        seat = result.summary.hero_seat
        report_path = _write_json(
            sweep_dir / f"hero_seat_{seat}.json",
            {
                "mode": "hero_seat_sweep_item",
                "summary": asdict(result.summary),
                "records": [asdict(record) for record in result.records],
                "hero_training_summary": asdict(training_result.summary) if training_result is not None else None,
            },
        )
        scoreboard.append(
            {
                "hero_seat": seat,
                "hero_display_name": result.summary.hero_display_name,
                "win_rate": result.summary.win_rate,
                "top2_rate": result.summary.top2_rate,
                "mean_rank": result.summary.mean_rank,
                "mean_utility": result.summary.mean_utility,
                "report_path": str(report_path),
            }
        )

    scoreboard.sort(key=lambda item: (-float(item["win_rate"] or 0.0), float(item["mean_rank"] or 99.0)))
    suite_path = _write_json(
        sweep_dir / "sweep_report.json",
        {
            "mode": "hero_seat_sweep",
            "hero_mode": args.hero_mode,
            "hero_training_summary": asdict(training_result.summary) if training_result is not None else None,
            "scoreboard": scoreboard,
        },
    )
    print(
        json.dumps(
            {
                "mode": "hero_seat_sweep",
                "suite_report_path": str(suite_path),
                "scoreboard": scoreboard,
            },
            ensure_ascii=True,
        )
    )


def _baseline_suite(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    suite_dir = output_dir / args.suite_name
    suite_dir.mkdir(parents=True, exist_ok=True)

    experiments = [
        {
            "name": "random_seed_42",
            "kind": "random_eval",
            "runner": lambda: _run_random_eval(
                games=args.games,
                seed_start=42,
                max_steps=args.max_steps,
                output_dir=str(suite_dir),
                include_all_snapshots=args.include_all_snapshots,
            ),
        },
        {
            "name": "random_seed_314",
            "kind": "random_eval",
            "runner": lambda: _run_random_eval(
                games=args.games,
                seed_start=314,
                max_steps=args.max_steps,
                output_dir=str(suite_dir),
                include_all_snapshots=args.include_all_snapshots,
            ),
        },
        {
            "name": "tabular_depth_3_iter_2",
            "kind": "tabular_mccfr_train",
            "runner": lambda: _run_tabular_mccfr_train(
                name="tabular_suite_d3_i2",
                iterations=2,
                seed=42,
                max_depth=3,
                eval_games=args.games,
                eval_seed_start=200,
                policy_seed_base=800,
                max_steps=args.max_steps,
                output_dir=str(suite_dir),
                include_all_snapshots=args.include_all_snapshots,
            ),
        },
        {
            "name": "tabular_depth_4_iter_4",
            "kind": "tabular_mccfr_train",
            "runner": lambda: _run_tabular_mccfr_train(
                name="tabular_suite_d4_i4",
                iterations=4,
                seed=84,
                max_depth=4,
                eval_games=args.games,
                eval_seed_start=400,
                policy_seed_base=1200,
                max_steps=args.max_steps,
                output_dir=str(suite_dir),
                include_all_snapshots=args.include_all_snapshots,
            ),
        },
    ]

    results = []
    for experiment in experiments:
        results.append(
            {
                "name": experiment["name"],
                "kind": experiment["kind"],
                "result": experiment["runner"](),
            }
        )

    scoreboard = [
        {
            "name": item["name"],
            "kind": item["kind"],
            "mean_terminal_utility": item["result"]["mean_terminal_utility"],
            "mean_modeling_degree": item["result"]["mean_modeling_degree"],
            "report_path": item["result"]["report_path"],
        }
        for item in results
    ]
    scoreboard.sort(key=lambda item: (-item["mean_terminal_utility"], -item["mean_modeling_degree"]))

    suite_report = {
        "mode": "baseline_suite",
        "suite_name": args.suite_name,
        "games": args.games,
        "max_steps": args.max_steps,
        "include_all_snapshots": args.include_all_snapshots,
        "results": results,
        "scoreboard": scoreboard,
    }
    suite_path = _write_json(suite_dir / "suite_report.json", suite_report)
    print(
        json.dumps(
            {
                "mode": "baseline_suite",
                "suite_report_path": str(suite_path),
                "scoreboard": scoreboard,
            },
            ensure_ascii=True,
        )
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Training runner for the survival card backend")
    subparsers = parser.add_subparsers(dest="command", required=True)

    random_parser = subparsers.add_parser("random_eval", help="Run random-policy self-play evaluation")
    random_parser.add_argument("--games", type=int, default=4)
    random_parser.add_argument("--seed-start", type=int, default=42)
    random_parser.add_argument("--max-steps", type=int, default=512)
    random_parser.add_argument("--output-dir", type=str, default="runs")
    random_parser.add_argument("--include-all-snapshots", action="store_true")
    random_parser.set_defaults(handler=_random_eval)

    response_parser = subparsers.add_parser(
        "style_response_train",
        help="Train public-style pressure-response priors from bot self-play",
    )
    response_parser.add_argument("--games", type=int, default=24)
    response_parser.add_argument("--seed-start", type=int, default=42)
    response_parser.add_argument("--max-steps", type=int, default=512)
    response_parser.add_argument("--output-dir", type=str, default="runs")
    response_parser.add_argument("--run-name", type=str, default="style_response_train")
    response_parser.set_defaults(handler=_style_response_train)

    train_parser = subparsers.add_parser("tabular_mccfr_train", help="Train and evaluate a tabular MCCFR scaffold")
    train_parser.add_argument("--name", type=str, default="tabular_mccfr")
    train_parser.add_argument("--iterations", type=int, default=8)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--max-depth", type=int, default=4)
    train_parser.add_argument("--eval-games", type=int, default=4)
    train_parser.add_argument("--eval-seed-start", type=int, default=100)
    train_parser.add_argument("--policy-seed-base", type=int, default=500)
    train_parser.add_argument("--max-steps", type=int, default=512)
    train_parser.add_argument("--output-dir", type=str, default="runs")
    train_parser.add_argument("--include-all-snapshots", action="store_true")
    train_parser.set_defaults(handler=_tabular_mccfr_train)

    pool_parser = subparsers.add_parser("pool_self_play_eval", help="Evaluate a default opponent pool without a hero seat")
    pool_parser.add_argument("--games", type=int, default=8)
    pool_parser.add_argument("--seed-start", type=int, default=42)
    pool_parser.add_argument("--max-steps", type=int, default=512)
    pool_parser.add_argument("--output-dir", type=str, default="runs")
    pool_parser.add_argument("--run-name", type=str, default="pool_self_play_eval")
    pool_parser.add_argument("--player-ids-json", type=str, default=None)
    pool_parser.set_defaults(handler=_pool_self_play_eval)

    hero_parser = subparsers.add_parser("hero_matchup_eval", help="Evaluate hero-vs-default-opponents from the hero perspective")
    hero_parser.add_argument("--hero-mode", type=str, choices=("tabular", "adaptive_tabular", "random", "uniform_solver"), default="tabular")
    hero_parser.add_argument("--name", type=str, default="hero_tabular")
    hero_parser.add_argument("--iterations", type=int, default=6)
    hero_parser.add_argument("--seed", type=int, default=42)
    hero_parser.add_argument("--max-depth", type=int, default=4)
    hero_parser.add_argument("--games", type=int, default=12)
    hero_parser.add_argument("--hero-seat", type=int, default=1)
    hero_parser.add_argument("--hero-label", type=str, default="我")
    hero_parser.add_argument("--seed-start", type=int, default=500)
    hero_parser.add_argument("--policy-seed-base", type=int, default=1000)
    hero_parser.add_argument("--max-steps", type=int, default=512)
    hero_parser.add_argument("--output-dir", type=str, default="runs")
    hero_parser.add_argument("--run-name", type=str, default="hero_matchup_eval")
    hero_parser.add_argument("--player-ids-json", type=str, default=None)
    hero_parser.add_argument("--response-prior-path", type=str, default=None)
    hero_parser.set_defaults(handler=_hero_matchup_eval)

    sweep_parser = subparsers.add_parser("hero_seat_sweep", help="Evaluate the hero in seats 1-4 against the default opponent pool")
    sweep_parser.add_argument("--hero-mode", type=str, choices=("tabular", "adaptive_tabular", "random", "uniform_solver"), default="tabular")
    sweep_parser.add_argument("--name", type=str, default="hero_sweep_tabular")
    sweep_parser.add_argument("--iterations", type=int, default=6)
    sweep_parser.add_argument("--seed", type=int, default=42)
    sweep_parser.add_argument("--max-depth", type=int, default=4)
    sweep_parser.add_argument("--games", type=int, default=8)
    sweep_parser.add_argument("--hero-label", type=str, default="我")
    sweep_parser.add_argument("--seed-start", type=int, default=900)
    sweep_parser.add_argument("--policy-seed-base", type=int, default=1600)
    sweep_parser.add_argument("--max-steps", type=int, default=512)
    sweep_parser.add_argument("--output-dir", type=str, default="runs")
    sweep_parser.add_argument("--run-name", type=str, default="hero_seat_sweep")
    sweep_parser.add_argument("--player-ids-json", type=str, default=None)
    sweep_parser.add_argument("--response-prior-path", type=str, default=None)
    sweep_parser.set_defaults(handler=_hero_seat_sweep)

    suite_parser = subparsers.add_parser("baseline_suite", help="Run a small baseline suite over random and tabular trainers")
    suite_parser.add_argument("--games", type=int, default=6)
    suite_parser.add_argument("--max-steps", type=int, default=512)
    suite_parser.add_argument("--output-dir", type=str, default="runs")
    suite_parser.add_argument("--suite-name", type=str, default="baseline_suite")
    suite_parser.add_argument("--include-all-snapshots", action="store_true")
    suite_parser.set_defaults(handler=_baseline_suite)

    deep_cfr_parser = subparsers.add_parser(
        "deep_cfr_train",
        help="Train Deep CFR neural network via multi-style self-play",
    )
    deep_cfr_parser.add_argument("--run-name", type=str, default="deep_cfr")
    deep_cfr_parser.add_argument("--iterations", type=int, default=500)
    deep_cfr_parser.add_argument("--traversals", type=int, default=150)
    deep_cfr_parser.add_argument("--buffer-size", type=int, default=500_000)
    deep_cfr_parser.add_argument("--batch-size", type=int, default=8192)
    deep_cfr_parser.add_argument("--lr", type=float, default=1e-3)
    deep_cfr_parser.add_argument("--train-epochs", type=int, default=150)
    deep_cfr_parser.add_argument("--hidden-dim", type=int, default=512)
    deep_cfr_parser.add_argument("--num-layers", type=int, default=4)
    deep_cfr_parser.add_argument("--dropout", type=float, default=0.1)
    deep_cfr_parser.add_argument("--eval-interval", type=int, default=50)
    deep_cfr_parser.add_argument("--eval-games", type=int, default=100)
    deep_cfr_parser.add_argument("--checkpoint-interval", type=int, default=100)
    deep_cfr_parser.add_argument("--max-depth", type=int, default=8, help="Max CFR traversal search depth (default: 8)")
    deep_cfr_parser.add_argument("--seed", type=int, default=42)
    deep_cfr_parser.add_argument("--num-workers", type=int, default=4, help="Number of worker threads for parallel self-play data collection")
    deep_cfr_parser.add_argument("--device", type=str, default="auto")
    deep_cfr_parser.add_argument("--output-dir", type=str, default="runs_deep_cfr")
    deep_cfr_parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint .pt to resume training from")
    deep_cfr_parser.set_defaults(handler=_deep_cfr_train)

    endgame_parser = subparsers.add_parser(
        "endgame_train",
        help="Train Deep CFR specialized on endgame and underdog situations",
    )
    endgame_parser.add_argument("--run-name", type=str, default="deep_cfr_v16_endgame")
    endgame_parser.add_argument("--iterations", type=int, default=200)
    endgame_parser.add_argument("--traversals", type=int, default=150)
    endgame_parser.add_argument("--buffer-size", type=int, default=500_000)
    endgame_parser.add_argument("--batch-size", type=int, default=8192)
    endgame_parser.add_argument("--lr", type=float, default=3e-4)
    endgame_parser.add_argument("--train-epochs", type=int, default=120)
    endgame_parser.add_argument("--hidden-dim", type=int, default=512)
    endgame_parser.add_argument("--num-layers", type=int, default=4)
    endgame_parser.add_argument("--dropout", type=float, default=0.1)
    endgame_parser.add_argument("--eval-interval", type=int, default=25)
    endgame_parser.add_argument("--eval-games", type=int, default=100)
    endgame_parser.add_argument("--checkpoint-interval", type=int, default=50)
    endgame_parser.add_argument("--max-depth", type=int, default=8)
    endgame_parser.add_argument("--seed", type=int, default=42)
    endgame_parser.add_argument("--num-workers", type=int, default=6)
    endgame_parser.add_argument("--endgame-ratio", type=float, default=0.70)
    endgame_parser.add_argument("--device", type=str, default="auto")
    endgame_parser.add_argument("--output-dir", type=str, default="runs_deep_cfr")
    endgame_parser.add_argument("--resume", type=str, default="runs_deep_cfr/deep_cfr_v15_league/policy_best.pt")
    endgame_parser.set_defaults(handler=_endgame_train)

    endgame_eval_parser = subparsers.add_parser(
        "endgame_eval",
        help="Evaluate policy on 100 benchmark endgame/underdog situations",
    )
    endgame_eval_parser.add_argument("--checkpoint", type=str, required=True)
    endgame_eval_parser.add_argument("--games", type=int, default=100)
    endgame_eval_parser.add_argument("--seed", type=int, default=42000)
    endgame_eval_parser.set_defaults(handler=_endgame_eval)

    # v21 终局名次积分对局续演训练器
    rollout_parser = subparsers.add_parser(
        "rank_rollout_train",
        help="Train Deep CFR with terminal rank rollout and zero artificial bonuses (v21)",
    )
    rollout_parser.add_argument("--run-name", type=str, default="deep_cfr_v21_rank_rollout_512")
    rollout_parser.add_argument("--iterations", type=int, default=25)
    rollout_parser.add_argument("--traversals", type=int, default=90)
    rollout_parser.add_argument("--buffer-size", type=int, default=1_000_000)
    rollout_parser.add_argument("--batch-size", type=int, default=16384)
    rollout_parser.add_argument("--lr", type=float, default=3e-4)
    rollout_parser.add_argument("--train-epochs", type=int, default=120)
    rollout_parser.add_argument("--hidden-dim", type=int, default=512)
    rollout_parser.add_argument("--num-layers", type=int, default=4)
    rollout_parser.add_argument("--dropout", type=float, default=0.1)
    rollout_parser.add_argument("--checkpoint-interval", type=int, default=25)
    rollout_parser.add_argument("--max-depth", type=int, default=8)
    rollout_parser.add_argument("--seed", type=int, default=42)
    rollout_parser.add_argument("--num-workers", type=int, default=6)
    rollout_parser.add_argument("--endgame-ratio", type=float, default=0.70)
    rollout_parser.add_argument("--device", type=str, default="auto")
    rollout_parser.add_argument("--output-dir", type=str, default="runs_deep_cfr")
    rollout_parser.add_argument("--resume", type=str, default=None)
    rollout_parser.add_argument(
        "--rollout-mode",
        type=str,
        choices=["fixed_rule", "frozen_net"],
        default="fixed_rule",
        help="Rollout mechanism at leaf nodes: 'fixed_rule' or 'frozen_net'",
    )
    rollout_parser.set_defaults(handler=_rank_rollout_train)

    return parser


def _deep_cfr_train(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir) / args.run_name

    total_iterations = args.iterations
    if args.resume:
        try:
            import torch as _torch
            _ckpt = _torch.load(args.resume, map_location="cpu", weights_only=False)
            _done = _ckpt.get("iteration", 0)
            total_iterations = _done + args.iterations
            print(f"[run_training] Resume 检测：已完成 {_done} 轮，本次再跑 {args.iterations} 轮，目标总轮次 => {total_iterations}")
        except Exception as e:
            print(f"[run_training] Resume 轮次探测失败 ({e})，使用原始 iterations={args.iterations}")

    config = DeepCFRConfig(
        name=args.run_name,
        cfr_iterations=total_iterations,
        traversals_per_iteration=args.traversals,
        advantage_buffer_size=args.buffer_size,
        strategy_buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        train_epochs=args.train_epochs,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        eval_interval=args.eval_interval,
        eval_games=args.eval_games,
        checkpoint_interval=args.checkpoint_interval,
        seed=args.seed,
        max_traverse_depth=args.max_depth if args.max_depth > 0 else None,
        num_workers=args.num_workers,
        device=args.device,
        resume_from=args.resume,
    )
    trainer = DeepCFRTrainer(config=config)
    trainer.train(output_dir=output_dir)


def _endgame_train(args: argparse.Namespace) -> None:
    from card_ai.endgame_cfr import EndgameDeepCFRTrainer

    output_dir = Path(args.output_dir) / args.run_name

    resume = args.resume
    if resume in ("", "none", "None", "null"):
        resume = None

    total_iterations = args.iterations
    if resume:
        try:
            import torch as _torch
            _ckpt = _torch.load(resume, map_location="cpu", weights_only=False)
            _done = _ckpt.get("iteration", 0)
            total_iterations = _done + args.iterations
            print(f"[run_training] 残局微调启动：承接基底 {_done} 轮，本次专项注水 {args.iterations} 轮，目标总轮次 => {total_iterations}")
        except Exception as e:
            print(f"[run_training] Resume 轮次探测失败 ({e})，使用原始 iterations={args.iterations}")

    config = DeepCFRConfig(
        name=args.run_name,
        cfr_iterations=total_iterations,
        traversals_per_iteration=args.traversals,
        advantage_buffer_size=args.buffer_size,
        strategy_buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        train_epochs=args.train_epochs,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        eval_interval=args.eval_interval,
        eval_games=args.eval_games,
        checkpoint_interval=args.checkpoint_interval,
        seed=args.seed,
        max_traverse_depth=args.max_depth if args.max_depth > 0 else None,
        num_workers=args.num_workers,
        device=args.device,
        resume_from=resume,
    )
    trainer = EndgameDeepCFRTrainer(config=config, endgame_ratio=args.endgame_ratio)
    trainer.train(output_dir=output_dir)


def _endgame_eval(args: argparse.Namespace) -> None:
    from card_ai.endgame_eval import EndgameBenchmark
    from card_ai.neural_policy import NeuralPolicy

    print(f"[Endgame Eval] 加载待测策略模型: {args.checkpoint}")
    policy = NeuralPolicy.load(args.checkpoint)
    bench = EndgameBenchmark()
    print(f"[Endgame Eval] 正在进行 {args.games} 场极端残局与劣势生死博弈盲测...")
    result = bench.evaluate_policy(policy, game_count=args.games, seed_start=args.seed)

    print("\n" + "=" * 50)
    print("      🏆 残局与劣势博弈专项评测报告 (Endgame Benchmark) 🏆")
    print("=" * 50)
    print(f"评测局数 (总样本):       {result.total_games} 场")
    print(f"🔥 逆风绝地吃鸡胜率:      {result.win_rate * 100:.1f}%  ({result.wins}/{result.total_games})")
    print(f"🛡️ 残局保底前二率:        {result.top2_rate * 100:.1f}%  ({result.top2_count}/{result.total_games})")
    print(f"💰 终局场均效用得分:      {result.mean_utility:+.2f}")
    print(f"🎯 生死关键质疑准确率:    {result.challenge_accuracy * 100:.1f}%  ({result.challenge_success}/{result.challenge_count})")
    print("=" * 50 + "\n")


def _rank_rollout_train(args: argparse.Namespace) -> None:
    from card_ai.rank_rollout_cfr import RankRolloutDeepCFRTrainer

    output_dir = Path(args.output_dir) / args.run_name

    resume = args.resume
    if resume in ("", "none", "None", "null"):
        resume = None

    total_iterations = args.iterations
    if resume:
        try:
            import torch as _torch
            _ckpt = _torch.load(resume, map_location="cpu", weights_only=False)
            _done = _ckpt.get("iteration", 0)
            total_iterations = _done + args.iterations
            print(f"[run_training] 终局名次续演训练启动：承接基底 {_done} 轮，本次运行 {args.iterations} 轮，目标总轮次 => {total_iterations}")
        except Exception as e:
            print(f"[run_training] Resume 轮次探测失败 ({e})，使用原始 iterations={args.iterations}")

    config = DeepCFRConfig(
        name=args.run_name,
        cfr_iterations=total_iterations,
        traversals_per_iteration=args.traversals,
        advantage_buffer_size=args.buffer_size,
        strategy_buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        train_epochs=args.train_epochs,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        checkpoint_interval=args.checkpoint_interval,
        seed=args.seed,
        max_traverse_depth=args.max_depth if args.max_depth > 0 else None,
        num_workers=args.num_workers,
        device=args.device,
        resume_from=resume,
    )
    trainer = RankRolloutDeepCFRTrainer(
        config=config,
        endgame_ratio=args.endgame_ratio,
        max_depth=args.max_depth,
        rollout_mode=args.rollout_mode,
    )
    trainer.train(output_dir=output_dir)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()

