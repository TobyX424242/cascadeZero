from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from run_teacher_iteration import (
    PROJECT_ROOT,
    build_arena_command,
    build_export_command,
    build_replay_dataset_specs,
    build_train_command,
    choose_selfplay_command,
    compute_player_score,
    copy_model_bundle,
    load_json,
    prune_expired_datasets,
    resolve_root_dirichlet_alpha_numerator,
    resolve_model_checkpoint,
    resolve_path,
    resolve_replay_policy,
    resolve_training_feature_layout_version,
    run_command,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run teacher iteration with an additional post-selfplay leagueplay stage against a historical model pool."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/leagueplay.json",
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--skip-next-selfplay", action="store_true")
    parser.add_argument("--skip-next-leagueplay", action="store_true")
    return parser.parse_args()


def append_unique_path(paths: list[Path], path: Path | None) -> None:
    if path is None:
        return
    resolved = path.resolve()
    if any(existing.resolve() == resolved for existing in paths):
        return
    paths.append(resolved)


def resolve_league_pool(
    leagueplay: dict[str, Any],
    history_manifests: list[Path],
    active_model_manifest: Path | None,
) -> tuple[list[Path], list[Path]]:
    if active_model_manifest is None:
        return [], []

    window_size = int(leagueplay.get("window_size", 0))
    if window_size < 0:
        raise ValueError("next_leagueplay_on_accept.window_size must be >= 0")

    history_pool = list(history_manifests)
    if window_size > 0:
        history_pool = history_pool[-window_size:]

    resolved_active = active_model_manifest.resolve()
    history_models: list[Path] = []
    elite_models: list[Path] = []
    seen: set[str] = set()

    for history_model in history_pool:
        resolved = history_model.resolve()
        key = str(resolved)
        if resolved == resolved_active or key in seen:
            continue
        seen.add(key)
        history_models.append(resolved)

    for value in leagueplay.get("elite_models", []):
        resolved = resolve_path(str(value))
        if resolved is None:
            continue
        resolved = resolved.resolve()
        key = str(resolved)
        if resolved == resolved_active or key in seen:
            continue
        seen.add(key)
        elite_models.append(resolved)

    return history_models, elite_models


def build_leagueplay_command(
    release_flag: list[str],
    leagueplay: dict[str, Any],
    output_path: Path,
    report_path: Path,
    seed: int,
    candidate_model: Path,
    history_models: list[Path],
    elite_models: list[Path],
) -> list[str]:
    root_dirichlet_alpha = resolve_root_dirichlet_alpha_numerator(leagueplay)
    command = [
        "cargo",
        "run",
        *release_flag,
        "--bin",
        "leagueplay",
        "--",
        "--games",
        str(leagueplay.get("games", 256)),
        "--simulations",
        str(leagueplay.get("simulations", 64)),
        "--c-puct",
        str(leagueplay.get("c_puct", 1.25)),
        "--temperature-plies",
        str(leagueplay.get("temperature_plies", 10)),
        "--temperature",
        str(leagueplay.get("temperature", 1.0)),
        "--root-dirichlet-alpha",
        str(root_dirichlet_alpha),
        "--root-exploration-fraction",
        str(leagueplay.get("root_exploration_fraction", 0.25)),
        "--seed",
        str(seed),
        "--output",
        str(output_path),
        "--candidate-name",
        str(leagueplay.get("candidate_name", "teacher_candidate")),
        "--candidate-model",
        str(candidate_model),
        "--report-json",
        str(report_path),
    ]
    if leagueplay.get("threads") is not None:
        command.extend(["--threads", str(leagueplay["threads"])])
    if leagueplay.get("quiet_progress", False):
        command.append("--quiet-progress")
    if leagueplay.get("model_policy_temperature") is not None:
        command.extend(
            [
                "--model-policy-temperature",
                str(leagueplay["model_policy_temperature"]),
            ]
        )
    if bool(leagueplay.get("record_league_moves", False)):
        command.append("--record-league-moves")
    for history_model in history_models:
        command.extend(["--league-model", str(history_model)])
    for elite_model in elite_models:
        command.extend(["--elite-model", str(elite_model)])
    return command


def build_parallel_leagueplay_command(
    leagueplay: dict[str, Any],
    output_path: Path,
    report_path: Path,
    seed: int,
    candidate_model: Path,
    history_models: list[Path],
    elite_models: list[Path],
    cargo_release: bool,
) -> list[str]:
    workers = int(leagueplay.get("workers", 1))
    root_dirichlet_alpha = resolve_root_dirichlet_alpha_numerator(leagueplay)
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_parallel_leagueplay.py"),
        "--workers",
        str(workers),
        "--games",
        str(leagueplay.get("games", 256)),
        "--simulations",
        str(leagueplay.get("simulations", 64)),
        "--c-puct",
        str(leagueplay.get("c_puct", 1.25)),
        "--temperature-plies",
        str(leagueplay.get("temperature_plies", 10)),
        "--temperature",
        str(leagueplay.get("temperature", 1.0)),
        "--root-dirichlet-alpha",
        str(root_dirichlet_alpha),
        "--root-exploration-fraction",
        str(leagueplay.get("root_exploration_fraction", 0.25)),
        "--seed",
        str(seed),
        "--output",
        str(output_path),
        "--report-json",
        str(report_path),
        "--candidate-name",
        str(leagueplay.get("candidate_name", "teacher_candidate")),
        "--candidate-model",
        str(candidate_model),
    ]
    threads = leagueplay.get("threads")
    if threads is not None:
        threads_per_worker = max(1, int(threads) // workers)
        command.extend(["--threads-per-worker", str(threads_per_worker)])
    if leagueplay.get("quiet_progress", False):
        command.append("--quiet-progress")
    if leagueplay.get("model_policy_temperature") is not None:
        command.extend(
            [
                "--model-policy-temperature",
                str(leagueplay["model_policy_temperature"]),
            ]
        )
    if bool(leagueplay.get("record_league_moves", False)):
        command.append("--record-league-moves")
    if not cargo_release:
        command.append("--no-cargo-release")
    for history_model in history_models:
        command.extend(["--league-model", str(history_model)])
    for elite_model in elite_models:
        command.extend(["--elite-model", str(elite_model)])
    return command


def leagueplay_wants_multiprocess(leagueplay: dict[str, Any]) -> bool:
    return int(leagueplay.get("workers", 1)) > 1


def choose_leagueplay_command(
    release_flag: list[str],
    leagueplay: dict[str, Any],
    output_path: Path,
    report_path: Path,
    seed: int,
    candidate_model: Path,
    history_models: list[Path],
    elite_models: list[Path],
    cargo_release: bool,
) -> list[str]:
    if leagueplay_wants_multiprocess(leagueplay):
        return build_parallel_leagueplay_command(
            leagueplay,
            output_path,
            report_path,
            seed,
            candidate_model,
            history_models,
            elite_models,
            cargo_release,
        )
    return build_leagueplay_command(
        release_flag,
        leagueplay,
        output_path,
        report_path,
        seed,
        candidate_model,
        history_models,
        elite_models,
    )


def main() -> int:
    args = parse_args()
    config = load_json(args.config)

    output_root = (
        args.output_root
        or resolve_path(config.get("output_root"))
        or (PROJECT_ROOT / "artifacts/iterations/teacher_iteration_leagueplay")
    )
    output_root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "config_path": str(args.config),
        "output_root": str(output_root),
        "commands": [],
    }
    logs_dir = output_root / "logs"

    release_flag = ["--release"] if config.get("cargo_release", True) else []
    python_executable = sys.executable

    iteration = config.get("iteration", {})
    max_rounds = int(iteration.get("rounds", 1))
    max_consecutive_rejections = int(iteration.get("max_consecutive_rejections", 1))
    accumulate_datasets = bool(iteration.get("accumulate_datasets", True))
    generate_selfplay_on_rejection = bool(iteration.get("generate_selfplay_on_rejection", False))
    generate_leagueplay_on_rejection = bool(iteration.get("generate_leagueplay_on_rejection", False))
    prune_expired = bool(iteration.get("prune_expired_datasets", False))
    replay_window, replay_decay = resolve_replay_policy(iteration, accumulate_datasets)
    summary["replay_policy"] = {
        "accumulate_datasets": accumulate_datasets,
        "replay_window": replay_window,
        "replay_decay": replay_decay,
    }

    dataset_path = resolve_path(config.get("dataset_path"))
    bootstrap = config.get("bootstrap_selfplay", {})
    if dataset_path is None:
        dataset_path = output_root / bootstrap.get("output_name", "bootstrap_selfplay.jsonl")
        bootstrap_command = choose_selfplay_command(
            release_flag,
            bootstrap,
            dataset_path,
            int(bootstrap.get("seed", 20260421)),
            resolve_path(bootstrap.get("model")),
            cargo_release=config.get("cargo_release", True),
        )
        bootstrap_result = run_command(
            bootstrap_command,
            label="bootstrap_selfplay",
            log_path=logs_dir / "01_bootstrap_selfplay.log",
        )
        summary["commands"].append(bootstrap_result.to_dict())
    summary["dataset_path"] = str(dataset_path)

    dataset_paths: list[Path] = [dataset_path]

    training = config.get("training", {})
    summary["training_feature_layout_version"] = resolve_training_feature_layout_version(training)
    arena = config.get("arena", {})
    gating = config.get("gating", {})
    incumbent_manifest = resolve_path(arena.get("baseline_model"))
    configured_init_checkpoint = resolve_path(training.get("init_checkpoint"))
    incumbent_checkpoint = configured_init_checkpoint or resolve_model_checkpoint(incumbent_manifest)
    league_history_manifests: list[Path] = []
    append_unique_path(league_history_manifests, incumbent_manifest)

    consecutive_rejections = 0
    accepted_bundle: dict[str, Any] | None = None
    next_selfplay = config.get("next_selfplay_on_accept", {})
    next_selfplay_seed = int(next_selfplay.get("seed", 20260422))
    next_leagueplay = config.get("next_leagueplay_on_accept", {})
    next_leagueplay_seed = int(next_leagueplay.get("seed", 20260423))

    summary["rounds"] = []

    for round_index in range(1, max_rounds + 1):
        round_dir = output_root if max_rounds == 1 else output_root / f"round_{round_index:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        round_label = f"round_{round_index:02d}"

        train_dataset_specs = build_replay_dataset_specs(
            dataset_paths, accumulate_datasets, replay_window, replay_decay
        )
        training_output = round_dir / training.get("output_dir_name", "candidate_training")
        print(
            f">>> [{round_label}] replay_pool={len(train_dataset_specs)} datasets "
            f"weights={[(str(path), round(weight, 4)) for path, weight in train_dataset_specs]}",
            flush=True,
        )
        train_command = build_train_command(
            python_executable,
            training,
            train_dataset_specs,
            training_output,
            init_checkpoint=incumbent_checkpoint,
        )
        train_result = run_command(
            train_command,
            label=f"{round_label}_train" if max_rounds > 1 else "train_teacher",
            log_path=logs_dir
            / (
                f"{round_index:02d}_{round_label}_train.log"
                if max_rounds > 1
                else "02_train_teacher.log"
            ),
        )
        summary["commands"].append(train_result.to_dict())

        deleted_paths = prune_expired_datasets(dataset_paths, replay_window, prune_expired)
        if deleted_paths:
            print(
                f"[{round_label}] pruned {len(deleted_paths)} expired dataset(s): {[str(path) for path in deleted_paths]}",
                flush=True,
            )

        export_output = round_dir / training.get("export_dir_name", "candidate_export")
        export_command = build_export_command(
            python_executable,
            training_output / training.get("checkpoint_name", "best.pt"),
            export_output,
            str(training.get("manifest_name", "teacher_resnet_onnx.json")),
            str(training.get("execution_provider", "cuda")),
        )
        export_result = run_command(
            export_command,
            label=f"{round_label}_export" if max_rounds > 1 else "export_teacher",
            log_path=logs_dir
            / (
                f"{round_index:02d}_{round_label}_export.log"
                if max_rounds > 1
                else "03_export_teacher.log"
            ),
        )
        summary["commands"].append(export_result.to_dict())
        candidate_manifest = export_output / training.get("manifest_name", "teacher_resnet_onnx.json")

        round_arena = dict(arena)
        if incumbent_manifest is not None:
            round_arena["baseline_model"] = str(incumbent_manifest)
        arena_report = round_dir / arena.get("report_name", "arena_report.json")
        arena_command = build_arena_command(release_flag, round_arena, candidate_manifest, arena_report)
        arena_result = run_command(
            arena_command,
            label=f"{round_label}_arena" if max_rounds > 1 else "arena_gating",
            log_path=logs_dir
            / (
                f"{round_index:02d}_{round_label}_arena.log"
                if max_rounds > 1
                else "04_arena_gating.log"
            ),
        )
        summary["commands"].append(arena_result.to_dict())

        arena_payload = load_json(arena_report)
        arena_summary = arena_payload["summary"]
        games_played = int(arena_summary["games_played"])
        draws = int(arena_summary["draws"])
        candidate_player = arena_summary["player_b"]
        baseline_player = arena_summary["player_a"]
        candidate_score = compute_player_score(candidate_player, draws, games_played)
        baseline_score = compute_player_score(baseline_player, draws, games_played)
        accepted = (
            candidate_score >= float(gating.get("min_candidate_score", 0.55))
            and int(candidate_player.get("wins", 0)) >= int(gating.get("min_candidate_wins", 0))
        )

        round_summary: dict[str, Any] = {
            "round": round_index,
            "datasets": [str(path) for path, _ in train_dataset_specs],
            "dataset_weights": [
                {
                    "path": str(path),
                    "weight": weight,
                    "age": len(train_dataset_specs) - index - 1,
                }
                for index, (path, weight) in enumerate(train_dataset_specs)
            ],
            "candidate_manifest": str(candidate_manifest),
            "baseline_model": str(incumbent_manifest) if incumbent_manifest is not None else None,
            "init_checkpoint": str(incumbent_checkpoint) if incumbent_checkpoint is not None else None,
            "arena": {
                "report_json": str(arena_report),
                "candidate_score": candidate_score,
                "baseline_score": baseline_score,
                "candidate_wins": int(candidate_player.get("wins", 0)),
                "baseline_wins": int(baseline_player.get("wins", 0)),
                "draws": draws,
                "games_played": games_played,
            },
            "accepted": accepted,
        }

        active_model_manifest: Path | None = incumbent_manifest
        if accepted:
            previous_incumbent = incumbent_manifest
            accepted_destination = output_root / (
                "accepted_model" if max_rounds == 1 else f"accepted_model_round_{round_index:02d}"
            )
            accepted_bundle = copy_model_bundle(
                candidate_manifest,
                accepted_destination,
                checkpoint_path=training_output / training.get("checkpoint_name", "best.pt"),
            )
            incumbent_manifest = Path(accepted_bundle["manifest"])
            incumbent_checkpoint_value = accepted_bundle.get("checkpoint")
            incumbent_checkpoint = (
                Path(incumbent_checkpoint_value)
                if incumbent_checkpoint_value is not None
                else resolve_model_checkpoint(incumbent_manifest)
            )
            active_model_manifest = incumbent_manifest
            consecutive_rejections = 0
            if previous_incumbent is not None:
                append_unique_path(league_history_manifests, previous_incumbent)
            round_summary["accepted_model"] = accepted_bundle
            summary["accepted"] = True
            summary["accepted_model"] = accepted_bundle
        else:
            consecutive_rejections += 1
            round_summary["consecutive_rejections"] = consecutive_rejections

        selfplay_should_run = (
            next_selfplay.get("enabled", False)
            and not args.skip_next_selfplay
            and (accepted or generate_selfplay_on_rejection)
        )
        if selfplay_should_run:
            next_output_name = next_selfplay.get("output_name", "accepted_selfplay.jsonl")
            next_output = round_dir / next_output_name if max_rounds > 1 else output_root / next_output_name
            next_command = choose_selfplay_command(
                release_flag,
                next_selfplay,
                next_output,
                next_selfplay_seed,
                active_model_manifest,
                cargo_release=config.get("cargo_release", True),
            )
            next_result = run_command(
                next_command,
                label=(
                    f"{round_label}_next_selfplay"
                    if max_rounds > 1
                    else ("accepted_selfplay" if accepted else "rejected_selfplay")
                ),
                log_path=logs_dir
                / (
                    f"{round_index:02d}_{round_label}_next_selfplay.log"
                    if max_rounds > 1
                    else ("05_accepted_selfplay.log" if accepted else "05_rejected_selfplay.log")
                ),
            )
            summary["commands"].append(next_result.to_dict())
            dataset_paths.append(next_output)
            round_summary["next_selfplay_output"] = str(next_output)
            round_summary["next_selfplay_model"] = str(active_model_manifest)
            next_selfplay_seed += 1

        leagueplay_should_run = (
            next_leagueplay.get("enabled", False)
            and not args.skip_next_leagueplay
            and active_model_manifest is not None
            and (accepted or generate_leagueplay_on_rejection)
        )
        if leagueplay_should_run:
            history_models, elite_models = resolve_league_pool(
                next_leagueplay,
                league_history_manifests,
                active_model_manifest,
            )
            if not history_models and not elite_models:
                raise ValueError(
                    f"{round_label}: leagueplay enabled but no historical or elite models were resolved"
                )

            league_output_name = next_leagueplay.get("output_name", "accepted_leagueplay.jsonl")
            league_output = round_dir / league_output_name if max_rounds > 1 else output_root / league_output_name
            league_report = round_dir / next_leagueplay.get("report_name", "leagueplay_report.json")
            league_command = choose_leagueplay_command(
                release_flag,
                next_leagueplay,
                league_output,
                league_report,
                next_leagueplay_seed,
                active_model_manifest,
                history_models,
                elite_models,
                cargo_release=config.get("cargo_release", True),
            )
            league_result = run_command(
                league_command,
                label=(
                    f"{round_label}_next_leagueplay"
                    if max_rounds > 1
                    else ("accepted_leagueplay" if accepted else "rejected_leagueplay")
                ),
                log_path=logs_dir
                / (
                    f"{round_index:02d}_{round_label}_next_leagueplay.log"
                    if max_rounds > 1
                    else ("06_accepted_leagueplay.log" if accepted else "06_rejected_leagueplay.log")
                ),
            )
            summary["commands"].append(league_result.to_dict())
            dataset_paths.append(league_output)
            round_summary["next_leagueplay_output"] = str(league_output)
            round_summary["next_leagueplay_report"] = str(league_report)
            round_summary["next_leagueplay_model"] = str(active_model_manifest)
            round_summary["next_leagueplay_history_models"] = [str(path) for path in history_models]
            round_summary["next_leagueplay_elite_models"] = [str(path) for path in elite_models]
            next_leagueplay_seed += 1

        summary["rounds"].append(round_summary)

        if consecutive_rejections >= max_consecutive_rejections:
            summary["stopped_reason"] = f"reached {consecutive_rejections} consecutive rejections"
            break

    if incumbent_manifest is not None:
        summary["incumbent_model"] = str(incumbent_manifest)
    if incumbent_checkpoint is not None:
        summary["incumbent_checkpoint"] = str(incumbent_checkpoint)
    summary["final_datasets"] = [str(path) for path in dataset_paths]
    summary["league_history_models"] = [str(path) for path in league_history_manifests]
    summary["consecutive_rejections"] = consecutive_rejections

    summary_path = output_root / "iteration_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if summary.get("rounds"):
        last_round = summary["rounds"][-1]
        last_arena = last_round["arena"]
        print(
            f"accepted={last_round['accepted']} candidate_score={last_arena['candidate_score']:.4f} "
            f"baseline_score={last_arena['baseline_score']:.4f} datasets={len(dataset_paths)} "
            f"summary={summary_path} consecutive_rejections={consecutive_rejections}"
        )
    else:
        print(f"summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
