from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_LAYOUT_HEIGHT_BUCKETS_V2 = "height_buckets_v2"


@dataclass(slots=True)
class CommandResult:
    label: str
    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    duration_seconds: float
    log_path: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "command": self.command,
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_seconds": self.duration_seconds,
            "log_path": self.log_path,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run teacher iteration for Cascade with optional multi-round strict gating.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/iteration.json")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--skip-next-selfplay", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(path_value: str | None, default_root: Path | None = None) -> Path | None:
    if path_value is None:
        return default_root
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def resolve_training_feature_layout_version(training: dict[str, Any]) -> str:
    feature_layout_version = str(
        training.get("feature_layout_version", FEATURE_LAYOUT_HEIGHT_BUCKETS_V2)
    ).strip().lower()
    if not feature_layout_version:
        feature_layout_version = FEATURE_LAYOUT_HEIGHT_BUCKETS_V2
    if feature_layout_version != FEATURE_LAYOUT_HEIGHT_BUCKETS_V2:
        raise ValueError(
            "teacher iteration only supports feature_layout_version=height_buckets_v2"
        )
    return feature_layout_version


def run_command(
    command: list[str],
    label: str,
    cwd: Path = PROJECT_ROOT,
    log_path: Path | None = None,
) -> CommandResult:
    started_at = time.perf_counter()
    print(f">>> [{label}] starting: {' '.join(command)}", flush=True)
    output_lines: list[str] = []

    completed = subprocess.Popen(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    assert completed.stdout is not None
    log_handle = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("w", encoding="utf-8")

    try:
        for line in completed.stdout:
            output_lines.append(line)
            if log_handle is not None:
                log_handle.write(line)
            print(f"[{label}] {line}", end="", flush=True)
    finally:
        if log_handle is not None:
            log_handle.close()

    returncode = completed.wait()
    duration_seconds = time.perf_counter() - started_at
    combined_output = "".join(output_lines)
    result = CommandResult(
        label=label,
        command=command,
        returncode=returncode,
        stdout=combined_output,
        stderr="",
        duration_seconds=duration_seconds,
        log_path=str(log_path) if log_path is not None else None,
    )
    print(
        f"<<< [{label}] completed rc={returncode} duration_s={duration_seconds:.2f}"
        + (f" log={log_path}" if log_path is not None else ""),
        flush=True,
    )
    if returncode != 0:
        raise RuntimeError(
            f"command failed ({returncode}): {' '.join(command)}\n"
            f"combined_output:\n{combined_output}\n"
            + (f"log_path: {log_path}\n" if log_path is not None else "")
        )
    return result


def add_optional_args(command: list[str], options: dict[str, Any], mapping: dict[str, str]) -> None:
    for key, flag in mapping.items():
        value = options.get(key)
        if value is None:
            continue
        command.extend([flag, str(value)])


def resolve_replay_policy(section: dict[str, Any], accumulate_datasets: bool) -> tuple[int | None, float]:
    replay_window_value = section.get("replay_window")
    replay_window = None if replay_window_value is None else int(replay_window_value)
    if replay_window is not None and replay_window < 1:
        raise ValueError("replay_window must be >= 1 when provided")

    replay_decay = float(section.get("replay_decay", 1.0))
    if replay_decay <= 0.0 or replay_decay > 1.0:
        raise ValueError("replay_decay must be in the interval (0, 1]")

    if not accumulate_datasets:
        return 1, 1.0

    return replay_window, replay_decay


def build_replay_dataset_specs(
    dataset_paths: list[Path],
    accumulate_datasets: bool,
    replay_window: int | None,
    replay_decay: float,
) -> list[tuple[Path, float]]:
    if not dataset_paths:
        return []

    if not accumulate_datasets:
        return [(dataset_paths[-1], 1.0)]

    replay_pool = list(dataset_paths if replay_window is None else dataset_paths[-replay_window:])
    newest_index = len(replay_pool) - 1
    return [
        (dataset_path, replay_decay ** (newest_index - index))
        for index, dataset_path in enumerate(replay_pool)
    ]


def prune_expired_datasets(
    all_dataset_paths: list[Path],
    replay_window: int | None,
    enabled: bool,
) -> list[Path]:
    """Delete dataset files that have slid fully outside the replay window to reclaim disk space.
    Returns the list of paths that were actually deleted."""
    if not enabled or replay_window is None or len(all_dataset_paths) <= replay_window:
        return []

    expired_paths = all_dataset_paths[:-replay_window]
    deleted: list[Path] = []
    for path in expired_paths:
        if path.exists():
            try:
                path.unlink()
                deleted.append(path)
                print(f"[prune] deleted expired dataset: {path}", flush=True)
            except OSError as exc:
                print(f"[prune] WARNING: could not delete {path}: {exc}", flush=True)
        else:
            print(f"[prune] skipped (already gone): {path}", flush=True)
    return deleted


def compute_player_score(player_summary: dict[str, Any], draws: int, games_played: int) -> float:
    if games_played <= 0:
        return 0.0
    return (float(player_summary.get("wins", 0)) + 0.5 * float(draws)) / float(games_played)


def resolve_model_artifacts(manifest_path: Path, manifest: dict[str, Any]) -> list[Path]:
    artifact_values: list[str] = []
    if isinstance(manifest.get("artifact_files"), list):
        artifact_values.extend(str(value) for value in manifest["artifact_files"])
    for key in ("onnx_file", "weights_file"):
        value = manifest.get(key)
        if value:
            artifact_values.append(str(value))

    onnx_file = manifest.get("onnx_file")
    if onnx_file:
        sidecar_value = f"{onnx_file}.data"
        sidecar_path = Path(sidecar_value)
        if not sidecar_path.is_absolute():
            sidecar_path = manifest_path.parent / sidecar_path
        if sidecar_path.exists():
            artifact_values.append(sidecar_value)

    artifact_paths: list[Path] = []
    seen: set[str] = set()
    for value in artifact_values:
        if value in seen:
            continue
        seen.add(value)
        artifact_path = Path(value)
        if not artifact_path.is_absolute():
            artifact_path = manifest_path.parent / artifact_path
        artifact_paths.append(artifact_path)

    if not artifact_paths:
        raise ValueError(f"manifest {manifest_path} does not declare any model artifacts")

    return artifact_paths


def resolve_model_checkpoint(manifest_path: Path | None) -> Path | None:
    if manifest_path is None:
        return None

    for checkpoint_name in ("best.pt", "final.pt"):
        checkpoint_path = manifest_path.parent / checkpoint_name
        if checkpoint_path.exists():
            return checkpoint_path

    return None


def copy_model_bundle(
    manifest_path: Path,
    destination_dir: Path,
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    destination_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_json(manifest_path)
    dest_manifest = destination_dir / manifest_path.name
    shutil.copy2(manifest_path, dest_manifest)
    copied_bundle = {
        "manifest": str(dest_manifest),
    }

    for source_artifact in resolve_model_artifacts(manifest_path, manifest):
        dest_artifact = destination_dir / source_artifact.name
        shutil.copy2(source_artifact, dest_artifact)
        copied_bundle.setdefault("artifacts", []).append(str(dest_artifact))

    if manifest.get("onnx_file"):
        copied_bundle["onnx"] = str(destination_dir / Path(str(manifest["onnx_file"])).name)
    if manifest.get("weights_file"):
        copied_bundle["weights"] = str(destination_dir / Path(str(manifest["weights_file"])).name)
    if checkpoint_path is not None:
        dest_checkpoint = destination_dir / checkpoint_path.name
        shutil.copy2(checkpoint_path, dest_checkpoint)
        copied_bundle["checkpoint"] = str(dest_checkpoint)

    return copied_bundle


def resolve_root_dirichlet_alpha_numerator(selfplay: dict[str, Any]) -> float:
    numerator = selfplay.get("root_dirichlet_alpha_numerator")
    if numerator is not None:
        return float(numerator)

    legacy_alpha = selfplay.get("root_dirichlet_alpha")
    if legacy_alpha is None:
        return 10.0

    legacy_alpha = float(legacy_alpha)
    if legacy_alpha > 1.0:
        return legacy_alpha

    return 10.0


def build_selfplay_command(
    release_flag: list[str],
    selfplay: dict[str, Any],
    output_path: Path,
    seed: int,
    model_path: Path | None,
) -> list[str]:
    root_dirichlet_alpha = resolve_root_dirichlet_alpha_numerator(selfplay)
    command = [
        "cargo",
        "run",
        *release_flag,
        "--bin",
        "selfplay",
        "--",
        "--games",
        str(selfplay.get("games", 4)),
        "--simulations",
        str(selfplay.get("simulations", 8)),
        "--c-puct",
        str(selfplay.get("c_puct", 1.35)),
        "--temperature-plies",
        str(selfplay.get("temperature_plies", 12)),
        "--temperature",
        str(selfplay.get("temperature", 1.0)),
        "--root-dirichlet-alpha",
        str(root_dirichlet_alpha),
        "--root-exploration-fraction",
        str(selfplay.get("root_exploration_fraction", 0.25)),
        "--seed",
        str(seed),
        "--output",
        str(output_path),
    ]
    if selfplay.get("threads") is not None:
        command.extend(["--threads", str(selfplay["threads"])])
    if model_path is not None:
        command.extend(["--model", str(model_path)])
    return command


def build_parallel_selfplay_command(
    selfplay: dict[str, Any],
    output_path: Path,
    seed: int,
    model_path: Path | None,
    cargo_release: bool,
) -> list[str]:
    """Build a command that calls run_parallel_selfplay.py to spawn N independent
    selfplay processes.  Use this when selfplay["workers"] > 1 to avoid the
    CUDA-context serialisation that limits throughput with in-process threads."""
    workers: int = int(selfplay.get("workers", 1))
    root_dirichlet_alpha = resolve_root_dirichlet_alpha_numerator(selfplay)
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_parallel_selfplay.py"),
        "--workers", str(workers),
        "--games", str(selfplay.get("games", 4)),
        "--simulations", str(selfplay.get("simulations", 8)),
        "--c-puct", str(selfplay.get("c_puct", 1.35)),
        "--temperature-plies", str(selfplay.get("temperature_plies", 12)),
        "--temperature", str(selfplay.get("temperature", 1.0)),
        "--root-dirichlet-alpha", str(root_dirichlet_alpha),
        "--root-exploration-fraction", str(selfplay.get("root_exploration_fraction", 0.25)),
        "--seed", str(seed),
        "--output", str(output_path),
    ]
    # threads_per_worker: honoured if present; otherwise the launcher auto-divides
    threads = selfplay.get("threads")
    if threads is not None:
        threads_per_worker = max(1, int(threads) // workers)
        command.extend(["--threads-per-worker", str(threads_per_worker)])
    if model_path is not None:
        command.extend(["--model", str(model_path)])
    if not cargo_release:
        command.append("--no-cargo-release")
    return command


def selfplay_wants_multiprocess(selfplay: dict[str, Any]) -> bool:
    """Return True when the selfplay config requests multi-process mode."""
    return int(selfplay.get("workers", 1)) > 1


def choose_selfplay_command(
    release_flag: list[str],
    selfplay: dict[str, Any],
    output_path: Path,
    seed: int,
    model_path: Path | None,
    cargo_release: bool,
) -> list[str]:
    """Return the appropriate selfplay command based on whether multi-process
    mode is requested (selfplay.workers > 1)."""
    if selfplay_wants_multiprocess(selfplay):
        return build_parallel_selfplay_command(
            selfplay, output_path, seed, model_path, cargo_release
        )
    return build_selfplay_command(release_flag, selfplay, output_path, seed, model_path)


def build_train_command(
    python_executable: str,
    training: dict[str, Any],
    dataset_specs: list[tuple[Path, float]],
    output_dir: Path,
    init_checkpoint: Path | None = None,
) -> list[str]:
    command = [
        python_executable,
        str(PROJECT_ROOT / "scripts" / "train_teacher_resnet.py"),
        "--config",
        str(resolve_path(training.get("config", "configs/training.json"))),
    ]
    command.extend([
        "--feature-layout-version",
        resolve_training_feature_layout_version(training),
    ])
    for dataset_path, dataset_weight in dataset_specs:
        command.extend(["--dataset", str(dataset_path)])
        command.extend(["--dataset-weight", str(dataset_path), f"{dataset_weight:.12g}"])
    if init_checkpoint is not None:
        command.extend(["--init-checkpoint", str(init_checkpoint)])
    command.extend([
        "--output-dir",
        str(output_dir),
    ])
    training_options = dict(training)
    if init_checkpoint is not None:
        if training_options.get("warm_start_epochs") is not None:
            training_options["epochs"] = training_options["warm_start_epochs"]
        if training_options.get("warm_start_learning_rate") is not None:
            training_options["learning_rate"] = training_options["warm_start_learning_rate"]
    add_optional_args(
        command,
        training_options,
        {
            "epochs": "--epochs",
            "batch_size": "--batch-size",
            "learning_rate": "--learning-rate",
            "weight_decay": "--weight-decay",
            "device": "--device",
            "num_workers": "--num-workers",
            "log_every_batches": "--log-every-batches",
        },
    )
    return command


def build_export_command(
    python_executable: str,
    checkpoint_path: Path,
    output_dir: Path,
    manifest_name: str,
    execution_provider: str,
) -> list[str]:
    return [
        python_executable,
        str(PROJECT_ROOT / "scripts" / "export_teacher_resnet.py"),
        "--checkpoint",
        str(checkpoint_path),
        "--output-dir",
        str(output_dir),
        "--manifest-name",
        manifest_name,
        "--execution-provider",
        execution_provider,
    ]


def build_arena_command(
    release_flag: list[str],
    arena: dict[str, Any],
    candidate_manifest: Path,
    report_path: Path,
) -> list[str]:
    command = [
        "cargo",
        "run",
        *release_flag,
        "--bin",
        "arena",
        "--",
        "--games",
        str(arena.get("games", 12)),
        "--simulations",
        str(arena.get("simulations", 8)),
        "--c-puct",
        str(arena.get("c_puct", 1.35)),
        "--seed",
        str(arena.get("seed", 20260421)),
        "--opening-temperature-plies",
        str(arena.get("opening_temperature_plies", 8)),
        "--opening-temperature",
        str(arena.get("opening_temperature", 1.0)),
        "--player-a-name",
        str(arena.get("baseline_name", "heuristic")),
        "--player-b-name",
        str(arena.get("candidate_name", "teacher_candidate")),
        "--player-b-model",
        str(candidate_manifest),
        "--report-json",
        str(report_path),
    ]
    if arena.get("threads") is not None:
        command.extend(["--threads", str(arena["threads"])])
    baseline_model = arena.get("baseline_model")
    if baseline_model:
        command.extend(["--player-a-model", str(resolve_path(baseline_model))])
    return command


def main() -> int:
    args = parse_args()
    config = load_json(args.config)

    output_root = args.output_root or resolve_path(config.get("output_root")) or (PROJECT_ROOT / "artifacts/iterations/teacher_iteration_v1")
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
        command = choose_selfplay_command(
            release_flag,
            bootstrap,
            dataset_path,
            int(bootstrap.get("seed", 20260421)),
            resolve_path(bootstrap.get("model")),
            cargo_release=config.get("cargo_release", True),
        )
        result = run_command(
            command,
            label="bootstrap_selfplay",
            log_path=logs_dir / "01_bootstrap_selfplay.log",
        )
        summary["commands"].append(result.to_dict())
    summary["dataset_path"] = str(dataset_path)

    dataset_paths: list[Path] = [dataset_path]

    training = config.get("training", {})
    summary["training_feature_layout_version"] = resolve_training_feature_layout_version(training)
    arena = config.get("arena", {})
    gating = config.get("gating", {})
    incumbent_manifest = resolve_path(arena.get("baseline_model"))
    configured_init_checkpoint = resolve_path(training.get("init_checkpoint"))
    incumbent_checkpoint = configured_init_checkpoint or resolve_model_checkpoint(incumbent_manifest)
    consecutive_rejections = 0
    accepted_bundle: dict[str, Any] | None = None
    next_selfplay = config.get("next_selfplay_on_accept", {})
    next_selfplay_seed = int(next_selfplay.get("seed", 20260422))

    summary["rounds"] = []

    for round_index in range(1, max_rounds + 1):
        round_dir = output_root if max_rounds == 1 else output_root / f"round_{round_index:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        round_label = f"round_{round_index:02d}"

        train_dataset_specs = build_replay_dataset_specs(dataset_paths, accumulate_datasets, replay_window, replay_decay)
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
            log_path=logs_dir / (f"{round_index:02d}_{round_label}_train.log" if max_rounds > 1 else "02_train_teacher.log"),
        )
        summary["commands"].append(train_result.to_dict())

        deleted_paths = prune_expired_datasets(dataset_paths, replay_window, prune_expired)
        if deleted_paths:
            round_summary_pruned = [str(p) for p in deleted_paths]
            print(f"[{round_label}] pruned {len(deleted_paths)} expired dataset(s): {round_summary_pruned}", flush=True)

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
            log_path=logs_dir / (f"{round_index:02d}_{round_label}_export.log" if max_rounds > 1 else "03_export_teacher.log"),
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
            log_path=logs_dir / (f"{round_index:02d}_{round_label}_arena.log" if max_rounds > 1 else "04_arena_gating.log"),
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

        round_summary = {
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
            accepted_destination = output_root / ("accepted_model" if max_rounds == 1 else f"accepted_model_round_{round_index:02d}")
            accepted_bundle = copy_model_bundle(
                candidate_manifest,
                accepted_destination,
                checkpoint_path=training_output / training.get("checkpoint_name", "best.pt"),
            )
            incumbent_manifest = Path(accepted_bundle["manifest"])
            incumbent_checkpoint_value = accepted_bundle.get("checkpoint")
            incumbent_checkpoint = Path(incumbent_checkpoint_value) if incumbent_checkpoint_value is not None else resolve_model_checkpoint(incumbent_manifest)
            active_model_manifest = incumbent_manifest
            consecutive_rejections = 0
            round_summary["accepted_model"] = accepted_bundle
            summary["accepted"] = True
            summary["accepted_model"] = accepted_bundle
        else:
            consecutive_rejections += 1
            round_summary["consecutive_rejections"] = consecutive_rejections

        if next_selfplay.get("enabled", False) and not args.skip_next_selfplay and (accepted or generate_selfplay_on_rejection):
            next_output_name = next_selfplay.get("output_name", "accepted_selfplay.jsonl")
            if max_rounds > 1:
                next_output = round_dir / next_output_name
            else:
                next_output = output_root / next_output_name
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
                label=f"{round_label}_next_selfplay" if max_rounds > 1 else ("accepted_selfplay" if accepted else "rejected_selfplay"),
                log_path=logs_dir / (f"{round_index:02d}_{round_label}_next_selfplay.log" if max_rounds > 1 else ("05_accepted_selfplay.log" if accepted else "05_rejected_selfplay.log")),
            )
            summary["commands"].append(next_result.to_dict())
            dataset_paths.append(next_output)
            round_summary["next_selfplay_output"] = str(next_output)
            round_summary["next_selfplay_model"] = str(active_model_manifest)
            next_selfplay_seed += 1

        summary["rounds"].append(round_summary)

        if consecutive_rejections >= max_consecutive_rejections:
            summary["stopped_reason"] = f"reached {consecutive_rejections} consecutive rejections"
            break

    if incumbent_manifest is not None:
        summary["incumbent_model"] = str(incumbent_manifest)
    if incumbent_checkpoint is not None:
        summary["incumbent_checkpoint"] = str(incumbent_checkpoint)
    summary["final_datasets"] = [str(path) for path in dataset_paths]
    summary["consecutive_rejections"] = consecutive_rejections

    summary_path = output_root / "iteration_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if summary.get("rounds"):
        last_round = summary["rounds"][-1]
        last_arena = last_round["arena"]
        print(
            f"accepted={last_round['accepted']} candidate_score={last_arena['candidate_score']:.4f} baseline_score={last_arena['baseline_score']:.4f} "
            f"datasets={len(dataset_paths)} summary={summary_path} consecutive_rejections={consecutive_rejections}"
        )
    else:
        print(f"summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
