from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class WorkerResult:
    worker_id: int
    returncode: int
    shard_path: Path
    report_path: Path
    games_assigned: int
    seed: int
    duration_seconds: float
    stdout_tail: str


@dataclass
class ParallelLeaguePlaySummary:
    workers: int
    total_games: int
    output: Path
    report_json: Path
    worker_results: list[WorkerResult] = field(default_factory=list)
    merged: bool = False
    merge_lines: int = 0
    report_merged: bool = False
    total_duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-process parallel leagueplay launcher for cascade_data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workers", "-n", type=int, default=4)
    parser.add_argument("--threads-per-worker", type=int, default=None)
    parser.add_argument("--games", type=int, required=True)
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--c-puct", type=float, default=1.25)
    parser.add_argument("--temperature-plies", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--root-dirichlet-alpha", type=float, default=10.0)
    parser.add_argument("--root-exploration-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260423)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, default=None)
    parser.add_argument("--no-merge", action="store_true")
    parser.add_argument("--keep-shards", action="store_true")
    parser.add_argument("--candidate-name", type=str, default="teacher_candidate")
    parser.add_argument("--candidate-model", type=Path, required=True)
    parser.add_argument("--league-model", action="append", type=Path, default=[])
    parser.add_argument("--elite-model", action="append", type=Path, default=[])
    parser.add_argument("--quiet-progress", action="store_true")
    parser.add_argument("--model-policy-temperature", type=float, default=None)
    parser.add_argument("--record-league-moves", action="store_true")
    parser.add_argument("--cargo-release", action="store_true", default=True)
    parser.add_argument("--no-cargo-release", dest="cargo_release", action="store_false")
    parser.add_argument("--summary-json", type=Path, default=None)
    return parser.parse_args()


def resolve_workspace_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def distribute_games(total: int, workers: int) -> list[int]:
    base = total // workers
    remainder = total % workers
    return [base + (1 if index < remainder else 0) for index in range(workers)]


def build_worker_command(
    *,
    games: int,
    simulations: int,
    c_puct: float,
    temperature_plies: int,
    temperature: float,
    root_dirichlet_alpha: float,
    root_exploration_fraction: float,
    seed: int,
    output: Path,
    report_json: Path,
    candidate_name: str,
    candidate_model: Path,
    league_models: list[Path],
    elite_models: list[Path],
    threads: int,
    cargo_release: bool,
    quiet_progress: bool,
    model_policy_temperature: float | None,
    record_league_moves: bool,
) -> list[str]:
    release_flag = ["--release"] if cargo_release else []
    command = [
        "cargo",
        "run",
        *release_flag,
        "--bin",
        "leagueplay",
        "--",
        "--games",
        str(games),
        "--simulations",
        str(simulations),
        "--c-puct",
        str(c_puct),
        "--temperature-plies",
        str(temperature_plies),
        "--temperature",
        str(temperature),
        "--root-dirichlet-alpha",
        str(root_dirichlet_alpha),
        "--root-exploration-fraction",
        str(root_exploration_fraction),
        "--threads",
        str(threads),
        "--seed",
        str(seed),
        "--output",
        str(output),
        "--candidate-name",
        candidate_name,
        "--candidate-model",
        str(candidate_model),
        "--report-json",
        str(report_json),
    ]
    if quiet_progress:
        command.append("--quiet-progress")
    if model_policy_temperature is not None:
        command.extend(["--model-policy-temperature", str(model_policy_temperature)])
    if record_league_moves:
        command.append("--record-league-moves")
    for history_model in league_models:
        command.extend(["--league-model", str(history_model)])
    for elite_model in elite_models:
        command.extend(["--elite-model", str(elite_model)])
    return command


_PRINT_LOCK = threading.Lock()


def tprint(message: str) -> None:
    with _PRINT_LOCK:
        print(message, flush=True)


def run_worker(
    worker_id: int,
    command: list[str],
    shard_path: Path,
    report_path: Path,
    games_assigned: int,
    seed: int,
    log_path: Path | None,
) -> WorkerResult:
    label = f"worker_{worker_id:02d}"
    tprint(
        f">>> [{label}] starting: games={games_assigned} seed={seed} output={shard_path} report={report_path}"
    )
    tprint(f">>> [{label}] cmd: {' '.join(command)}")

    started_at = time.perf_counter()
    buffer: list[str] = []
    log_handle = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("w", encoding="utf-8")

    env = os.environ.copy()
    worker_miopen_dir = shard_path.parent / f"miopen_worker_{worker_id:04d}"
    worker_miopen_dir.mkdir(parents=True, exist_ok=True)
    env["MIOPEN_USER_DB_PATH"] = str(worker_miopen_dir)
    env["MIOPEN_CUSTOM_CACHE_DIR"] = str(worker_miopen_dir)

    try:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            buffer.append(line)
            if log_handle is not None:
                log_handle.write(line)
            tprint(f"[{label}] {line}")
        returncode = process.wait()
    finally:
        if log_handle is not None:
            log_handle.close()

    duration = time.perf_counter() - started_at
    tail = "".join(buffer[-200:])
    status = "OK" if returncode == 0 else f"FAILED(rc={returncode})"
    tprint(f"<<< [{label}] {status} duration_s={duration:.1f} shard={shard_path}")

    return WorkerResult(
        worker_id=worker_id,
        returncode=returncode,
        shard_path=shard_path,
        report_path=report_path,
        games_assigned=games_assigned,
        seed=seed,
        duration_seconds=duration,
        stdout_tail=tail,
    )


def launch_workers(
    worker_commands: list[tuple[list[str], Path, Path, int, int]],
    logs_dir: Path | None,
) -> list[WorkerResult]:
    results: list[WorkerResult | None] = [None] * len(worker_commands)
    threads: list[threading.Thread] = []

    def run_slot(index: int) -> None:
        command, shard_path, report_path, games_assigned, seed = worker_commands[index]
        log_path = (logs_dir / f"leagueplay_worker_{index:02d}.log") if logs_dir else None
        results[index] = run_worker(
            worker_id=index,
            command=command,
            shard_path=shard_path,
            report_path=report_path,
            games_assigned=games_assigned,
            seed=seed,
            log_path=log_path,
        )

    for index in range(len(worker_commands)):
        thread = threading.Thread(
            target=run_slot,
            args=(index,),
            daemon=True,
            name=f"leagueplay-worker-{index:02d}",
        )
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    assert all(result is not None for result in results)
    return results  # type: ignore[return-value]


def merge_shards(shards: list[Path], output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    total_lines = 0
    with output.open("w", encoding="utf-8", newline="\n") as output_handle:
        for shard in shards:
            if not shard.exists():
                print(f"[merge] WARNING: shard not found, skipping: {shard}", flush=True)
                continue
            with shard.open("r", encoding="utf-8") as input_handle:
                for raw_line in input_handle:
                    line = raw_line.rstrip("\r\n")
                    if not line:
                        continue
                    output_handle.write(line + "\n")
                    total_lines += 1
    return total_lines


def load_report(report_path: Path) -> dict[str, Any]:
    return json.loads(report_path.read_text(encoding="utf-8"))


def merge_reports(
    report_paths: list[Path],
    output: Path,
    *,
    total_games: int,
    base_seed: int,
    workers: int,
    threads_per_worker: int,
    wall_elapsed_seconds: float,
    succeeded_workers: list[WorkerResult],
) -> None:
    reports = [load_report(report_path) for report_path in report_paths if report_path.exists()]
    if not reports:
        raise ValueError("no worker leagueplay reports were produced")

    base_report = copy.deepcopy(reports[0])
    base_opponents = base_report.get("summary", {}).get("opponents", [])
    merged_summary = {
        "games_played": 0,
        "candidate_wins": 0,
        "league_wins": 0,
        "draws": 0,
        "samples_written": 0,
        "total_plies": 0,
        "opponents": [
            {
                "name": opponent.get("name"),
                "tier": opponent.get("tier"),
                "games": 0,
                "candidate_wins": 0,
                "league_wins": 0,
                "draws": 0,
                "samples_written": 0,
            }
            for opponent in base_opponents
        ],
    }

    for report in reports:
        if report.get("candidate") != base_report.get("candidate"):
            raise ValueError("worker leagueplay reports disagree on candidate metadata")
        if report.get("league_members") != base_report.get("league_members"):
            raise ValueError("worker leagueplay reports disagree on league member metadata")

        summary = report.get("summary", {})
        merged_summary["games_played"] += int(summary.get("games_played", 0))
        merged_summary["candidate_wins"] += int(summary.get("candidate_wins", 0))
        merged_summary["league_wins"] += int(summary.get("league_wins", 0))
        merged_summary["draws"] += int(summary.get("draws", 0))
        merged_summary["samples_written"] += int(summary.get("samples_written", 0))
        merged_summary["total_plies"] += int(summary.get("total_plies", 0))

        opponents = summary.get("opponents", [])
        if len(opponents) != len(merged_summary["opponents"]):
            raise ValueError("worker leagueplay reports disagree on opponent summary length")

        for index, opponent in enumerate(opponents):
            merged_opponent = merged_summary["opponents"][index]
            if (
                opponent.get("name") != merged_opponent.get("name")
                or opponent.get("tier") != merged_opponent.get("tier")
            ):
                raise ValueError("worker leagueplay reports disagree on opponent ordering")
            merged_opponent["games"] += int(opponent.get("games", 0))
            merged_opponent["candidate_wins"] += int(opponent.get("candidate_wins", 0))
            merged_opponent["league_wins"] += int(opponent.get("league_wins", 0))
            merged_opponent["draws"] += int(opponent.get("draws", 0))
            merged_opponent["samples_written"] += int(opponent.get("samples_written", 0))

    merged_config = copy.deepcopy(base_report.get("config", {}))
    merged_config["games"] = total_games
    merged_config["seed"] = base_seed
    merged_config["threads"] = threads_per_worker

    merged_report = {
        "config": merged_config,
        "candidate": base_report.get("candidate"),
        "league_members": base_report.get("league_members", []),
        "simulations": base_report.get("simulations"),
        "c_puct": base_report.get("c_puct"),
        "elapsed_ms": int(round(wall_elapsed_seconds * 1000.0)),
        "worker_elapsed_ms_total": sum(int(report.get("elapsed_ms", 0)) for report in reports),
        "parallel": {
            "workers": workers,
            "successful_workers": len(reports),
            "threads_per_worker": threads_per_worker,
            "worker_reports": [str(path) for path in report_paths],
            "worker_shards": [str(worker.shard_path) for worker in succeeded_workers],
            "worker_seeds": [worker.seed for worker in succeeded_workers],
        },
        "summary": merged_summary,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(merged_report, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()

    workers = args.workers
    if workers < 1:
        print("[error] --workers must be >= 1", flush=True)
        return 1
    if args.games < 1:
        print("[error] --games must be >= 1", flush=True)
        return 1
    if not args.league_model and not args.elite_model:
        print("[error] at least one --league-model or --elite-model is required", flush=True)
        return 1

    output = resolve_workspace_path(args.output)
    report_json = resolve_workspace_path(args.report_json)
    candidate_model = resolve_workspace_path(args.candidate_model)
    league_models = [resolve_workspace_path(path) for path in args.league_model]
    elite_models = [resolve_workspace_path(path) for path in args.elite_model]

    if args.shard_dir is None:
        shard_dir = output.parent / f"{output.stem}_shards"
    else:
        shard_dir = resolve_workspace_path(args.shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = shard_dir / "logs"

    if args.summary_json is not None:
        summary_json = resolve_workspace_path(args.summary_json)
    else:
        summary_json = None

    if args.threads_per_worker is not None:
        threads_per_worker = max(1, args.threads_per_worker)
    else:
        cpu_count = os.cpu_count() or 1
        threads_per_worker = max(1, cpu_count // workers)

    game_distribution = distribute_games(args.games, workers)
    print(
        f"[plan] workers={workers} total_games={args.games} threads_per_worker={threads_per_worker} "
        f"game_distribution={game_distribution} base_seed={args.seed} output={output} report={report_json} shard_dir={shard_dir}",
        flush=True,
    )

    worker_commands: list[tuple[list[str], Path, Path, int, int]] = []
    for worker_id, worker_games in enumerate(game_distribution):
        shard_path = shard_dir / f"shard_{worker_id:04d}.jsonl"
        worker_report = shard_dir / f"report_{worker_id:04d}.json"
        worker_seed = args.seed + worker_id
        command = build_worker_command(
            games=worker_games,
            simulations=args.simulations,
            c_puct=args.c_puct,
            temperature_plies=args.temperature_plies,
            temperature=args.temperature,
            root_dirichlet_alpha=args.root_dirichlet_alpha,
            root_exploration_fraction=args.root_exploration_fraction,
            seed=worker_seed,
            output=shard_path,
            report_json=worker_report,
            candidate_name=args.candidate_name,
            candidate_model=candidate_model,
            league_models=league_models,
            elite_models=elite_models,
            threads=threads_per_worker,
            cargo_release=args.cargo_release,
            quiet_progress=args.quiet_progress,
            model_policy_temperature=args.model_policy_temperature,
            record_league_moves=args.record_league_moves,
        )
        worker_commands.append((command, shard_path, worker_report, worker_games, worker_seed))

    wall_start = time.perf_counter()
    worker_results = launch_workers(worker_commands, logs_dir)
    wall_elapsed = time.perf_counter() - wall_start

    failed_workers = [result for result in worker_results if result.returncode != 0]
    succeeded_workers = [result for result in worker_results if result.returncode == 0]

    summary = ParallelLeaguePlaySummary(
        workers=workers,
        total_games=args.games,
        output=output,
        report_json=report_json,
        worker_results=worker_results,
        total_duration_seconds=wall_elapsed,
    )

    if failed_workers:
        for result in failed_workers:
            message = (
                f"worker_{result.worker_id:02d} failed (rc={result.returncode}), "
                f"shard={result.shard_path} report={result.report_path}"
            )
            summary.errors.append(message)
            print(f"[error] {message}", flush=True)
            print(f"[error] worker_{result.worker_id:02d} tail:\n{result.stdout_tail}", flush=True)

    if not args.no_merge and succeeded_workers:
        shard_paths = sorted((result.shard_path for result in succeeded_workers), key=lambda path: path.name)
        report_paths = sorted((result.report_path for result in succeeded_workers), key=lambda path: path.name)

        if failed_workers:
            print(
                f"[merge] WARNING: {len(failed_workers)} worker(s) failed; merging {len(succeeded_workers)} successful shards only.",
                flush=True,
            )

        print(f"[merge] merging {len(shard_paths)} shards -> {output}", flush=True)
        merge_start = time.perf_counter()
        total_lines = merge_shards(shard_paths, output)
        merge_reports(
            report_paths,
            report_json,
            total_games=sum(result.games_assigned for result in succeeded_workers),
            base_seed=args.seed,
            workers=workers,
            threads_per_worker=threads_per_worker,
            wall_elapsed_seconds=wall_elapsed,
            succeeded_workers=succeeded_workers,
        )
        merge_elapsed = time.perf_counter() - merge_start
        summary.merged = True
        summary.merge_lines = total_lines
        summary.report_merged = True
        print(
            f"[merge] done: lines={total_lines} merge_elapsed_s={merge_elapsed:.2f} output={output} report={report_json}",
            flush=True,
        )

        if not args.keep_shards:
            if not failed_workers:
                for path in [*shard_paths, *report_paths]:
                    try:
                        path.unlink()
                    except OSError as exc:
                        print(f"[merge] WARNING: could not delete shard artifact {path}: {exc}", flush=True)
            else:
                print(
                    "[merge] keeping shard files because some workers failed (remove manually after investigation).",
                    flush=True,
                )

    print(
        f"[summary] workers={workers} succeeded={len(succeeded_workers)} failed={len(failed_workers)} "
        f"merged={summary.merged} report_merged={summary.report_merged} lines={summary.merge_lines} "
        f"wall_elapsed_s={wall_elapsed:.1f}",
        flush=True,
    )

    if summary_json is not None:
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary_json.write_text(
            json.dumps(
                {
                    "workers": summary.workers,
                    "total_games": summary.total_games,
                    "output": str(summary.output),
                    "report_json": str(summary.report_json),
                    "merged": summary.merged,
                    "merge_lines": summary.merge_lines,
                    "report_merged": summary.report_merged,
                    "total_duration_seconds": summary.total_duration_seconds,
                    "errors": summary.errors,
                    "worker_results": [
                        {
                            "worker_id": result.worker_id,
                            "returncode": result.returncode,
                            "shard_path": str(result.shard_path),
                            "report_path": str(result.report_path),
                            "games_assigned": result.games_assigned,
                            "seed": result.seed,
                            "duration_seconds": result.duration_seconds,
                        }
                        for result in summary.worker_results
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    return 1 if failed_workers else 0


if __name__ == "__main__":
    raise SystemExit(main())
