"""
run_parallel_selfplay.py
------------------------
Launches N independent `cargo selfplay` processes, each generating
  ceil(total_games / N)  games into a private shard file, then
concatenates the shards into the final output file and removes the
shards.

Each process owns its evaluator and inference context, allowing CPU search
and model inference to overlap across workers. Throughput depends on the
runtime, model, hardware, and worker count; profile before scaling up.

Usage
-----
    python scripts/run_parallel_selfplay.py \
        --games 2048 \
        --simulations 100 \
        --threads-per-worker 4 \
        --workers 4 \
        --output artifacts/selfplay_out.jsonl \
        [--model path/to/model.json] \
        [--seed 20260421] \
        [--c-puct 1.25] \
        [--temperature-plies 10] \
        [--temperature 1.0] \
        [--root-dirichlet-alpha 10.0] \
        [--root-exploration-fraction 0.25] \
        [--quiet-progress] \
        [--cargo-release] \
        [--no-merge]

Config-file mode (compatible with teacher_iteration configs)
------------------------------------------------------------
    python scripts/run_parallel_selfplay.py \
        --from-config configs/iteration.json \
        --config-section bootstrap_selfplay \
        --output artifacts/selfplay_out.jsonl \
        [--workers 4]

Exit codes
----------
  0  all workers succeeded, merge done
  1  one or more workers failed (partial shards are kept for inspection)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class WorkerResult:
    worker_id: int
    returncode: int
    shard_path: Path
    games_assigned: int
    seed: int
    duration_seconds: float
    stdout_tail: str  # last ~4 KB for error diagnosis


@dataclass
class ParallelSelfPlaySummary:
    workers: int
    total_games: int
    output: Path
    worker_results: list[WorkerResult] = field(default_factory=list)
    merged: bool = False
    merge_lines: int = 0
    total_duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-process parallel self-play launcher for cascade_data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ------------------------------------------------------------------
    # Core parallelism knobs
    # ------------------------------------------------------------------
    p.add_argument("--workers", "-n", type=int, default=4,
                   help="Number of independent selfplay processes to spawn.")
    p.add_argument("--threads-per-worker", type=int, default=None,
                   help="Rayon threads per worker process (--threads passed to the binary). "
                        "Defaults to: max(1, total_system_threads // workers).")

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    p.add_argument("--output", type=Path, default=None,
                   help="Final merged JSONL output path.")
    p.add_argument("--shard-dir", type=Path, default=None,
                   help="Directory for temporary shard files. Defaults to <output>.shards/")
    p.add_argument("--no-merge", action="store_true",
                   help="Skip merging; keep shards as-is (useful for debugging).")
    p.add_argument("--keep-shards", action="store_true",
                   help="Keep shard files even after a successful merge.")

    # ------------------------------------------------------------------
    # Config-file shortcut
    # ------------------------------------------------------------------
    p.add_argument("--from-config", type=Path, default=None,
                   help="Load selfplay parameters from a teacher iteration JSON config file.")
    p.add_argument("--config-section", type=str, default="bootstrap_selfplay",
                   help="Which section of the config to read (e.g. bootstrap_selfplay, next_selfplay_on_accept).")

    # ------------------------------------------------------------------
    # Selfplay binary parameters (mirror those in build_selfplay_command)
    # ------------------------------------------------------------------
    p.add_argument("--games", type=int, default=None,
                   help="Total number of self-play games to generate across all workers.")
    p.add_argument("--simulations", type=int, default=100)
    p.add_argument("--c-puct", type=float, default=1.25)
    p.add_argument("--temperature-plies", type=int, default=10)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--root-dirichlet-alpha", type=float, default=None,
                   help="Dirichlet numerator for dynamic root noise; actual alpha is value / legal_actions.")
    p.add_argument("--root-exploration-fraction", type=float, default=0.25)
    p.add_argument("--model", type=Path, default=None,
                   help="ONNX model manifest path (optional; omit for heuristic self-play).")
    p.add_argument("--seed", type=int, default=20260421,
                   help="Base RNG seed; each worker receives seed + worker_id.")
    p.add_argument("--quiet-progress", action="store_true",
                   help="Pass --quiet-progress to each worker binary.")

    # ------------------------------------------------------------------
    # Build flags
    # ------------------------------------------------------------------
    p.add_argument("--cargo-release", action="store_true", default=True,
                   help="Build / run with --release (default: true).")
    p.add_argument("--no-cargo-release", dest="cargo_release", action="store_false")

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------
    p.add_argument("--summary-json", type=Path, default=None,
                   help="Write a JSON summary to this path after completion.")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config_section(config_path: Path, section: str) -> dict[str, Any]:
    data = json.loads(config_path.read_text(encoding="utf-8"))
    if section not in data:
        raise KeyError(f"Section '{section}' not found in {config_path}. "
                       f"Available: {list(data.keys())}")
    return data[section]


def resolve_path_from_config(value: str | None) -> Path | None:
    if value is None:
        return None
    p = Path(value)
    if p.is_absolute():
        return p
    return (PROJECT_ROOT / p).resolve()


def resolve_root_dirichlet_alpha_numerator(
    selfplay_params: dict[str, Any],
    cli_value: float | None,
) -> float:
    if cli_value is not None:
        return cli_value

    numerator = selfplay_params.get("root_dirichlet_alpha_numerator")
    if numerator is not None:
        return float(numerator)

    legacy_alpha = selfplay_params.get("root_dirichlet_alpha")
    if legacy_alpha is None:
        return 10.0

    legacy_alpha = float(legacy_alpha)
    if legacy_alpha > 1.0:
        return legacy_alpha

    return 10.0


# ---------------------------------------------------------------------------
# Game distribution
# ---------------------------------------------------------------------------

def distribute_games(total: int, workers: int) -> list[int]:
    """Return a list of length `workers` that sums to `total`."""
    base = total // workers
    remainder = total % workers
    return [base + (1 if i < remainder else 0) for i in range(workers)]


# ---------------------------------------------------------------------------
# Command builder
# ---------------------------------------------------------------------------

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
    model: Path | None,
    threads: int,
    cargo_release: bool,
    quiet_progress: bool,
) -> list[str]:
    release_flag = ["--release"] if cargo_release else []
    cmd = [
        "cargo", "run", *release_flag,
        "--bin", "selfplay",
        "--",
        "--games", str(games),
        "--simulations", str(simulations),
        "--c-puct", str(c_puct),
        "--temperature-plies", str(temperature_plies),
        "--temperature", str(temperature),
        "--root-dirichlet-alpha", str(root_dirichlet_alpha),
        "--root-exploration-fraction", str(root_exploration_fraction),
        "--threads", str(threads),
        "--seed", str(seed),
        "--output", str(output),
    ]
    if model is not None:
        cmd.extend(["--model", str(model)])
    if quiet_progress:
        cmd.append("--quiet-progress")
    return cmd


# ---------------------------------------------------------------------------
# Worker execution
# ---------------------------------------------------------------------------

_print_lock = threading.Lock()


def _tprint(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


def run_worker(
    worker_id: int,
    cmd: list[str],
    shard_path: Path,
    games_assigned: int,
    seed: int,
    log_path: Path | None,
) -> WorkerResult:
    label = f"worker_{worker_id:02d}"
    _tprint(f">>> [{label}] starting: games={games_assigned} seed={seed} output={shard_path}")
    _tprint(f">>> [{label}] cmd: {' '.join(cmd)}")

    started_at = time.perf_counter()
    buf: list[str] = []

    log_handle = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("w", encoding="utf-8")

    env = os.environ.copy()
    # Isolate MIOpen cache directories per worker to prevent sqlite database corruption/locking
    # when multiple ROCm workers run concurrently on the same GPU.
    worker_miopen_dir = shard_path.parent / f"miopen_worker_{worker_id:04d}"
    worker_miopen_dir.mkdir(parents=True, exist_ok=True)
    env["MIOPEN_USER_DB_PATH"] = str(worker_miopen_dir)
    env["MIOPEN_CUSTOM_CACHE_DIR"] = str(worker_miopen_dir)

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            buf.append(line)
            if log_handle is not None:
                log_handle.write(line)
            # Prefix each line with the worker label
            _tprint(f"[{label}] {line}", )
        returncode = proc.wait()
    finally:
        if log_handle is not None:
            log_handle.close()

    duration = time.perf_counter() - started_at
    tail = "".join(buf[-200:])  # keep last ~200 lines for diagnosis

    status_str = "OK" if returncode == 0 else f"FAILED(rc={returncode})"
    _tprint(f"<<< [{label}] {status_str} duration_s={duration:.1f} shard={shard_path}")

    return WorkerResult(
        worker_id=worker_id,
        returncode=returncode,
        shard_path=shard_path,
        games_assigned=games_assigned,
        seed=seed,
        duration_seconds=duration,
        stdout_tail=tail,
    )


# ---------------------------------------------------------------------------
# Parallel launch
# ---------------------------------------------------------------------------

def launch_workers(
    worker_commands: list[tuple[list[str], Path, int, int]],  # (cmd, shard, games, seed)
    logs_dir: Path | None,
) -> list[WorkerResult]:
    """Spawn all workers in background threads, wait for all to finish."""
    results: list[WorkerResult | None] = [None] * len(worker_commands)
    threads: list[threading.Thread] = []

    def _run(idx: int) -> None:
        cmd, shard, games, seed = worker_commands[idx]
        log_path = (logs_dir / f"selfplay_worker_{idx:02d}.log") if logs_dir else None
        results[idx] = run_worker(
            worker_id=idx,
            cmd=cmd,
            shard_path=shard,
            games_assigned=games,
            seed=seed,
            log_path=log_path,
        )

    for i in range(len(worker_commands)):
        t = threading.Thread(target=_run, args=(i,), daemon=True, name=f"worker-{i:02d}")
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    # Satisfy type checker: all slots filled
    assert all(r is not None for r in results)
    return results  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def merge_shards(shards: list[Path], output: Path) -> int:
    """Concatenate shard JSONL files → output. Returns total line count."""
    output.parent.mkdir(parents=True, exist_ok=True)
    total_lines = 0
    with output.open("w", encoding="utf-8", newline="\n") as out_f:
        for shard in shards:
            if not shard.exists():
                print(f"[merge] WARNING: shard not found, skipping: {shard}", flush=True)
                continue
            with shard.open("r", encoding="utf-8") as in_f:
                for raw_line in in_f:
                    line = raw_line.rstrip("\r\n")
                    if not line:
                        continue  # skip blank lines
                    out_f.write(line + "\n")
                    total_lines += 1
    return total_lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    # ------------------------------------------------------------------
    # Resolve parameters (config file takes precedence, then CLI)
    # ------------------------------------------------------------------
    selfplay_params: dict[str, Any] = {}
    if args.from_config is not None:
        cfg_path = args.from_config if args.from_config.is_absolute() else (PROJECT_ROOT / args.from_config)
        selfplay_params = load_config_section(cfg_path, args.config_section)
        print(f"[config] loaded section '{args.config_section}' from {cfg_path}", flush=True)

    def param(cli_val: Any, cfg_key: str, default: Any) -> Any:
        if cli_val is not None:
            return cli_val
        if cfg_key in selfplay_params:
            return selfplay_params[cfg_key]
        return default

    total_games: int = param(args.games, "games", 2048)
    simulations: int = param(args.simulations, "simulations", 100)
    c_puct: float = param(args.c_puct, "c_puct", 1.25)
    temperature_plies: int = param(args.temperature_plies, "temperature_plies", 10)
    temperature: float = param(args.temperature, "temperature", 1.0)
    root_dirichlet_alpha = resolve_root_dirichlet_alpha_numerator(
        selfplay_params,
        args.root_dirichlet_alpha,
    )
    root_exploration_fraction: float = param(args.root_exploration_fraction, "root_exploration_fraction", 0.25)
    base_seed: int = param(args.seed, "seed", 20260421)
    model_path: Path | None = (
        args.model
        if args.model is not None
        else resolve_path_from_config(selfplay_params.get("model"))
    )
    cargo_release: bool = args.cargo_release

    if args.output is None:
        cfg_output_name: str = selfplay_params.get("output_name", "parallel_selfplay.jsonl")
        output = PROJECT_ROOT / "artifacts" / cfg_output_name
    else:
        output = args.output if args.output.is_absolute() else (PROJECT_ROOT / args.output)

    workers = args.workers
    if workers < 1:
        print("[error] --workers must be >= 1", flush=True)
        return 1

    # Threads per worker
    if args.threads_per_worker is not None:
        threads_per_worker = max(1, args.threads_per_worker)
    elif "threads" in selfplay_params:
        # Divide the config's thread count across workers
        threads_per_worker = max(1, int(selfplay_params["threads"]) // workers)
    else:
        import os as _os
        cpu_count = _os.cpu_count() or 1
        threads_per_worker = max(1, cpu_count // workers)

    # Shard directory
    shard_dir = args.shard_dir if args.shard_dir is not None else output.parent / f"{output.stem}_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    logs_dir = shard_dir / "logs"

    # ------------------------------------------------------------------
    # Print plan
    # ------------------------------------------------------------------
    game_distribution = distribute_games(total_games, workers)
    print(
        f"[plan] workers={workers} total_games={total_games} "
        f"threads_per_worker={threads_per_worker} "
        f"game_distribution={game_distribution} "
        f"base_seed={base_seed} "
        f"output={output} "
        f"shard_dir={shard_dir}",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Build per-worker commands
    # ------------------------------------------------------------------
    worker_commands: list[tuple[list[str], Path, int, int]] = []
    for wid, wgames in enumerate(game_distribution):
        shard_path = shard_dir / f"shard_{wid:04d}.jsonl"
        wseed = base_seed + wid
        cmd = build_worker_command(
            games=wgames,
            simulations=simulations,
            c_puct=c_puct,
            temperature_plies=temperature_plies,
            temperature=temperature,
            root_dirichlet_alpha=root_dirichlet_alpha,
            root_exploration_fraction=root_exploration_fraction,
            seed=wseed,
            output=shard_path,
            model=model_path,
            threads=threads_per_worker,
            cargo_release=cargo_release,
            quiet_progress=args.quiet_progress,
        )
        worker_commands.append((cmd, shard_path, wgames, wseed))

    # ------------------------------------------------------------------
    # Launch all workers in parallel (via threads wrapping subprocesses)
    # ------------------------------------------------------------------
    wall_start = time.perf_counter()
    worker_results = launch_workers(worker_commands, logs_dir)
    wall_elapsed = time.perf_counter() - wall_start

    # ------------------------------------------------------------------
    # Check results
    # ------------------------------------------------------------------
    failed_workers = [r for r in worker_results if r.returncode != 0]
    succeeded_workers = [r for r in worker_results if r.returncode == 0]

    summary = ParallelSelfPlaySummary(
        workers=workers,
        total_games=total_games,
        output=output,
        worker_results=worker_results,
        total_duration_seconds=wall_elapsed,
    )

    if failed_workers:
        for r in failed_workers:
            msg = f"worker_{r.worker_id:02d} failed (rc={r.returncode}), shard={r.shard_path}"
            summary.errors.append(msg)
            print(f"[error] {msg}", flush=True)
            print(f"[error] worker_{r.worker_id:02d} tail:\n{r.stdout_tail}", flush=True)

    # ------------------------------------------------------------------
    # Merge shards (only if at least one succeeded and merge not disabled)
    # ------------------------------------------------------------------
    if not args.no_merge and succeeded_workers:
        shard_paths = [r.shard_path for r in succeeded_workers]
        shard_paths_sorted = sorted(shard_paths, key=lambda p: p.name)

        if failed_workers:
            print(
                f"[merge] WARNING: {len(failed_workers)} worker(s) failed; "
                f"merging {len(succeeded_workers)} successful shards only.",
                flush=True,
            )

        print(f"[merge] merging {len(shard_paths_sorted)} shards → {output}", flush=True)
        merge_start = time.perf_counter()
        total_lines = merge_shards(shard_paths_sorted, output)
        merge_elapsed = time.perf_counter() - merge_start
        summary.merged = True
        summary.merge_lines = total_lines
        print(
            f"[merge] done: lines={total_lines} merge_elapsed_s={merge_elapsed:.2f} output={output}",
            flush=True,
        )

        # Clean up shards (only on full success unless --keep-shards)
        if not args.keep_shards:
            if not failed_workers:
                for shard in shard_paths_sorted:
                    try:
                        shard.unlink()
                    except OSError as exc:
                        print(f"[merge] WARNING: could not delete shard {shard}: {exc}", flush=True)
                # Remove shard dir if empty
                try:
                    shard_dir.rmdir()
                except OSError:
                    pass
            else:
                print(
                    "[merge] keeping shard files because some workers failed "
                    "(remove manually after investigation).",
                    flush=True,
                )

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    total_game_lines = sum(
        r.games_assigned for r in succeeded_workers
    )
    print(
        f"[summary] workers={workers} succeeded={len(succeeded_workers)} "
        f"failed={len(failed_workers)} "
        f"merged={summary.merged} lines={summary.merge_lines} "
        f"wall_elapsed_s={wall_elapsed:.1f} "
        f"approx_games_collected~{total_game_lines}",
        flush=True,
    )

    if args.summary_json is not None:
        summary_dict = {
            "workers": summary.workers,
            "total_games": summary.total_games,
            "output": str(summary.output),
            "merged": summary.merged,
            "merge_lines": summary.merge_lines,
            "total_duration_seconds": summary.total_duration_seconds,
            "errors": summary.errors,
            "worker_results": [
                {
                    "worker_id": r.worker_id,
                    "returncode": r.returncode,
                    "shard_path": str(r.shard_path),
                    "games_assigned": r.games_assigned,
                    "seed": r.seed,
                    "duration_seconds": r.duration_seconds,
                }
                for r in summary.worker_results
            ],
        }
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary_dict, indent=2), encoding="utf-8")
        print(f"[summary] written to {args.summary_json}", flush=True)

    return 1 if failed_workers else 0


if __name__ == "__main__":
    raise SystemExit(main())
