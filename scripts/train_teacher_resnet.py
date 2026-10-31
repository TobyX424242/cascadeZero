from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from cascade_zero.teacher_training import (  # noqa: E402
    TeacherResNet,
    TeacherResNetConfig,
    choose_device,
    count_parameters,
    create_data_loaders,
    resolve_num_workers,
    run_epoch,
    save_checkpoint,
)


def resolve_bool_flag(value: object, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


def resolve_autocast_dtype(device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def create_grad_scaler(device: torch.device, enabled: bool, autocast_dtype: torch.dtype):
    scaler_enabled = device.type == "cuda" and enabled and autocast_dtype == torch.float16
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    return torch.cuda.amp.GradScaler(enabled=scaler_enabled)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a dual-head teacher ResNet on Cascade self-play JSONL data.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/training.json")
    parser.add_argument("--dataset", dest="datasets", action="append", default=None)
    parser.add_argument("--dataset-weight", dest="dataset_weights", action="append", nargs=2, metavar=("DATASET", "WEIGHT"), default=None)
    parser.add_argument("--feature-layout-version", type=str, default=None)
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--log-every-batches", type=int, default=None)
    return parser.parse_args()


def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def parse_dataset_weights(raw_weights: object) -> dict[str, float]:
    if raw_weights is None:
        return {}

    if isinstance(raw_weights, dict):
        items = raw_weights.items()
    elif isinstance(raw_weights, list):
        normalized_items: list[tuple[object, object]] = []
        for entry in raw_weights:
            if isinstance(entry, dict):
                normalized_items.append((entry.get("path"), entry.get("weight")))
            elif isinstance(entry, (list, tuple)) and len(entry) == 2:
                normalized_items.append((entry[0], entry[1]))
            else:
                raise ValueError("dataset_weights entries must be {path, weight} objects or [path, weight] pairs")
        items = normalized_items
    else:
        raise ValueError("dataset_weights must be a mapping or a list of entries")

    dataset_weights: dict[str, float] = {}
    for dataset_path, raw_weight in items:
        if dataset_path is None:
            raise ValueError("dataset_weights entries must include a dataset path")
        weight = float(raw_weight)
        if weight <= 0.0:
            raise ValueError(f"dataset weight must be positive: {dataset_path} -> {weight}")
        dataset_weights[str(Path(str(dataset_path)))] = weight

    return dataset_weights


def build_persisted_training_config(
    config: dict[str, object],
    model_config: TeacherResNetConfig,
    dataset_paths: list[str],
    dataset_weights: dict[str, float],
    output_dir: Path,
    init_checkpoint_path: Path | None,
    device: torch.device,
    resolved_num_workers: int,
) -> dict[str, object]:
    persisted_config = dict(config)
    persisted_config["dataset_paths"] = list(dataset_paths)
    persisted_config["dataset_weights"] = [
        {"path": dataset_path, "weight": weight}
        for dataset_path, weight in dataset_weights.items()
    ]
    persisted_config["output_dir"] = str(output_dir)
    persisted_config["device"] = persisted_config.get("device", "auto")
    persisted_config["resolved_device"] = str(device)
    persisted_config["num_workers"] = resolved_num_workers
    persisted_config["model"] = model_config.to_dict()
    if init_checkpoint_path is not None:
        persisted_config["init_checkpoint"] = str(init_checkpoint_path)
    return persisted_config


def load_initial_checkpoint(
    checkpoint_path: Path,
    model: TeacherResNet,
    model_config: TeacherResNetConfig,
) -> dict[str, object]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"init checkpoint not found: {checkpoint_path}")

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_config = TeacherResNetConfig.from_dict(payload["model_config"])
    if checkpoint_config.to_dict() != model_config.to_dict():
        raise ValueError(
            "init checkpoint model_config does not match requested config: "
            f"checkpoint={checkpoint_config.to_dict()} requested={model_config.to_dict()}"
        )

    model.load_state_dict(payload["model_state_dict"])
    return payload


def main() -> int:
    args = parse_args()
    config = load_config(args.config)

    if args.feature_layout_version is not None:
        model_config_section = dict(config.get("model", {}))
        model_config_section["feature_layout_version"] = args.feature_layout_version
        config["model"] = model_config_section

    if args.datasets:
        config["dataset_paths"] = args.datasets
    if args.dataset_weights:
        config["dataset_weights"] = [
            {"path": dataset_path, "weight": float(weight)}
            for dataset_path, weight in args.dataset_weights
        ]
    if args.init_checkpoint is not None:
        config["init_checkpoint"] = str(args.init_checkpoint)
    if args.output_dir is not None:
        config["output_dir"] = str(args.output_dir)
    for key, value in {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "device": args.device,
        "num_workers": args.num_workers,
        "log_every_batches": args.log_every_batches,
    }.items():
        if value is not None:
            config[key] = value

    dataset_paths = config.get("dataset_paths", [])
    if not dataset_paths:
        raise ValueError("dataset_paths must contain at least one JSONL file")
    dataset_weights = parse_dataset_weights(config.get("dataset_weights"))
    normalized_dataset_paths = {str(Path(dataset_path)) for dataset_path in dataset_paths}
    unknown_weight_paths = sorted(set(dataset_weights) - normalized_dataset_paths)
    if unknown_weight_paths:
        joined_paths = ", ".join(unknown_weight_paths)
        raise ValueError(f"dataset_weights contains paths not present in dataset_paths: {joined_paths}")

    output_dir = Path(config.get("output_dir", PROJECT_ROOT / "artifacts/training/teacher_resnet_v1"))
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(config.get("device", "auto"))
    resolved_num_workers = resolve_num_workers(config.get("num_workers"), device)
    use_amp = resolve_bool_flag(config.get("amp"), device.type == "cuda")
    use_channels_last = resolve_bool_flag(config.get("channels_last"), device.type == "cuda")
    autocast_dtype = resolve_autocast_dtype(device)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    model_config = TeacherResNetConfig.from_dict(config.get("model", {}))
    model = TeacherResNet(model_config)
    init_checkpoint = config.get("init_checkpoint")
    init_checkpoint_path = Path(str(init_checkpoint)) if init_checkpoint else None
    init_checkpoint_payload = None
    if init_checkpoint_path is not None:
        init_checkpoint_payload = load_initial_checkpoint(init_checkpoint_path, model, model_config)

    model = model.to(device)
    if use_channels_last:
        model = model.to(memory_format=torch.channels_last)

    mhc_params = []
    base_params = []
    for name, param in model.named_parameters():
        if "mhc_matrix" in name:
            mhc_params.append(param)
        else:
            base_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": base_params, "lr": float(config.get("learning_rate", 1e-3))},
            {"params": mhc_params, "lr": float(config.get("learning_rate", 1e-3)) * 0.1, "weight_decay": 0.0}
        ],
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    grad_scaler = None
    if device.type == "cuda":
        grad_scaler = create_grad_scaler(device, use_amp, autocast_dtype)

    train_loader, val_loader, dataset_stats = create_data_loaders(
        dataset_paths=dataset_paths,
        batch_size=int(config.get("batch_size", 64)),
        train_fraction=float(config.get("train_fraction", 0.95)),
        seed=int(config.get("seed", 20260421)),
        num_workers=resolved_num_workers,
        pin_memory=device.type == "cuda",
        dataset_weights=dataset_weights or None,
        feature_layout_version=model_config.feature_layout_version,
    )
    effective_num_workers = int(dataset_stats.get("num_workers", resolved_num_workers))
    if dataset_stats["samples_total"] == 0:
        raise ValueError("no JSONL samples were found in dataset_paths")

    history: list[dict] = []
    best_val_loss = float("inf")
    best_metrics: dict | None = None
    epochs = int(config.get("epochs", 8))
    policy_loss_weight = float(config.get("policy_loss_weight", 1.0))
    value_loss_weight = float(config.get("value_loss_weight", 1.0))
    log_every_batches = int(config.get("log_every_batches", 0))

    print(
        f"device={device} samples_total={dataset_stats['samples_total']} samples_train={dataset_stats['samples_train']} "
        f"samples_val={dataset_stats['samples_val']} parameters={count_parameters(model)} "
        f"num_workers={effective_num_workers} amp={str(use_amp).lower()} amp_dtype={str(autocast_dtype).replace('torch.', '')} "
        f"channels_last={str(use_channels_last).lower()} weighted_sampling={str(bool(dataset_stats.get('weighted_sampling', 0))).lower()} "
        f"init_checkpoint={(str(init_checkpoint_path) if init_checkpoint_path is not None else 'none')}"
    )

    mhc_warmup_epochs = int(model_config.to_dict().get("warmup_epochs", 2)) if model_config.to_dict().get("use_mhc", False) else 0

    for epoch in range(1, epochs + 1):
        if mhc_warmup_epochs > 0:
            is_warmup = epoch <= mhc_warmup_epochs
            for param in mhc_params:
                param.requires_grad = not is_warmup
            if is_warmup:
                print(f"[mHC Warmup] epoch={epoch}/{mhc_warmup_epochs}: mhc_matrix gradients frozen.", flush=True)
            elif epoch == mhc_warmup_epochs + 1:
                print(f"[mHC Warmup] epoch={epoch}: warmup complete, mhc_matrix unfrozen.", flush=True)

        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            optimizer,
            policy_loss_weight,
            value_loss_weight,
            progress_label=f"phase=train epoch={epoch}",
            log_every_batches=log_every_batches,
            use_autocast=use_amp,
            autocast_dtype=autocast_dtype,
            channels_last=use_channels_last,
            grad_scaler=grad_scaler,
        )

        val_metrics = None
        if val_loader is not None:
            with torch.inference_mode():
                val_metrics = run_epoch(
                    model,
                    val_loader,
                    device,
                    None,
                    policy_loss_weight,
                    value_loss_weight,
                    progress_label=f"phase=val epoch={epoch}",
                    log_every_batches=log_every_batches,
                    use_autocast=use_amp,
                    autocast_dtype=autocast_dtype,
                    channels_last=use_channels_last,
                )

        epoch_metrics = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(epoch_metrics)

        monitored_loss = val_metrics["loss"] if val_metrics is not None else train_metrics["loss"]
        if monitored_loss < best_val_loss:
            best_val_loss = monitored_loss
            best_metrics = epoch_metrics
            save_checkpoint(output_dir / "best.pt", model, model_config, optimizer, epoch, epoch_metrics)

        print(
            f"epoch={epoch} train_loss={train_metrics['loss']:.4f} train_top1={train_metrics['top1']:.4f} "
            f"train_sps={train_metrics['samples_per_second']:.2f} train_sec={train_metrics['elapsed_seconds']:.2f}"
            + (
                f" val_loss={val_metrics['loss']:.4f} val_top1={val_metrics['top1']:.4f} "
                f"val_sps={val_metrics['samples_per_second']:.2f} val_sec={val_metrics['elapsed_seconds']:.2f}"
                if val_metrics is not None
                else ""
            )
        )

    final_metrics = history[-1]
    save_checkpoint(output_dir / "final.pt", model, model_config, optimizer, epochs, final_metrics)
    summary = {
        "dataset_paths": dataset_paths,
        "dataset_weights": dataset_weights,
        "dataset_stats": dataset_stats,
        "device": str(device),
        "epochs": epochs,
        "init_checkpoint": str(init_checkpoint_path) if init_checkpoint_path is not None else None,
        "init_checkpoint_epoch": int(init_checkpoint_payload["epoch"]) if init_checkpoint_payload is not None and "epoch" in init_checkpoint_payload else None,
        "model": model_config.to_dict(),
        "best": best_metrics,
        "history": history,
    }
    persisted_config = build_persisted_training_config(
        config=config,
        model_config=model_config,
        dataset_paths=dataset_paths,
        dataset_weights=dataset_weights,
        output_dir=output_dir,
        init_checkpoint_path=init_checkpoint_path,
        device=device,
        resolved_num_workers=effective_num_workers,
    )
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "train_config.json").write_text(json.dumps(persisted_config, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
