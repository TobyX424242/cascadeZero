from __future__ import annotations

import json
import math
import os
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .teacher_features import (
    FEATURE_LAYOUT_HEIGHT_BUCKETS_V2,
    TEACHER_FEATURE_PLANES,
    TOTAL_ACTIONS,
    encode_teacher_feature_planes,
    flip_action_index,
    resolve_feature_layout_version,
)


@dataclass(slots=True)
class TeacherResNetConfig:
    input_planes: int = TEACHER_FEATURE_PLANES
    board_side: int = 8
    action_dim: int = TOTAL_ACTIONS
    feature_layout_version: str = FEATURE_LAYOUT_HEIGHT_BUCKETS_V2
    channels: int = 96
    residual_blocks: int = 6
    policy_channels: int = 48
    value_channels: int = 32
    value_hidden_dim: int = 128
    num_streams: int = 4
    sinkhorn_iterations: int = 7
    use_mhc: bool = False
    warmup_epochs: int = 2

    def __post_init__(self) -> None:
        self.feature_layout_version = resolve_feature_layout_version(self.feature_layout_version)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "TeacherResNetConfig":
        return cls(**{key: raw[key] for key in cls.__dataclass_fields__ if key in raw})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def log_sinkhorn(log_alpha: torch.Tensor, iterations: int = 7) -> torch.Tensor:
    """Approximate a doubly stochastic matrix with log-space Sinkhorn iterations."""
    for _ in range(iterations):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=1, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=0, keepdim=True)
    return torch.exp(log_alpha)


class mHCBlock(nn.Module):
    def __init__(self, channels: int, num_streams: int = 4, iterations: int = 7) -> None:
        super().__init__()
        self.num_streams = num_streams
        self.iterations = iterations
        self.stream_channels = channels // num_streams

        # Start the stream mixer close to the identity.
        self.mhc_matrix = nn.Parameter(torch.eye(num_streams) * 5.0)

        # Each stream has its own convolution group.
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=num_streams, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=num_streams, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.is_reparameterized = False

    def reparameterize(self) -> None:
        if self.is_reparameterized:
            return

        B = log_sinkhorn(self.mhc_matrix, self.iterations).detach()
        C = self.num_streams * self.stream_channels

        fused_conv = nn.Conv2d(C, C, kernel_size=1, bias=False).to(self.mhc_matrix.device)
        fused_weight = torch.zeros(C, C, 1, 1, device=B.device, dtype=B.dtype)

        for i in range(self.num_streams):
            for j in range(self.num_streams):
                start_i, end_i = i * self.stream_channels, (i + 1) * self.stream_channels
                start_j, end_j = j * self.stream_channels, (j + 1) * self.stream_channels
                fused_weight[start_i:end_i, start_j:end_j, 0, 0] = B[i, j] * torch.eye(self.stream_channels, device=B.device, dtype=B.dtype)

        fused_conv.weight = nn.Parameter(fused_weight)
        self.fused_mixer = fused_conv
        self.is_reparameterized = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_reparameterized:
            mixed = self.fused_mixer(x)
        else:
            b, c, h, w = x.shape
            streams = x.view(b, self.num_streams, self.stream_channels, h, w)
            B = log_sinkhorn(self.mhc_matrix, self.iterations)
            mixed_streams = torch.einsum('kj, bjchw -> bkchw', B, streams)
            mixed = mixed_streams.reshape(b, c, h, w)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        out = self.bn2(out)

        # Combine the transformed features with the mixed residual streams.
        return F.relu(out + mixed, inplace=True)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs
        outputs = self.conv1(inputs)
        outputs = self.bn1(outputs)
        outputs = F.relu(outputs, inplace=True)
        outputs = self.conv2(outputs)
        outputs = self.bn2(outputs)
        return F.relu(outputs + residual, inplace=True)


class TeacherResNet(nn.Module):
    def __init__(self, config: TeacherResNetConfig) -> None:
        super().__init__()
        self.config = config
        board_area = config.board_side * config.board_side

        self.stem_conv = nn.Conv2d(config.input_planes, config.channels, kernel_size=3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(config.channels)
        if getattr(config, "use_mhc", False):
            self.blocks = nn.ModuleList(
                mHCBlock(
                    config.channels,
                    num_streams=getattr(config, "num_streams", 4),
                    iterations=getattr(config, "sinkhorn_iterations", 7)
                ) for _ in range(config.residual_blocks)
            )
        else:
            self.blocks = nn.ModuleList(ResidualBlock(config.channels) for _ in range(config.residual_blocks))

        self.policy_conv = nn.Conv2d(config.channels, config.policy_channels, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(config.policy_channels)
        self.policy_linear = nn.Linear(config.policy_channels * board_area, config.action_dim)

        self.value_conv = nn.Conv2d(config.channels, config.value_channels, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(config.value_channels)
        self.value_hidden = nn.Linear(config.value_channels * board_area, config.value_hidden_dim)
        self.value_out = nn.Linear(config.value_hidden_dim, 1)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.stem_conv(inputs)
        outputs = self.stem_bn(outputs)
        outputs = F.relu(outputs, inplace=True)
        for block in self.blocks:
            outputs = block(outputs)

        policy = self.policy_conv(outputs)
        policy = self.policy_bn(policy)
        policy = F.relu(policy, inplace=True)
        policy = torch.flatten(policy, start_dim=1)
        policy_logits = self.policy_linear(policy)

        value = self.value_conv(outputs)
        value = self.value_bn(value)
        value = F.relu(value, inplace=True)
        value = torch.flatten(value, start_dim=1)
        value = F.relu(self.value_hidden(value), inplace=True)
        value = torch.tanh(self.value_out(value)).squeeze(-1)
        return policy_logits, value

    def reparameterize(self) -> None:
        for block in self.blocks:
            if hasattr(block, "reparameterize"):
                block.reparameterize()


class JsonlTeacherDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        offsets: list[tuple[str, int]],
        feature_layout_version: str = FEATURE_LAYOUT_HEIGHT_BUCKETS_V2,
    ) -> None:
        self.offsets = offsets
        self.feature_layout_version = resolve_feature_layout_version(feature_layout_version)
        self._handles: dict[str, Any] = {}

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        path, offset = self.offsets[index]
        handle = self._handles.get(path)
        if handle is None:
            handle = open(path, "r", encoding="utf-8")
            self._handles[path] = handle

        handle.seek(offset)
        raw_line = handle.readline()
        sample = json.loads(raw_line)

        features = encode_teacher_feature_planes(
            sample["root_state"],
            self.feature_layout_version,
        )
        turn = sample["root_state"]["turn"]
        policy_target = torch.zeros(TOTAL_ACTIONS, dtype=torch.float32)
        legal_mask = torch.zeros(TOTAL_ACTIONS, dtype=torch.bool)
        for action_target in sample["actions"]:
            action_index = int(action_target["action_index"])
            if turn == "blue":
                action_index = flip_action_index(action_index)
            policy_target[action_index] = float(action_target["policy_probability"])
            legal_mask[action_index] = True

        if not bool(legal_mask.any()):
            legal_mask[:] = True

        value_target = sample.get("outcome_target")
        if value_target is None:
            value_target = sample.get("root_value", 0.0)

        return {
            "features": torch.from_numpy(features),
            "policy_target": policy_target,
            "legal_mask": legal_mask,
            "value_target": torch.tensor(float(value_target), dtype=torch.float32),
        }

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_handles"] = {}
        return state

    def __del__(self) -> None:
        for handle in getattr(self, "_handles", {}).values():
            handle.close()


def build_offset_index(dataset_paths: list[str]) -> list[tuple[str, int]]:
    offsets: list[tuple[str, int]] = []
    for dataset_path in dataset_paths:
        path = str(Path(dataset_path))
        with open(path, "r", encoding="utf-8") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if line.strip():
                    offsets.append((path, offset))
    return offsets


def normalize_dataset_weights(dataset_weights: dict[str, float] | None) -> dict[str, float]:
    normalized: dict[str, float] = {}
    if not dataset_weights:
        return normalized

    for dataset_path, raw_weight in dataset_weights.items():
        weight = float(raw_weight)
        if weight <= 0.0:
            raise ValueError(f"dataset weight must be positive: {dataset_path} -> {weight}")
        normalized[str(Path(dataset_path))] = weight

    return normalized


def build_train_sample_weights(
    train_offsets: list[tuple[str, int]],
    dataset_weights: dict[str, float],
) -> list[float] | None:
    if not train_offsets or not dataset_weights:
        return None

    sample_count_by_path = Counter(path for path, _ in train_offsets)
    sample_weights = [
        dataset_weights.get(path, 1.0) / float(sample_count_by_path[path])
        for path, _ in train_offsets
    ]
    if not sample_weights:
        return None

    reference_weight = sample_weights[0]
    if all(abs(weight - reference_weight) <= 1e-12 for weight in sample_weights[1:]):
        return None

    return sample_weights


def split_dataset_offsets(
    offsets: list[tuple[str, int]],
    train_fraction: float,
    seed: int,
) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    if not offsets:
        return [], []

    shuffled = list(offsets)
    random.Random(seed).shuffle(shuffled)
    if len(shuffled) == 1:
        return shuffled, []

    train_fraction = min(max(train_fraction, 0.5), 0.999)
    train_size = int(len(shuffled) * train_fraction)
    train_size = min(max(train_size, 1), len(shuffled) - 1)
    return shuffled[:train_size], shuffled[train_size:]


def create_data_loaders(
    dataset_paths: list[str],
    batch_size: int,
    train_fraction: float,
    seed: int,
    num_workers: int,
    pin_memory: bool,
    dataset_weights: dict[str, float] | None = None,
    feature_layout_version: str = FEATURE_LAYOUT_HEIGHT_BUCKETS_V2,
) -> tuple[DataLoader, DataLoader | None, dict[str, int]]:
    offsets = build_offset_index(dataset_paths)
    normalized_dataset_weights = normalize_dataset_weights(dataset_weights)
    if normalized_dataset_weights:
        available_paths = {path for path, _ in offsets}
        unknown_weight_paths = sorted(set(normalized_dataset_weights) - available_paths)
        if unknown_weight_paths:
            joined_paths = ", ".join(unknown_weight_paths)
            raise ValueError(f"dataset_weights contains paths not present in dataset_paths: {joined_paths}")

    train_offsets, val_offsets = split_dataset_offsets(offsets, train_fraction, seed)
    max_batches = max(math.ceil(len(train_offsets) / max(batch_size, 1)), 1) if train_offsets else 0
    effective_num_workers = min(num_workers, max_batches)
    train_sample_weights = build_train_sample_weights(train_offsets, normalized_dataset_weights)
    train_sampler = None
    if train_sample_weights is not None:
        train_sampler = WeightedRandomSampler(
            train_sample_weights,
            num_samples=len(train_offsets),
            replacement=True,
            generator=torch.Generator().manual_seed(seed),
        )

    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": effective_num_workers,
        "pin_memory": pin_memory,
    }
    if effective_num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4

    train_loader = DataLoader(
        JsonlTeacherDataset(train_offsets, feature_layout_version),
        shuffle=train_sampler is None,
        sampler=train_sampler,
        **loader_kwargs,
    )
    val_loader = None
    if val_offsets:
        val_loader = DataLoader(
            JsonlTeacherDataset(val_offsets, feature_layout_version),
            shuffle=False,
            **loader_kwargs,
        )

    return train_loader, val_loader, {
        "samples_total": len(offsets),
        "samples_train": len(train_offsets),
        "samples_val": len(val_offsets),
        "num_workers": effective_num_workers,
        "weighted_sampling": int(train_sampler is not None),
    }


def choose_device(requested_device: str | None) -> torch.device:
    if requested_device and requested_device != "auto":
        return torch.device(requested_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_num_workers(requested_num_workers: Any, device: torch.device) -> int:
    if requested_num_workers is None:
        requested_num_workers = "auto"

    if isinstance(requested_num_workers, str):
        normalized = requested_num_workers.strip().lower()
        if normalized == "auto":
            cpu_count = os.cpu_count() or 1
            if device.type != "cuda" or cpu_count <= 2:
                return 0
            return min(max(cpu_count // 2, 1), 8)
        requested_num_workers = int(normalized)

    return max(int(requested_num_workers), 0)


def masked_policy_cross_entropy(
    policy_logits: torch.Tensor,
    policy_target: torch.Tensor,
    legal_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    masked_logits = policy_logits.masked_fill(~legal_mask, torch.finfo(policy_logits.dtype).min)
    log_probs = F.log_softmax(masked_logits, dim=1)
    loss = -(policy_target * log_probs).sum(dim=1).mean()
    return loss, masked_logits


def run_epoch(
    model: TeacherResNet,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    policy_loss_weight: float,
    value_loss_weight: float,
    progress_label: str | None = None,
    log_every_batches: int = 0,
    use_autocast: bool = False,
    autocast_dtype: torch.dtype = torch.float16,
    channels_last: bool = False,
    grad_scaler: Any | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    started_at = time.perf_counter()
    total_batches_estimate = len(loader)
    autocast_enabled = use_autocast and device.type == "cuda"

    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_top1 = 0.0
    total_batches = 0
    total_examples = 0

    for batch_index, batch in enumerate(loader, start=1):
        features = batch["features"].to(device=device, dtype=torch.float32, non_blocking=True)
        if channels_last:
            features = features.contiguous(memory_format=torch.channels_last)
        policy_target = batch["policy_target"].to(device=device, dtype=torch.float32, non_blocking=True)
        legal_mask = batch["legal_mask"].to(device=device, dtype=torch.bool, non_blocking=True)
        value_target = batch["value_target"].to(device=device, dtype=torch.float32, non_blocking=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=autocast_enabled,
            ):
                policy_logits, value_pred = model(features)
                policy_loss, masked_logits = masked_policy_cross_entropy(policy_logits, policy_target, legal_mask)
                value_loss = F.mse_loss(value_pred, value_target)
                loss = policy_loss_weight * policy_loss + value_loss_weight * value_loss

        if training:
            optimizer.zero_grad(set_to_none=True)
            if grad_scaler is not None and grad_scaler.is_enabled():
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                loss.backward()
                optimizer.step()

        batch_size = features.shape[0]
        target_action = policy_target.argmax(dim=1)
        predicted_action = masked_logits.argmax(dim=1)
        total_top1 += (predicted_action == target_action).float().sum().item()
        total_loss += loss.item() * batch_size
        total_policy_loss += policy_loss.item() * batch_size
        total_value_loss += value_loss.item() * batch_size
        total_examples += batch_size
        total_batches += 1

        should_log = (
            progress_label is not None
            and log_every_batches > 0
            and (
                batch_index == 1
                or batch_index % log_every_batches == 0
                or batch_index == total_batches_estimate
            )
        )
        if should_log:
            elapsed_seconds = max(time.perf_counter() - started_at, 1e-9)
            samples_per_second = total_examples / elapsed_seconds
            average_loss = total_loss / total_examples
            average_top1 = total_top1 / total_examples
            print(
                f"{progress_label} batch={batch_index}/{total_batches_estimate} "
                f"loss={average_loss:.4f} top1={average_top1:.4f} "
                f"samples={total_examples} samples_per_sec={samples_per_second:.2f}",
                flush=True,
            )

    elapsed_seconds = time.perf_counter() - started_at
    if total_examples == 0:
        return {
            "loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "top1": 0.0,
            "examples": 0.0,
            "batches": 0.0,
            "elapsed_seconds": elapsed_seconds,
            "samples_per_second": 0.0,
        }

    return {
        "loss": total_loss / total_examples,
        "policy_loss": total_policy_loss / total_examples,
        "value_loss": total_value_loss / total_examples,
        "top1": total_top1 / total_examples,
        "examples": float(total_examples),
        "batches": float(total_batches),
        "elapsed_seconds": elapsed_seconds,
        "samples_per_second": total_examples / max(elapsed_seconds, 1e-9),
    }


def save_checkpoint(
    checkpoint_path: Path,
    model: TeacherResNet,
    model_config: TeacherResNetConfig,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, Any],
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_config": model_config.to_dict(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        checkpoint_path,
    )


def export_checkpoint(
    checkpoint_path: str,
    output_dir: str,
    manifest_name: str = "teacher_resnet_onnx.json",
    execution_provider: str = "cuda",
) -> tuple[Path, Path]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = TeacherResNetConfig.from_dict(payload["model_config"])
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model = TeacherResNet(model_config)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    if hasattr(model, "reparameterize"):
        model.reparameterize()

    onnx_file = output_path / "teacher_resnet.onnx"
    manifest_path = output_path / manifest_name

    dummy_input = torch.zeros(
        (1, model_config.input_planes, model_config.board_side, model_config.board_side),
        dtype=torch.float32,
    )
    with torch.inference_mode():
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_file),
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["policy_logits", "value"],
            dynamic_axes={
                "input": {0: "batch"},
                "policy_logits": {0: "batch"},
                "value": {0: "batch"},
            },
        )

    artifact_files = [onnx_file.name]
    onnx_sidecar = output_path / f"{onnx_file.name}.data"
    if onnx_sidecar.exists():
        artifact_files.append(onnx_sidecar.name)

    manifest = {
        "format": "cascade_teacher_onnx_v1",
        "onnx_file": onnx_file.name,
        "artifact_files": artifact_files,
        "feature_layout_version": model_config.feature_layout_version,
        "feature_planes": model_config.input_planes,
        "board_side": model_config.board_side,
        "action_dim": model_config.action_dim,
        "input_name": "input",
        "policy_output_name": "policy_logits",
        "value_output_name": "value",
        "execution_provider": execution_provider,
        "device_id": 0,
        "allow_cpu_fallback": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path, onnx_file


def count_parameters(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))
