from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from cascade_zero.teacher_training import export_checkpoint  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a teacher ResNet checkpoint to an ONNX model + runtime manifest.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest-name", type=str, default="teacher_resnet_onnx.json")
    parser.add_argument("--execution-provider", type=str, default="cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path, onnx_path = export_checkpoint(
        checkpoint_path=str(args.checkpoint),
        output_dir=str(args.output_dir),
        manifest_name=args.manifest_name,
        execution_provider=args.execution_provider,
    )
    print(f"manifest={manifest_path} model={onnx_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
