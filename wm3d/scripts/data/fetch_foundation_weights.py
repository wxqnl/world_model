"""Fetch and verify WSA-Large and its Wan2.2 foundation assets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download
from safetensors import safe_open


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True, help="WSA-Large output")
    parser.add_argument("--wan-output", type=Path, required=True)
    parser.add_argument("--repo", default="zaleni/WSA-Large")
    parser.add_argument("--wan-repo", default="Wan-AI/Wan2.2-TI2V-5B")
    parser.add_argument("--endpoint", default="https://huggingface.co")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(
        snapshot_download(
            args.repo,
            local_dir=str(args.output.expanduser().absolute()),
            endpoint=args.endpoint,
        )
    ).resolve(strict=True)
    required = {
        "config.json",
        "model.safetensors",
        "stats.json",
        "train_config.json",
    }
    missing = sorted(name for name in required if not (root / name).is_file())
    if missing:
        raise RuntimeError(f"WSA joint checkpoint is incomplete: {missing}")
    with safe_open(root / "model.safetensors", framework="pt", device="cpu") as file:
        keys = list(file.keys())
    required_prefixes = (
        "model.mot.mixtures.video.",
        "model.mot.mixtures.action.",
        "model.mot.mixtures.future_3d.",
    )
    absent = [
        prefix
        for prefix in required_prefixes
        if not any(key.startswith(prefix) for key in keys)
    ]
    if absent:
        raise RuntimeError(f"WSA checkpoint misses joint expert prefixes: {absent}")
    wan_root = Path(
        snapshot_download(
            args.wan_repo,
            local_dir=str(args.wan_output.expanduser().absolute()),
            endpoint=args.endpoint,
        )
    ).resolve(strict=True)
    required_wan = {
        "config.json",
        "Wan2.2_VAE.pth",
        "models_t5_umt5-xxl-enc-bf16.pth",
        "diffusion_pytorch_model.safetensors.index.json",
        "google/umt5-xxl/spiece.model",
    }
    missing_wan = sorted(
        name for name in required_wan if not (wan_root / name).is_file()
    )
    if missing_wan:
        raise RuntimeError(f"Wan2.2 foundation assets are incomplete: {missing_wan}")
    print(
        json.dumps(
            {
                "wsa_output": str(root),
                "wsa_checkpoint_bytes": (root / "model.safetensors").stat().st_size,
                "wsa_tensor_keys": len(keys),
                "wan_output": str(wan_root),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
