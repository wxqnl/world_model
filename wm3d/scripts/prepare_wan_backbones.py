"""Thin CPU entry point for upstream Wan-derived expert initialization."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy
import sys

import yaml

from wm3d.models.foundation_world_model import _load_upstream_symbols, foundation_config_from_mapping


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    cfg = foundation_config_from_mapping(yaml.safe_load(args.model.read_text())["model"])
    cfg.validate()
    if cfg.initialization != "wan_video":
        raise ValueError("this command only prepares generic Wan-derived initialization")
    action, state = Path(cfg.wan_action_backbone_path), Path(cfg.wan_3d_backbone_path)
    if action.parent != state.parent:
        raise ValueError("expert backbones must share an explicit output directory")
    receipt = action.parent / "initialization.json"
    if any(path.exists() for path in (action, state, receipt)):
        raise FileExistsError("refusing to overwrite existing initialization assets")
    # Use the same narrow import boundary as the model factory, avoiding
    # unrelated WSA-Base/Qwen/Cosmos dependencies in lerobot.policies.__init__.
    _load_upstream_symbols(cfg.upstream_source_root)
    script = Path(cfg.upstream_source_root) / "tools/preprocess_expert_backbones.py"
    argv = [str(script), "--expert", "both", "--device", "cpu", "--dtype", "bfloat16",
            "--model-id", cfg.wan_model_path, "--tokenizer-model-id", cfg.wan_tokenizer_model_path,
            "--action-dim", str(cfg.action_width), "--da3-num-views", str(cfg.num_views),
            "--da3-tokens-per-view", str(cfg.K * cfg.P), "--da3-query-dim", str(cfg.token_dim),
            "--future-3d-tokens-per-view", str(cfg.K * cfg.P),
            "--redirect-common-files", "false", "--apply-alpha-scaling", "true",
            "--action-output", str(action), "--future-3d-output", str(state)]
    previous = sys.argv
    try:
        sys.argv = argv
        runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = previous
    if not all(path.is_file() and path.stat().st_size > 0 for path in (action, state)):
        raise RuntimeError("upstream preprocessing did not produce both backbones")
    with receipt.open("x") as handle:
        json.dump({
            "schema": "wm3d_wan_derived_initialization_v1",
            "source_model": cfg.wan_model_path,
            "source_is_wsa_checkpoint": False,
            "method": "upstream_linear_interpolation_alpha_scaling",
            "action_backbone": str(action), "native_3d_backbone": str(state),
            "new_robot_and_3d_interfaces": True, "command": argv,
        }, handle, indent=2)
    print(f"Prepared Wan-derived expert backbones; receipt: {receipt}")


if __name__ == "__main__":
    main()
