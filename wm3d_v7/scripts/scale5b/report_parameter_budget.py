#!/usr/bin/env python3
"""在 meta device 上生成 Native WM3D-V7 5B 精确参数预算。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from wm3d_v3.data.scale5b_contracts import resolve_regular_file
from wm3d_v3.models.native5b import NativeWM3D5B, config_from_mapping


FORMAL_TOTAL = 4_956_589_929


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--format", choices=("json", "markdown"), default="markdown")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expect-total", type=int, default=FORMAL_TOTAL)
    return parser.parse_args()


def _report(config_path: Path, expected: int) -> dict[str, object]:
    path = resolve_regular_file(config_path.parent, config_path.name)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg = config_from_mapping(dict(raw["model"]))
    cfg.validate()
    with torch.device("meta"):
        model = NativeWM3D5B(cfg)
    top_level: dict[str, int] = {}
    for name, parameter in model.named_parameters():
        prefix = name.split(".", 1)[0]
        top_level[prefix] = top_level.get(prefix, 0) + parameter.numel()
    total = sum(top_level.values())
    if expected > 0 and total != expected:
        raise ValueError(f"参数总数 {total:,} 与期望 {expected:,} 不一致")
    major = model.parameter_counts()
    categorized = sum(value for key, value in major.items() if key != "total")
    groups = {
        "state_trunk": major["state_trunk"],
        "action_trunk": major["action_trunk"],
        "state_action_bridges": major["bridges"],
        "multiview_fuser": major["multiview_fuser"],
        "rgb_head": major["rgb_head"],
        "geometry_head": major["geometry_head"],
        "action_head": major["action_head"],
        "interfaces_memory_positions": total - categorized,
    }
    return {
        "pass": True,
        "config": str(path),
        "total": total,
        "groups": {
            name: {"parameters": value, "percent": 100.0 * value / total}
            for name, value in groups.items()
        },
        "top_level_modules": {
            name: {"parameters": value, "percent": 100.0 * value / total}
            for name, value in sorted(
                top_level.items(), key=lambda item: (-item[1], item[0])
            )
        },
        "shape_contract": {
            "T": cfg.T,
            "P": cfg.P,
            "K": cfg.K,
            "external_token_dim": cfg.token_dim,
            "state_hidden": cfg.state_hidden,
            "state_layers": cfg.state_layers,
            "action_hidden": cfg.action_hidden,
            "action_layers": cfg.action_layers,
            "bridge_count": len(cfg.bridge_layers_state),
            "state_token_positions_per_sample": (cfg.T + cfg.K) * cfg.P,
            "visual_context_seconds_at_5hz": cfg.T / 5.0,
            "future_seconds_at_5hz": cfg.K / 5.0,
        },
    }


def _markdown(report: dict[str, object]) -> str:
    groups = report["groups"]
    assert isinstance(groups, dict)
    lines = [
        "# WM3D-V7 Native 5B 参数预算（程序生成）",
        "",
        f"精确总参数：**{int(report['total']):,}**。",
        "",
        "| 模块 | 参数量 | 占比 |",
        "|---|---:|---:|",
    ]
    labels = {
        "state_trunk": "原生 3D/world state trunk",
        "action_trunk": "原生 grouped-action trunk",
        "state_action_bridges": "state↔action 双向桥",
        "multiview_fuser": "三视角融合",
        "rgb_head": "显式 RGB head",
        "geometry_head": "显式 depth/point/camera/confidence head",
        "action_head": "动作分布 head",
        "interfaces_memory_positions": "接口投影、长期记忆、位置/查询参数",
    }
    for name, value in groups.items():
        assert isinstance(value, dict)
        lines.append(
            f"| {labels[name]} | {int(value['parameters']):,} | "
            f"{float(value['percent']):.4f}% |"
        )
    lines.extend(
        ["", "```json", json.dumps(report["shape_contract"], indent=2), "```", ""]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    report = _report(args.config, args.expect_total)
    payload = (
        json.dumps(report, sort_keys=True, indent=2) + "\n"
        if args.format == "json"
        else _markdown(report)
    )
    if args.output is None:
        print(payload, end="")
    else:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as handle:
            handle.write(payload)


if __name__ == "__main__":
    main()
