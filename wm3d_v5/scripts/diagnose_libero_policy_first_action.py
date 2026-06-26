#!/usr/bin/env python3
"""Compare cached LIBERO first actions against a policy checkpoint."""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import yaml

from wm3d_v3.eval.run_eval import build_model
from wm3d_v3.training.train_libero_success_p0 import LiberoExpertCacheDataset, _load_action_stats


def _make_dataset(cfg: dict[str, Any]) -> LiberoExpertCacheDataset:
    data_cfg = cfg["data"]
    model_cfg = yaml.safe_load(Path(cfg["base_cfg"]).read_text()).get("model", {})
    kwargs: dict[str, Any] = {"plan_state_dim": int(data_cfg.get("plan_state_dim", 8))}
    if "include_action_history" in inspect.signature(LiberoExpertCacheDataset).parameters:
        kwargs["include_action_history"] = int(model_cfg.get("policy_action_history_len", 0) or 0) > 0
    return LiberoExpertCacheDataset(Path(data_cfg["manifest"]), **kwargs)


def _load_model(cfg: dict[str, Any], ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    base_cfg = yaml.safe_load(Path(cfg["base_cfg"]).read_text())
    model = build_model(base_cfg)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    bad_missing = [key for key in missing if not key.startswith(("action_policy.", "geom."))]
    bad_unexpected = [key for key in unexpected if not key.startswith(("action_policy.", "geom."))]
    if bad_missing or bad_unexpected:
        raise RuntimeError({"bad_missing": bad_missing[:20], "unexpected": bad_unexpected[:20]})
    model = model.to(device).eval()
    _load_action_stats(model, Path(cfg["data"]["action_stats"]), device)
    return model


def _task_match(row: dict[str, Any], terms: list[str]) -> bool:
    if not terms:
        return True
    text = " ".join(str(row.get(key, "")) for key in ("task_name", "instruction")).lower()
    return all(term.lower() in text for term in terms)


def _tensor_sample(sample: dict[str, Any], key: str, device: torch.device) -> torch.Tensor | None:
    value = sample.get(key)
    if value is None:
        return None
    return value.unsqueeze(0).to(device)


def _match_horizon(value: torch.Tensor, horizon: int) -> torch.Tensor:
    if value.shape[1] == horizon:
        return value
    if value.shape[1] > horizon:
        return value[:, :horizon]
    pad = value[:, -1:].expand(-1, horizon - value.shape[1], *value.shape[2:])
    return torch.cat([value, pad], dim=1)


def _manual_act_policy(
    model: torch.nn.Module,
    s: torch.Tensor,
    c: torch.Tensor,
    kwargs: dict[str, torch.Tensor | None],
) -> dict[str, torch.Tensor]:
    dual_out = model.dual(s, c, action_cond=None)
    proj = model.action_proj(dual_out["z_a"])
    policy_kwargs = {
        "lowdim_state": kwargs.get("lowdim_state"),
        "object_state": kwargs.get("object_state"),
        "plan_state": kwargs.get("plan_state"),
        "action_history": kwargs.get("action_history"),
        "progress_state": kwargs.get("progress_state"),
    }
    if "context_rgb" in inspect.signature(model.action_policy.forward).parameters:
        policy_kwargs["context_rgb"] = kwargs.get("context_rgb")
    pol = model.action_policy(
        dual_out["pred_tokens"],
        task_emb=c,
        **policy_kwargs,
    )
    policy_horizon = int(pol["policy_pose_norm"].shape[1])
    pose_norm = _match_horizon(proj["pose_norm"], policy_horizon) + pol["policy_pose_norm"]
    gripper_logit = _match_horizon(proj["gripper_logit"], policy_horizon) + pol["policy_gripper_logit"]
    out = dict(pol)
    out["policy_pose_norm"] = pose_norm
    out["policy_gripper_logit"] = gripper_logit
    return out


@torch.no_grad()
def _predict_first(model: torch.nn.Module, sample: dict[str, Any], device: torch.device) -> torch.Tensor:
    s = sample["s_in"].unsqueeze(0).to(device)
    c = sample["c"].unsqueeze(0).to(device)
    kwargs = {
        "lowdim_state": _tensor_sample(sample, "lowdim_state", device),
        "object_state": _tensor_sample(sample, "object_state", device),
        "plan_state": _tensor_sample(sample, "plan_state", device),
        "action_history": _tensor_sample(sample, "action_history", device),
        "progress_state": sample["progress_tgt"].float().reshape(1, 1).to(device) if "progress_tgt" in sample else None,
        "context_rgb": _tensor_sample(sample, "context_rgb", device),
    }
    act_policy = getattr(model, "act_policy", None)
    if callable(act_policy):
        out = act_policy(s, c, **kwargs)
    else:
        out = _manual_act_policy(model, s, c, kwargs)
    pose_norm = out["policy_pose_norm"].float()
    mean = model.action_proj.mean.detach().to(device=device, dtype=pose_norm.dtype)
    std = model.action_proj.std.detach().to(device=device, dtype=pose_norm.dtype).clamp_min(1e-6)
    pose_raw = pose_norm * std + mean
    grip = torch.sigmoid(out["policy_gripper_logit"].float())[..., None]
    return torch.cat([pose_raw, grip], dim=-1)[0, 0].detach().cpu()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--task_terms", default="cream,butter,basket")
    ap.add_argument("--target_start", type=int, default=0)
    ap.add_argument("--max_rows", type=int, default=20)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.cfg.read_text())
    device = torch.device(args.device)
    ds = _make_dataset(cfg)
    terms = [term.strip() for term in args.task_terms.split(",") if term.strip()]
    indices = [
        idx
        for idx, row in enumerate(ds.rows)
        if int(row.get("target_start", -1)) == int(args.target_start) and _task_match(row, terms)
    ][: int(args.max_rows)]
    if not indices:
        raise RuntimeError(f"no rows matched target_start={args.target_start} terms={terms}")
    model = _load_model(cfg, args.ckpt, device)

    records: list[dict[str, Any]] = []
    abs_errors = []
    for idx in indices:
        row = ds.rows[idx]
        sample = ds[idx]
        pred = _predict_first(model, sample, device)
        target = sample["action_tgt"][0].detach().float().cpu()
        err = (pred[:6] - target[:6]).abs()
        abs_errors.append(err)
        records.append(
            {
                "idx": int(idx),
                "task_name": row.get("task_name"),
                "demo_id": row.get("demo_id"),
                "context_start": row.get("context_start"),
                "target_start": row.get("target_start"),
                "target_first": [float(x) for x in target.tolist()],
                "pred_first": [float(x) for x in pred.tolist()],
                "abs_err6": [float(x) for x in err.tolist()],
                "l1_6": float(err.mean()),
                "l1_xyz": float(err[:3].mean()),
                "err_y": float(err[1]),
            }
        )
    err_t = torch.stack(abs_errors, dim=0)
    summary = {
        "cfg": str(args.cfg),
        "ckpt": str(args.ckpt),
        "rows": len(records),
        "target_start": int(args.target_start),
        "task_terms": terms,
        "mean_l1_6": float(err_t.mean()),
        "mean_l1_xyz": float(err_t[:, :3].mean()),
        "mean_err_y": float(err_t[:, 1].mean()),
        "axis_l1": [float(x) for x in err_t.mean(dim=0).tolist()],
        "records": records,
    }
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
