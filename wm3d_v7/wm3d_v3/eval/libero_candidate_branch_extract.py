"""Extract real proposer candidates for simulator-branched outcome labeling."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import default_collate

from wm3d_v3.data.action_condition import make_action_condition
from wm3d_v3.eval.run_eval import build_model
from wm3d_v3.policy.world_model_policy import (
    ScoreWeights,
    denormalize_action_cond,
    select_action_chunk,
)
from wm3d_v3.training.train_libero_success_p0 import (
    LiberoExpertCacheDataset,
    _load_action_stats,
    _score,
)


_BATCH_KEYS = (
    "s_in",
    "c",
    "context_rgb",
    "action_tgt",
    "action_tgt_norm",
)
_OPTIONAL_BATCH_KEYS = ("lowdim_state", "action_history")


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise RuntimeError(f"empty manifest: {path}")
    return rows


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--overlay_ckpt", type=Path, default=None)
    ap.add_argument("--allow_missing_prefix", action="append", default=[])
    ap.add_argument("--include_policy_anchor", action="store_true")
    ap.add_argument(
        "--selection_mode",
        choices=("proposer", "ranked_residual"),
        default="proposer",
    )
    ap.add_argument("--score_progress_weight", type=float, default=1.0)
    ap.add_argument("--score_terminal_weight", type=float, default=1.0)
    ap.add_argument("--score_plausibility_weight", type=float, default=0.0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shard_index", type=int, required=True)
    ap.add_argument("--num_shards", type=int, required=True)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")
    if args.selection_mode == "ranked_residual" and args.include_policy_anchor:
        raise ValueError(
            "ranked_residual already includes the serving policy anchor"
        )

    cfg = yaml.safe_load(args.cfg.read_text())
    base_cfg = yaml.safe_load(Path(cfg["base_cfg"]).read_text())
    rows = _load_rows(args.manifest)
    indices = list(range(args.shard_index, len(rows), args.num_shards))
    if not indices:
        raise RuntimeError("shard selected no rows")

    device = torch.device(args.device)
    model = build_model(base_cfg).to(device).eval()
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False, mmap=True)
    if args.allow_missing_prefix:
        incompatible = model.load_state_dict(state["model"], strict=False)
        bad_missing = [
            key
            for key in incompatible.missing_keys
            if not any(key.startswith(prefix) for prefix in args.allow_missing_prefix)
        ]
        if bad_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "checkpoint incompatibility outside allowed missing prefixes: "
                f"missing={bad_missing} unexpected={incompatible.unexpected_keys}"
            )
    else:
        model.load_state_dict(state["model"], strict=True)
    del state
    if args.overlay_ckpt is not None:
        overlay = torch.load(
            args.overlay_ckpt, map_location="cpu", weights_only=False, mmap=True
        )
        overlay_state = overlay["model"]
        expected = {
            name for name in model.state_dict() if name.startswith("action_proposer.")
        }
        actual = set(overlay_state)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise RuntimeError(
                f"action proposer overlay mismatch: missing={missing} extra={extra}"
            )
        model.load_state_dict(overlay_state, strict=False)
        del overlay, overlay_state
    _load_action_stats(model, Path(cfg["data"]["action_stats"]), device)

    dataset = LiberoExpertCacheDataset(args.manifest)
    candidate_cond_parts: list[np.ndarray] = []
    candidate_raw_parts: list[np.ndarray] = []
    candidate_score_parts: list[np.ndarray] = []
    expert_action_parts: list[np.ndarray] = []
    selected_rows: list[dict[str, Any]] = []

    for offset in range(0, len(indices), args.batch_size):
        batch_indices = indices[offset : offset + args.batch_size]
        samples = [dataset[index] for index in batch_indices]
        batch_keys = list(_BATCH_KEYS)
        batch_keys.extend(
            key for key in _OPTIONAL_BATCH_KEYS if all(key in sample for sample in samples)
        )
        batch = default_collate(
            [{key: sample[key] for key in batch_keys} for sample in samples]
        )
        s = batch["s_in"].to(device, non_blocking=True)
        c = batch["c"].to(device, non_blocking=True)
        context_rgb = batch["context_rgb"].to(device, non_blocking=True)
        action_tgt = batch["action_tgt"].to(device, non_blocking=True)
        action_tgt_norm = batch["action_tgt_norm"].to(device, non_blocking=True)
        factual = make_action_condition(action_tgt, action_tgt_norm)
        model_kwargs = {
            key: batch[key].to(device, non_blocking=True)
            for key in _OPTIONAL_BATCH_KEYS
            if key in batch
        }
        if "action_history" in model_kwargs:
            history_len = int(base_cfg["model"].get("policy_action_history_len", 0))
            if history_len > 0:
                model_kwargs["action_history"] = model_kwargs["action_history"][:, -history_len:]

        if args.selection_mode == "ranked_residual":
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                decision = select_action_chunk(
                    model,
                    s,
                    c,
                    context_rgb=context_rgb,
                    pixel=False,
                    score_weights=ScoreWeights(
                        progress=args.score_progress_weight,
                        terminal=args.score_terminal_weight,
                        plausibility=args.score_plausibility_weight,
                    ),
                    selection_mode="ranked_residual",
                    **model_kwargs,
                )
            candidates = decision["candidate_action_cond"].float()
            score_t = decision["candidate_scores"].float()
        else:
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                proposal_out = model(
                    s,
                    c,
                    action_cond=factual,
                    context_rgb=context_rgb,
                    pixel=False,
                    bridging=False,
                    **model_kwargs,
                )
            candidates = proposal_out["proposer_action_cond"].float()
            if args.include_policy_anchor:
                anchor = proposal_out["policy_action_cond"].float()[:, None]
                candidates = torch.cat([anchor, candidates], dim=1)
            scores = []
            for candidate_index in range(candidates.shape[1]):
                with torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
                ):
                    candidate_out = model(
                        s,
                        c,
                        action_cond=candidates[:, candidate_index].to(dtype=s.dtype),
                        context_rgb=context_rgb,
                        pixel=False,
                        bridging=False,
                        **model_kwargs,
                    )
                scores.append(_score(candidate_out))
            score_t = torch.stack(scores, dim=1)
        raw = denormalize_action_cond(candidates, model=model)

        candidate_cond_parts.append(candidates.cpu().numpy().astype(np.float32))
        candidate_raw_parts.append(raw.cpu().numpy().astype(np.float32))
        candidate_score_parts.append(score_t.cpu().numpy().astype(np.float32))
        expert_action_parts.append(action_tgt.cpu().numpy().astype(np.float32))
        selected_rows.extend(rows[index] for index in batch_indices)
        print(
            json.dumps(
                {
                    "shard": args.shard_index,
                    "processed": min(offset + len(batch_indices), len(indices)),
                    "total": len(indices),
                }
            ),
            flush=True,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        candidate_cond=np.concatenate(candidate_cond_parts, axis=0),
        candidate_raw=np.concatenate(candidate_raw_parts, axis=0),
        candidate_score=np.concatenate(candidate_score_parts, axis=0),
        expert_action=np.concatenate(expert_action_parts, axis=0),
        rows_json=np.asarray(
            [json.dumps(row, sort_keys=True) for row in selected_rows], dtype=np.str_
        ),
        shard_index=np.asarray(args.shard_index, dtype=np.int64),
        num_shards=np.asarray(args.num_shards, dtype=np.int64),
        selection_mode=np.asarray(args.selection_mode),
        score_weights=np.asarray(
            [
                args.score_progress_weight,
                args.score_terminal_weight,
                args.score_plausibility_weight,
            ],
            dtype=np.float32,
        ),
    )
    print(json.dumps({"out": str(args.out), "rows": len(selected_rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
