"""Extract frozen world-model consequence features for fixed candidates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import default_collate

from wm3d_v3.eval.run_eval import build_model
from wm3d_v3.training.train_libero_success_p0 import _progress_score, _score


def _load_context(row: dict) -> dict[str, torch.Tensor]:
    with np.load(row["cache_path"]) as data:
        return {
            "s": torch.from_numpy(data["s_in"].astype(np.float32)),
            "c": torch.from_numpy(data["c"].astype(np.float32)),
        }


@torch.no_grad()
def _extract(model, input_path: Path, output_path: Path, device, batch_size: int, score_mode: str) -> None:
    score_fn = _progress_score if score_mode == "progress" else _score
    payload = np.load(input_path)
    rows = [json.loads(str(item)) for item in payload["rows_json"]]
    candidates = np.asarray(payload["candidate_cond"], dtype=np.float32)
    if candidates.ndim != 4 or candidates.shape[-1] != 7:
        raise ValueError(f"candidate_cond must be [N,K,T,7], got {candidates.shape}")
    if len(rows) != len(candidates):
        raise ValueError("row and candidate counts differ")
    token_parts, score_parts, task_parts = [], [], []
    for start in range(0, len(rows), batch_size):
        stop = min(start + batch_size, len(rows))
        batch = default_collate([_load_context(row) for row in rows[start:stop]])
        s = batch["s"].to(device, non_blocking=True)
        c = batch["c"].to(device, non_blocking=True)
        candidate_batch = torch.from_numpy(candidates[start:stop]).to(device)
        token_candidates, score_candidates = [], []
        for index in range(candidate_batch.shape[1]):
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                out = model(
                    s,
                    c,
                    action_cond=candidate_batch[:, index].to(dtype=s.dtype),
                    pixel=False,
                    bridging=False,
                    skip_action_proposer=True,
                    skip_action_policy=True,
                )
            token_candidates.append(out["pred_tokens"].float().mean(dim=2))
            score_candidates.append(score_fn(out).float())
        token_parts.append(torch.stack(token_candidates, 1).cpu().numpy().astype(np.float16))
        score_parts.append(torch.stack(score_candidates, 1).cpu().numpy().astype(np.float32))
        task_parts.append(c.cpu().numpy().astype(np.float16))
        print(json.dumps({"input": str(input_path), "processed": stop, "total": len(rows)}), flush=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        world_token_mean=np.concatenate(token_parts),
        current_score=np.concatenate(score_parts),
        candidate_cond=candidates,
        task_emb=np.concatenate(task_parts),
        rows_json=payload["rows_json"],
        score_mode=np.asarray(score_mode),
    )
    print(json.dumps({"out": str(output_path), "rows": len(rows)}, sort_keys=True))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--input", type=Path, action="append", required=True)
    ap.add_argument("--out", type=Path, action="append", required=True)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--overlay_ckpt", type=Path)
    ap.add_argument("--score_mode", choices=("outcome", "progress"), default="outcome")
    args = ap.parse_args()
    if len(args.input) != len(args.out):
        raise ValueError("--input and --out counts must match")
    cfg = yaml.safe_load(args.cfg.read_text())
    base_cfg = yaml.safe_load(Path(cfg["base_cfg"]).read_text())
    device = torch.device(args.device)
    model = build_model(base_cfg).to(device).eval()
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False, mmap=True)
    model.load_state_dict(state["model"], strict=True)
    del state
    if args.overlay_ckpt is not None:
        overlay = torch.load(args.overlay_ckpt, map_location="cpu", weights_only=False, mmap=True)
        overlay_state = overlay["model"]
        expected = {
            name for name in model.state_dict() if name.startswith("progress_head.")
        }
        actual = set(overlay_state)
        if actual != expected:
            raise RuntimeError(
                "overlay checkpoint must contain the complete progress_head state: "
                f"missing={sorted(expected - actual)} extra={sorted(actual - expected)}"
            )
        incompatible = model.load_state_dict(overlay_state, strict=False)
        if incompatible.unexpected_keys:
            raise RuntimeError(f"unexpected overlay keys: {incompatible.unexpected_keys}")
        del overlay, overlay_state
    for input_path, output_path in zip(args.input, args.out, strict=True):
        _extract(model, input_path, output_path, device, args.batch_size, args.score_mode)


if __name__ == "__main__":
    main()
