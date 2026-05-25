"""One-time token caching for Phase 1.

Rationale: Training the predictive head is cheap compared to running VGGT on
every epoch. We run VGGT once per clip, save pooled tokens + decoded depth +
camera extrinsic + action/state sequences aligned to the same frames, then
the training loop reads npz shards directly.

Shard layout (one .npz per clip):
    tokens      (n_frames, G*G, D)      bf16 → stored as int16 view
    depth       (n_frames, Hs, Hs)       fp16
    extrinsic   (n_frames, 3, 4)         fp32
    intrinsic   (n_frames, 3, 3)         fp32
    actions     (n_frames, A)            fp32
    states      (n_frames, S)            fp32
    frame_ids   (n_frames,)              int32  absolute frame idx in clip
    meta.json-ish side-car with clip_id / episode_index / task / window plan.

We cache at *frame* granularity (not window-level). The dataset module then
composes (s_t, a_t, s_{t+1}) pairs freely. Frames that fall in multiple
sliding windows are written from the window whose center is closest.

Design choices
--------------
- **Last aggregator layer only.** Exp 3 on Phase 0 used this; keeps memory low.
- **Spatial pool to G=8 (64 patch tokens).** 1374 VGGT tokens include a camera
  token, a register token, and a 37 x 37 patch grid. We split off the 4 leading
  "special" tokens, adaptive-avg-pool the remaining grid to G x G, and discard
  the specials (they do not carry per-frame geometry).
- **bf16 storage via int16 view.** numpy has no bf16; we bitcast tokens to int16
  for storage and reverse on load. No precision loss.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml

log = logging.getLogger("phase1.cache")


# ----------------------------------------------------------------------- utils
def bf16_to_int16(x: torch.Tensor) -> np.ndarray:
    if x.dtype != torch.bfloat16:
        x = x.to(torch.bfloat16)
    return x.contiguous().view(torch.int16).cpu().numpy()


def int16_to_bf16(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(a).view(torch.bfloat16)


def pool_tokens(tokens: torch.Tensor, grid: int, n_specials: int = 4) -> torch.Tensor:
    """Pool VGGT aggregator tokens to a (G, G) patch grid per frame.

    tokens: [B, T, N, D]  where N = n_specials + P*P.
    Returns [B, T, grid*grid, D].
    """
    B, T, N, D = tokens.shape
    patches = tokens[:, :, n_specials:, :]  # drop leading special tokens
    p2 = patches.shape[2]
    side = int(math.isqrt(p2))
    if side * side != p2:
        # Fall back to 1-D adaptive pool over the token axis.
        x = patches.permute(0, 1, 3, 2).reshape(B * T, D, p2)
        x = F.adaptive_avg_pool1d(x.float(), grid * grid).to(tokens.dtype)
        return x.reshape(B, T, D, grid * grid).permute(0, 1, 3, 2).contiguous()
    # 2-D pool over the patch grid.
    x = patches.reshape(B * T, side, side, D).permute(0, 3, 1, 2)  # [BT, D, S, S]
    x = F.adaptive_avg_pool2d(x.float(), (grid, grid)).to(tokens.dtype)
    x = x.reshape(B, T, D, grid * grid).permute(0, 1, 3, 2).contiguous()
    return x


# -------------------------------------------------------- DROID action lookup
@dataclass
class DroidActions:
    """Lazy loader for per-episode action/state arrays from the LeRobot parquet."""

    parquet_path: str
    _cache: dict[int, dict[str, np.ndarray]] | None = None

    def _ensure(self) -> None:
        if self._cache is not None:
            return
        import pyarrow.parquet as pq  # local import

        log.info("reading DROID parquet: %s", self.parquet_path)
        t = pq.read_table(self.parquet_path)
        df = t.to_pandas()
        by_ep: dict[int, dict[str, np.ndarray]] = {}
        for ep, sub in df.groupby("episode_index"):
            sub = sub.sort_values("frame_index")
            a = np.stack(sub["action"].to_numpy()).astype(np.float32)
            s = np.stack(sub["observation.state"].to_numpy()).astype(np.float32)
            by_ep[int(ep)] = {"action": a, "state": s}
        self._cache = by_ep
        log.info("  loaded %d episodes", len(by_ep))

    def __call__(self, episode_index: int, frame_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self._ensure()
        assert self._cache is not None
        ep = self._cache[int(episode_index)]
        # clip indices in case extracted frames overshoot parquet length
        fi = np.clip(frame_ids, 0, ep["action"].shape[0] - 1)
        return ep["action"][fi].copy(), ep["state"][fi].copy()


# ------------------------------------------------------------- main processor
class VGGTCachier:
    def __init__(self, cfg: dict[str, Any], device: str = "cuda"):
        self.cfg = cfg
        self.device = device
        from src.vggt_wrapper import VGGTConfig, VGGTWrapper

        self.vggt = VGGTWrapper(VGGTConfig(device=device))
        self.grid = cfg["cache"]["token_grid"]
        self.save_hw = cfg["cache"]["save_depth_hw"]
        self.window = cfg["cache"]["window"]
        self.stride = cfg["cache"]["stride"]
        self.max_frames = cfg["cache"]["max_frames_per_clip"]

    @torch.no_grad()
    def process_clip(
        self,
        clip: dict[str, Any],
        actions: DroidActions,
        out_dir: Path,
    ) -> dict[str, Any]:
        from src.data_loader import Clip, load_frames

        frame_paths = clip["frames"][: self.max_frames]
        n = len(frame_paths)
        if n < self.window:
            log.warning("skip %s (only %d frames)", clip["clip_id"], n)
            return {"skipped": True}

        # Plan windows: stride S, window W, covering [0, n).
        windows: list[tuple[int, int]] = []
        start = 0
        while start + self.window <= n:
            windows.append((start, start + self.window))
            start += self.stride
        if windows[-1][1] < n:
            windows.append((n - self.window, n))

        # For each frame we keep the window-local output from the "closest-center"
        # window. That gives a single token/depth/extrinsic per frame.
        closest_window = np.full(n, -1, dtype=np.int32)
        centers = np.array([(a + b - 1) / 2 for (a, b) in windows])
        for fi in range(n):
            closest_window[fi] = int(np.argmin(np.abs(centers - fi)))

        # Outputs at per-frame granularity.
        D = self.cfg["cache"]["token_dim"]
        G = self.grid
        tokens_out = np.zeros((n, G * G, D), dtype=np.int16)  # bf16 via view
        depth_out = np.zeros((n, self.save_hw, self.save_hw), dtype=np.float16)
        extri_out = np.zeros((n, 3, 4), dtype=np.float32)
        intri_out = np.zeros((n, 3, 3), dtype=np.float32)
        conf_out = np.zeros((n,), dtype=np.float32)
        filled = np.zeros(n, dtype=bool)

        clip_obj = Clip(
            clip_id=clip["clip_id"],
            set_name=clip.get("set", "A"),
            frames=frame_paths,
            fps=float(clip.get("fps", 15)),
            meta=clip.get("meta", {}),
        )

        t0 = time.time()
        for wi, (a, b) in enumerate(windows):
            idx = list(range(a, b))
            imgs = load_frames(clip_obj, indices=idx)  # [W, 3, H, W_]
            enc = self.vggt.encode(imgs)
            last = enc["aggregated_tokens"][-1]  # [1, W, N, D]
            pooled = pool_tokens(last, self.grid).squeeze(0)  # [W, G*G, D]
            geo = self.vggt.decode_geometry(enc["aggregated_tokens"], enc["ps_idx"], enc["images"])
            depth = geo["depth"].squeeze(0).squeeze(-1).float()  # [W, H, W_]
            conf = geo["depth_conf"].squeeze(0).float()  # [W, H, W_]
            extri = geo["camera_extrinsic"].squeeze(0).float().cpu().numpy()
            intri = geo["camera_intrinsic"].squeeze(0).float().cpu().numpy()
            depth_small = F.adaptive_avg_pool2d(
                depth.unsqueeze(0), (self.save_hw, self.save_hw)
            ).squeeze(0).cpu().numpy().astype(np.float16)
            conf_mean = conf.mean(dim=(-1, -2)).cpu().numpy()

            for k, fi in enumerate(range(a, b)):
                if closest_window[fi] != wi:
                    continue
                tokens_out[fi] = bf16_to_int16(pooled[k])
                depth_out[fi] = depth_small[k]
                extri_out[fi] = extri[k]
                intri_out[fi] = intri[k]
                conf_out[fi] = float(conf_mean[k])
                filled[fi] = True

        missing = int((~filled).sum())
        if missing:
            # Shouldn't happen by construction, but guard against off-by-one.
            log.warning("%s: %d frames not filled", clip["clip_id"], missing)

        # Actions / states for these frames.
        episode_index = int(clip["meta"]["episode_index"])
        frame_ids = np.arange(n, dtype=np.int32)
        act, state = actions(episode_index, frame_ids)

        out_path = out_dir / f"{clip['clip_id']}.npz"
        meta = {
            "clip_id": clip["clip_id"],
            "episode_index": episode_index,
            "n_frames": n,
            "n_windows": len(windows),
            "window": self.window,
            "stride": self.stride,
            "token_grid": self.grid,
            "token_dim": D,
            "save_depth_hw": self.save_hw,
            "task": clip["meta"].get("task", ""),
        }
        np.savez_compressed(
            out_path,
            tokens=tokens_out,
            depth=depth_out,
            extrinsic=extri_out,
            intrinsic=intri_out,
            conf=conf_out,
            actions=act,
            states=state,
            frame_ids=frame_ids,
        )
        (out_path.with_suffix(".json")).write_text(json.dumps(meta, indent=2))
        dt = time.time() - t0
        log.info(
            "  cached %s: n=%d windows=%d elapsed=%.1fs  → %s",
            clip["clip_id"], n, len(windows), dt, out_path.name,
        )
        return {"skipped": False, "elapsed": dt, "n_frames": n, "path": str(out_path)}


def run(cfg_path: str, limit: int | None = None, force: bool = False) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg = yaml.safe_load(open(cfg_path))
    manifest = json.load(open(cfg["cache"]["manifest"]))
    out_dir = Path(cfg["cache"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    actions = DroidActions(cfg["cache"]["droid_parquet"])

    cacher = VGGTCachier(cfg)
    clips = manifest if limit is None else manifest[:limit]
    summary = {"n": len(clips), "clips": []}
    for i, clip in enumerate(clips):
        out_path = out_dir / f"{clip['clip_id']}.npz"
        if out_path.exists() and not force:
            log.info("[%d/%d] %s  already cached, skipping", i + 1, len(clips), clip["clip_id"])
            summary["clips"].append({"clip_id": clip["clip_id"], "cached": True, "skipped": True})
            continue
        log.info("[%d/%d] %s", i + 1, len(clips), clip["clip_id"])
        info = cacher.process_clip(clip, actions, out_dir)
        summary["clips"].append({"clip_id": clip["clip_id"], **info})
    (out_dir / "_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="configs/phase1/default.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run(args.cfg, args.limit, args.force)
