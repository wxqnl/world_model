# VLA Stage D Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Push wm3d_v3 VLA past the three strict gates (bridge pose -30%, bridge grip switch F1 ≥ 0.6, fractal pose -15%) while keeping the native-3D thesis intact — only VGGT-derived signals as input.

**Architecture:** Add VGGT point-map cache → scene-flow encoder fed into IDM (carries angular motion); teacher-force the IDM with GT future during early training, anneal to predicted future; freeze all dense heads in the final 5 epochs and concentrate gradient on the action head; fix gripper supervision (switch-rate pos_weight + auxiliary L_switch).

**Tech Stack:** PyTorch 2.x, bf16 autocast, DDP (2× H100 on cards 2/3 or 0/1), VGGT backbone (cached), OXE bridge + fractal manifest.

---

## Task D-0: Compute gripper switch statistics

**Files:**
- Create: `wm3d_v3/scripts/compute_switch_stats.py`
- Test: ad-hoc (script run)

**Why:** Stage C used `pos_weight ≈ 0.85` against currently-closed rate, but the gate measures switches. Need the empirical switch rate over the train manifest to set the right pos_weight.

- [ ] **Step 1: Create the script**

Write `wm3d_v3/scripts/compute_switch_stats.py`:
```python
"""Compute gripper switch event rate over OXE train manifest."""
from pathlib import Path
import argparse, json
import numpy as np
from wm3d_v3.data.manifest import read_manifest

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--cache_root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    records = read_manifest(args.manifest)
    n_total = 0
    n_switch = 0
    n_closed = 0
    for r in records:
        cache_file = args.cache_root / "actions" / f"{r['safe_id']}.npz"
        if not cache_file.exists():
            continue
        a = np.load(cache_file)["action"]    # [n, 7]
        g = (a[:, 6] > 0.5).astype(np.int8)
        n_total += len(g) - 1
        n_switch += int(np.abs(np.diff(g)).sum())
        n_closed += int(g[1:].sum())
    out = {
        "n_total_transitions": int(n_total),
        "n_switches": int(n_switch),
        "switch_rate": float(n_switch / max(1, n_total)),
        "closed_rate": float(n_closed / max(1, n_total)),
        "pos_weight_switch": float((1 - n_switch / max(1, n_total)) / max(1e-9, n_switch / max(1, n_total))),
    }
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run and inspect output**

```bash
cd /home/user01/Minko/newwm/wm3d_v3 && python scripts/compute_switch_stats.py \
  --manifest manifests/oxe_train.jsonl \
  --cache_root /home/user01/Minko/datasets/cache/wm3d_v3 \
  --out /home/user01/Minko/datasets/cache/wm3d_v3/switch_stats.json
```

Expected: `switch_rate` in range 0.03–0.07, `pos_weight_switch` in 13–30.

- [ ] **Step 3: Commit**

```bash
git add wm3d_v3/scripts/compute_switch_stats.py
git commit -m "feat(vla,D): one-shot script to compute gripper switch frequency"
```

---

## Task D-1: VGGT point-map cache script (with subset validation)

**Files:**
- Create: `wm3d_v3/scripts/cache_vggt_pointmap.py`
- Output: `cache/wm3d_v3/vggt_pointmap/<safe_id>.npz`

**Why:** Scene flow needs per-frame 3D point maps. VGGT produces them but we did not previously cache them.

- [ ] **Step 1: Write the cache script**

Pattern after the existing `cache_oxe.py`. Inputs: list of safe_ids from manifest. For each episode load frames, run VGGT, extract `point_map [n,H,W,3]` and `intrinsic [n,3,3]`, save as fp16 npz.

```python
"""Re-cache VGGT point_map for all manifest episodes."""
# (full implementation mirrors scripts/cache_oxe.py structure)
# Key changes:
# - load existing pooled-token cache to skip already-encoded frames is NOT possible
#   (VGGT must run full to produce pmap); run full inference.
# - save dict: {"point": fp16[n,224,224,3], "intrinsic": fp32[n,3,3]}
# - skip if output file already exists (for resume).
```

- [ ] **Step 2: Smoke test on 10 fractal episodes**

```bash
cd /home/user01/Minko/newwm/wm3d_v3 && CUDA_VISIBLE_DEVICES=2 python scripts/cache_vggt_pointmap.py \
  --manifest manifests/oxe_train.jsonl \
  --cache_root /home/user01/Minko/datasets/cache/wm3d_v3 \
  --limit 10 \
  --dataset fractal20220817_data
```

Expected: ~10 npz files in `cache/wm3d_v3/vggt_pointmap/`. Total size ~50 MB.

- [ ] **Step 3: Validate scene flow non-degenerate**

```python
# in a quick repl block or a tiny script
import numpy as np
from pathlib import Path
files = sorted(Path("/home/user01/Minko/datasets/cache/wm3d_v3/vggt_pointmap").glob("*.npz"))[:3]
for f in files:
    d = np.load(f)
    pmap = d["point"]                       # [n, 224, 224, 3]
    flow = pmap[1:] - pmap[:-1]              # [n-1, 224, 224, 3]
    print(f.name, "flow norm pct50/95:", np.percentile(np.linalg.norm(flow, axis=-1), [50, 95]))
```

Expected: 50th pct flow norm > 0.001, 95th pct > 0.01 (signal exists). All-zero or all-huge means broken.

- [ ] **Step 4: Commit script**

```bash
git add wm3d_v3/scripts/cache_vggt_pointmap.py
git commit -m "feat(vla,D): VGGT point-map cache script + smoke validation"
```

---

## Task D-2: Dataset emits `pmap_in / pmap_tgt / intrinsic`

**Files:**
- Modify: `wm3d_v3/wm3d_v3/data/window_dataset.py:1-92`
- Test: `wm3d_v3/tests/test_stage_d.py` (new file)

- [ ] **Step 1: Write the failing test**

Create `wm3d_v3/tests/test_stage_d.py`:
```python
"""Stage D unit tests."""
import numpy as np
import torch
from pathlib import Path

def test_dataset_emits_pmap(tmp_path):
    from wm3d_v3.data.window_dataset import OXEWindowDataset, WindowConfig
    # construct a tiny synthetic cache
    sid = "test_ep"
    cache = tmp_path
    (cache / "vggt_pool").mkdir()
    (cache / "vggt_geom").mkdir()
    (cache / "vggt_pointmap").mkdir()
    (cache / "actions").mkdir()
    (cache / "task_emb").mkdir()
    n = 24
    np.savez(cache/"vggt_pool"/f"{sid}.npz", pooled=np.zeros((n,64,2048),np.float16))
    np.savez(cache/"vggt_geom"/f"{sid}.npz", depth=np.zeros((n,224,224),np.float16))
    np.savez(cache/"vggt_pointmap"/f"{sid}.npz",
             point=np.random.randn(n,224,224,3).astype(np.float16),
             intrinsic=np.tile(np.eye(3).astype(np.float32), (n,1,1)))
    np.savez(cache/"actions"/f"{sid}.npz", action=np.zeros((n,7),np.float32))
    np.savez(cache/"task_emb"/f"{sid}.npz", emb=np.zeros((2048,),np.float32))
    rec = {"safe_id": sid, "dataset": "test", "n_frames": n}
    cfg = WindowConfig(T=16, k=8, stride=1, cache_root=cache,
                       action_stats=None, with_pointmap=True)
    ds = OXEWindowDataset([rec], cfg)
    item = ds[0]
    assert "pmap_in" in item and item["pmap_in"].shape == (16, 224, 224, 3)
    assert "pmap_tgt" in item and item["pmap_tgt"].shape == (8, 224, 224, 3)
    assert "intrinsic" in item and item["intrinsic"].shape == (24, 3, 3)
```

- [ ] **Step 2: Run test, verify failure**

```bash
cd /home/user01/Minko/newwm/wm3d_v3 && pytest tests/test_stage_d.py::test_dataset_emits_pmap -x
```
Expected: FAIL — `with_pointmap` not a valid WindowConfig field, or `pmap_in` missing.

- [ ] **Step 3: Modify WindowConfig and OXEWindowDataset**

In `window_dataset.py`:
- Add `with_pointmap: bool = False` to `WindowConfig`.
- In `__getitem__`, when `with_pointmap`: load `vggt_pointmap/<safe_id>.npz`, slice, emit `pmap_in [T,H,W,3]`, `pmap_tgt [k,H,W,3]`, `intrinsic [T+k,3,3]`. Cast point to float32 on load (fp16 → fp32 to control diff noise).

- [ ] **Step 4: Run test, verify pass**

```bash
pytest tests/test_stage_d.py::test_dataset_emits_pmap -x
```

- [ ] **Step 5: Commit**

```bash
git add wm3d_v3/wm3d_v3/data/window_dataset.py wm3d_v3/tests/test_stage_d.py
git commit -m "feat(vla,D): emit pmap_in / pmap_tgt / intrinsic from OXEWindowDataset"
```

---

## Task D-3: Backproject helper

**Files:**
- Create: `wm3d_v3/wm3d_v3/models/backproject.py`
- Test: `wm3d_v3/tests/test_stage_d.py::test_backproject`

- [ ] **Step 1: Write the failing test**

```python
def test_backproject():
    from wm3d_v3.models.backproject import depth_to_pointmap
    B, T, H, W = 2, 4, 16, 16
    depth = torch.ones(B, T, H, W) * 2.0
    K = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(B, T, 3, 3).clone()
    K[..., 0, 0] = K[..., 1, 1] = 8.0   # focal
    K[..., 0, 2] = W / 2
    K[..., 1, 2] = H / 2
    pmap = depth_to_pointmap(depth, K)
    assert pmap.shape == (B, T, H, W, 3)
    assert torch.allclose(pmap[..., 2], torch.full_like(pmap[..., 2], 2.0))
    # principal point pixel should have x≈0, y≈0
    cx, cy = W // 2, H // 2
    assert pmap[0, 0, cy, cx, 0].abs() < 0.5
    assert pmap[0, 0, cy, cx, 1].abs() < 0.5
```

- [ ] **Step 2: Verify failure**

```bash
pytest tests/test_stage_d.py::test_backproject -x
```
Expected: FAIL (module missing).

- [ ] **Step 3: Implement**

```python
"""Camera-frame backprojection: depth + intrinsic -> 3D point map."""
import torch

def depth_to_pointmap(depth: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """
    depth : [B, T, H, W]
    K     : [B, T, 3, 3]   per-frame intrinsic
    -> point map [B, T, H, W, 3] in camera frame.
    """
    B, T, H, W = depth.shape
    dev, dt = depth.device, depth.dtype
    v, u = torch.meshgrid(
        torch.arange(H, device=dev, dtype=dt),
        torch.arange(W, device=dev, dtype=dt),
        indexing="ij",
    )
    uv1 = torch.stack([u, v, torch.ones_like(u)], dim=0)        # [3, H, W]
    K_inv = torch.linalg.inv(K.float()).to(dt)                  # [B, T, 3, 3]
    # ray direction in camera frame
    rays = torch.einsum("btij,jhw->bthwi", K_inv, uv1)          # [B, T, H, W, 3]
    return rays * depth.unsqueeze(-1)
```

- [ ] **Step 4: Verify pass**

```bash
pytest tests/test_stage_d.py::test_backproject -x
```

- [ ] **Step 5: Commit**

```bash
git add wm3d_v3/wm3d_v3/models/backproject.py wm3d_v3/tests/test_stage_d.py
git commit -m "feat(vla,D): depth_to_pointmap backprojection helper"
```

---

## Task D-4: SceneFlowEncoder

**Files:**
- Create: `wm3d_v3/wm3d_v3/models/scene_flow_encoder.py`
- Test: `wm3d_v3/tests/test_stage_d.py::test_scene_flow_encoder`

- [ ] **Step 1: Write the failing test**

```python
def test_scene_flow_encoder():
    from wm3d_v3.models.scene_flow_encoder import SceneFlowEncoder, SceneFlowEncoderConfig
    cfg = SceneFlowEncoderConfig(in_size=224, out_grid=8, hidden_d=256, base_ch=16)
    enc = SceneFlowEncoder(cfg)
    B, T = 2, 16
    pmap = torch.randn(B, T, 224, 224, 3)
    tok = enc(pmap)
    # flow is computed inside; output token at every t (last frame uses zero-flow)
    assert tok.shape == (B, T, 64, 256)
```

- [ ] **Step 2: Verify failure**

```bash
pytest tests/test_stage_d.py::test_scene_flow_encoder -x
```

- [ ] **Step 3: Implement**

```python
"""SceneFlowEncoder: 3D point-map sequence -> patch tokens."""
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SceneFlowEncoderConfig:
    in_size: int = 224
    out_grid: int = 8
    hidden_d: int = 256
    base_ch: int = 16
    smooth: bool = True     # apply 3x3 box filter to flow before encoding


class SceneFlowEncoder(nn.Module):
    def __init__(self, cfg: SceneFlowEncoderConfig):
        super().__init__()
        self.cfg = cfg
        c0 = cfg.base_ch
        # 4-stage stride-2 CNN: 224 -> 112 -> 56 -> 28 -> 14 -> pool to 8
        self.stem = nn.Conv2d(3, c0, kernel_size=3, stride=2, padding=1)
        self.b1 = nn.Sequential(nn.GELU(), nn.Conv2d(c0,   c0*2, 3, 2, 1))
        self.b2 = nn.Sequential(nn.GELU(), nn.Conv2d(c0*2, c0*4, 3, 2, 1))
        self.b3 = nn.Sequential(nn.GELU(), nn.Conv2d(c0*4, c0*8, 3, 2, 1))
        self.pool = nn.AdaptiveAvgPool2d(cfg.out_grid)
        self.proj = nn.Linear(c0 * 8, cfg.hidden_d)

    def forward(self, pmap: torch.Tensor) -> torch.Tensor:
        """pmap : [B, T, H, W, 3] -> tokens [B, T, out_grid^2, hidden_d]"""
        B, T, H, W, _ = pmap.shape
        # flow[t] = pmap[t+1] - pmap[t]; flow[T-1] = 0
        flow = torch.zeros_like(pmap)
        flow[:, :-1] = pmap[:, 1:] - pmap[:, :-1]
        if self.cfg.smooth:
            f = flow.permute(0, 1, 4, 2, 3).reshape(B*T, 3, H, W)
            f = F.avg_pool2d(f, kernel_size=3, stride=1, padding=1)
        else:
            f = flow.permute(0, 1, 4, 2, 3).reshape(B*T, 3, H, W)
        x = self.stem(f)
        x = self.b1(x); x = self.b2(x); x = self.b3(x)
        x = self.pool(x)                                     # [BT, c, g, g]
        x = x.flatten(-2).transpose(-1, -2)                  # [BT, g*g, c]
        x = self.proj(x)                                     # [BT, g*g, d]
        return x.view(B, T, self.cfg.out_grid ** 2, self.cfg.hidden_d)
```

- [ ] **Step 4: Verify pass**

```bash
pytest tests/test_stage_d.py::test_scene_flow_encoder -x
```

- [ ] **Step 5: Commit**

```bash
git add wm3d_v3/wm3d_v3/models/scene_flow_encoder.py wm3d_v3/tests/test_stage_d.py
git commit -m "feat(vla,D): SceneFlowEncoder (point-map diff -> patch tokens)"
```

---

## Task D-5: Switch loss

**Files:**
- Modify: `wm3d_v3/wm3d_v3/losses.py`
- Test: `wm3d_v3/tests/test_stage_d.py::test_switch_loss`

- [ ] **Step 1: Write the failing test**

```python
def test_switch_loss():
    from wm3d_v3.losses import compute_switch_loss
    B, k = 4, 8
    logits = torch.zeros(B, k)         # all "open" predicted
    grip_tgt = torch.zeros(B, k)
    grip_tgt[:, 3] = 1.0; grip_tgt[:, 4:] = 1.0    # one switch at t=2->3
    l = compute_switch_loss(logits, grip_tgt)
    assert l.item() > 0  # missed switch -> nonzero loss
    # all correct: prediction transitions at the same place with margin
    logits = torch.zeros(B, k); logits[:, 3:] = 5.0
    l_good = compute_switch_loss(logits, grip_tgt)
    assert l_good.item() < l.item()
```

- [ ] **Step 2: Verify failure**

```bash
pytest tests/test_stage_d.py::test_switch_loss -x
```

- [ ] **Step 3: Implement**

Append to `wm3d_v3/wm3d_v3/losses.py`:
```python
def compute_switch_loss(grip_logit: torch.Tensor, grip_tgt: torch.Tensor,
                        alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    """Focal BCE on first-differenced logits vs GT switch events."""
    diff = (grip_logit[:, 1:] - grip_logit[:, :-1]).abs()    # [B, k-1], "switch likelihood"
    gt = (grip_tgt[:, 1:] != grip_tgt[:, :-1]).float()
    return focal_bce(diff.float(), gt, alpha=alpha, gamma=gamma)
```

- [ ] **Step 4: Verify pass**

```bash
pytest tests/test_stage_d.py::test_switch_loss -x
```

- [ ] **Step 5: Commit**

```bash
git add wm3d_v3/wm3d_v3/losses.py wm3d_v3/tests/test_stage_d.py
git commit -m "feat(vla,D): compute_switch_loss (auxiliary switch detector)"
```

---

## Task D-6: JointWorldModelD (teacher-forcing + scene flow)

**Files:**
- Create: `wm3d_v3/wm3d_v3/models/joint_model_d.py`
- Modify: `wm3d_v3/wm3d_v3/models/idm_stream.py` (extend extra_token_dim to accept 512)
- Test: `wm3d_v3/tests/test_stage_d.py::test_joint_d_forward`

- [ ] **Step 1: Write the failing test**

```python
def test_joint_d_forward():
    import torch
    from wm3d_v3.models.state_stream import StateConfig
    from wm3d_v3.models.idm_stream import IDMStreamConfig
    from wm3d_v3.models.depth_encoder import DepthEncoderConfig
    from wm3d_v3.models.scene_flow_encoder import SceneFlowEncoderConfig
    from wm3d_v3.models.joint_model_d import JointDConfig, JointWorldModelD

    sc = StateConfig(T=4, P=16, D=128, hidden=128, n_layers=2, n_heads=4, k=2, cond_dim=128)
    ic = IDMStreamConfig(T=4, k=2, P=16, D=128, hidden=128, n_layers=2,
                         n_heads=4, z_dim=64, cond_dim=128, extra_token_dim=512)
    de = DepthEncoderConfig(in_size=224, out_grid=4, hidden_d=256, base_ch=8)
    fe = SceneFlowEncoderConfig(in_size=224, out_grid=4, hidden_d=256, base_ch=8)
    cfg = JointDConfig(state=sc, idm=ic, depth_enc=de, flow_enc=fe,
                       action_proj_hidden=64, action_proj_layers=2,
                       geom_hidden=64, pixel_hidden=64, pixel_n_res=1,
                       enable_pixel=False)
    m = JointWorldModelD(cfg)
    B = 2
    s = torch.randn(B, 4, 16, 128)
    c = torch.randn(B, 128)
    depth_in = torch.randn(B, 4, 224, 224)
    pmap_in  = torch.randn(B, 4, 224, 224, 3)
    # train path (teacher-forcing on)
    s_tgt = torch.randn(B, 2, 16, 128)
    depth_tgt = torch.randn(B, 2, 224, 224)
    pmap_tgt = torch.randn(B, 2, 224, 224, 3)
    out = m(s, c, depth_in=depth_in, pmap_in=pmap_in,
            s_tgt=s_tgt, depth_tgt=depth_tgt, pmap_tgt=pmap_tgt,
            teacher_mix=1.0, pixel=False)
    assert out["pose"].shape == (B, 2, 6)
    assert out["gripper_logit"].shape == (B, 2)
    # eval path (teacher_mix=0)
    out2 = m(s, c, depth_in=depth_in, pmap_in=pmap_in,
             teacher_mix=0.0, pixel=False)
    assert out2["pose"].shape == (B, 2, 6)
```

- [ ] **Step 2: Verify failure**

```bash
pytest tests/test_stage_d.py::test_joint_d_forward -x
```

- [ ] **Step 3: Extend IDMStream (if not already supporting larger extra_token_dim)**

Read `wm3d_v3/wm3d_v3/models/idm_stream.py`. If the in_proj already handles arbitrary `extra_token_dim` (verified in Stage C), no change required — just confirm.

- [ ] **Step 4: Implement JointWorldModelD**

Pattern after `joint_model_c.py`. Key differences:
- Constructor adds `self.flow_enc = SceneFlowEncoder(cfg.flow_enc)`.
- `forward` accepts `pmap_in`, `pmap_tgt` (optional), `s_tgt` (optional), `depth_tgt` (optional), `teacher_mix: float = 0.0`.
- Pred path computes everything as in C: state→pred, geom→depth_pred, backproject(depth_pred, K_pred) → pmap_pred (need intrinsic for k future frames; passed in or copied from last input intrinsic).
- Teacher path uses `s_tgt`, `depth_tgt`, `pmap_tgt` directly.
- Mix: `fut_state = (1 - α)·pred + α·s_tgt`, same for depth and pmap, when training. Inference `α = 0`.
- Concat `depth_tok` ⊕ `flow_tok` per patch → IDM's `extra_in`, `extra_pred`.

```python
"""JointWorldModelD — scene-flow augmented IDM + teacher-forcing schedule."""
from dataclasses import dataclass, field
import torch
import torch.nn as nn
from .state_stream import StateStream, StateConfig
from .action_proj import ActionProjHead
from .geom_decoder import GeomDecoder
from .pixel_decoder import PixelDecoder, PixelDecoderConfig
from .idm_stream import IDMStream, IDMStreamConfig
from .depth_encoder import DepthEncoder, DepthEncoderConfig
from .scene_flow_encoder import SceneFlowEncoder, SceneFlowEncoderConfig
from .backproject import depth_to_pointmap


@dataclass
class JointDConfig:
    state: StateConfig = field(default_factory=StateConfig)
    idm: IDMStreamConfig = field(default_factory=IDMStreamConfig)
    depth_enc: DepthEncoderConfig = field(default_factory=DepthEncoderConfig)
    flow_enc: SceneFlowEncoderConfig = field(default_factory=SceneFlowEncoderConfig)
    action_proj_hidden: int = 1024
    action_proj_layers: int = 5
    geom_hidden: int = 384
    pixel_hidden: int = 768
    pixel_n_res: int = 2
    enable_pixel: bool = False


class JointWorldModelD(nn.Module):
    def __init__(self, cfg: JointDConfig):
        super().__init__()
        self.cfg = cfg
        assert cfg.idm.extra_token_dim == cfg.depth_enc.hidden_d + cfg.flow_enc.hidden_d, \
            "IDM.extra_token_dim must equal depth_enc.hidden_d + flow_enc.hidden_d"
        self.state = StateStream(cfg.state)
        self.depth_enc = DepthEncoder(cfg.depth_enc)
        self.flow_enc = SceneFlowEncoder(cfg.flow_enc)
        self.idm = IDMStream(cfg.idm)
        self.action_proj = ActionProjHead(
            z_dim=cfg.idm.z_dim,
            hidden=cfg.action_proj_hidden,
            n_layers=cfg.action_proj_layers,
        )
        self.geom = GeomDecoder(
            token_dim=cfg.state.D,
            token_grid=int(cfg.state.P ** 0.5),
            hidden=cfg.geom_hidden,
        )
        self.pixel = PixelDecoder(
            PixelDecoderConfig(
                token_dim=cfg.state.D,
                token_grid=int(cfg.state.P ** 0.5),
                hidden=cfg.pixel_hidden,
                n_res=cfg.pixel_n_res,
            )
        ) if cfg.enable_pixel else None

    def load_action_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.action_proj.mean.copy_(mean.float().view(1, 1, 6))
        self.action_proj.std.copy_(std.float().view(1, 1, 6))

    @staticmethod
    def _mix(pred, tgt, alpha):
        if tgt is None or alpha == 0.0:
            return pred
        if alpha == 1.0:
            return tgt
        return (1 - alpha) * pred + alpha * tgt

    def forward(self, s, c, depth_in, pmap_in,
                s_tgt=None, depth_tgt=None, pmap_tgt=None,
                intrinsic=None, teacher_mix: float = 0.0,
                pixel: bool = False) -> dict:
        # ① state stream
        h_s = self.state.encode(s, c)
        for layer in self.state.layers:
            h_s = layer(h_s)
        h_s = self.state.norm(h_s)
        pred = self.state.decode(h_s)                        # [B, k, P, D]
        # ② geom -> depth_pred
        geom = self.geom(pred)
        depth_pred = geom["depth"]                           # [B, k, 224, 224]
        # ③ pmap_pred via backproject (camera-frame)
        if intrinsic is not None:
            B, T = depth_in.shape[:2]
            k = depth_pred.shape[1]
            K_fut = intrinsic[:, T:T+k]                       # [B, k, 3, 3]
        else:
            # fallback: identity intrinsic, principal point in middle
            B, _, H, W = depth_pred.shape
            K_fut = torch.eye(3, device=depth_pred.device).expand(B, depth_pred.shape[1], 3, 3).clone()
            K_fut[..., 0, 0] = K_fut[..., 1, 1] = max(H, W)
            K_fut[..., 0, 2] = W / 2; K_fut[..., 1, 2] = H / 2
        pmap_pred = depth_to_pointmap(depth_pred, K_fut)     # [B, k, 224, 224, 3]

        # ④ teacher-forcing mix for fut signals
        fut_state = self._mix(pred,       s_tgt,    teacher_mix)
        fut_depth = self._mix(depth_pred, depth_tgt, teacher_mix)
        fut_pmap  = self._mix(pmap_pred,  pmap_tgt,  teacher_mix)

        # ⑤ encode depth + flow to patch tokens
        d_in   = self.depth_enc(depth_in)                    # [B, T, P, d_d]
        d_fut  = self.depth_enc(fut_depth)                   # [B, k, P, d_d]
        f_in   = self.flow_enc(pmap_in)                      # [B, T, P, d_f]
        f_fut  = self.flow_enc(fut_pmap)                     # [B, k, P, d_f]
        extra_in  = torch.cat([d_in,  f_in],  dim=-1)        # [B, T, P, d_d+d_f]
        extra_fut = torch.cat([d_fut, f_fut], dim=-1)

        # ⑥ IDM + action head
        z_a = self.idm(s, fut_state, c,
                       extra_in=extra_in,
                       extra_pred=extra_fut)["z_a"]
        proj = self.action_proj(z_a)
        out = {
            "pred_tokens": pred,
            "z_a": z_a,
            "pose_norm": proj["pose_norm"],
            "pose": proj["pose"],
            "gripper_logit": proj["gripper_logit"],
            "depth": depth_pred,
            "pmap": pmap_pred,
            "point": geom["point"],
            "pose_geom": geom["pose"],
        }
        if pixel and self.pixel is not None:
            out["rgb"] = self.pixel(pred)
        return out

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
```

- [ ] **Step 5: Verify pass**

```bash
pytest tests/test_stage_d.py::test_joint_d_forward -x
```

- [ ] **Step 6: Commit**

```bash
git add wm3d_v3/wm3d_v3/models/joint_model_d.py wm3d_v3/tests/test_stage_d.py
git commit -m "feat(vla,D): JointWorldModelD — scene-flow + teacher-forcing IDM"
```

---

## Task D-7: Stage D trainer

**Files:**
- Create: `wm3d_v3/wm3d_v3/training/train_vla_d.py`
- Create: `wm3d_v3/configs/v3_vla_d.yaml`
- Create: `wm3d_v3/scripts/train_v3_vla_d.sh`

- [ ] **Step 1: Write config**

`wm3d_v3/configs/v3_vla_d.yaml`:
```yaml
data:
  manifest: /home/user01/Minko/newwm/wm3d_v3/manifests/oxe_train.jsonl
  cache_root: /home/user01/Minko/datasets/cache/wm3d_v3
  action_stats: /home/user01/Minko/datasets/cache/wm3d_v3/action_stats.npz
  switch_stats: /home/user01/Minko/datasets/cache/wm3d_v3/switch_stats.json
  T: 16
  k: 8
  stride: 4
  val_frac: 0.05
  seed: 0
  with_pointmap: true

model:
  state: { T: 16, P: 64, D: 2048, hidden: 1152, n_layers: 14, n_heads: 16, k: 8 }
  idm:   { T: 16, k: 8, P: 64, D: 2048, hidden: 896, n_layers: 10, n_heads: 14,
           z_dim: 192, cond_dim: 2048, extra_token_dim: 512 }
  depth_enc: { in_size: 224, out_grid: 8, hidden_d: 256, base_ch: 16 }
  flow_enc:  { in_size: 224, out_grid: 8, hidden_d: 256, base_ch: 16, smooth: true }
  action_proj_hidden: 1024
  action_proj_layers: 5
  geom_hidden: 384
  pixel_hidden: 768
  pixel_n_res: 2
  enable_pixel: false

train:
  epochs: 15
  batch_size_per_gpu: 8
  num_workers: 4
  lr: 1.5e-4
  weight_decay: 0.02
  warmup_steps: 1500
  grad_clip: 1.0
  log_every: 50
  ckpt_every_epochs: 2
  teacher_full_until_epoch: 5     # ep 0..4 teacher_mix=1.0
  teacher_anneal_until_epoch: 10  # ep 5..9 linear 1.0 -> 0.0
  freeze_dense_from_epoch: 10     # ep 10..14 freeze state/geom/encoders

loss:
  action_pose: 10.0
  action_pose_late: 30.0          # weight after freeze_dense_from_epoch
  action_grip: 2.0
  action_switch: 1.0
  state_mse: 1.0
  state_cos: 0.1
  geom_depth: 0.5
  grip_focal_alpha: 0.25
  grip_focal_gamma: 2.0
  huber_delta: 1.0

out:
  root: /home/user01/Minko/newwm/results/wm3d_v3_vla_d
  tb_dir: tb
  ckpt_dir: ckpt
```

- [ ] **Step 2: Write trainer**

Pattern after `train_vla_c.py`. Key additions:
- Load `switch_stats.json` → set `pos_weight` for grip.
- Each epoch compute `teacher_mix` from schedule and pass into model.forward.
- Epoch ≥ `freeze_dense_from_epoch`: `requires_grad_(False)` on state / geom / depth_enc / flow_enc; bump pose weight.
- Compute losses: pose, grip (with new pos_weight), switch, state, cos, geom_depth.

- [ ] **Step 3: Write launcher**

`wm3d_v3/scripts/train_v3_vla_d.sh`:
```bash
#!/usr/bin/env bash
cd /home/user01/Minko/newwm/wm3d_v3
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3}
exec torchrun --standalone --nproc_per_node=2 -m wm3d_v3.training.train_vla_d --cfg "${1:-configs/v3_vla_d.yaml}" "${@:2}"
```

- [ ] **Step 4: Verify trainer can import + parse**

```bash
cd /home/user01/Minko/newwm/wm3d_v3 && python -c "from wm3d_v3.training.train_vla_d import main; print('ok')"
```

- [ ] **Step 5: Commit**

```bash
git add wm3d_v3/wm3d_v3/training/train_vla_d.py wm3d_v3/configs/v3_vla_d.yaml wm3d_v3/scripts/train_v3_vla_d.sh
git commit -m "feat(vla,D): train_vla_d + config + launcher"
```

---

## Task D-8: Extend analyze_vla.py for variant d

**Files:**
- Modify: `wm3d_v3/scripts/analyze_vla.py`

- [ ] **Step 1: Add `c` variant build + `d` variant build**

After the `variant == "c"` branch, add `variant == "d"` branch building `JointWorldModelD` with flow_enc config.

- [ ] **Step 2: Collect loop emits pmap_in / intrinsic for variant d**

```python
if args.variant == "d":
    depth_in = batch["depth_in"].to(device, non_blocking=True)
    pmap_in  = batch["pmap_in"].to(device, non_blocking=True)
    intr     = batch["intrinsic"].to(device, non_blocking=True)
    out = model(s, c, depth_in=depth_in, pmap_in=pmap_in,
                intrinsic=intr, teacher_mix=0.0, pixel=False)
```

- [ ] **Step 3: Update arg parser**

```python
ap.add_argument("--variant", choices=["a", "b", "c", "d"], default="a", ...)
```

- [ ] **Step 4: Commit**

```bash
git add wm3d_v3/scripts/analyze_vla.py
git commit -m "feat(vla,D): analyze_vla.py supports variant d"
```

---

## Task D-9: Full point-map cache (background)

- [ ] **Step 1: Launch full cache on free GPU(s)**

```bash
# Pick a free card (check nvidia-smi). Use cards 0/1 if available, else queue.
cd /home/user01/Minko/newwm/wm3d_v3 && CUDA_VISIBLE_DEVICES=0,1 nohup python scripts/cache_vggt_pointmap.py \
  --manifest manifests/oxe_train.jsonl \
  --cache_root /home/user01/Minko/datasets/cache/wm3d_v3 \
  --num_gpus 2 \
  > /home/user01/Minko/datasets/cache/wm3d_v3/pointmap_cache.log 2>&1 &
echo $! > /home/user01/Minko/datasets/cache/wm3d_v3/pointmap_cache.pid
```

- [ ] **Step 2: Monitor**

```bash
tail -f /home/user01/Minko/datasets/cache/wm3d_v3/pointmap_cache.log
```

- [ ] **Step 3: Verify completion (count files)**

```bash
ls /home/user01/Minko/datasets/cache/wm3d_v3/vggt_pointmap/ | wc -l
# expected: ~22000 (matches manifest size)
```

---

## Task D-10: Stage D smoke run (1 epoch, 5% data)

- [ ] **Step 1: Run smoke**

```bash
cd /home/user01/Minko/newwm/wm3d_v3 && CUDA_VISIBLE_DEVICES=2,3 torchrun --standalone --nproc_per_node=2 \
  -m wm3d_v3.training.train_vla_d --cfg configs/v3_vla_d.yaml \
  --override "train.epochs=1 data.val_frac=0.5 train.log_every=20"
```

Expected: training runs, loss decreases over ~50 steps, ckpt saved.

- [ ] **Step 2: Run quick eval on smoke ckpt**

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/analyze_vla.py \
  --cfg configs/v3_vla_d.yaml \
  --ckpt /home/user01/Minko/newwm/results/wm3d_v3_vla_d/ckpt/epoch_000.pt \
  --out /home/user01/Minko/newwm/results/wm3d_v3_vla_d/eval/smoke \
  --variant d --max_batches 20
```

Expected: report.json non-empty, finite numbers.

- [ ] **Step 3: If smoke fails, fix; otherwise commit smoke artifacts location**

(No code commit — smoke verifies the pipeline only.)

---

## Task D-11: Full Stage D training

- [ ] **Step 1: Launch full training**

```bash
cd /home/user01/Minko/newwm/wm3d_v3 && CUDA_VISIBLE_DEVICES=2,3 nohup bash scripts/train_v3_vla_d.sh \
  > /home/user01/Minko/newwm/results/wm3d_v3_vla_d/train.log 2>&1 &
echo $! > /home/user01/Minko/newwm/results/wm3d_v3_vla_d/train.pid
```

- [ ] **Step 2: Monitor epoch boundaries**

Tail log; verify val_total decreases each epoch.

- [ ] **Step 3: Intermediate gate at epoch 5 (peak teacher)**

```bash
python scripts/analyze_vla.py --cfg configs/v3_vla_d.yaml \
  --ckpt /home/user01/Minko/newwm/results/wm3d_v3_vla_d/ckpt/epoch_005.pt \
  --out /home/user01/Minko/newwm/results/wm3d_v3_vla_d/eval/ep5 \
  --variant d --device cuda:0
```

Inspect bridge `pose_mse_overall`. Gate: ≤ 0.000674 (i.e. -20% vs zero-baseline 0.000843). If above this, **STOP** — even teacher-forced GT can't drive useful learning, and the structural diagnosis is confirmed. Write final report (Task D-13).

- [ ] **Step 4: Wait for training to finish (~7-8h)**

---

## Task D-12: Final eval on best.pt

- [ ] **Step 1: Run full eval**

```bash
cd /home/user01/Minko/newwm/wm3d_v3 && CUDA_VISIBLE_DEVICES=2 python scripts/analyze_vla.py \
  --cfg configs/v3_vla_d.yaml \
  --ckpt /home/user01/Minko/newwm/results/wm3d_v3_vla_d/ckpt/best.pt \
  --out /home/user01/Minko/newwm/results/wm3d_v3_vla_d/eval/vla_analysis \
  --variant d --device cuda:0
```

- [ ] **Step 2: Inspect report.json**

Check the three gates:
- bridge pose_mse_overall < 0.000590
- bridge grip_switch_f1 ≥ 0.60
- fractal pose_mse_overall < 0.01011

---

## Task D-13: Final report

**Files:**
- Create: `results/wm3d_v3_vla_d/eval/vla_analysis/STAGE_D_RESULTS.md`
- Update: `results/wm3d_v3_vla_c/eval/vla_analysis/FINAL_VLA_REPORT.md` (replace with v3 final encompassing all of A/B/C/D)

- [ ] **Step 1: Write STAGE_D_RESULTS.md**

Same structure as `STAGE_C_RESULTS.md`. Mirror the table with the three gates and per-axis breakdown.

- [ ] **Step 2: Rewrite the encompassing FINAL_VLA_REPORT.md**

If gates pass: declare native-3D VLA validated. If fail: declare ceiling reached and hand off to v4.

- [ ] **Step 3: Commit**

```bash
git add results/wm3d_v3_vla_d/eval/vla_analysis/STAGE_D_RESULTS.md \
        results/wm3d_v3_vla_c/eval/vla_analysis/FINAL_VLA_REPORT.md
git commit -m "docs(vla,D): Stage D results and v3 final VLA report"
```

---

## Self-review checklist

- [ ] All required files for each task explicitly listed
- [ ] Each implementation step has code; no "TBD" or "similar to above"
- [ ] Tests precede implementation in tasks D-2 through D-6
- [ ] Type / config name consistency: `JointDConfig`, `SceneFlowEncoderConfig`, `flow_enc`, `pmap_in`, `pmap_tgt`, `teacher_mix`, `with_pointmap`, `extra_token_dim=512`
- [ ] Intermediate gate at epoch 5 explicitly defined with stop condition (Task D-11 step 3)
- [ ] Failure mode (smoke broken, ep5 gate failed, ep10 gate failed) leads to defined action (stop + write final report)
- [ ] Spec covered: ✅ pointmap cache, ✅ SceneFlowEncoder, ✅ teacher-forcing IDM, ✅ phased freezing, ✅ switch loss + pos_weight
