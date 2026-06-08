"""Text-controllability demo through the COMPLETE merged world model.

Holds ONE scene fixed (its observation tokens s_in, action sequence, and
context RGB frame) and varies ONLY the instruction text. For each instruction
we feed:
  * the cached Qwen embedding of that instruction -> JointWorldModel (task_emb c)
  * the raw instruction string -> HunyuanVideo text encoder
and generate a video through the merged WM -> DiT-control -> HunyuanVideo path.

If the merged system is text-controllable, the SAME initial frame + actions
should yield DIFFERENT videos as the instruction changes.

Layout: one row per instruction = context frame + sampled generated frames,
labeled with the instruction text. All rows share the same context frame.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from wm3d_v3.data.action_condition import make_action_condition
from wm3d_v3.data.manifest import read_manifest
from wm3d_v3.data.window_dataset import OXEWindowDataset
from wm3d_v3.eval.make_demo_gif import window_config_from_cfg
from wm3d_v3.eval.run_eval import build_model
from wm3d_v3.video_backends.base import VideoConditionBundle
from wm3d_v3.video_backends.hunyuan_dit_control_video import (
    HunyuanDiTControlVideoBackend,
    HunyuanDiTControlVideoBackendConfig,
)

try:
    from PIL import Image, ImageDraw
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False


_TEXT_MAP: dict[str, str] = {}


def _task_text(sample: dict[str, Any]) -> str:
    for key in ("task_text", "language_instruction", "instruction"):
        v = sample.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    # window_config_from_cfg does not propagate load_task_text into the sample,
    # so fall back to the manifest record's task_text keyed by clip_id.
    cid = sample.get("clip_id", "")
    return _TEXT_MAP.get(str(cid), "").strip()


def _caption(img: np.ndarray, text: str) -> np.ndarray:
    """Stack a black caption band (the task being executed) on top of a frame.
    No context, no side-by-side — this is pure text2video output."""
    if not _HAS_PIL:
        return img
    import textwrap
    w = img.shape[1]
    lines = textwrap.wrap(text, width=max(20, w // 7)) or [text]
    band_h = 7 + 13 * len(lines)
    band = Image.new("RGB", (w, band_h), (12, 12, 12))
    d = ImageDraw.Draw(band)
    for i, ln in enumerate(lines):
        d.text((6, 4 + 13 * i), ln, fill=(255, 224, 0))
    return np.concatenate([np.asarray(band), img], axis=0)


def _video_prompt(instruction: str) -> str:
    """Wrap a raw LIBERO manipulation command into a descriptive video-generation
    prompt. The raw command ("put X in Y") is an action spec, not a scene
    description, so HunyuanVideo renders disconnected object close-ups. We frame
    it as a continuous third-person robot-manipulation clip and pin scene/style/
    camera so the model produces a coherent tabletop video instead.
    """
    instr = instruction.strip().rstrip(".")
    return (
        "Third-person video of a single robotic arm performing a tabletop "
        f"manipulation task in a tidy scene. The gripper proceeds to {instr}. "
        "One robot arm over the same wooden table and objects throughout, "
        "smooth continuous and precise motion, consistent layout, soft even "
        "lighting, realistic rendering, sharp focus, stable camera, high detail."
    )


_NEG_PROMPT = (
    "static image, still frame, extreme close-up, single object only, no robot, "
    "blurry, low quality, distorted, text, watermark, camera shake, jump cut, "
    "duplicated arms, warped geometry"
)


def _resize_f3hw(v: torch.Tensor, h: int, w: int) -> torch.Tensor:
    v = v.float()
    if v.shape[-2:] != (h, w):
        v = F.interpolate(v, size=(h, w), mode="bilinear", align_corners=False)
    return v.clamp(0, 1)


def _to_uint8_fhwc(v_f3hw: torch.Tensor) -> np.ndarray:
    arr = v_f3hw.permute(0, 2, 3, 1).detach().cpu().numpy()
    return (arr * 255.0).round().astype(np.uint8)


def _label(img: np.ndarray, text: str) -> np.ndarray:
    if not _HAS_PIL:
        return img
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.text((4, 3), text, fill=(0, 0, 0))
    d.text((3, 2), text, fill=(255, 255, 0))
    return np.asarray(im)


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description="Text-controllability demo through merged WM.")
    ap.add_argument("--cfg", type=Path, required=True)
    ap.add_argument("--wm_ckpt", type=Path, required=True)
    ap.add_argument("--control_ckpt", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--n_instructions", type=int, default=4)
    ap.add_argument("--control_scale", type=float, default=1.0)
    ap.add_argument("--control_scales", type=str, default="", help="comma-separated list to sweep, e.g. '0.3,0.5,0.7'; overrides --control_scale")
    ap.add_argument("--scene_rank", type=int, default=0, help="which distinct-instruction clip provides the fixed scene")
    ap.add_argument("--height", type=int, default=320)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--frames", type=int, default=9)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fixed_seed", action="store_true", help="use the SAME seed for every instruction so instruction text is the only variable (rigorous text-controllability)")
    ap.add_argument("--rich_prompt", action="store_true", help="wrap raw LIBERO command into a descriptive video-generation prompt (recommended for text2video mode)")
    ap.add_argument("--gif_clean", action="store_true", help="output one clean GIF per instruction: generated frames only (no context/compare), task text captioned on top")
    ap.add_argument("--embedded_cfg", type=float, default=6.0, help="HunyuanVideo embedded guidance scale; higher = follow prompt more")
    ap.add_argument("--fps", type=int, default=6)
    ap.add_argument("--sheet_cols", type=int, default=5)
    args = ap.parse_args()

    H, W = int(args.height), int(args.width)
    nf = int(args.frames)
    cfg = yaml.safe_load(args.cfg.read_text())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)
    torch.manual_seed(args.seed)

    records = read_manifest(cfg["data"]["manifest"])
    _TEXT_MAP.update({str(r.clip_id): str(getattr(r, "task_text", "") or "") for r in records})
    ds = OXEWindowDataset(records, window_config_from_cfg(cfg))
    g = torch.Generator().manual_seed(int(cfg["data"].get("seed", args.seed)))
    perm = torch.randperm(len(ds), generator=g).tolist()

    model = build_model(cfg).to(device).eval()
    ckpt = torch.load(args.wm_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    for p in model.parameters():
        p.requires_grad_(False)

    backend = HunyuanDiTControlVideoBackend(
        HunyuanDiTControlVideoBackendConfig(
            control_ckpt=str(args.control_ckpt),
            control_scale=float(args.control_scale),
            infer_steps=int(args.steps),
        ),
        device=device,
    )

    # ---- collect N distinct instructions (each carries its own cached Qwen c) ----
    picked: list[dict[str, Any]] = []
    seen_text: set[str] = set()
    for vi in perm:
        sample = ds[vi]
        txt = _task_text(sample)
        if not txt or txt in seen_text:
            continue
        seen_text.add(txt)
        picked.append({"idx": vi, "sample": sample, "text": txt})
        if len(picked) >= max(args.n_instructions, args.scene_rank + 1):
            break
    if len(picked) < 2:
        raise RuntimeError(f"need >=2 distinct instructions, found {len(picked)}")

    instructions = picked[: args.n_instructions]
    scene = picked[min(args.scene_rank, len(picked) - 1)]["sample"]

    # ---- fixed scene inputs (observation + actions + context frame) ----
    s = scene["s_in"].unsqueeze(0).to(device)
    action_tgt = scene["action_tgt"].unsqueeze(0).to(device)
    action_tgt_norm = scene["action_tgt_norm"].unsqueeze(0).to(device)
    context_rgb = scene["rgb_in"][-1].permute(2, 0, 1).unsqueeze(0).to(device)
    action_cond = make_action_condition(action_tgt, action_tgt_norm)
    ctx = _resize_f3hw(context_rgb, H, W)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    scene_text = _task_text(scene)
    print(f"[scene] fixed initial frame from clip with instruction: {scene_text!r}", flush=True)
    ctx_u = _to_uint8_fhwc(ctx)[0]

    # ---- precompute the WM bundle for each instruction ONCE (control_scale-independent) ----
    bundles: list[dict[str, Any]] = []
    for j, item in enumerate(instructions):
        c_j = item["sample"]["c"].unsqueeze(0).to(device)  # cached Qwen emb of THIS instruction
        text_j = item["text"]
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            out = model(s, c_j, action_cond=action_cond, context_rgb=context_rgb, pixel=True, bridging=False)
        bundle = VideoConditionBundle(
            context_rgb=context_rgb.float(),
            action_cond=action_cond.float(),
            task_emb=c_j.float(),
            task_text=[text_j],
            pred_tokens=out["pred_tokens"].float(),
            depth=out["depth"].float(),
            motion_hint=out.get("motion_hint"),
            contact_hint=out.get("contact_hint"),
            rough_rgb=out.get("rgb"),
        )
        bundles.append({"bundle": bundle, "text": text_j})

    # ---- sweep control_scale: each scale gets its own contact sheet (rows = instructions) ----
    if args.control_scales:
        scales = [float(x) for x in str(args.control_scales).split(",") if x.strip() != ""]
    else:
        scales = [float(args.control_scale)]
    print(f"[sweep] control_scales={scales}", flush=True)

    for cs in scales:
        backend.cfg.control_scale = float(cs)
        rows: list[np.ndarray] = []
        cs_tag = f"cs{cs:.2f}".replace(".", "p")
        for j, bj in enumerate(bundles):
            text_j = bj["text"]
            gen_seed = int(args.seed) if args.fixed_seed else int(args.seed) + j
            vprompt = _video_prompt(text_j) if args.rich_prompt else text_j
            print(f"[cs={cs:.2f}][{j}] prompt -> {vprompt}", flush=True)
            result = backend.generate(
                bj["bundle"], num_frames=nf, height=H, width=W, seed=gen_seed,
                infer_steps=int(args.steps),
                prompt=vprompt,
                negative_prompt=_NEG_PROMPT,
                embedded_guidance_scale=float(args.embedded_cfg),
            )
            hun = _resize_f3hw(result.rgb[0].to(device), H, W)[:nf]
            if hun.shape[0] < nf:
                hun = torch.cat([hun, hun[-1:].expand(nf - hun.shape[0], -1, -1, -1)], dim=0)
            hun_u = _to_uint8_fhwc(hun)

            cols = np.linspace(0, nf - 1, num=min(args.sheet_cols - 1, nf)).round().astype(int)
            strip = np.concatenate([ctx_u] + [hun_u[ci] for ci in cols], axis=1)
            strip = _label(strip, f"[{j}] {text_j[:70]}")
            rows.append(strip)

            safe = "".join(ch if ch.isalnum() else "_" for ch in text_j)[:48]
            base = args.out_dir / f"{cs_tag}_instr_{j:02d}_{safe}"
            if args.gif_clean:
                # pure text2video: generated frames only, task captioned on top
                frames_g = [_caption(hun_u[f], text_j) for f in range(nf)]
                imageio.mimsave(base.with_suffix(".gif"), frames_g, duration=1.0 / args.fps, loop=0)
                imageio.mimsave(base.with_suffix(".mp4"), frames_g, fps=args.fps)
            else:
                panel = [np.concatenate([ctx_u, hun_u[f]], axis=1) for f in range(nf)]
                imageio.mimsave(base.with_suffix(".mp4"), panel, fps=args.fps)
            print(f"wrote {base} task={text_j!r}", flush=True)

        maxw = max(r.shape[1] for r in rows)
        padded = [np.pad(r, ((4, 4), (0, maxw - r.shape[1]), (0, 0)), constant_values=255) for r in rows]
        sheet = np.concatenate(padded, axis=0)
        title = np.full((22, maxw, 3), 30, dtype=np.uint8)
        title = _label(title, f"control_scale={cs:.2f}  col0=initial frame; only instruction varies. scene={scene_text[:46]}")
        out_png = args.out_dir / f"text_control_{cs_tag}.png"
        imageio.imwrite(out_png, np.concatenate([title, sheet], axis=0))
        print(f"wrote {out_png} control_scale={cs:.2f} instructions={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
