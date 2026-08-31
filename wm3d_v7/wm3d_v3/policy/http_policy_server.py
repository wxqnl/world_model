"""HTTP policy server for external benchmark runtimes.

This server keeps WM3D, VGGT, and Qwen in the current training environment while
older simulator stacks such as LIBERO run in a separate Python environment. It
uses only the Python standard library for transport.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image

from wm3d_v3.benchmarks.online_tokenizer import OnlineObservationTokenizer
from wm3d_v3.policy import ScoreWeights, WM3DTokenPolicy
from wm3d_v3.policy.world_model_policy import denormalize_action_cond


def _decode_frame(payload: str) -> np.ndarray:
    if "," in payload and payload.lstrip().startswith("data:"):
        payload = payload.split(",", 1)[1]
    raw = base64.b64decode(payload)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return np.asarray(img)


def _decode_frames_by_camera(request: dict[str, Any]) -> list[np.ndarray] | dict[str, list[np.ndarray]]:
    frames_by_camera = request.get("frames_by_camera")
    if frames_by_camera is not None:
        if not isinstance(frames_by_camera, dict) or not frames_by_camera:
            raise ValueError("frames_by_camera must be a non-empty object")
        decoded: dict[str, list[np.ndarray]] = {}
        for camera_name, payloads in frames_by_camera.items():
            if not isinstance(payloads, list) or not payloads:
                raise ValueError(f"frames_by_camera[{camera_name!r}] must be a non-empty list")
            decoded[str(camera_name)] = [_decode_frame(str(item)) for item in payloads]
        return decoded
    frames_payload = request.get("frames") or []
    if not isinstance(frames_payload, list) or not frames_payload:
        raise ValueError("request must include non-empty frames list or frames_by_camera object")
    return [_decode_frame(str(item)) for item in frames_payload]


def _optional_tensor(request: dict[str, Any], key: str, shape_last: int | None = None) -> torch.Tensor | None:
    value = request.get(key)
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32)
    if shape_last is not None and (arr.ndim == 0 or arr.shape[-1] != shape_last):
        raise ValueError(f"{key} must end with dim {shape_last}, got {arr.shape}")
    if arr.ndim == 1 or (key == "action_history" and arr.ndim == 2):
        arr = arr[None]
    return torch.from_numpy(arr)


def _optional_progress_tensor(request: dict[str, Any]) -> torch.Tensor | None:
    value = request.get("progress_state")
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[-1] != 1:
        raise ValueError(f"progress_state must be scalar, [1], or [B,1], got {arr.shape}")
    return torch.from_numpy(arr)


def _optional_bool(request: dict[str, Any], key: str) -> bool | None:
    value = request.get(key)
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{key} must be boolean-like, got {value!r}")


def _optional_int(request: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = request.get(key)
        if value is not None:
            return int(value)
    return None


def _optional_float(request: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = request.get(key)
        if value is not None:
            return float(value)
    return None


def _flow_sample_from_arg(value: str) -> bool | None:
    value = value.strip().lower()
    if value == "auto":
        return None
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"unsupported flow sample mode {value!r}")


def _selection_mode_from_cfg(cfg: dict[str, Any], requested: str) -> str:
    if requested != "auto":
        return requested
    train_cfg = cfg.get("train", {})
    if bool(train_cfg.get("candidate_planner_joint_action", False)):
        return "ranked_residual"
    model_cfg = cfg.get("model", {})
    if bool(model_cfg.get("enable_action_policy", False)):
        return "direct"
    return "ranked"


def _score_weights_from_cfg(
    cfg: dict[str, Any],
    *,
    progress: float | None,
    terminal: float | None,
    plausibility: float | None,
) -> ScoreWeights:
    train_cfg = cfg.get("train", {})
    joint_candidate = bool(train_cfg.get("candidate_planner_joint_action", False))
    legacy_defaults = (1.0, 1.0, 0.0)
    candidate_defaults = (
        float(train_cfg.get("evaluator_score_progress_weight", 0.0)),
        float(train_cfg.get("evaluator_score_terminal_weight", 0.0)),
        float(train_cfg.get("evaluator_score_plausibility_weight", 1.0)),
    )
    defaults = candidate_defaults if joint_candidate else legacy_defaults
    return ScoreWeights(
        progress=defaults[0] if progress is None else float(progress),
        terminal=defaults[1] if terminal is None else float(terminal),
        plausibility=defaults[2] if plausibility is None else float(plausibility),
    )


class WM3DPolicyHTTPServer(ThreadingHTTPServer):
    policy: WM3DTokenPolicy
    tokenizer: OnlineObservationTokenizer


class Handler(BaseHTTPRequestHandler):
    server: WM3DPolicyHTTPServer

    def log_message(self, fmt: str, *args: Any) -> None:
        return None

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json(200, {"ok": True, "service": "wm3d_policy"})
            return
        self._send_json(404, {"error": "unknown endpoint"})

    def do_POST(self) -> None:
        if self.path != "/act":
            self._send_json(404, {"error": "unknown endpoint"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length).decode("utf-8"))
            frames = _decode_frames_by_camera(request)
            task_text = str(request.get("task_text") or "robot manipulation")
            obs = self.server.tokenizer.tokenize(frames, task_text)
            decision = self.server.policy.act_from_tokens(
                obs.context_tokens,
                obs.task_emb,
                wrist_tokens=obs.wrist_tokens,
                view_mask=obs.view_mask,
                context_rgb=obs.context_rgb,
                lowdim_state=_optional_tensor(request, "lowdim_state"),
                object_state=_optional_tensor(request, "object_state"),
                plan_state=_optional_tensor(request, "plan_state"),
                action_history=_optional_tensor(request, "action_history", 7),
                progress_state=_optional_progress_tensor(request),
                flow_sample=_optional_bool(request, "flow_sample"),
                flow_sample_steps=_optional_int(request, "flow_sample_steps", "flow_steps"),
                flow_noise_scale=_optional_float(request, "flow_noise_scale"),
                flow_seed=_optional_int(request, "flow_seed"),
            )
            action_chunk_continuous = denormalize_action_cond(
                decision.raw["selected_action_cond"],
                model=self.server.policy.action_stats_model,
                binarize_gripper=False,
            ).detach().float().cpu()
            self._send_json(
                200,
                {
                    "first_action_raw": decision.first_action_raw.reshape(-1, 7)[0].tolist(),
                    "action_chunk_raw": decision.action_chunk_raw.reshape(-1, 7).tolist(),
                    "first_action_continuous_raw": action_chunk_continuous.reshape(-1, 7)[0].tolist(),
                    "action_chunk_continuous_raw": action_chunk_continuous.reshape(-1, 7).tolist(),
                    "selected_gripper_prob": decision.raw["selected_gripper_prob"].detach().float().cpu().reshape(-1).tolist(),
                    "selected_idx": int(decision.selected_idx.reshape(-1)[0]),
                    "selected_score": float(decision.selected_score.reshape(-1)[0]),
                    "candidate_scores": decision.candidate_scores.reshape(-1).tolist(),
                    "action_space": "7D delta pose + gripper_closed",
                },
            )
        except Exception as exc:
            self._send_json(500, {"error": repr(exc)})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--overlay_ckpt", type=Path, default=None)
    ap.add_argument("--anchor_cfg", type=Path, default=None)
    ap.add_argument("--anchor_ckpt", type=Path, default=None)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--qwen_device", default=None)
    ap.add_argument("--task_cache_dir", type=Path, default=Path("/data/Minko/datasets/cache/wm3d_v3/online_taskemb"))
    ap.add_argument("--allow_zero_task_fallback", action="store_true")
    ap.add_argument("--camera_fusion", default="mean", choices=("mean", "concat", "separate"))
    ap.add_argument("--anchor_camera_key", default="agentview_image")
    ap.add_argument("--wrist_camera_key", default="robot0_eye_in_hand_image")
    ap.add_argument("--score_progress_weight", type=float, default=None)
    ap.add_argument("--score_terminal_weight", type=float, default=None)
    ap.add_argument("--score_plausibility_weight", type=float, default=None)
    ap.add_argument("--flow_sample", default="auto", choices=("auto", "true", "false"))
    ap.add_argument("--flow_sample_steps", "--flow_steps", dest="flow_sample_steps", type=int, default=None)
    ap.add_argument("--flow_noise_scale", type=float, default=None)
    ap.add_argument("--flow_seed", type=int, default=None)
    ap.add_argument(
        "--selection_mode",
        default="auto",
        choices=(
            "auto",
            "ranked",
            "ranked_residual",
            "anchor",
            "first",
            "candidate0",
            "direct",
            "policy",
            "bc",
            "action_policy",
            "direct_prior",
            "prior_policy",
            "direct_stage3_place",
            "direct_terminal_nn",
            "direct_terminal_linear",
            "direct_trace_linear",
            "plan_waypoint",
            "waypoint_servo",
        ),
    )
    ap.add_argument("--terminal_reference_json", type=Path, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.cfg.read_text())
    selection_mode = _selection_mode_from_cfg(cfg, args.selection_mode)
    data_cfg = cfg["data"]
    token_grid = 16 if data_cfg.get("tokens_subdir") == "vggt_p256" else 8
    tokenizer = OnlineObservationTokenizer(
        T=int(data_cfg["T"]),
        token_grid=token_grid,
        task_cache_dir=args.task_cache_dir,
        device=args.device,
        qwen_device=args.qwen_device,
        allow_zero_task_fallback=args.allow_zero_task_fallback,
        camera_fusion=args.camera_fusion,
        anchor_camera_key=args.anchor_camera_key,
        wrist_camera_key=args.wrist_camera_key,
    )
    policy = WM3DTokenPolicy.from_checkpoint(
        args.cfg,
        args.ckpt,
        device=args.device,
        overlay_ckpt_path=args.overlay_ckpt,
        anchor_cfg_path=args.anchor_cfg,
        anchor_ckpt_path=args.anchor_ckpt,
        score_weights=_score_weights_from_cfg(
            cfg,
            progress=args.score_progress_weight,
            terminal=args.score_terminal_weight,
            plausibility=args.score_plausibility_weight,
        ),
        selection_mode=selection_mode,
        terminal_reference_path=args.terminal_reference_json,
        flow_sample=_flow_sample_from_arg(args.flow_sample),
        flow_sample_steps=args.flow_sample_steps,
        flow_noise_scale=args.flow_noise_scale,
        flow_seed=args.flow_seed,
    )
    server = WM3DPolicyHTTPServer((args.host, args.port), Handler)
    server.policy = policy
    server.tokenizer = tokenizer
    print(
        json.dumps(
            {
                "host": args.host,
                "port": args.port,
                "status": "ready",
                "selection_mode": selection_mode,
                "flow_sample": args.flow_sample,
                "flow_sample_steps": args.flow_sample_steps,
                "flow_noise_scale": args.flow_noise_scale,
                "flow_seed": args.flow_seed,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
