"""Serve an official LIBERO BC policy to an older simulator environment."""
from __future__ import annotations

import argparse
import base64
import io
import json
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from wm3d_v3.benchmarks.libero_bc_teacher import (
    _bootstrap_libero,
    _load_rollout_algo,
    _task_id_for_name,
    _teacher_action,
)


def _decode_frame(payload: str) -> np.ndarray:
    raw = base64.b64decode(payload.split(",", 1)[-1])
    return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))


def _resize_frame(frame: np.ndarray, height: int, width: int) -> np.ndarray:
    if frame.shape[0] == height and frame.shape[1] == width:
        return frame
    return np.asarray(Image.fromarray(frame).resize((width, height), Image.BILINEAR))


class TeacherServer(ThreadingHTTPServer):
    cfg: Any
    benchmark: Any
    algo: Any
    raw_obs_to_tensor_obs: Any
    deterministic: bool
    force_task_id: int | None
    lock: threading.Lock


class Handler(BaseHTTPRequestHandler):
    server: TeacherServer

    def log_message(self, fmt: str, *args: Any) -> None:
        return None

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send(200, {"ok": True, "service": "libero_bc_teacher"})
            return
        self._send(404, {"error": "unknown endpoint"})

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length).decode("utf-8"))
            if self.path == "/reset":
                with self.server.lock:
                    self.server.algo.reset()
                self._send(200, {"ok": True})
                return
            if self.path != "/act":
                self._send(404, {"error": "unknown endpoint"})
                return
            task_name = str(request["task_name"])
            task_id = (
                self.server.force_task_id
                if self.server.force_task_id is not None
                else _task_id_for_name(self.server.benchmark, task_name)
            )
            task_emb = self.server.benchmark.get_task_emb(task_id)
            img_h = int(self.server.cfg.data.img_h)
            img_w = int(self.server.cfg.data.img_w)
            obs = {
                "agentview_image": _resize_frame(
                    _decode_frame(request["agentview_image"]), img_h, img_w
                ),
                "robot0_eye_in_hand_image": _resize_frame(
                    _decode_frame(request["eye_in_hand_image"]), img_h, img_w
                ),
                "robot0_gripper_qpos": np.asarray(
                    request["gripper_qpos"], dtype=np.float32
                ),
                "robot0_joint_pos": np.asarray(
                    request["joint_pos"], dtype=np.float32
                ),
            }
            with self.server.lock:
                data = self.server.raw_obs_to_tensor_obs(
                    [obs],
                    task_emb,
                    self.server.cfg,
                )
                action = _teacher_action(
                    self.server.algo.policy,
                    data,
                    deterministic=self.server.deterministic,
                ).reshape(-1, 7)[0]
            self._send(200, {"action": action.astype(float).tolist()})
        except Exception as exc:
            traceback.print_exc()
            self._send(500, {"error": repr(exc)})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--low_eval_noise", action="store_true")
    ap.add_argument("--deterministic_action", action="store_true")
    ap.add_argument("--force_task_id", type=int, default=None)
    ap.add_argument(
        "--libero_root",
        type=Path,
        default=Path("/data/Minko/benchmarks/LIBERO"),
    )
    args = ap.parse_args()

    _bootstrap_libero(args.libero_root)
    from libero.lifelong.metric import raw_obs_to_tensor_obs
    import robomimic.utils.obs_utils as ObsUtils

    cfg, benchmark, algo = _load_rollout_algo(args)
    ObsUtils.initialize_obs_utils_with_obs_specs({"obs": cfg.data.obs.modality})
    server = TeacherServer((args.host, args.port), Handler)
    server.cfg = cfg
    server.benchmark = benchmark
    server.algo = algo
    server.raw_obs_to_tensor_obs = raw_obs_to_tensor_obs
    server.deterministic = bool(args.deterministic_action)
    server.force_task_id = args.force_task_id
    server.lock = threading.Lock()
    print(
        json.dumps(
            {
                "host": args.host,
                "port": args.port,
                "checkpoint": str(args.ckpt),
                "status": "ready",
            },
            sort_keys=True,
        ),
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
