"""Checkpoint-backed WM3D policy for tokenized observations."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from wm3d_v3.eval.run_eval import build_model
from wm3d_v3.policy.world_model_policy import ScoreWeights, select_action_chunk, selected_first_action


@dataclass
class PolicyDecision:
    first_action_raw: torch.Tensor
    action_chunk_raw: torch.Tensor
    selected_idx: torch.Tensor
    selected_score: torch.Tensor
    candidate_scores: torch.Tensor
    raw: dict[str, Any]


def load_terminal_reference(path: str | Path) -> dict[str, torch.Tensor]:
    """Load stage3 terminal actions from a successful LIBERO trace."""
    data = json.loads(Path(path).read_text())
    features: list[list[float]] = []
    actions: list[list[float]] = []
    trace_features: list[list[float]] = []
    trace_actions: list[list[float]] = []
    for episode in data.get("results") or []:
        for item in episode.get("step_trace") or []:
            plan = item.get("plan_state")
            if not isinstance(plan, list) or len(plan) < 17:
                continue
            action = item.get("policy_action") or item.get("action")
            if not isinstance(action, list) or len(action) < 7:
                continue
            raw = [float(v) for v in action[:7]]
            raw[6] = 1.0 if raw[6] > 0.5 else 0.0
            trace_features.append([float(v) for v in plan[:17]])
            trace_actions.append(raw)
            stage = int(item.get("plan_stage", max(range(4), key=lambda idx: float(plan[idx]))))
            if stage != 3:
                continue
            features.append([float(v) for v in plan[8:17]])
            actions.append(raw)
    if not features:
        raise RuntimeError(f"no stage3 terminal reference rows found in {path}")
    feature_t = torch.tensor(features, dtype=torch.float32)
    action_t = torch.tensor(actions, dtype=torch.float32)
    design = torch.cat([torch.ones(feature_t.shape[0], 1), feature_t], dim=1)
    ridge = 1e-3 * torch.eye(design.shape[1], dtype=torch.float32)
    ridge[0, 0] = 0.0
    linear_weights = torch.linalg.solve(design.T @ design + ridge, design.T @ action_t)
    trace_feature_t = torch.tensor(trace_features, dtype=torch.float32)
    trace_action_t = torch.tensor(trace_actions, dtype=torch.float32)
    trace_design = torch.cat([torch.ones(trace_feature_t.shape[0], 1), trace_feature_t], dim=1)
    trace_ridge = 1e-3 * torch.eye(trace_design.shape[1], dtype=torch.float32)
    trace_ridge[0, 0] = 0.0
    trace_linear_weights = torch.linalg.solve(trace_design.T @ trace_design + trace_ridge, trace_design.T @ trace_action_t)
    return {
        "features": feature_t,
        "actions": action_t,
        "linear_weights": linear_weights,
        "trace_features": trace_feature_t,
        "trace_actions": trace_action_t,
        "trace_linear_weights": trace_linear_weights,
    }


class WM3DTokenPolicy:
    """Policy adapter for already-tokenized observations.

    External benchmark runners still need an observation adapter that turns RGB
    frames into VGGT token windows. Once they have `[B,T,P,D]` tokens and one
    Qwen task embedding, this class owns the common WM3D action-selection path.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        device: str | torch.device = "cuda:0",
        score_weights: ScoreWeights | None = None,
        selection_mode: str = "ranked",
        terminal_reference: dict[str, torch.Tensor] | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.score_weights = score_weights or ScoreWeights()
        self.selection_mode = selection_mode
        self.terminal_reference = terminal_reference

    @classmethod
    def from_checkpoint(
        cls,
        cfg_path: str | Path,
        ckpt_path: str | Path,
        *,
        device: str | torch.device = "cuda:0",
        score_weights: ScoreWeights | None = None,
        selection_mode: str = "ranked",
        terminal_reference_path: str | Path | None = None,
    ) -> "WM3DTokenPolicy":
        cfg = yaml.safe_load(Path(cfg_path).read_text())
        model = build_model(cfg)
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(sd["model"])
        terminal_reference = load_terminal_reference(terminal_reference_path) if terminal_reference_path else None
        return cls(
            model,
            device=device,
            score_weights=score_weights,
            selection_mode=selection_mode,
            terminal_reference=terminal_reference,
        )

    @torch.no_grad()
    def act_from_tokens(
        self,
        context_tokens: torch.Tensor,
        task_emb: torch.Tensor,
        *,
        context_rgb: torch.Tensor | None = None,
        lowdim_state: torch.Tensor | None = None,
        object_state: torch.Tensor | None = None,
        plan_state: torch.Tensor | None = None,
        action_history: torch.Tensor | None = None,
        progress_state: torch.Tensor | None = None,
        return_rollouts: bool = False,
    ) -> PolicyDecision:
        context_tokens = context_tokens.to(self.device, non_blocking=True)
        task_emb = task_emb.to(self.device, non_blocking=True)
        if context_rgb is not None:
            context_rgb = context_rgb.to(self.device, non_blocking=True)
        if lowdim_state is not None:
            lowdim_state = lowdim_state.to(self.device, non_blocking=True)
        if object_state is not None:
            object_state = object_state.to(self.device, non_blocking=True)
        if plan_state is not None:
            plan_state = plan_state.to(self.device, non_blocking=True)
        if action_history is not None:
            action_history = action_history.to(self.device, non_blocking=True)
        if progress_state is not None:
            progress_state = progress_state.to(self.device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            raw = select_action_chunk(
                self.model,
                context_tokens,
                task_emb,
                context_rgb=context_rgb,
                lowdim_state=lowdim_state,
                object_state=object_state,
                plan_state=plan_state,
                action_history=action_history,
                progress_state=progress_state,
                pixel=False,
                score_weights=self.score_weights,
                selection_mode=self.selection_mode,
                terminal_reference=self.terminal_reference,
                return_rollouts=return_rollouts,
            )
        return PolicyDecision(
            first_action_raw=selected_first_action(raw, raw=True).detach().float().cpu(),
            action_chunk_raw=raw["selected_action_raw"].detach().float().cpu(),
            selected_idx=raw["selected_idx"].detach().cpu(),
            selected_score=raw["selected_score"].detach().float().cpu(),
            candidate_scores=raw["candidate_scores"].detach().float().cpu(),
            raw=raw,
        )
