"""JointWorldModel — one forward, end-to-end.

Input: pooled VGGT tokens [B, T, 64, 2048] (pre-cached) + task embedding c [B, 2048]
Output dict (always):
    pred_tokens:   [B, k, 64, 2048]
    z_a:           [B, k, z_dim]
    pose_norm:     [B, k, 6]   (standardized 6-DoF Δpose)
    pose:          [B, k, 6]   (de-normalized = pose_norm * std + mean)
    gripper_logit: [B, k]
    depth:         [B, k, 224, 224]
    point:         [B, k, 224, 224, 3] if geometry extras are enabled
    pose_geom:     [B, k, 9] if geometry extras are enabled
Conditional:
    rgb:                if pixel=True and self.pixel is not None
    cosmos_depth_input: if bridging=True and self.bridging is not None
    aux_pose_norm:      if aux_idm=True and self.aux_idm is not None
    aux_grip:
"""
from __future__ import annotations
from dataclasses import dataclass, field
import torch
import torch.nn as nn
from .dual_stream import DualStreamDynamics, DualConfig
from .action_proj import ActionProjHead
from .geom_decoder import GeomDecoder
from .pixel_decoder import PixelDecoder, PixelDecoderConfig
from .context_residual_pixel_decoder import (
    ContextResidualPixelDecoder,
    ContextResidualPixelDecoderConfig,
)
from .control_head import ControlHead, ControlHeadConfig
from .progress_head import ProgressHead, ProgressHeadConfig
from .action_proposer import ActionProposer, ActionProposerConfig
from .action_policy import ActionChunkPolicy, ActionChunkPolicyConfig
from .bridging_adapter import BridgingAdapter
from .aux_idm import AuxIDM, AuxIDMConfig
from .world_token_prior import WorldTokenPrior, WorldTokenPriorConfig


@dataclass
class JointConfig:
    dual: DualConfig = field(default_factory=DualConfig)
    action_proj_hidden: int = 768
    action_proj_layers: int = 5
    geom_hidden: int = 384
    geom_upsample_mode: str = "transpose"
    enable_geom_extra: bool = True
    pixel_hidden: int = 768
    pixel_n_res: int = 2
    enable_pixel: bool = True
    enable_context_pixel: bool = False
    context_pixel_hidden: int = 384
    context_pixel_action_dim: int = 7
    context_pixel_task_dim: int | None = None
    context_pixel_residual_scale: float = 0.75
    context_pixel_use_action: bool = True
    context_pixel_use_task: bool = True
    context_pixel_predict_motion: bool = False
    context_pixel_motion_blend_gain: float = 0.0
    enable_control_head: bool = False
    control_hidden: int = 128
    control_output_size: int = 256
    control_fuse_size: int = 64
    control_refine_channels: int = 16
    control_use_refine: bool = True
    control_action_dim: int = 7
    control_task_dim: int | None = None
    control_use_context: bool = True
    control_use_action: bool = True
    control_use_task: bool = True
    enable_progress_head: bool = False
    progress_hidden: int = 256
    progress_layers: int = 2
    progress_heads: int = 4
    progress_action_dim: int = 7
    progress_task_dim: int | None = None
    progress_max_horizon: int = 32
    progress_use_action: bool = True
    progress_use_task: bool = True
    enable_action_proposer: bool = False
    proposer_hidden: int = 512
    proposer_layers: int = 3
    proposer_candidates: int = 4
    proposer_horizon: int | None = None
    proposer_task_dim: int | None = None
    proposer_dropout: float = 0.0
    proposer_use_task: bool = True
    enable_action_policy: bool = False
    policy_hidden: int = 768
    policy_layers: int = 6
    policy_heads: int = 8
    policy_chunk_layers: int = 2
    policy_horizon: int | None = None
    policy_task_dim: int | None = None
    policy_max_context: int | None = None
    policy_dropout: float = 0.1
    policy_use_task: bool = True
    policy_lowdim_dim: int = 0
    policy_object_state_dim: int = 0
    policy_plan_state_dim: int = 0
    policy_action_history_len: int = 0
    policy_action_history_dim: int = 7
    policy_use_progress: bool = False
    policy_progress_dim: int = 1
    policy_progress_mode: str = "token"
    policy_enable_local_residual: bool = False
    policy_local_hidden: int = 256
    policy_local_layers: int = 2
    policy_local_residual_scale: float = 1.0
    policy_local_use_lowdim: bool = True
    policy_local_use_plan_state: bool = True
    policy_local_use_progress: bool = True
    policy_local_use_action_history: bool = True
    policy_enable_waypoint_head: bool = False
    policy_waypoint_hidden: int = 256
    policy_waypoint_layers: int = 2
    policy_waypoint_num_stages: int = 4
    policy_waypoint_stage_dim: int = 4
    policy_waypoint_active_stages: tuple[int, ...] = ()
    policy_waypoint_residual_scale: float = 1.0
    policy_waypoint_mode: str = "residual"
    policy_waypoint_use_summary: bool = True
    policy_waypoint_use_lowdim: bool = True
    policy_waypoint_use_plan_state: bool = True
    policy_waypoint_use_progress: bool = True
    policy_waypoint_use_action_history: bool = True
    policy_enable_prior: bool = False
    policy_prior_chunk_layers: int = 1
    enable_bridging: bool = True
    enable_aux_idm: bool = False
    aux_idm_hidden: int = 1024
    aux_idm_layers: int = 3
    enable_world_prior: bool = False
    world_prior_hidden: int = 768
    world_prior_layers: int = 4
    world_prior_heads: int = 8
    world_prior_mlp_mult: int = 4
    world_prior_dropout: float = 0.0
    world_prior_task_dim: int | None = None
    world_prior_action_dim: int = 7
    world_prior_use_context: bool = True
    world_prior_use_action: bool = True
    world_prior_predict_initial: bool = True


class JointWorldModel(nn.Module):
    def __init__(self, cfg: JointConfig):
        super().__init__()
        self.cfg = cfg
        self.dual = DualStreamDynamics(cfg.dual)
        self.action_proj = ActionProjHead(
            z_dim=cfg.dual.action.z_dim,
            hidden=cfg.action_proj_hidden,
            n_layers=cfg.action_proj_layers,
        )
        self.geom = GeomDecoder(
            token_dim=cfg.dual.state.D,
            token_grid=int(cfg.dual.state.P ** 0.5),
            hidden=cfg.geom_hidden,
            enable_extra=cfg.enable_geom_extra,
            upsample_mode=cfg.geom_upsample_mode,
        )
        token_grid = int(cfg.dual.state.P ** 0.5)
        self.pixel = PixelDecoder(
            PixelDecoderConfig(
                token_dim=cfg.dual.state.D,
                token_grid=token_grid,
                hidden=cfg.pixel_hidden,
                n_res=cfg.pixel_n_res,
            )
        ) if cfg.enable_pixel and not cfg.enable_context_pixel else None
        self.context_pixel = ContextResidualPixelDecoder(
            ContextResidualPixelDecoderConfig(
                token_dim=cfg.dual.state.D,
                token_grid=token_grid,
                hidden=cfg.context_pixel_hidden,
                action_dim=cfg.context_pixel_action_dim,
                task_dim=cfg.context_pixel_task_dim or cfg.dual.state.cond_dim,
                residual_scale=cfg.context_pixel_residual_scale,
                use_action=cfg.context_pixel_use_action,
                use_task=cfg.context_pixel_use_task,
                predict_motion=cfg.context_pixel_predict_motion,
                motion_blend_gain=cfg.context_pixel_motion_blend_gain,
            )
        ) if cfg.enable_pixel and cfg.enable_context_pixel else None
        self.control_head = ControlHead(
            ControlHeadConfig(
                token_dim=cfg.dual.state.D,
                hidden=cfg.control_hidden,
                output_size=cfg.control_output_size,
                fuse_size=cfg.control_fuse_size,
                refine_channels=cfg.control_refine_channels,
                use_refine=cfg.control_use_refine,
                action_dim=cfg.control_action_dim,
                task_dim=cfg.control_task_dim or cfg.dual.state.cond_dim,
                use_context=cfg.control_use_context,
                use_action=cfg.control_use_action,
                use_task=cfg.control_use_task,
            )
        ) if cfg.enable_control_head else None
        self.progress_head = ProgressHead(
            ProgressHeadConfig(
                token_dim=cfg.dual.state.D,
                hidden=cfg.progress_hidden,
                n_layers=cfg.progress_layers,
                n_heads=cfg.progress_heads,
                action_dim=cfg.progress_action_dim,
                task_dim=cfg.progress_task_dim or cfg.dual.state.cond_dim,
                max_horizon=cfg.progress_max_horizon,
                use_action=cfg.progress_use_action,
                use_task=cfg.progress_use_task,
            )
        ) if cfg.enable_progress_head else None
        self.action_proposer = ActionProposer(
            ActionProposerConfig(
                token_dim=cfg.dual.state.D,
                task_dim=cfg.proposer_task_dim or cfg.dual.state.cond_dim,
                hidden=cfg.proposer_hidden,
                n_layers=cfg.proposer_layers,
                n_candidates=cfg.proposer_candidates,
                horizon=cfg.proposer_horizon or cfg.dual.state.k,
                dropout=cfg.proposer_dropout,
                use_task=cfg.proposer_use_task,
            )
        ) if cfg.enable_action_proposer else None
        self.action_policy = ActionChunkPolicy(
            ActionChunkPolicyConfig(
                token_dim=cfg.dual.state.D,
                task_dim=cfg.policy_task_dim or cfg.dual.state.cond_dim,
                hidden=cfg.policy_hidden,
                n_layers=cfg.policy_layers,
                n_heads=cfg.policy_heads,
                chunk_layers=cfg.policy_chunk_layers,
                horizon=cfg.policy_horizon or cfg.dual.state.k,
                max_context=cfg.policy_max_context or cfg.dual.state.T,
                dropout=cfg.policy_dropout,
                use_task=cfg.policy_use_task,
                lowdim_dim=cfg.policy_lowdim_dim,
                object_state_dim=cfg.policy_object_state_dim,
                plan_state_dim=cfg.policy_plan_state_dim,
                action_history_len=cfg.policy_action_history_len,
                action_history_dim=cfg.policy_action_history_dim,
                use_progress=cfg.policy_use_progress,
                progress_dim=cfg.policy_progress_dim,
                progress_mode=cfg.policy_progress_mode,
                enable_local_residual=cfg.policy_enable_local_residual,
                local_hidden=cfg.policy_local_hidden,
                local_layers=cfg.policy_local_layers,
                local_residual_scale=cfg.policy_local_residual_scale,
                local_use_lowdim=cfg.policy_local_use_lowdim,
                local_use_plan_state=cfg.policy_local_use_plan_state,
                local_use_progress=cfg.policy_local_use_progress,
                local_use_action_history=cfg.policy_local_use_action_history,
                enable_waypoint_head=cfg.policy_enable_waypoint_head,
                waypoint_hidden=cfg.policy_waypoint_hidden,
                waypoint_layers=cfg.policy_waypoint_layers,
                waypoint_num_stages=cfg.policy_waypoint_num_stages,
                waypoint_stage_dim=cfg.policy_waypoint_stage_dim,
                waypoint_active_stages=tuple(cfg.policy_waypoint_active_stages),
                waypoint_residual_scale=cfg.policy_waypoint_residual_scale,
                waypoint_mode=cfg.policy_waypoint_mode,
                waypoint_use_summary=cfg.policy_waypoint_use_summary,
                waypoint_use_lowdim=cfg.policy_waypoint_use_lowdim,
                waypoint_use_plan_state=cfg.policy_waypoint_use_plan_state,
                waypoint_use_progress=cfg.policy_waypoint_use_progress,
                waypoint_use_action_history=cfg.policy_waypoint_use_action_history,
                enable_prior_policy=cfg.policy_enable_prior,
                prior_chunk_layers=cfg.policy_prior_chunk_layers,
            )
        ) if cfg.enable_action_policy else None
        self.bridging = BridgingAdapter() if cfg.enable_bridging else None
        if cfg.enable_aux_idm:
            self.aux_idm = AuxIDM(AuxIDMConfig(
                token_dim=cfg.dual.state.D,
                n_tokens=cfg.dual.state.P,
                hidden=cfg.aux_idm_hidden,
                n_layers=cfg.aux_idm_layers,
                k=cfg.dual.state.k,
            ))
        else:
            self.aux_idm = None
        self.world_prior = WorldTokenPrior(
            WorldTokenPriorConfig(
                token_dim=cfg.dual.state.D,
                n_tokens=cfg.dual.state.P,
                horizon=cfg.dual.state.k,
                context_frames=cfg.dual.state.T,
                hidden=cfg.world_prior_hidden,
                n_layers=cfg.world_prior_layers,
                n_heads=cfg.world_prior_heads,
                mlp_mult=cfg.world_prior_mlp_mult,
                dropout=cfg.world_prior_dropout,
                task_dim=cfg.world_prior_task_dim or cfg.dual.state.cond_dim,
                action_dim=cfg.world_prior_action_dim,
                use_context=cfg.world_prior_use_context,
                use_action=cfg.world_prior_use_action,
                predict_initial=cfg.world_prior_predict_initial,
            )
        ) if cfg.enable_world_prior else None

    def load_action_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Copy pre-computed per-axis mean[6]/std[6] into ActionProjHead buffers."""
        self.action_proj.mean.copy_(mean.float().view(1, 1, 6))
        self.action_proj.std.copy_(std.float().view(1, 1, 6))

    def _decode_prior_tokens(
        self,
        out: dict,
        prior_tokens: torch.Tensor,
        context_rgb: torch.Tensor | None,
        action_cond: torch.Tensor | None,
        task_emb: torch.Tensor,
        *,
        pixel: bool,
    ) -> None:
        prior_geom = self.geom(prior_tokens)
        out["prior_depth"] = prior_geom["depth"]
        out["prior_hunyuan_tokens"] = prior_tokens
        out["prior_hunyuan_depth"] = prior_geom["depth"]
        if "point" in prior_geom:
            out["prior_point"] = prior_geom["point"]
        if "pose" in prior_geom:
            out["prior_pose_geom"] = prior_geom["pose"]
        if not pixel:
            return
        if self.context_pixel is not None:
            if context_rgb is None:
                context_rgb = prior_tokens.new_zeros(prior_tokens.shape[0], 3, 256, 256)
            if action_cond is None and self.cfg.context_pixel_use_action:
                action_cond = prior_tokens.new_zeros(
                    prior_tokens.shape[0], prior_tokens.shape[1], self.cfg.context_pixel_action_dim
                )
            render = self.context_pixel(
                prior_tokens,
                context_rgb,
                action_cond=action_cond,
                task_emb=task_emb,
                return_aux=False,
            )
            out["prior_rgb"] = render["rgb"] if isinstance(render, dict) else render
        elif self.pixel is not None:
            out["prior_rgb"] = self.pixel(prior_tokens)

    @torch.no_grad()
    def generate_world_prior(
        self,
        c: torch.Tensor,
        context_tokens: torch.Tensor | None = None,
        action_cond: torch.Tensor | None = None,
        context_rgb: torch.Tensor | None = None,
        *,
        steps: int = 8,
        pixel: bool = False,
    ) -> dict:
        if self.world_prior is None:
            raise RuntimeError("generate_world_prior requires enable_world_prior=True")
        out = self.world_prior.sample_flow(
            c,
            context_tokens=context_tokens,
            action_cond=action_cond,
            steps=steps,
        )
        self._decode_prior_tokens(
            out,
            out["prior_future_tokens"],
            context_rgb,
            action_cond,
            c,
            pixel=pixel,
        )
        return out

    def forward(self, s: torch.Tensor, c: torch.Tensor,
                action_cond: torch.Tensor | None = None,
                context_rgb: torch.Tensor | None = None,
                lowdim_state: torch.Tensor | None = None,
                object_state: torch.Tensor | None = None,
                plan_state: torch.Tensor | None = None,
                action_history: torch.Tensor | None = None,
                progress_state: torch.Tensor | None = None,
                prior_clean_tokens: torch.Tensor | None = None,
                pixel: bool = True, bridging: bool = False,
                aux_idm: bool = False) -> dict:
        dual_out = self.dual(s, c, action_cond=action_cond)
        pred = dual_out["pred_tokens"]
        proj = self.action_proj(dual_out["z_a"])
        geom = self.geom(pred)
        out = {
            "pred_tokens": pred,
            "z_a": dual_out["z_a"],
            "pose_norm": proj["pose_norm"],
            "pose": proj["pose"],
            "gripper_logit": proj["gripper_logit"],
            "depth": geom["depth"],
        }
        if "point" in geom:
            out["point"] = geom["point"]
        if "pose" in geom:
            out["pose_geom"] = geom["pose"]
        if pixel and self.context_pixel is not None:
            if context_rgb is None:
                raise ValueError("context_rgb is required when context pixel renderer is enabled")
            render = self.context_pixel(
                pred,
                context_rgb,
                action_cond=action_cond,
                task_emb=c,
                return_aux=self.cfg.context_pixel_predict_motion,
            )
            if isinstance(render, dict):
                out.update(render)
            else:
                out["rgb"] = render
        elif pixel and self.pixel is not None:
            out["rgb"] = self.pixel(pred)
        if self.world_prior is not None:
            prior_out = self.world_prior(
                c,
                context_tokens=s,
                action_cond=action_cond,
                clean_tokens=prior_clean_tokens,
            )
            out.update(prior_out)
            self._decode_prior_tokens(
                out,
                prior_out["prior_future_tokens"],
                context_rgb,
                action_cond,
                c,
                pixel=pixel,
            )
        if self.control_head is not None:
            out.update(self.control_head(
                pred,
                geom["depth"],
                context_rgb=context_rgb,
                action_cond=action_cond,
                task_emb=c,
            ))
        if self.progress_head is not None:
            out.update(self.progress_head(
                pred,
                action_cond=action_cond,
                task_emb=c,
            ))
        if self.action_proposer is not None:
            out.update(self.action_proposer(s, task_emb=c))
        if self.action_policy is not None:
            out.update(self.action_policy(
                s,
                task_emb=c,
                lowdim_state=lowdim_state,
                object_state=object_state,
                plan_state=plan_state,
                action_history=action_history,
                progress_state=progress_state,
            ))
        if bridging and self.bridging is not None:
            out["cosmos_depth_input"] = self.bridging(geom["depth"])
        if aux_idm and self.aux_idm is not None:
            s_last = s[:, -1]
            pred_last = pred[:, -1]
            aux = self.aux_idm(s_last, pred_last)
            out["aux_pose_norm"] = aux["aux_pose_norm"]
            out["aux_grip"] = aux["aux_grip"]
        return out

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
