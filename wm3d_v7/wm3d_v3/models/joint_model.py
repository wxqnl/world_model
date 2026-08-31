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
from .future_value_head import FutureValueConfig, FutureValueHead
from .action_proposer import ActionProposer, ActionProposerConfig
from .action_policy import ActionChunkPolicy, ActionChunkPolicyConfig
from .bridging_adapter import BridgingAdapter
from .aux_idm import AuxIDM, AuxIDMConfig
from .multiview_fuser import MultiViewFuserConfig, MultiViewTokenFuser
from .token_codec import PCATokenCodec, TokenCodecConfig


@dataclass
class JointConfig:
    dual: DualConfig = field(default_factory=DualConfig)
    enable_multiview_fuser: bool = False
    multiview_heads: int = 16
    multiview_dropout: float = 0.0
    multiview_use_camera_pose: bool = True
    multiview_pose_dim: int = 16
    enable_token_codec: bool = False
    token_codec_latent_dim: int = 384
    token_codec_checkpoint: str | None = None
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
    enable_future_value: bool = False
    future_value_hidden: int = 256
    future_value_layers: int = 2
    future_value_heads: int = 4
    future_value_task_dim: int | None = None
    future_value_max_horizon: int = 32
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
    policy_patch_pool: str = "mean"
    policy_max_spatial_tokens: int = 64
    policy_context_source: str = "input"
    policy_core_action_cond: str = "same"
    policy_use_context_rgb: bool = False
    policy_rgb_spatial_tokens: int = 64
    policy_lowdim_dim: int = 0
    policy_object_state_dim: int = 0
    policy_plan_state_dim: int = 0
    policy_action_history_len: int = 0
    policy_action_history_dim: int = 7
    policy_action_history_as_token: bool = True
    policy_grip_history_adapter: bool = False
    policy_grip_history_hidden: int = 128
    policy_grip_history_zero_init: bool = True
    policy_enable_grip_delta_head: bool = False
    policy_grip_delta_hidden: int = 256
    policy_grip_delta_zero_init: bool = True
    policy_grip_delta_use_composed_action_cond: bool = False
    policy_grip_delta_soft_compose_action_cond: bool = False
    policy_grip_delta_straight_through_action_cond: bool = False
    policy_grip_owner: str = "auto"
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
    policy_action_add_trunk: bool = True
    policy_zero_init_output: bool = False
    policy_enable_flow_head: bool = False
    policy_flow_use_as_policy: bool = False
    policy_flow_layers: int = 2
    policy_flow_hidden: int = 768
    policy_flow_action_dim: int = 7
    policy_flow_default_steps: int = 8
    policy_flow_noise_scale: float = 1.0
    policy_flow_zero_init_output: bool = False
    policy_head_type: str = "native"
    policy_oft_max_horizon: int = 16
    policy_oft_query_layers: int = 2
    policy_oft_mlp_hidden: int = 0
    policy_oft_adapter_name: str = "canonical_7d"
    policy_oft_action_dim: int = 7
    policy_oft_grip_indices: tuple[int, ...] = (6,)
    policy_oft_normalization_version: str = "wm3d_d7_norm_v1"
    policy_oft_grip_loss: str = "bce_logits"
    policy_oft_grip_threshold: float = 0.5
    policy_oft_adapters: tuple[dict[str, object], ...] = ()
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
        self.multiview_fuser = (
            MultiViewTokenFuser(
                MultiViewFuserConfig(
                    token_dim=cfg.dual.state.D,
                    n_heads=cfg.multiview_heads,
                    dropout=cfg.multiview_dropout,
                    use_camera_pose=cfg.multiview_use_camera_pose,
                    pose_dim=cfg.multiview_pose_dim,
                )
            )
            if cfg.enable_multiview_fuser
            else None
        )
        self.token_codec = (
            PCATokenCodec(
                TokenCodecConfig(
                    token_dim=cfg.dual.state.D,
                    latent_dim=cfg.token_codec_latent_dim,
                )
            )
            if cfg.enable_token_codec
            else None
        )
        if self.token_codec is not None and cfg.token_codec_checkpoint:
            self.load_token_codec(cfg.token_codec_checkpoint)
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
        self.future_value_head = FutureValueHead(
            FutureValueConfig(
                token_dim=cfg.dual.state.D,
                task_dim=cfg.future_value_task_dim or cfg.dual.state.cond_dim,
                hidden=cfg.future_value_hidden,
                n_layers=cfg.future_value_layers,
                n_heads=cfg.future_value_heads,
                max_horizon=cfg.future_value_max_horizon,
            )
        ) if cfg.enable_future_value else None
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
                patch_pool=cfg.policy_patch_pool,
                max_spatial_tokens=cfg.policy_max_spatial_tokens,
                use_context_rgb=cfg.policy_use_context_rgb,
                rgb_spatial_tokens=cfg.policy_rgb_spatial_tokens,
                lowdim_dim=cfg.policy_lowdim_dim,
                object_state_dim=cfg.policy_object_state_dim,
                plan_state_dim=cfg.policy_plan_state_dim,
                action_history_len=cfg.policy_action_history_len,
                action_history_dim=cfg.policy_action_history_dim,
                action_history_as_token=cfg.policy_action_history_as_token,
                grip_history_adapter=cfg.policy_grip_history_adapter,
                grip_history_hidden=cfg.policy_grip_history_hidden,
                grip_history_zero_init=cfg.policy_grip_history_zero_init,
                enable_grip_delta_head=cfg.policy_enable_grip_delta_head,
                grip_delta_hidden=cfg.policy_grip_delta_hidden,
                grip_delta_zero_init=cfg.policy_grip_delta_zero_init,
                grip_delta_use_composed_action_cond=cfg.policy_grip_delta_use_composed_action_cond,
                grip_delta_soft_compose_action_cond=cfg.policy_grip_delta_soft_compose_action_cond,
                grip_delta_straight_through_action_cond=cfg.policy_grip_delta_straight_through_action_cond,
                grip_owner=cfg.policy_grip_owner,
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
                zero_init_output=cfg.policy_zero_init_output,
                enable_flow_head=cfg.policy_enable_flow_head,
                flow_use_as_policy=cfg.policy_flow_use_as_policy,
                flow_layers=cfg.policy_flow_layers,
                flow_hidden=cfg.policy_flow_hidden,
                flow_action_dim=cfg.policy_flow_action_dim,
                flow_default_steps=cfg.policy_flow_default_steps,
                flow_noise_scale=cfg.policy_flow_noise_scale,
                flow_zero_init_output=cfg.policy_flow_zero_init_output,
                head_type=cfg.policy_head_type,
                oft_max_horizon=cfg.policy_oft_max_horizon,
                oft_query_layers=cfg.policy_oft_query_layers,
                oft_mlp_hidden=cfg.policy_oft_mlp_hidden,
                oft_adapter_name=cfg.policy_oft_adapter_name,
                oft_action_dim=cfg.policy_oft_action_dim,
                oft_grip_indices=tuple(cfg.policy_oft_grip_indices),
                oft_normalization_version=cfg.policy_oft_normalization_version,
                oft_grip_loss=cfg.policy_oft_grip_loss,
                oft_grip_threshold=cfg.policy_oft_grip_threshold,
                oft_adapters=tuple(cfg.policy_oft_adapters),
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

    def load_action_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Copy pre-computed per-axis mean[6]/std[6] into ActionProjHead buffers."""
        self.action_proj.mean.copy_(mean.float().view(1, 1, 6))
        self.action_proj.std.copy_(std.float().view(1, 1, 6))

    def load_token_codec(self, checkpoint: str) -> None:
        if self.token_codec is None:
            raise RuntimeError("enable_token_codec must be true before loading a codec")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.token_codec.set_basis(payload["mean"], payload["components"])
        for parameter in self.token_codec.parameters():
            parameter.requires_grad_(False)

    def decode_input_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.shape[-1] == self.cfg.dual.state.D:
            return tokens
        if self.token_codec is None:
            raise RuntimeError(
                f"received compressed D={tokens.shape[-1]} tokens but token codec is disabled"
            )
        return self.token_codec.decode(tokens)

    def fuse_views(
        self,
        anchor_tokens: torch.Tensor,
        wrist_tokens: torch.Tensor | None = None,
        *,
        view_mask: torch.Tensor | None = None,
        anchor_camera_pose: torch.Tensor | None = None,
        wrist_camera_pose: torch.Tensor | None = None,
    ) -> torch.Tensor:
        anchor_tokens = self.decode_input_tokens(anchor_tokens)
        if wrist_tokens is None:
            return anchor_tokens
        wrist_tokens = self.decode_input_tokens(wrist_tokens)
        if self.multiview_fuser is None:
            raise RuntimeError("wrist tokens were supplied but multiview fuser is disabled")
        return self.multiview_fuser(
            anchor_tokens,
            wrist_tokens,
            view_mask=view_mask,
            anchor_camera_pose=anchor_camera_pose,
            wrist_camera_pose=wrist_camera_pose,
        )

    def imagine_candidates(
        self,
        s: torch.Tensor,
        c: torch.Tensor,
        candidate_actions: torch.Tensor,
        *,
        include_geometry: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Vectorized native rollout for candidate actions ``[B,K,H,7]``."""
        if s.ndim != 4 or c.ndim != 2 or candidate_actions.ndim != 4:
            raise ValueError("expected s[B,T,P,D], c[B,D], candidate_actions[B,K,H,7]")
        s = self.decode_input_tokens(s)
        bsz, candidates, horizon, action_dim = candidate_actions.shape
        if bsz != s.shape[0] or bsz != c.shape[0]:
            raise ValueError("batch dimensions must match")
        if candidates < 2:
            raise ValueError("candidate imagination requires K>=2")
        if action_dim != self.cfg.dual.state.action_cond_dim:
            raise ValueError(
                f"expected action dim {self.cfg.dual.state.action_cond_dim}, got {action_dim}"
            )
        core_horizon = int(self.cfg.dual.state.k)
        if horizon > core_horizon:
            candidate_actions = candidate_actions[:, :, :core_horizon]
        elif horizon < core_horizon:
            pad = candidate_actions[:, :, -1:].expand(-1, -1, core_horizon - horizon, -1)
            candidate_actions = torch.cat((candidate_actions, pad), dim=2)
        expanded_s = s[:, None].expand(-1, candidates, -1, -1, -1).reshape(
            bsz * candidates, *s.shape[1:]
        )
        expanded_c = c[:, None].expand(-1, candidates, -1).reshape(bsz * candidates, -1)
        flat_actions = candidate_actions.reshape(bsz * candidates, core_horizon, action_dim)
        dual_out = self.dual(expanded_s, expanded_c, action_cond=flat_actions)

        def unflatten(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.reshape(bsz, candidates, *tensor.shape[1:])

        result = {
            "pred_tokens": unflatten(dual_out["pred_tokens"]),
            "z_a": unflatten(dual_out["z_a"]),
            "candidate_action_cond": candidate_actions,
        }
        projected = self.action_proj(dual_out["z_a"])
        result.update({name: unflatten(value) for name, value in projected.items()})
        if include_geometry:
            geometry = self.geom(dual_out["pred_tokens"])
            result.update({name: unflatten(value) for name, value in geometry.items()})
        return result

    def forward(self, s: torch.Tensor, c: torch.Tensor,
                action_cond: torch.Tensor | None = None,
                context_rgb: torch.Tensor | None = None,
                lowdim_state: torch.Tensor | None = None,
                object_state: torch.Tensor | None = None,
                plan_state: torch.Tensor | None = None,
                action_history: torch.Tensor | None = None,
                progress_state: torch.Tensor | None = None,
                flow_action: torch.Tensor | None = None,
                flow_t: torch.Tensor | None = None,
                flow_sample: bool | None = None,
                flow_sample_steps: int | None = None,
                flow_noise: torch.Tensor | None = None,
                flow_noise_scale: float | None = None,
                oft_adapter_name: str | None = None,
                oft_horizon: int | None = None,
                oft_state: torch.Tensor | None = None,
                oft_action_history: torch.Tensor | None = None,
                prior_clean_tokens: torch.Tensor | None = None,
                pixel: bool = True, bridging: bool = False,
                aux_idm: bool = False,
                return_rgb_features: bool = False,
                skip_action_proposer: bool = False,
                skip_action_policy: bool = False,
                skip_native_prediction_heads: bool = False,
                detach_progress_input: bool = False,
                wrist_s: torch.Tensor | None = None,
                view_mask: torch.Tensor | None = None,
                anchor_camera_pose: torch.Tensor | None = None,
                wrist_camera_pose: torch.Tensor | None = None,
                candidate_actions: torch.Tensor | None = None,
                candidate_include_geometry: bool = False,
                native_action_no_teacher: bool = False) -> dict:
        s = self.fuse_views(
            s,
            wrist_s,
            view_mask=view_mask,
            anchor_camera_pose=anchor_camera_pose,
            wrist_camera_pose=wrist_camera_pose,
        )
        core_action_cond = action_cond
        if core_action_cond is not None:
            core_horizon = int(self.cfg.dual.state.k)
            if int(core_action_cond.shape[1]) > core_horizon:
                core_action_cond = core_action_cond[:, :core_horizon]
            elif int(core_action_cond.shape[1]) < core_horizon:
                pad = core_action_cond[:, -1:].expand(
                    -1,
                    core_horizon - int(core_action_cond.shape[1]),
                    -1,
                )
                core_action_cond = torch.cat([core_action_cond, pad], dim=1)
        dual_out = self.dual(s, c, action_cond=core_action_cond)
        pred = dual_out["pred_tokens"]
        out = {
            "pred_tokens": pred,
            "z_a": dual_out["z_a"],
        }
        geom = None
        if not skip_native_prediction_heads:
            proj = self.action_proj(dual_out["z_a"])
            geom = self.geom(pred)
            out.update({
                "pose_norm": proj["pose_norm"],
                "pose": proj["pose"],
                "gripper_logit": proj["gripper_logit"],
                "depth": geom["depth"],
            })
            if "point" in geom:
                out["point"] = geom["point"]
            if "pose" in geom:
                out["pose_geom"] = geom["pose"]
        if native_action_no_teacher:
            # Keep the main world rollout action-conditioned while training the
            # native state+task -> action path in a second internal pass.  This
            # avoids turning factual world prediction into an action-ambiguous
            # average future merely to obtain a behavior-cloning signal.
            no_teacher_dual = (
                dual_out
                if core_action_cond is None
                else self.dual(s, c, action_cond=None)
            )
            no_teacher_action = self.action_proj(no_teacher_dual["z_a"])
            out.update(
                {
                    "native_action_no_teacher_pose_norm": no_teacher_action[
                        "pose_norm"
                    ],
                    "native_action_no_teacher_pose": no_teacher_action["pose"],
                    "native_action_no_teacher_gripper_logit": no_teacher_action[
                        "gripper_logit"
                    ],
                    "native_action_no_teacher_z_a": no_teacher_dual["z_a"],
                }
            )
        if pixel and self.context_pixel is not None:
            if context_rgb is None:
                raise ValueError("context_rgb is required when context pixel renderer is enabled")
            render = self.context_pixel(
                pred,
                context_rgb,
                action_cond=core_action_cond,
                task_emb=c,
                return_aux=self.cfg.context_pixel_predict_motion,
                return_features=return_rgb_features,
            )
            if isinstance(render, dict):
                out.update(render)
            else:
                out["rgb"] = render
        elif pixel and self.pixel is not None:
            render = self.pixel(pred, return_features=return_rgb_features)
            if isinstance(render, dict):
                out.update(render)
            else:
                out["rgb"] = render
        if self.control_head is not None and geom is not None:
            out.update(self.control_head(
                pred,
                geom["depth"],
                context_rgb=context_rgb,
                action_cond=core_action_cond,
                task_emb=c,
            ))
        if self.progress_head is not None:
            out.update(self.progress_head(
                pred.detach() if detach_progress_input else pred,
                action_cond=(
                    core_action_cond.detach()
                    if detach_progress_input and core_action_cond is not None
                    else core_action_cond
                ),
                task_emb=c.detach() if detach_progress_input else c,
            ))
        if self.action_proposer is not None and not skip_action_proposer:
            out.update(self.action_proposer(s, task_emb=c))
        if self.action_policy is not None and not skip_action_policy:
            policy_source = str(getattr(self.cfg, "policy_context_source", "input")).strip().lower()
            if policy_source in {"", "input", "s", "tokens", "cached"}:
                policy_tokens = s
            elif policy_source in {"core", "core_pred", "pred", "pred_tokens", "serving"}:
                policy_core_action_mode = str(getattr(self.cfg, "policy_core_action_cond", "same")).strip().lower()
                if policy_core_action_mode in {"", "same", "teacher", "gt", "action", "action_cond"}:
                    policy_tokens = pred
                elif policy_core_action_mode in {"none", "no_action", "off", "disabled"}:
                    if core_action_cond is None:
                        policy_tokens = pred
                    else:
                        policy_tokens = self.dual(s, c, action_cond=None)["pred_tokens"]
                else:
                    raise ValueError(f"unsupported policy_core_action_cond={policy_core_action_mode!r}")
            elif policy_source in {"core_detach", "core_pred_detach", "pred_detach", "pred_tokens_detach", "serving_detach"}:
                policy_core_action_mode = str(getattr(self.cfg, "policy_core_action_cond", "same")).strip().lower()
                if policy_core_action_mode in {"", "same", "teacher", "gt", "action", "action_cond"}:
                    policy_tokens = pred.detach()
                elif policy_core_action_mode in {"none", "no_action", "off", "disabled"}:
                    if core_action_cond is None:
                        policy_tokens = pred.detach()
                    else:
                        with torch.no_grad():
                            policy_tokens = self.dual(s, c, action_cond=None)["pred_tokens"].detach()
                else:
                    raise ValueError(f"unsupported policy_core_action_cond={policy_core_action_mode!r}")
            else:
                raise ValueError(f"unsupported policy_context_source={policy_source!r}")
            out["policy_context_tokens"] = policy_tokens
            out.update(self.action_policy(
                policy_tokens,
                task_emb=c,
                lowdim_state=lowdim_state,
                object_state=object_state,
                plan_state=plan_state,
                action_history=action_history,
                progress_state=progress_state,
                context_rgb=context_rgb,
                flow_action=flow_action,
                flow_t=flow_t,
                flow_sample=flow_sample,
                flow_sample_steps=flow_sample_steps,
                flow_noise=flow_noise,
                flow_noise_scale=flow_noise_scale,
                oft_adapter_name=oft_adapter_name,
                oft_horizon=oft_horizon,
                oft_state=oft_state,
                oft_action_history=oft_action_history,
            ))
        if bridging and self.bridging is not None:
            if geom is None:
                raise RuntimeError("bridging requires native geometry prediction heads")
            out["cosmos_depth_input"] = self.bridging(geom["depth"])
        if aux_idm and self.aux_idm is not None:
            s_last = s[:, -1]
            pred_last = pred[:, -1]
            aux = self.aux_idm(s_last, pred_last)
            out["aux_pose_norm"] = aux["aux_pose_norm"]
            out["aux_grip"] = aux["aux_grip"]
        if candidate_actions is not None:
            candidate_out = self.imagine_candidates(
                s,
                c,
                candidate_actions,
                include_geometry=candidate_include_geometry,
            )
            for name, value in candidate_out.items():
                out[f"candidate_{name}"] = value
            if self.future_value_head is not None:
                # The value head performs an architectural stop-gradient on
                # candidate futures: value/ranking supervision cannot mutate
                # dynamics. Joint S1 may still update dynamics intentionally
                # through audited true-branch reconstruction/effect losses.
                out.update(self.future_value_head(candidate_out["pred_tokens"], c))
        return out

    def act_policy(
        self,
        s: torch.Tensor,
        c: torch.Tensor,
        *,
        lowdim_state: torch.Tensor | None = None,
        object_state: torch.Tensor | None = None,
        plan_state: torch.Tensor | None = None,
        action_history: torch.Tensor | None = None,
        progress_state: torch.Tensor | None = None,
        context_rgb: torch.Tensor | None = None,
        action_cond: torch.Tensor | None = None,
        flow_sample: bool | None = None,
        flow_sample_steps: int | None = None,
        flow_noise: torch.Tensor | None = None,
        flow_noise_scale: float | None = None,
        oft_adapter_name: str | None = None,
        oft_horizon: int | None = None,
        oft_state: torch.Tensor | None = None,
        oft_action_history: torch.Tensor | None = None,
        wrist_s: torch.Tensor | None = None,
        view_mask: torch.Tensor | None = None,
        anchor_camera_pose: torch.Tensor | None = None,
        wrist_camera_pose: torch.Tensor | None = None,
    ) -> dict:
        """Run the closed-loop action policy path used by LIBERO serving."""
        if self.action_policy is None:
            raise RuntimeError("act_policy requires enable_action_policy=True")
        s = self.fuse_views(
            s,
            wrist_s,
            view_mask=view_mask,
            anchor_camera_pose=anchor_camera_pose,
            wrist_camera_pose=wrist_camera_pose,
        )
        dual_out = self.dual(s, c, action_cond=action_cond)
        policy_source = str(getattr(self.cfg, "policy_context_source", "input")).strip().lower()
        policy_dual_out = dual_out
        if policy_source in {"", "input", "s", "tokens", "cached"}:
            policy_tokens = s
        elif policy_source in {"core", "core_pred", "pred", "pred_tokens", "serving"}:
            policy_core_action_mode = str(getattr(self.cfg, "policy_core_action_cond", "same")).strip().lower()
            if policy_core_action_mode in {"", "same", "teacher", "gt", "action", "action_cond"}:
                policy_tokens = dual_out["pred_tokens"]
            elif policy_core_action_mode in {"none", "no_action", "off", "disabled"}:
                if action_cond is None:
                    policy_tokens = dual_out["pred_tokens"]
                else:
                    policy_dual_out = self.dual(s, c, action_cond=None)
                    policy_tokens = policy_dual_out["pred_tokens"]
            else:
                raise ValueError(f"unsupported policy_core_action_cond={policy_core_action_mode!r}")
        elif policy_source in {"core_detach", "core_pred_detach", "pred_detach", "pred_tokens_detach", "serving_detach"}:
            policy_core_action_mode = str(getattr(self.cfg, "policy_core_action_cond", "same")).strip().lower()
            if policy_core_action_mode in {"", "same", "teacher", "gt", "action", "action_cond"}:
                policy_tokens = dual_out["pred_tokens"].detach()
            elif policy_core_action_mode in {"none", "no_action", "off", "disabled"}:
                if action_cond is None:
                    policy_tokens = dual_out["pred_tokens"].detach()
                else:
                    with torch.no_grad():
                        policy_dual_out = self.dual(s, c, action_cond=None)
                    policy_tokens = policy_dual_out["pred_tokens"].detach()
            else:
                raise ValueError(f"unsupported policy_core_action_cond={policy_core_action_mode!r}")
        else:
            raise ValueError(f"unsupported policy_context_source={policy_source!r}")
        proj = self.action_proj(policy_dual_out["z_a"])
        pol = self.action_policy(
            policy_tokens,
            task_emb=c,
            lowdim_state=lowdim_state,
            object_state=object_state,
            plan_state=plan_state,
            action_history=action_history,
            progress_state=progress_state,
            context_rgb=context_rgb,
            flow_sample=flow_sample,
            flow_sample_steps=flow_sample_steps,
            flow_noise=flow_noise,
            flow_noise_scale=flow_noise_scale,
            oft_adapter_name=oft_adapter_name,
            oft_horizon=oft_horizon,
            oft_state=oft_state,
            oft_action_history=oft_action_history,
        )

        if "policy_pose_norm" not in pol:
            if oft_adapter_name is None:
                raise RuntimeError("noncanonical OFT output requires an explicit adapter name")
            return pol

        def match_horizon(value: torch.Tensor, horizon: int) -> torch.Tensor:
            if value.shape[1] == horizon:
                return value
            if value.shape[1] > horizon:
                return value[:, :horizon]
            pad = value[:, -1:].expand(-1, horizon - value.shape[1], *value.shape[2:])
            return torch.cat([value, pad], dim=1)

        policy_horizon = int(pol["policy_pose_norm"].shape[1])
        trunk_pose_norm = match_horizon(proj["pose_norm"], policy_horizon)
        trunk_gripper_logit = match_horizon(proj["gripper_logit"], policy_horizon)
        if self.cfg.policy_action_add_trunk:
            pose_norm = trunk_pose_norm + pol["policy_pose_norm"]
            gripper_logit = trunk_gripper_logit + pol["policy_gripper_logit"]
            action_cond_out = torch.cat(
                [pose_norm, torch.sigmoid(gripper_logit)[..., None]],
                dim=-1,
            )
        else:
            pose_norm = pol["policy_pose_norm"]
            gripper_logit = pol["policy_gripper_logit"]
            action_cond_out = pol.get("policy_action_cond")
            if action_cond_out is None:
                action_cond_out = torch.cat(
                    [pose_norm, torch.sigmoid(gripper_logit)[..., None]],
                    dim=-1,
                )
        out = dict(pol)
        out["policy_pose_norm"] = pose_norm
        out["policy_gripper_logit"] = gripper_logit
        out["policy_action_cond"] = action_cond_out
        out["trunk_pose_norm"] = trunk_pose_norm
        out["trunk_gripper_logit"] = trunk_gripper_logit
        out["trunk_pose"] = proj["pose"]
        return out

    def register_oft_adapter(
        self,
        name: str,
        *,
        action_dim: int,
        grip_indices: tuple[int, ...] = (),
        state_dim: int = 0,
        history_dim: int = 0,
        history_len: int = 0,
        normalization_version: str = "identity_v1",
        grip_loss: str = "bce_logits",
        grip_threshold: float = 0.5,
    ) -> None:
        if self.action_policy is None:
            raise RuntimeError("register_oft_adapter requires enable_action_policy=True")
        self.action_policy.register_oft_adapter(
            name,
            action_dim=action_dim,
            grip_indices=tuple(grip_indices),
            state_dim=state_dim,
            history_dim=history_dim,
            history_len=history_len,
            normalization_version=normalization_version,
            grip_loss=grip_loss,
            grip_threshold=grip_threshold,
        )

    def action_policy_checkpoint_contract(self) -> dict[str, object]:
        if self.action_policy is None:
            raise RuntimeError("action policy contract requires enable_action_policy=True")
        contract = dict(self.action_policy.checkpoint_contract())
        contract["joint_behavior"] = {
            "policy_context_source": str(self.cfg.policy_context_source),
            "policy_core_action_cond": str(self.cfg.policy_core_action_cond),
            "policy_action_add_trunk": bool(self.cfg.policy_action_add_trunk),
        }
        return contract

    def act_oft(
        self,
        s: torch.Tensor,
        c: torch.Tensor,
        *,
        adapter_name: str,
        horizon: int,
        **policy_kwargs,
    ) -> dict:
        """Explicit noncanonical OFT entrypoint for benchmark-specific adapters."""
        out = self.act_policy(
            s,
            c,
            oft_adapter_name=adapter_name,
            oft_horizon=horizon,
            **policy_kwargs,
        )
        if "policy_oft_actions" not in out:
            raise RuntimeError(f"OFT adapter {adapter_name!r} did not return generic actions")
        return out

    def load_oft_benchmark_state_dict(
        self,
        source_state: dict[str, torch.Tensor],
        source_contract: dict[str, object],
    ) -> dict[str, list[str]]:
        """Strictly inherit WM3D-OFT while initializing only new typed adapters."""
        if self.action_policy is None or self.action_policy.oft_head is None:
            raise RuntimeError("benchmark OFT load requires an OFT action policy")
        target_contract = self.action_policy_checkpoint_contract()
        fixed_fields = (
            "version",
            "head_type",
            "horizon",
            "max_horizon",
            "default_adapter",
            "trunk_feature_dim",
            "context_schema",
            "joint_behavior",
        )
        mismatched_contract = {
            key: {"source": source_contract.get(key), "target": target_contract.get(key)}
            for key in fixed_fields
            if source_contract.get(key) != target_contract.get(key)
        }
        source_adapters = source_contract.get("adapters")
        target_adapters = target_contract.get("adapters")
        if not isinstance(source_adapters, dict) or not isinstance(target_adapters, dict):
            raise RuntimeError("OFT checkpoint is missing typed adapter manifests")
        for name, spec in source_adapters.items():
            if target_adapters.get(name) != spec:
                mismatched_contract[f"adapter:{name}"] = {
                    "source": spec,
                    "target": target_adapters.get(name),
                }
        if mismatched_contract:
            raise RuntimeError(f"OFT benchmark contract mismatch: {mismatched_contract}")

        target_state = self.state_dict()
        unexpected = sorted(set(source_state).difference(target_state))
        shape_mismatch = sorted(
            key
            for key in set(source_state).intersection(target_state)
            if source_state[key].shape != target_state[key].shape
        )
        new_adapters = sorted(set(target_adapters).difference(source_adapters))
        allowed_prefixes = tuple(
            f"action_policy.oft_head.adapters.{name}." for name in new_adapters
        )
        missing = sorted(set(target_state).difference(source_state))
        unauthorized_missing = [
            key for key in missing if not any(key.startswith(prefix) for prefix in allowed_prefixes)
        ]
        if unexpected or shape_mismatch or unauthorized_missing:
            raise RuntimeError(
                "OFT benchmark state load rejected differences: "
                f"unexpected={unexpected} shape_mismatch={shape_mismatch} "
                f"unauthorized_missing={unauthorized_missing}"
            )
        complete = {
            key: source_state[key] if key in source_state else value
            for key, value in target_state.items()
        }
        self.load_state_dict(complete, strict=True)
        return {
            "loaded": sorted(source_state),
            "initialized_adapter_keys": missing,
            "new_adapters": new_adapters,
        }

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
