"""WM3D native trajectories on the upstream WSA MoT/flow training pipeline.

Only generic Wan priors are loaded. Narrow expert backbones use the upstream
Wan interpolation/alpha initialization; robot and native-state interfaces are
fresh. The K trajectory states participate in mixed attention, rather than
being an auxiliary rollout computed after RGB/action have finished.
"""
from __future__ import annotations

import types
from typing import Any, Mapping

import torch
from torch import nn
import torch.nn.functional as F

from .foundation_world_model import (
    ActionTimeConditioner, FoundationModelError, FoundationWorldModel,
    FoundationWorldModelConfig, Native3DStateConditioner, WSABackendAdapter,
)


class TrajectoryStateConditioner(Native3DStateConditioner):
    """Expand observed spatial states to view-major [view, horizon, patch]."""

    def __init__(self, config: FoundationWorldModelConfig, hidden_dim: int):
        super().__init__(token_dim=config.token_dim, hidden_dim=hidden_dim,
                         num_views=config.num_views, tokens_per_view=config.P)
        self.horizons = config.K
        self.views = config.num_views
        self.patches = config.P
        # There is no trained robot/native-state interface prior to preserve.
        nn.init.ones_(self.gate)

    def forward(self, current: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        observed = super().forward(current, mask)
        b, _, d = observed.shape
        return observed.view(b, self.views, 1, self.patches, d).expand(
            -1, -1, self.horizons, -1, -1
        ).reshape(b, self.views * self.horizons * self.patches, d)


class TrajectoryReadout(nn.Module):
    """One-to-one readout: never resample or mix horizon/view identities."""

    def __init__(self, hidden_dim: int, token_dim: int):
        super().__init__()
        self.output_proj = nn.Linear(hidden_dim, token_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.output_proj(tokens)


def trajectory_loss(
    predictions: list[torch.Tensor], target: torch.Tensor,
    valid: torch.Tensor, current: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Upstream cosine + normalized-MSE objective, over all valid K states.

    Inputs are [B,V,K,P,D]; labels are detached here as well as at the frozen
    VGGT boundary. Temporal metrics are diagnostic, not extra loss terms.
    """
    if target.ndim != 5 or valid.shape != target.shape[:3] or not predictions:
        raise FoundationModelError("native trajectory target/mask shape mismatch")
    if current.shape != (target.shape[0], target.shape[1], target.shape[3], target.shape[4]):
        raise FoundationModelError("native trajectory current-state shape mismatch")
    mask = valid.bool().unsqueeze(-1).expand(target.shape[:-1])
    if not bool(mask.any()):
        raise FoundationModelError("native trajectory batch has no valid targets")
    if not bool(torch.isfinite(target[mask]).all()):
        raise FoundationModelError("native trajectory target is non-finite")
    truth = target.detach().float()
    truth_ln = F.layer_norm(truth, (truth.shape[-1],))
    losses = []
    for pred in predictions:
        if pred.shape != target.shape:
            raise FoundationModelError("native trajectory output layout changed")
        p, t = pred[mask].float(), truth[mask]
        losses.append((1 - F.cosine_similarity(p, t, dim=-1)).mean()
                      + F.mse_loss(F.layer_norm(p, (p.shape[-1],)), truth_ln[mask]))
    loss = torch.stack(losses).mean()
    with torch.no_grad():
        pred_ln = F.layer_norm(predictions[-1].float(), (truth.shape[-1],))
        anchor = F.layer_norm(current.detach().float(), (truth.shape[-1],))[:, :, None]
        pd, td = pred_ln - anchor, truth_ln - anchor
        logs = {}
        for h in range(target.shape[2]):
            hm = mask[:, :, h]
            if not bool(hm.any()):
                continue
            p, t = pd[:, :, h][hm], td[:, :, h][hm]
            moving = t.square().mean(-1) > 1e-8
            prefix = f"native_3d_h{h + 1}"
            logs[prefix + "_delta_error"] = (p - t).square().mean().sqrt()
            logs[prefix + "_copy_error"] = t.square().mean().sqrt()
            logs[prefix + "_amplitude_ratio"] = p.square().mean().sqrt() / t.square().mean().sqrt().clamp_min(1e-8)
            logs[prefix + "_direction"] = (
                F.cosine_similarity(p[moving], t[moving], dim=-1).mean()
                if bool(moving.any()) else p.new_zeros(())
            )
    return loss, logs


class NativeTrajectoryBackend(WSABackendAdapter):
    """Reuse WSA training_loss and samplers; replace only its 3D contract."""

    def __init__(self, wsa_model: nn.Module, *, config: FoundationWorldModelConfig):
        nn.Module.__init__(self)
        self.wsa_model = wsa_model
        self.cfg = config
        expert = wsa_model.future_3d_expert
        expected = config.num_views * config.K * config.P
        if expert.num_query_tokens != expected:
            raise FoundationModelError("native trajectory query layout changed")
        self.action_time = ActionTimeConditioner(
            wsa_model.action_expert.hidden_dim, config.action_time_fourier_dim)
        self.native_3d_state = TrajectoryStateConditioner(config, expert.hidden_dim)
        self.native_3d_rollout = nn.Identity()  # No second dynamics owner/loss.
        expert.output_decoder = TrajectoryReadout(expert.hidden_dim, config.token_dim)
        del expert.output_queries
        self._active_time_update = None
        self._active_native_3d_update = None
        self._targets = None
        # Upstream uses presence of a teacher to collect intermediate states.
        # The callback below owns VGGT supervision; no DA3 model is loaded.
        wsa_model.da3_teacher = TrajectoryObjective()
        self._time_hook = wsa_model.action_expert.action_encoder.register_forward_hook(self._inject_action_time)
        base_pre_dit = expert.pre_dit

        def pre_dit(_expert, *args, **kwargs):
            pre = dict(base_pre_dit(*args, **kwargs))
            update = self._active_native_3d_update
            if update is None or update.shape != pre["tokens"].shape:
                raise FoundationModelError("native observation was not bound to trajectory queries")
            pre["tokens"] = pre["tokens"] + update.to(pre["tokens"])
            return pre

        def project(_expert, layers):
            if len(layers) != len(_expert.layer_norms):
                raise FoundationModelError("native query-layer count changed")
            return [
                _expert.output_decoder(norm(tokens)).reshape(
                    tokens.shape[0], config.num_views, config.K, config.P, config.token_dim)
                for norm, tokens in zip(_expert.layer_norms, layers, strict=True)
            ]

        def compute_loss(_model, future_3d_layer_tokens, future_images, img_masks, collect_logs=True):
            if self._targets is None or future_3d_layer_tokens is None:
                raise FoundationModelError("native trajectory supervision was not bound")
            target, valid, current = self._targets
            loss, metrics = _model.da3_teacher(expert.project_query_layers(future_3d_layer_tokens), target, valid, current)
            return loss, metrics if collect_logs else {}

        expert.pre_dit = types.MethodType(pre_dit, expert)
        expert.project_query_layers = types.MethodType(project, expert)
        wsa_model._compute_future_3d_loss = types.MethodType(compute_loss, wsa_model)
        wsa_model.loss_lambda_3d = float(config.native_3d_loss_weight)
        self._last_native_state = None

        def capture_state(_module, _args, output):
            if self._targets is not None or not torch.is_inference_mode_enabled():
                return
            states = output[0] if isinstance(output, tuple) else output
            tokens = states["future_3d"]
            self._last_native_state = expert.output_decoder(expert.layer_norms[-1](tokens)).reshape(
                tokens.shape[0], config.num_views, config.K, config.P, config.token_dim)

        self._state_hook = wsa_model.mot.register_forward_hook(capture_state)

    @torch.inference_mode()
    def infer_world(self, **kwargs):
        self._last_native_state = None
        device_type = self.wsa_model.device.type
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=device_type == "cuda"):
            result = dict(super().infer_world(**kwargs))
        if self._last_native_state is None:
            raise FoundationModelError("joint sampler produced no native trajectory state")
        result["native_3d_future_tokens"] = self._last_native_state
        result["native_3d_view_mask"] = kwargs["current_native_3d_view_mask"]
        self._last_native_state = None
        return result

    @torch.inference_mode()
    def infer_action(self, **kwargs):
        device_type = self.wsa_model.device.type
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=device_type == "cuda"):
            return super().infer_action(**kwargs)

    def forward(self, sample: Mapping[str, torch.Tensor], _final, current, current_mask,
                _rollout, _horizon, _rollout_mask) -> dict[str, torch.Tensor]:
        if self._targets is not None:
            raise RuntimeError("concurrent native trajectory forwards are forbidden")
        # Labels are bound only for the loss callback after mixed attention.
        # pre_dit sees observations, queries and noisy action, never these labels.
        self._targets = (sample["native_3d_trajectory_target"].detach(),
                         sample["native_3d_trajectory_mask"].bool(), current.detach())
        self._active_time_update = self.action_time(sample["action_time_s"], sample["action_element_mask"])
        self._active_native_3d_update = self.native_3d_state(current, current_mask)
        try:
            total, metrics = self.wsa_model.training_loss(
                sample, collect_loss_dict=True, collect_detailed_loss_dict=True,
                loss_dict_as_tensors=True)
            return {"total": total, **metrics}
        finally:
            self._targets = None
            self._active_time_update = None
            self._active_native_3d_update = None


class TrajectoryObjective(nn.Module):
    def forward(self, predictions, target, valid, current):
        return trajectory_loss(predictions, target, valid, current)


def build_native_pretraining_model(config: FoundationWorldModelConfig, *, device) -> FoundationWorldModel:
    """Use the official Wan loader; never resolve any WSA checkpoint/config."""
    from lerobot.policies.WSA_Large.configuration_wsa_large import (
        _default_video_dit_config, _default_action_dit_config, _default_future_3d_config,
    )
    from lerobot.policies.WSA_Large.core.models.wan22.wsa_large import WSALarge

    video = _default_video_dit_config(config.action_width)
    action = _default_action_dit_config(config.action_width)
    future = _default_future_3d_config(
        da3_query_dim=config.token_dim, da3_num_views=config.num_views,
        da3_tokens_per_view=config.K * config.P,
        future_3d_tokens_per_view=config.K * config.P,
        query_mode="query_token", query_noise_scale=0.0,
        query_noise_min_sigma=0.0, query_noise_max_sigma=0.0,
        query_sigma_source="constant", slot_pos_scale=0.5,
    )
    for mapping in (video, action, future):
        mapping["use_gradient_checkpointing"] = config.activation_checkpointing
    core = WSALarge.from_wan22_pretrained(
        model_id=config.wan_model_path, tokenizer_model_id=config.wan_tokenizer_model_path,
        device=str(device), torch_dtype=torch.bfloat16, load_text_encoder=False,
        redirect_common_files=False, video_dit_config=video,
        action_dit_config=action, future_3d_config=future,
        action_dit_pretrained_path=config.wan_action_backbone_path,
        future_3d_pretrained_path=config.wan_3d_backbone_path,
        skip_dit_load_from_pretrain=False, proprio_dim=config.state_width,
        mot_checkpoint_mixed_attn=config.activation_checkpointing,
        loss_lambda_video=config.video_loss_weight, loss_lambda_action=config.action_loss_weight,
        mask_action_dim_padding_loss=True, loss_lambda_3d=0.0,
        future_3d_view_attention_layout="robotwin",
    )
    # Keep the upstream API aliases but register each expert exactly once.
    # Otherwise a fresh FP32 state_dict stores the same MoT weights under
    # video_expert, mot.mixtures and dit.mixtures, multiplying checkpoint I/O.
    for name in ("video_expert", "action_expert", "future_3d_expert", "dit"):
        module = getattr(core, name)
        delattr(core, name)
        object.__setattr__(core, name, module)
    core.vae.requires_grad_(False)
    model = FoundationWorldModel(config, backend=NativeTrajectoryBackend(core, config=config))
    # FP32 master parameters/moments, BF16 compute under the common trainer.
    # Root FSDP also requires a consistent original parameter dtype.
    return model.to(device=device, dtype=torch.float32)
