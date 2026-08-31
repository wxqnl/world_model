"""VLA stage-A fine-tune: warm-start from v3 best.pt, phased freezing."""
from __future__ import annotations
import argparse
from collections.abc import Callable
from dataclasses import dataclass
import importlib
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from wm3d_v3.data.manifest import read_manifest
from wm3d_v3.data.window_dataset import OXEWindowDataset, WindowConfig
from wm3d_v3.losses import VLALossWeights, compute_losses_vla
from wm3d_v3.models.action_stream import ActionConfig
from wm3d_v3.models.dual_stream import DualConfig
from wm3d_v3.models.joint_model import JointConfig, JointWorldModel
from wm3d_v3.models.model_factory import build_joint_world_model
from wm3d_v3.models.state_stream import StateConfig


def setup_ddp():
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world = dist.get_world_size()
        local = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local)
        return rank, world, local
    return 0, 1, 0


def _legacy_build_model(cfg: dict) -> JointWorldModel:
    sc = StateConfig(**cfg["model"]["state"])
    ac = ActionConfig(**cfg["model"]["action"])
    dc = DualConfig(state=sc, action=ac,
                    xattn_layers_state=tuple(cfg["model"]["xattn_layers_state"]),
                    xattn_n_heads=cfg["model"]["xattn_n_heads"])
    jc = JointConfig(
        dual=dc,
        action_proj_hidden=cfg["model"]["action_proj_hidden"],
        action_proj_layers=cfg["model"]["action_proj_layers"],
        geom_hidden=cfg["model"]["geom_hidden"],
        pixel_hidden=cfg["model"]["pixel_hidden"],
        pixel_n_res=cfg["model"]["pixel_n_res"],
        enable_pixel=cfg["model"].get("enable_pixel", True),
        enable_bridging=cfg["model"].get("enable_bridging", True),
        enable_aux_idm=cfg["model"].get("enable_aux_idm", True),
        aux_idm_hidden=cfg["model"].get("aux_idm_hidden", 1024),
        aux_idm_layers=cfg["model"].get("aux_idm_layers", 3),
    )
    return JointWorldModel(jc)


def build_model(cfg: dict) -> JointWorldModel:
    return build_joint_world_model(cfg["model"])


DatasetBuilder = Callable[[dict], tuple[Dataset, Dataset]]
_DATASET_BUILDERS: dict[str, DatasetBuilder] = {}


def register_vla_dataset_builder(name: str, builder: DatasetBuilder) -> None:
    if not name or name == "oxe":
        raise ValueError("custom VLA dataset builder name must be non-empty and cannot be 'oxe'")
    if name in _DATASET_BUILDERS and _DATASET_BUILDERS[name] is not builder:
        raise ValueError(f"VLA dataset builder {name!r} is already registered")
    _DATASET_BUILDERS[name] = builder


def _build_oxe_datasets(data_cfg: dict) -> tuple[Dataset, Dataset]:
    records = read_manifest(data_cfg["manifest"])
    wcfg = WindowConfig(T=data_cfg["T"], k=data_cfg["k"],
                        stride=data_cfg["stride"],
                        cache_root=Path(data_cfg["cache_root"]),
                        action_stats=Path(data_cfg["action_stats"]))
    ds = OXEWindowDataset(records, wcfg)
    n = len(ds)
    g = torch.Generator().manual_seed(data_cfg["seed"])
    perm = torch.randperm(n, generator=g).tolist()
    n_val = max(1, int(n * data_cfg["val_frac"]))
    return Subset(ds, perm[n_val:]), Subset(ds, perm[:n_val])


def _import_dataset_builder(path: str) -> DatasetBuilder:
    module_name, separator, attr = path.partition(":")
    if not separator or not module_name or not attr:
        raise ValueError("data.dataset_factory must use 'module.path:callable' syntax")
    builder = getattr(importlib.import_module(module_name), attr)
    if not callable(builder):
        raise TypeError(f"configured VLA dataset factory {path!r} is not callable")
    return builder


def build_datasets(cfg: dict) -> tuple[Dataset, Dataset]:
    data_cfg = cfg["data"]
    backend = str(data_cfg.get("dataset_backend", "oxe"))
    if backend == "oxe":
        datasets = _build_oxe_datasets(data_cfg)
    elif backend in _DATASET_BUILDERS:
        datasets = _DATASET_BUILDERS[backend](data_cfg)
    else:
        factory_path = data_cfg.get("dataset_factory")
        if not factory_path:
            raise KeyError(
                f"unknown VLA dataset backend {backend!r}; configure data.dataset_factory"
            )
        datasets = _import_dataset_builder(str(factory_path))(data_cfg)
    if not isinstance(datasets, tuple) or len(datasets) != 2:
        raise TypeError("VLA dataset builder must return (train_dataset, val_dataset)")
    if not all(isinstance(dataset, Dataset) for dataset in datasets):
        raise TypeError("VLA dataset builder outputs must be torch Dataset instances")
    return datasets


@dataclass(frozen=True)
class OFTBatchContract:
    input_tokens_key: str
    task_key: str
    target_key: str
    context_rgb_key: str | None
    lowdim_state_key: str | None
    canonical_history_key: str | None
    adapter_state_key: str | None
    adapter_history_key: str | None
    action_mask_key: str | None
    target_slice: str

    @classmethod
    def from_config(cls, train_cfg: dict) -> "OFTBatchContract":
        target_slice = str(train_cfg.get("oft_target_slice", "first"))
        if target_slice not in {"first", "last"}:
            raise ValueError("train.oft_target_slice must be first or last")
        return cls(
            input_tokens_key=str(train_cfg.get("oft_input_tokens_key", "s_in")),
            task_key=str(train_cfg.get("oft_task_key", "c")),
            target_key=str(train_cfg.get("oft_target_key", "action_tgt_norm")),
            context_rgb_key=(str(train_cfg["oft_context_rgb_key"])
                             if train_cfg.get("oft_context_rgb_key") else None),
            lowdim_state_key=(str(train_cfg["oft_lowdim_state_key"])
                              if train_cfg.get("oft_lowdim_state_key") else None),
            canonical_history_key=(str(train_cfg["oft_canonical_history_key"])
                                   if train_cfg.get("oft_canonical_history_key") else None),
            adapter_state_key=(str(train_cfg["oft_state_key"])
                               if train_cfg.get("oft_state_key") else None),
            adapter_history_key=(str(train_cfg["oft_history_key"])
                                 if train_cfg.get("oft_history_key") else None),
            action_mask_key=(str(train_cfg["oft_action_mask_key"])
                             if train_cfg.get("oft_action_mask_key") else None),
            target_slice=target_slice,
        )


def apply_freeze(model: torch.nn.Module, prefixes: list[str]) -> tuple[int, int]:
    """Set requires_grad on each param based on name prefix match.
    Returns (n_frozen_params, n_trainable_params).
    """
    n_frozen = 0
    n_train = 0
    for name, p in model.named_parameters():
        if any(name.startswith(pfx) for pfx in prefixes):
            p.requires_grad = False
            n_frozen += p.numel()
        else:
            p.requires_grad = True
            n_train += p.numel()
    return n_frozen, n_train


def load_warm_start(model: JointWorldModel, ckpt_path: Path, rank: int) -> None:
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = sd["model"]
    if model.action_policy is not None and model.action_policy.oft_head is not None:
        contract = sd.get("action_policy_contract")
        if not isinstance(contract, dict):
            raise RuntimeError("WM3D-OFT warm start requires action_policy_contract")
        report = model.load_oft_benchmark_state_dict(state, contract)
        if rank == 0:
            print(
                f"[warm-start] strict WM3D-OFT load {ckpt_path}; "
                f"new_adapters={report['new_adapters']} "
                f"initialized={len(report['initialized_adapter_keys'])}"
            )
        return
    drop = [k for k in state if k.startswith("action_proj.")]
    for k in drop:
        del state[k]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if rank == 0:
        print(f"[warm-start] loaded {ckpt_path}; dropped {len(drop)} old action_proj keys")
        print(f"  missing (fresh init): {sorted(missing)[:6]}{'...' if len(missing)>6 else ''}")
        print(f"  unexpected (ignored): {sorted(unexpected)[:6]}{'...' if len(unexpected)>6 else ''}")


def batch_to_device(batch: dict, device: torch.device) -> tuple:
    s = batch["s_in"].to(device, non_blocking=True)
    c = batch["c"].to(device, non_blocking=True)
    tgt = {
        "action_tgt": batch["action_tgt"].to(device, non_blocking=True),
        "action_tgt_norm": batch["action_tgt_norm"].to(device, non_blocking=True),
    }
    return s, c, tgt


def compute_oft_action_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    grip_indices: tuple[int, ...] = (),
    grip_threshold: float = 0.5,
    mask: torch.Tensor | None = None,
    loss_type: str = "l1",
    pose_weight: float = 1.0,
    grip_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    if pred.shape != target.shape:
        raise ValueError(f"OFT prediction/target shape mismatch {tuple(pred.shape)} != {tuple(target.shape)}")
    grip = tuple(sorted(set(int(index) for index in grip_indices)))
    if any(index < 0 or index >= pred.shape[-1] for index in grip):
        raise ValueError(f"OFT grip indices {grip} exceed action_dim={pred.shape[-1]}")
    continuous = tuple(index for index in range(pred.shape[-1]) if index not in grip)

    if mask is not None:
        if mask.shape != pred.shape[:-1]:
            raise ValueError(f"OFT action mask cannot broadcast from {tuple(mask.shape)} to {tuple(pred.shape)}")
        mask = mask.to(device=pred.device, dtype=pred.dtype)
    else:
        mask = torch.ones(pred.shape[:-1], device=pred.device, dtype=pred.dtype)

    def masked_mean(values: torch.Tensor) -> torch.Tensor:
        expanded_mask = mask
        while expanded_mask.ndim < values.ndim:
            expanded_mask = expanded_mask.unsqueeze(-1)
        return (values * expanded_mask).sum() / expanded_mask.expand_as(values).sum().clamp_min(1.0)

    if continuous:
        pose_pred = pred[..., list(continuous)]
        pose_target = target[..., list(continuous)]
        if loss_type == "l1":
            pose_elements = (pose_pred - pose_target).abs()
        elif loss_type == "smooth_l1":
            pose_elements = F.smooth_l1_loss(pose_pred, pose_target, reduction="none")
        else:
            raise ValueError(f"unsupported OFT loss_type={loss_type!r}")
        pose_loss = masked_mean(pose_elements)
    else:
        pose_loss = pred.sum() * 0.0

    if grip:
        grip_logits = pred[..., list(grip)]
        grip_target = target[..., list(grip)]
        if bool(((grip_target < 0.0) | (grip_target > 1.0)).any()):
            raise ValueError("OFT grip targets must be binary values in [0,1]")
        grip_loss_value = masked_mean(
            F.binary_cross_entropy_with_logits(grip_logits, grip_target, reduction="none")
        )
        grip_prediction = (torch.sigmoid(grip_logits) >= float(grip_threshold)).to(grip_target.dtype)
        grip_acc = masked_mean((grip_prediction == grip_target).to(pred.dtype))
    else:
        grip_loss_value = pred.sum() * 0.0
        grip_acc = pred.new_tensor(1.0)
    total = float(pose_weight) * pose_loss + float(grip_weight) * grip_loss_value
    return {
        "L_total": total,
        "L_action": total,
        "L_pose": pose_loss,
        "L_grip": grip_loss_value,
        "grip_acc": grip_acc,
    }


def forward_oft_benchmark_batch(
    model: torch.nn.Module,
    batch: dict,
    cfg: dict,
    device: torch.device,
) -> tuple[dict, dict[str, torch.Tensor]]:
    train_cfg = cfg["train"]
    contract = OFTBatchContract.from_config(train_cfg)
    adapter_name = str(train_cfg["oft_adapter_name"])
    horizon = int(train_cfg["oft_horizon"])
    target_model = model.module if hasattr(model, "module") else model
    policy = getattr(target_model, "action_policy", None)
    if policy is None or policy.oft_head is None:
        raise RuntimeError("OFT benchmark batch requires an OFT action policy")
    if adapter_name not in policy.oft_head.adapter_specs:
        raise KeyError(f"unknown configured OFT adapter {adapter_name!r}")
    spec = policy.oft_head.adapter_specs[adapter_name]

    def tensor(key: str | None, purpose: str, *, required: bool) -> torch.Tensor | None:
        if key is None:
            if required:
                raise KeyError(f"OFT batch contract is missing a configured key for {purpose}")
            return None
        if key not in batch:
            raise KeyError(f"OFT benchmark batch is missing {purpose} key {key!r}")
        value = batch[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"OFT benchmark field {key!r} must be a tensor")
        return value.to(device, non_blocking=True)

    s = tensor(contract.input_tokens_key, "input tokens", required=True)
    c = tensor(contract.task_key, "task embedding", required=bool(policy.cfg.use_task))
    if c is None:
        c = torch.zeros(s.shape[0], policy.cfg.task_dim, device=device, dtype=s.dtype)
    target = tensor(contract.target_key, "action target", required=True)
    assert s is not None and target is not None
    if target.shape[1] < horizon:
        raise ValueError(f"OFT target horizon {target.shape[1]} is shorter than requested K{horizon}")
    target = target[:, :horizon] if contract.target_slice == "first" else target[:, -horizon:]
    if target.shape[-1] != spec.action_dim:
        raise ValueError(
            f"OFT target action_dim={target.shape[-1]} does not match adapter {spec.action_dim}"
        )

    is_canonical = adapter_name == policy.cfg.oft_adapter_name
    context_rgb = tensor(
        contract.context_rgb_key,
        "context RGB",
        required=bool(policy.cfg.use_context_rgb),
    )
    lowdim_state = tensor(
        contract.lowdim_state_key,
        "canonical low-dimensional state",
        required=is_canonical and policy.cfg.lowdim_dim > 0,
    )
    canonical_history = tensor(
        contract.canonical_history_key,
        "canonical action history",
        required=is_canonical and policy.cfg.action_history_len > 0,
    )
    state = tensor(
        contract.adapter_state_key,
        "typed adapter state",
        required=(not is_canonical and spec.state_dim > 0),
    )
    history = tensor(
        contract.adapter_history_key,
        "typed adapter history",
        required=(not is_canonical and spec.history_len > 0),
    )
    mask = tensor(contract.action_mask_key, "action mask", required=False)
    out = model(
        s,
        c,
        pixel=False,
        bridging=False,
        aux_idm=False,
        lowdim_state=lowdim_state,
        action_history=canonical_history,
        context_rgb=context_rgb,
        oft_adapter_name=adapter_name,
        oft_horizon=horizon,
        oft_state=state,
        oft_action_history=history,
    )
    pred = out.get("policy_oft_actions")
    if pred is None:
        raise RuntimeError(f"OFT adapter {adapter_name!r} did not emit policy_oft_actions")
    losses = compute_oft_action_loss(
        pred,
        target.to(dtype=pred.dtype),
        grip_indices=spec.grip_indices,
        grip_threshold=spec.grip_threshold,
        mask=mask,
        loss_type=str(train_cfg.get("oft_loss_type", "l1")),
        pose_weight=float(train_cfg.get("oft_pose_weight", 1.0)),
        grip_weight=float(train_cfg.get("oft_grip_weight", 1.0)),
    )
    return out, losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=Path, required=True)
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--max_batches_per_epoch", type=int, default=0)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.cfg.read_text())
    rank, world, local = setup_ddp()
    device = torch.device(f"cuda:{local}")

    tr_ds, val_ds = build_datasets(cfg)
    bs = cfg["train"]["batch_size_per_gpu"]
    nw = cfg["train"]["num_workers"]
    if world > 1:
        tr_s = DistributedSampler(tr_ds, num_replicas=world, rank=rank,
                                   shuffle=True, drop_last=True)
        tr_loader = DataLoader(tr_ds, batch_size=bs, sampler=tr_s,
                                num_workers=nw, pin_memory=True, drop_last=True)
        val_s = DistributedSampler(val_ds, num_replicas=world, rank=rank, shuffle=False)
        val_loader = DataLoader(val_ds, batch_size=bs, sampler=val_s,
                                 num_workers=nw, pin_memory=True)
    else:
        tr_loader = DataLoader(tr_ds, batch_size=bs, shuffle=True,
                                num_workers=nw, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                                 num_workers=nw, pin_memory=True)

    model = build_model(cfg).to(device)
    is_oft_benchmark = bool(cfg["train"].get("oft_adapter_name"))

    action_stats_path = cfg["data"].get("action_stats")
    if action_stats_path:
        stats = np.load(action_stats_path)
        mean = torch.from_numpy(stats["mean"][:6].astype(np.float32)).to(device)
        std = torch.from_numpy(stats["std"][:6].astype(np.float32)).to(device)
        pos_rate = float(stats["pos_rate"][0])
        model.load_action_stats(mean, std)
        if not is_oft_benchmark and pos_rate > 0:
            cfg["loss"]["grip_pos_weight"] = float(
                min(5.0, max(0.5, (1.0 - pos_rate) / pos_rate))
            )
        if rank == 0:
            print(f"action_stats: pos_rate={pos_rate:.4f}")
    elif not is_oft_benchmark:
        raise KeyError("legacy VLA training requires data.action_stats")

    if args.resume is None and "warm_start" in cfg:
        load_warm_start(model, Path(cfg["warm_start"]), rank)

    cut = int(cfg["train"]["freeze_phase1_until_epoch"])
    phase1_pfx = cfg["train"]["freeze_phase1_prefixes"]
    phase2_pfx = cfg["train"]["freeze_phase2_prefixes"]

    start_epoch = 0
    step = 0
    best_val = float("inf")

    # Apply initial freeze based on starting epoch
    init_pfx = phase1_pfx if start_epoch < cut else phase2_pfx
    n_frozen, n_train = apply_freeze(model, init_pfx)
    if rank == 0:
        print(f"[freeze] phase={'A.1' if start_epoch<cut else 'A.2'} "
              f"frozen={n_frozen/1e6:.1f}M trainable={n_train/1e6:.1f}M")

    if world > 1:
        model = DDP(model, device_ids=[local], find_unused_parameters=True)

    weights = None if is_oft_benchmark else VLALossWeights(**cfg["loss"])

    def make_opt_sched(stp: int):
        target = model.module if world > 1 else model
        lr = cfg["train"]["lr"]
        op = torch.optim.AdamW([p for p in target.parameters() if p.requires_grad],
                                lr=lr,
                                weight_decay=cfg["train"]["weight_decay"],
                                betas=(0.9, 0.95))
        # LambdaLR with last_epoch>=0 requires initial_lr in each param group.
        for pg in op.param_groups:
            pg["initial_lr"] = lr
        warmup = int(cfg["train"]["warmup_steps"])
        total_steps = max(1, len(tr_loader) * cfg["train"]["epochs"])
        def lr_lambda(s):
            if s < warmup:
                return (s + 1) / warmup
            prog = (s - warmup) / max(1, total_steps - warmup)
            return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog))
        sch = torch.optim.lr_scheduler.LambdaLR(op, lr_lambda, last_epoch=stp - 1)
        return op, sch

    opt, sched = make_opt_sched(step)

    out_root = Path(cfg["out"]["root"])
    ckpt_dir = out_root / cfg["out"]["ckpt_dir"]
    if rank == 0:
        (out_root / cfg["out"]["tb_dir"]).mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        tb = SummaryWriter(out_root / cfg["out"]["tb_dir"])

    if args.resume is not None and args.resume.exists():
        sd = torch.load(args.resume, map_location=device, weights_only=False)
        (model.module if world > 1 else model).load_state_dict(sd["model"])
        opt.load_state_dict(sd["opt"])
        sched.load_state_dict(sd["sched"])
        start_epoch = sd["epoch"] + 1
        step = sd["step"]
        best_val = sd.get("best_val", best_val)
        if rank == 0:
            print(f"[resume] from {args.resume} at epoch {start_epoch}")

    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        prev_phase = "A.1" if (epoch - 1) < cut else "A.2"
        this_phase = "A.1" if epoch < cut else "A.2"
        if this_phase != prev_phase and epoch != start_epoch:
            target = model.module if world > 1 else model
            n_frozen, n_train = apply_freeze(target, phase2_pfx)
            opt, sched = make_opt_sched(step)
            if rank == 0:
                print(f"[freeze] phase=A.2 frozen={n_frozen/1e6:.1f}M trainable={n_train/1e6:.1f}M")
        if world > 1:
            tr_s.set_epoch(epoch)
        model.train()
        bi = 0
        for batch in tr_loader:
            if args.max_batches_per_epoch and bi >= args.max_batches_per_epoch:
                break
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                if is_oft_benchmark:
                    out, losses = forward_oft_benchmark_batch(model, batch, cfg, device)
                else:
                    s, c, tgt = batch_to_device(batch, device)
                    out = model(s, c, pixel=False, bridging=False, aux_idm=True)
                    assert weights is not None
                    losses = compute_losses_vla(out, tgt, weights)
            opt.zero_grad(set_to_none=True)
            losses["L_total"].backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                cfg["train"]["grad_clip"])
            opt.step()
            sched.step()
            if rank == 0 and step % cfg["train"]["log_every"] == 0:
                for kk, v in losses.items():
                    tb.add_scalar(f"train/{kk}", float(v.detach()), step)
                tb.add_scalar("lr", sched.get_last_lr()[0], step)
                tb.add_scalar("grad_norm", float(grad_norm), step)
                if "L_action" in losses:
                    detail = f"L_action={float(losses['L_action']):.4f}"
                else:
                    detail = (
                        f"L_pose={float(losses['L_pose']):.4f} L_grip={float(losses['L_grip']):.4f} "
                        f"L_aux_pose={float(losses['L_aux_pose']):.4f}"
                    )
                print(
                    f"  ep{epoch} step{step} L_total={float(losses['L_total'].detach()):.4f} "
                    f"{detail} gn={float(grad_norm):.2f}"
                )
            step += 1
            bi += 1
        model.eval()
        agg = {}
        nb = 0
        with torch.no_grad():
            for batch in val_loader:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    if is_oft_benchmark:
                        out, losses = forward_oft_benchmark_batch(model, batch, cfg, device)
                    else:
                        s, c, tgt = batch_to_device(batch, device)
                        out = model(s, c, pixel=False, bridging=False, aux_idm=True)
                        assert weights is not None
                        losses = compute_losses_vla(out, tgt, weights)
                for kk, v in losses.items():
                    agg[kk] = agg.get(kk, 0.0) + float(v.detach().float())
                nb += 1
        if world > 1:
            keys = sorted(agg.keys())
            v = torch.tensor([agg[kk] for kk in keys] + [float(nb)], device=device)
            dist.all_reduce(v)
            tot_nb = float(v[-1].item())
            for i, kk in enumerate(keys):
                agg[kk] = float(v[i].item()) / max(1.0, tot_nb)
            nb = 1
        if rank == 0:
            for kk, vv in agg.items():
                tb.add_scalar(f"val/{kk}", vv / max(1, nb), step)
            val_total = agg["L_total"] / max(1, nb)
            ckpt = {"model": (model.module if world > 1 else model).state_dict(),
                    "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": epoch, "step": step, "val_total": val_total,
                    "best_val": best_val, "cfg": cfg}
            target_model = model.module if world > 1 else model
            if target_model.action_policy is not None:
                ckpt["action_policy_contract"] = target_model.action_policy_checkpoint_contract()
            if (epoch + 1) % cfg["train"]["ckpt_every_epochs"] == 0:
                torch.save(ckpt, ckpt_dir / f"epoch_{epoch:03d}.pt")
            if val_total < best_val:
                best_val = val_total
                ckpt["best_val"] = best_val
                torch.save(ckpt, ckpt_dir / "best.pt")
            print(f"[rank0] epoch {epoch}: val_total {val_total:.4f} (best {best_val:.4f}, phase={this_phase})")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
