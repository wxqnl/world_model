"""Single-task LIBERO BC teacher utilities.

This module intentionally lives at the benchmark/tooling boundary. It reuses
LIBERO's official BC policies and datasets to train or roll out an external
teacher, without changing the WM3D world core or policy heads.
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import yaml
from PIL import Image

if "bool" not in np.__dict__:
    np.bool = np.bool_  # type: ignore[attr-defined]


def _enable_trusted_legacy_torch_load() -> None:
    """Allow LIBERO trusted local init-state files under PyTorch 2.6+."""
    if getattr(torch.load, "_wm3d_legacy_compat", False):
        return
    original_load = torch.load
    supports_weights_only = "weights_only" in inspect.signature(original_load).parameters

    def compatible_load(*args: Any, **kwargs: Any) -> Any:
        if supports_weights_only:
            kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    compatible_load._wm3d_legacy_compat = True  # type: ignore[attr-defined]
    torch.load = compatible_load  # type: ignore[assignment]


def _bootstrap_libero(root: Path) -> None:
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    benchmark_root = root / "libero" / "libero"
    config_dir = root / ".wm3d_libero_config"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.yaml"
    if not config_path.exists():
        config = {
            "benchmark_root": str(benchmark_root),
            "bddl_files": str(benchmark_root / "bddl_files"),
            "init_states": str(benchmark_root / "init_files"),
            "datasets": str(root / "datasets"),
            "assets": str(benchmark_root / "assets"),
        }
        config_path.write_text(yaml.safe_dump(config, sort_keys=True))
    os.environ.setdefault("LIBERO_CONFIG_PATH", str(config_dir))


def _prefer_glfw_rendering() -> None:
    """Keep LIBERO rollouts on the same Xvfb/glfw path as the py38 runner."""
    os.environ.setdefault("MUJOCO_GL", "glfw")
    try:
        import robosuite.macros as macros

        macros.MUJOCO_GPU_RENDERING = False
    except Exception:
        return


def _compose_cfg(args: argparse.Namespace) -> Any:
    from easydict import EasyDict
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    overrides = [
        f"seed={args.seed}",
        f"benchmark_name={args.benchmark_name}",
        f"policy={args.policy}",
        f"lifelong={args.algo}",
        f"device={args.device}",
        f"task_embedding_format={args.task_embedding_format}",
        "use_wandb=false",
        f"data.task_order_index={args.task_order_index}",
        f"data.seq_len={args.seq_len}",
        f"data.img_h={args.image_size}",
        f"data.img_w={args.image_size}",
        f"train.n_epochs={args.epochs}",
        f"train.batch_size={args.batch_size}",
        f"train.num_workers={args.num_workers}",
        f"train.use_augmentation={str(not args.no_augmentation).lower()}",
        f"eval.eval={str(not args.no_eval).lower()}",
        f"eval.n_eval={args.eval_episodes}",
        f"eval.num_procs={args.eval_num_procs}",
        f"eval.use_mp={str(args.eval_num_procs > 1).lower()}",
        f"eval.max_steps={args.eval_max_steps}",
        f"eval.eval_every={args.eval_every}",
    ]
    with initialize_config_dir(config_dir=str(args.libero_root / "libero" / "configs"), version_base=None):
        hydra_cfg = compose(config_name="config", overrides=overrides)
    cfg = EasyDict(yaml.safe_load(OmegaConf.to_yaml(hydra_cfg)))
    return cfg


def _resolve_paths(cfg: Any) -> None:
    from libero.libero import get_libero_path

    cfg.folder = cfg.folder or get_libero_path("datasets")
    cfg.bddl_folder = cfg.bddl_folder or get_libero_path("bddl_files")
    cfg.init_states_folder = cfg.init_states_folder or get_libero_path("init_states")


def _task_id_for_name(benchmark: Any, task_name: str) -> int:
    for task_id in range(benchmark.get_num_tasks()):
        if benchmark.get_task(task_id).name == task_name:
            return task_id
    raise RuntimeError(f"task {task_name!r} not found in benchmark {benchmark.name}")


def _task_name_from_hdf5(path: Path) -> str:
    name = path.stem
    return name[:-5] if name.endswith("_demo") else name


def _load_hdf5_init_state(path: Path, demo_id: str) -> np.ndarray:
    with h5py.File(path, "r") as h5:
        return np.asarray(h5["data"][demo_id].attrs["init_state"])


def _save_frame(frame: np.ndarray, path: Path) -> str:
    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="RGB").save(path)
    return str(path)


def _extract_lowdim(obs: dict[str, Any]) -> np.ndarray:
    parts = [
        np.asarray(obs["robot0_eef_pos"], dtype=np.float32).reshape(-1),
        np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1),
        np.asarray(obs["robot0_joint_pos"], dtype=np.float32).reshape(-1),
    ]
    out = np.concatenate(parts).astype(np.float32)
    if out.shape != (12,):
        raise ValueError(f"expected 12D lowdim state, got {out.shape}")
    return out


def _extract_object_state(obs: dict[str, Any]) -> np.ndarray:
    out = np.asarray(obs["object-state"], dtype=np.float32).reshape(-1)
    if out.size == 0:
        raise ValueError("object-state is empty")
    return out


def _extract_named_poses(obs: dict[str, Any]) -> dict[str, dict[str, list[float]]]:
    out: dict[str, dict[str, list[float]]] = {}
    suffixes = ("_to_robot0_eef_pos", "_to_robot0_eef_quat", "_pos", "_quat")
    explicit = {"robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"}
    for key, value in obs.items():
        if key not in explicit and not key.endswith(suffixes):
            continue
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            continue
        if key.startswith("robot0_"):
            entity = "robot0"
            field = key[len("robot0_") :]
        else:
            entity = key
            field = None
            for suffix in suffixes:
                if key.endswith(suffix):
                    entity = key[: -len(suffix)]
                    field = suffix[1:]
                    break
            if field is None:
                continue
        out.setdefault(entity, {})[field] = arr.astype(float).tolist()
    return out


def _entity_pos(named_poses: dict[str, dict[str, list[float]]] | None, entity: str) -> np.ndarray | None:
    if not named_poses:
        return None
    fields = named_poses.get(entity, {})
    value = fields.get("pos")
    if value is None and entity == "robot0":
        value = fields.get("eef_pos")
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    return arr[:3] if arr.size >= 3 else None


def _is_task1_put_cream_butter(task_text: str) -> bool:
    text = task_text.lower()
    return ("cream_cheese" in text or "cream cheese" in text) and "butter" in text and "basket" in text


def _object_eef_dist(named_poses: dict[str, dict[str, list[float]]] | None, entity: str) -> float | None:
    if not named_poses:
        return None
    rel = named_poses.get(entity, {}).get("to_robot0_eef_pos")
    if rel is not None:
        arr = np.asarray(rel, dtype=np.float32).reshape(-1)
        if arr.size >= 3:
            return float(np.linalg.norm(arr[:3]))
    obj = _entity_pos(named_poses, entity)
    eef = _entity_pos(named_poses, "robot0")
    if obj is None or eef is None:
        return None
    return float(np.linalg.norm(obj[:3] - eef[:3]))


def _object_in_receptacle_xy(
    named_poses: dict[str, dict[str, list[float]]] | None,
    entity: str,
    receptacle: str = "basket_1",
    threshold: float = 0.14,
) -> bool:
    obj = _entity_pos(named_poses, entity)
    rec = _entity_pos(named_poses, receptacle)
    if obj is None or rec is None:
        return False
    return bool(float(np.linalg.norm(obj[:2] - rec[:2])) <= threshold)


def _update_task1_plan_stage(
    stage: int,
    named_poses: dict[str, dict[str, list[float]]] | None,
    *,
    contact_threshold: float = 0.08,
) -> int:
    cream_in = _object_in_receptacle_xy(named_poses, "cream_cheese_1")
    butter_in = _object_in_receptacle_xy(named_poses, "butter_1")
    cream_dist = _object_eef_dist(named_poses, "cream_cheese_1")
    butter_dist = _object_eef_dist(named_poses, "butter_1")
    cream_contact = cream_dist is not None and cream_dist <= contact_threshold
    butter_contact = butter_dist is not None and butter_dist <= contact_threshold

    out = int(np.clip(stage, 0, 3))
    if cream_in:
        out = max(out, 2)
    elif out < 1 and cream_contact:
        out = 1
    if out >= 2:
        if butter_in:
            out = max(out, 3)
        elif out < 3 and butter_contact:
            out = 3
    return out


def _plan_state_from_stage(
    stage: int,
    named_poses: dict[str, dict[str, list[float]]] | None,
    *,
    dim: int,
) -> np.ndarray:
    if dim < 8:
        raise ValueError(f"plan_state_dim must be >= 8, got {dim}")
    stage = int(np.clip(stage, 0, 3))
    target = 0 if stage < 2 else 1
    subgoal = 0 if stage in (0, 2) else 1
    out = np.zeros(int(dim), dtype=np.float32)
    out[stage] = 1.0
    out[4 + target] = 1.0
    out[6 + subgoal] = 1.0
    if dim >= 17:
        target_entity = "cream_cheese_1" if target == 0 else "butter_1"
        target_pos = _entity_pos(named_poses, target_entity)
        eef_pos = _entity_pos(named_poses, "robot0")
        basket_pos = _entity_pos(named_poses, "basket_1")
        if target_pos is not None and eef_pos is not None:
            out[8:11] = np.clip(target_pos[:3] - eef_pos[:3], -1.0, 1.0)
        if target_pos is not None and basket_pos is not None:
            out[11:14] = np.clip(basket_pos[:3] - target_pos[:3], -1.0, 1.0)
        if eef_pos is not None and basket_pos is not None:
            out[14:17] = np.clip(basket_pos[:3] - eef_pos[:3], -1.0, 1.0)
    return out


def _gmm_mode_action(dist: Any) -> torch.Tensor:
    mix = dist.mixture_distribution
    comp = dist.component_distribution
    logits = mix.logits
    means = comp.base_dist.loc if hasattr(comp, "base_dist") else comp.mean
    best = logits.argmax(dim=-1)
    gather_idx = best.unsqueeze(-1).unsqueeze(-1).expand(*best.shape, 1, means.shape[-1])
    return means.gather(dim=-2, index=gather_idx).squeeze(-2)


def _teacher_action(policy: Any, data: dict[str, Any], *, deterministic: bool) -> np.ndarray:
    if not deterministic:
        return np.asarray(policy.get_action(data), dtype=np.float32).reshape(-1, 7)
    policy.eval()
    data = policy.preprocess_input(data, train_mode=False)
    with torch.no_grad():
        if hasattr(policy, "spatial_encode") and hasattr(policy, "temporal_encode"):
            x = policy.spatial_encode(data)
            policy.latent_queue.append(x)
            if len(policy.latent_queue) > policy.max_seq_len:
                policy.latent_queue.pop(0)
            x = torch.cat(policy.latent_queue, dim=1)
            x = policy.temporal_encode(x)
            dist = policy.policy_head(x[:, -1])
        else:
            dist = policy.forward(data, train_mode=False)
        action = _gmm_mode_action(dist).detach().cpu()
    return action.reshape(action.shape[0], -1).numpy().astype(np.float32)


def _prepare_task_dataset(cfg: Any, benchmark: Any, task_id: int) -> tuple[Any, Any]:
    import os
    from libero.lifelong.datasets import SequenceVLDataset, get_dataset

    dataset, shape_meta = get_dataset(
        dataset_path=os.path.join(cfg.folder, benchmark.get_task_demonstration(task_id)),
        obs_modality=cfg.data.obs.modality,
        initialize_obs_utils=True,
        seq_len=cfg.data.seq_len,
    )
    task_emb = benchmark.get_task_emb(task_id)
    return SequenceVLDataset(dataset, task_emb), shape_meta


def _init_benchmark_and_embs(cfg: Any) -> Any:
    from libero.libero.benchmark import get_benchmark
    from libero.lifelong.utils import get_task_embs

    benchmark = get_benchmark(cfg.benchmark_name)(cfg.data.task_order_index)
    if cfg.task_embedding_format == "onehot_no_bert":
        task_embs = torch.eye(benchmark.n_tasks, dtype=torch.float32)
        cfg.policy.language_encoder.network_kwargs.input_size = task_embs.shape[-1]
    elif cfg.task_embedding_format == "zero_no_bert":
        task_embs = torch.zeros(benchmark.n_tasks, 1, dtype=torch.float32)
        cfg.policy.language_encoder.network_kwargs.input_size = task_embs.shape[-1]
    else:
        descriptions = [benchmark.get_task(i).language for i in range(benchmark.n_tasks)]
        task_embs = get_task_embs(cfg, descriptions)
    benchmark.set_task_embs(task_embs)
    return benchmark


def _train(args: argparse.Namespace) -> None:
    _bootstrap_libero(args.libero_root)
    _enable_trusted_legacy_torch_load()
    _prefer_glfw_rendering()
    import numpy as np
    from torch.utils.data import DataLoader, RandomSampler
    from libero.lifelong.algos import get_algo_class
    from libero.lifelong.utils import NpEncoder, compute_flops, control_seed, safe_device, torch_save_model

    cfg = _compose_cfg(args)
    _resolve_paths(cfg)
    cfg.experiment_dir = str(args.out_dir)
    cfg.experiment_name = args.out_dir.name
    Path(cfg.experiment_dir).mkdir(parents=True, exist_ok=True)
    control_seed(cfg.seed)

    benchmark = _init_benchmark_and_embs(cfg)
    task_id = args.task_id
    if args.task_name:
        task_id = _task_id_for_name(benchmark, args.task_name)
    if task_id < 0:
        raise ValueError("--task_id or --task_name is required for train")
    dataset, shape_meta = _prepare_task_dataset(cfg, benchmark, task_id)
    cfg.shape_meta = shape_meta

    algo = safe_device(get_algo_class(cfg.lifelong.algo)(benchmark.n_tasks, cfg), cfg.device)
    try:
        gflops, mparams = compute_flops(algo, dataset, cfg)
    except Exception as exc:
        print(json.dumps({"warning": "compute_flops_failed", "error": repr(exc)}), flush=True)
        gflops, mparams = 0.0, 0.0
    result_summary = {
        "L_conf_mat": np.zeros((benchmark.n_tasks, benchmark.n_tasks)),
        "S_conf_mat": np.zeros((benchmark.n_tasks, benchmark.n_tasks)),
        "L_fwd": np.zeros((benchmark.n_tasks,)),
        "S_fwd": np.zeros((benchmark.n_tasks,)),
    }
    metadata = {
        "mode": "train",
        "benchmark": cfg.benchmark_name,
        "task_id": int(task_id),
        "task_name": benchmark.get_task(task_id).name,
        "instruction": benchmark.get_task(task_id).language,
        "policy": cfg.policy.policy_type,
        "algo": cfg.lifelong.algo,
        "dataset_sequences": int(len(dataset)),
        "dataset_demos": int(dataset.n_demos),
        "gflops": float(gflops),
        "mparams": float(mparams),
        "out_dir": str(args.out_dir),
    }
    print(json.dumps(metadata, sort_keys=True), flush=True)
    with (args.out_dir / "metadata.json").open("w") as fh:
        json.dump(metadata, fh, cls=NpEncoder, indent=2)
    with (args.out_dir / "config.json").open("w") as fh:
        json.dump(cfg, fh, cls=NpEncoder, indent=2)
    if args.no_eval:
        algo.start_task(int(task_id))
        loader = DataLoader(
            dataset,
            batch_size=cfg.train.batch_size,
            num_workers=cfg.train.num_workers,
            sampler=RandomSampler(dataset),
            persistent_workers=bool(cfg.train.num_workers > 0),
        )
        epoch_losses: list[float] = []
        for epoch in range(1, int(cfg.train.n_epochs) + 1):
            algo.train()
            total = 0.0
            count = 0
            t0 = time.time()
            for data in loader:
                total += float(algo.observe(data))
                count += 1
            if algo.scheduler is not None:
                algo.scheduler.step()
            loss = total / max(1, count)
            epoch_losses.append(loss)
            print(
                json.dumps(
                    {
                        "epoch": int(epoch),
                        "train_loss": float(loss),
                        "elapsed_sec": float(time.time() - t0),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        model_path = args.out_dir / f"task{int(task_id)}_model.pth"
        torch_save_model(algo.policy, str(model_path), cfg=cfg)
        result = {"offline_train_losses": epoch_losses, "checkpoint": str(model_path), **metadata}
        with (args.out_dir / "train_result.json").open("w") as fh:
            json.dump(result, fh, indent=2)
        print(json.dumps(result, sort_keys=True), flush=True)
        return

    s_fwd, l_fwd = algo.learn_one_task(dataset, int(task_id), benchmark, result_summary)
    result = {"S_fwd": float(s_fwd), "L_fwd": float(l_fwd), **metadata}
    with (args.out_dir / "train_result.json").open("w") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result, sort_keys=True), flush=True)


def _load_rollout_algo(args: argparse.Namespace) -> tuple[Any, Any, Any]:
    _bootstrap_libero(args.libero_root)
    _enable_trusted_legacy_torch_load()
    _prefer_glfw_rendering()
    from libero.lifelong.algos import get_algo_class
    from libero.lifelong.utils import safe_device

    model_dict = torch.load(args.ckpt, map_location=args.device)
    state_dict = model_dict["state_dict"]
    cfg = model_dict.get("cfg")
    previous_masks = model_dict.get("previous_masks")
    if cfg is None:
        raise RuntimeError(f"checkpoint {args.ckpt} does not contain LIBERO cfg")
    cfg.device = args.device
    _resolve_paths(cfg)
    benchmark = _init_benchmark_and_embs(cfg)
    algo = safe_device(get_algo_class(cfg.lifelong.algo)(benchmark.n_tasks, cfg), cfg.device)
    if previous_masks is not None:
        algo.policy.previous_mask = previous_masks
    algo.policy.load_state_dict(state_dict)
    if getattr(args, "low_eval_noise", False) and hasattr(algo.policy, "policy_head"):
        setattr(algo.policy.policy_head, "low_eval_noise", True)
    algo.eval()
    return cfg, benchmark, algo


def _rollout(args: argparse.Namespace) -> None:
    _bootstrap_libero(args.libero_root)
    _prefer_glfw_rendering()
    from libero.libero.envs import OffScreenRenderEnv
    from libero.lifelong.metric import raw_obs_to_tensor_obs
    import robomimic.utils.obs_utils as ObsUtils

    cfg, benchmark, algo = _load_rollout_algo(args)
    ObsUtils.initialize_obs_utils_with_obs_specs({"obs": cfg.data.obs.modality})
    task_id = int(args.task_id)
    if args.init_state_hdf5 is not None:
        hdf5_task_name = _task_name_from_hdf5(args.init_state_hdf5)
        task_id = _task_id_for_name(benchmark, hdf5_task_name)
    task = benchmark.get_task(task_id)
    bddl = Path(cfg.bddl_folder) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=int(args.camera_size),
        camera_widths=int(args.camera_size),
    )
    started = time.time()
    step_trace: list[dict[str, Any]] = []
    success = False
    last_info: dict[str, Any] = {}
    steps = 0
    try:
        env.seed(int(args.seed))
        env.reset()
        if args.init_state_hdf5 is None:
            init_states = benchmark.get_task_init_states(task_id)
            obs = env.set_init_state(init_states[int(args.init_state_id) % int(init_states.shape[0])])
            init_source = "suite"
        else:
            obs = env.set_init_state(_load_hdf5_init_state(args.init_state_hdf5, args.init_state_demo_id))
            init_source = "hdf5"
        for _ in range(int(args.warmup_steps)):
            obs, _reward, _done, _info = env.step(np.zeros(7, dtype=np.float32))
        algo.reset()
        task_emb = benchmark.get_task_emb(task_id)
        action_history = deque(
            [np.zeros(7, dtype=np.float32) for _ in range(max(0, int(args.action_history_len)))],
            maxlen=max(0, int(args.action_history_len)),
        )
        plan_stage = 0
        for steps in range(1, int(args.max_steps) + 1):
            frame_path = None
            if args.save_frames_dir is not None and (
                args.save_frame_every <= 1 or (steps - 1) % int(args.save_frame_every) == 0
            ):
                frame_path = _save_frame(
                    np.asarray(obs[args.camera_key]),
                    args.save_frames_dir / f"task{task_id:03d}_step{steps:04d}.png",
                )
            lowdim_state = _extract_lowdim(obs) if args.trace_lowdim else None
            object_state = _extract_object_state(obs) if args.trace_object_state else None
            named_poses = _extract_named_poses(obs) if (args.trace_object_state or args.trace_plan_state) else None
            plan_state = None
            if args.trace_plan_state:
                if _is_task1_put_cream_butter(task.language):
                    plan_stage = _update_task1_plan_stage(plan_stage, named_poses)
                    plan_state = _plan_state_from_stage(plan_stage, named_poses, dim=int(args.plan_state_dim))
                else:
                    plan_state = np.zeros(int(args.plan_state_dim), dtype=np.float32)
            hist_arr = np.stack(list(action_history), axis=0) if args.action_history_len > 0 else None
            progress_denominator = float(args.progress_denominator)
            if progress_denominator <= 0:
                progress_denominator = max(1.0, float(args.max_steps - 1))
            progress_state = (
                min(1.0, max(0.0, float(steps - 1) / progress_denominator))
                if args.trace_progress
                else None
            )
            data = raw_obs_to_tensor_obs([obs], task_emb, cfg)
            action = _teacher_action(
                algo.policy,
                data,
                deterministic=bool(args.deterministic_action),
            ).reshape(-1, 7)[0]
            obs, reward, done, info = env.step(action)
            if args.action_history_len > 0:
                action_history.append(action.astype(np.float32))
            success = bool(done) or bool(reward >= 1.0) or bool(env.check_success())
            last_info = dict(info or {})
            last_info.update({"reward": float(reward), "done": float(done)})
            action_chunk = np.repeat(action[None].astype(np.float32), int(args.action_chunk_len), axis=0)
            item = {
                "step": int(steps),
                "action": action.astype(float).tolist(),
                "action_chunk_raw": action_chunk.astype(float).tolist(),
                "reward": float(reward),
                "done": bool(done),
                "success": bool(success),
            }
            if lowdim_state is not None:
                item["lowdim_state"] = lowdim_state.astype(float).tolist()
            if object_state is not None:
                item["object_state"] = object_state.astype(float).tolist()
            if named_poses is not None:
                item["named_poses"] = named_poses
            if plan_state is not None:
                item["plan_state"] = plan_state.astype(float).tolist()
                item["plan_stage"] = int(plan_stage)
            if hist_arr is not None:
                item["action_history"] = hist_arr.astype(float).tolist()
            if progress_state is not None:
                item["progress_state"] = float(progress_state)
                item["progress_denominator"] = float(progress_denominator)
            if frame_path is not None:
                item["frame_path"] = frame_path
            step_trace.append(item)
            if success or done:
                break
    finally:
        env.close()

    report = {
        "mode": "rollout",
        "teacher_ckpt": str(args.ckpt),
        "benchmark": cfg.benchmark_name,
        "task_order_index": int(cfg.data.task_order_index),
        "task_id": int(task_id),
        "task_name": task.name,
        "instruction": task.language,
        "init_state_source": init_source,
        "init_state_id": int(args.init_state_id) if args.init_state_hdf5 is None else None,
        "init_state_hdf5": str(args.init_state_hdf5) if args.init_state_hdf5 is not None else None,
        "init_state_demo_id": args.init_state_demo_id if args.init_state_hdf5 is not None else None,
        "warmup_steps": int(args.warmup_steps),
        "success": bool(success),
        "steps": int(steps),
        "elapsed_sec": float(time.time() - started),
        "last_info": last_info,
        "step_trace": step_trace,
    }
    report["results"] = [
        {
            "suite": cfg.benchmark_name,
            "task_id": int(task_id),
            "task_name": task.name,
            "instruction": task.language,
            "init_state_id": int(args.init_state_id) if args.init_state_hdf5 is None else None,
            "init_state_source": init_source,
            "init_state_hdf5": str(args.init_state_hdf5) if args.init_state_hdf5 is not None else None,
            "init_state_demo_id": args.init_state_demo_id if args.init_state_hdf5 is not None else None,
            "success": bool(success),
            "steps": int(steps),
            "last_info": last_info,
            "step_trace": step_trace,
        }
    ]
    report["success_rate"] = float(success)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("success", "steps", "elapsed_sec", "task_id", "task_name")}, sort_keys=True), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    train = sub.add_parser("train")
    train.add_argument("--libero_root", type=Path, default=Path("/data/Minko/benchmarks/LIBERO"))
    train.add_argument("--benchmark_name", default="LIBERO_10")
    train.add_argument("--task_order_index", type=int, default=0)
    train.add_argument("--task_id", type=int, default=-1)
    train.add_argument("--task_name", default="")
    train.add_argument("--policy", default="bc_transformer_policy")
    train.add_argument("--algo", default="single_task")
    train.add_argument("--seed", type=int, default=10000)
    train.add_argument("--device", default="cuda:0")
    train.add_argument(
        "--task_embedding_format",
        default="onehot_no_bert",
        choices=("onehot_no_bert", "zero_no_bert", "bert", "clip", "gpt2", "roberta"),
    )
    train.add_argument("--seq_len", type=int, default=10)
    train.add_argument("--image_size", type=int, default=128)
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--batch_size", type=int, default=32)
    train.add_argument("--num_workers", type=int, default=4)
    train.add_argument("--eval_every", type=int, default=5)
    train.add_argument("--eval_episodes", type=int, default=5)
    train.add_argument("--eval_num_procs", type=int, default=1)
    train.add_argument("--eval_max_steps", type=int, default=300)
    train.add_argument("--no_eval", action="store_true")
    train.add_argument("--no_augmentation", action="store_true")
    train.add_argument("--out_dir", type=Path, required=True)
    train.set_defaults(func=_train)

    rollout = sub.add_parser("rollout")
    rollout.add_argument("--libero_root", type=Path, default=Path("/data/Minko/benchmarks/LIBERO"))
    rollout.add_argument("--ckpt", type=Path, required=True)
    rollout.add_argument("--task_id", type=int, default=1)
    rollout.add_argument("--init_state_id", type=int, default=0)
    rollout.add_argument("--init_state_hdf5", type=Path, default=None)
    rollout.add_argument("--init_state_demo_id", default="demo_0")
    rollout.add_argument("--max_steps", type=int, default=300)
    rollout.add_argument("--warmup_steps", type=int, default=0)
    rollout.add_argument("--seed", type=int, default=0)
    rollout.add_argument("--camera_key", default="agentview_image")
    rollout.add_argument("--camera_size", type=int, default=128)
    rollout.add_argument("--device", default="cuda:0")
    rollout.add_argument("--low_eval_noise", action="store_true")
    rollout.add_argument("--deterministic_action", action="store_true")
    rollout.add_argument("--save_frames_dir", type=Path, default=None)
    rollout.add_argument("--save_frame_every", type=int, default=0)
    rollout.add_argument("--trace_lowdim", action="store_true")
    rollout.add_argument("--trace_object_state", action="store_true")
    rollout.add_argument("--trace_plan_state", action="store_true")
    rollout.add_argument("--plan_state_dim", type=int, default=17)
    rollout.add_argument("--action_history_len", type=int, default=0)
    rollout.add_argument("--action_chunk_len", type=int, default=8)
    rollout.add_argument("--trace_progress", action="store_true")
    rollout.add_argument("--progress_denominator", type=float, default=0.0)
    rollout.add_argument("--out", type=Path, required=True)
    rollout.set_defaults(func=_rollout)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
