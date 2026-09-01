"""Fail-closed runtime gate for the exact V7 factual full-sequence topology.

This tool performs no optimizer step and never inspects model source text.  It
observes one real production-shaped batch at runtime and rejects candidates
that merely flatten tokens at a bridge while retaining factorized factual
blocks.  The required topology is the exact V7 contract adapted to grouped
actions: both factual streams carry ``task + T*P + K*G`` tokens through every
block, configured bridges update the two pre-bridge streams synchronously, and
the factual decoder consumes the complete post-block StateStream.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import inspect
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import torch
from torch import nn


SCHEMA = "wm3d_exact_v7_fullsequence_runtime_gate_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v8-code-root", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _unwrap(module: nn.Module) -> nn.Module:
    value = module
    seen: set[int] = set()
    while id(value) not in seen:
        seen.add(id(value))
        child = getattr(value, "_checkpoint_wrapped_module", None)
        if not isinstance(child, nn.Module):
            break
        value = child
    return value


def _tensor(value: Any) -> torch.Tensor | None:
    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            found = _tensor(item)
            if found is not None:
                return found
    if isinstance(value, Mapping):
        for item in value.values():
            found = _tensor(item)
            if found is not None:
                return found
    return None


def _cpu(value: torch.Tensor) -> torch.Tensor:
    return value.detach().to(device="cpu", copy=True)


def _rms(value: torch.Tensor) -> float:
    return float(value.float().square().mean().sqrt())


def _max_abs(value: torch.Tensor) -> float:
    return float(value.float().abs().max()) if value.numel() else 0.0


def _named_arguments(
    module: nn.Module,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> list[tuple[str, Any]]:
    try:
        parameters = list(inspect.signature(_unwrap(module).forward).parameters)
    except (TypeError, ValueError):
        parameters = []
    result = []
    for index, value in enumerate(args):
        name = parameters[index] if index < len(parameters) else f"arg{index}"
        result.append((name, value))
    result.extend((str(name), value) for name, value in kwargs.items())
    return result


def _extract_sequence_and_valid(
    module: nn.Module,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    *,
    expected_length: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None, str | None]:
    sequence = None
    masks: list[tuple[str, torch.Tensor]] = []
    for name, value in _named_arguments(module, args, kwargs):
        if not torch.is_tensor(value):
            continue
        if value.ndim == 3 and value.shape[1] == expected_length and sequence is None:
            sequence = value
        if (
            value.dtype == torch.bool
            and value.ndim == 2
            and value.shape[1] == expected_length
        ):
            masks.append((name, value))
    if sequence is None or not masks:
        return sequence, None, None
    preferred = [
        item
        for item in masks
        if item[0] in {"token_mask", "valid_mask", "sequence_valid"}
    ]
    if len(preferred) == 1:
        name, mask = preferred[0]
    elif len(masks) == 1:
        name, mask = masks[0]
    else:
        return sequence, None, "ambiguous_multiple_masks"
    lowered = name.lower()
    if "padding" in lowered:
        return sequence, ~mask, "padding_is_invalid"
    if "valid" in lowered or "token_mask" in lowered or lowered.endswith("mask"):
        return sequence, mask, "true_is_valid"
    return sequence, None, f"ambiguous:{name}"


def _expected_kg_mask(batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    fine = batch["future_factual_fine_sample_mask"] & batch[
        "future_factual_fine_action_mask"
    ].any(dim=-1)
    fine_present = fine.any(dim=-1)
    coarse_present = batch["future_factual_coarse_action_mask"].any(dim=-1)
    return (fine_present | coarse_present) & batch["action_group_mask"][:, None]


def _expected_full_mask(
    batch: Mapping[str, torch.Tensor], *, t: int, p: int
) -> torch.Tensor:
    kg = _expected_kg_mask(batch).reshape(batch["world_tokens"].shape[0], -1)
    prefix = torch.ones(
        kg.shape[0], 1 + t * p, dtype=torch.bool, device=kg.device
    )
    return torch.cat((prefix, kg), dim=1)


def action_variant(
    batch: Mapping[str, torch.Tensor], mode: str
) -> dict[str, torch.Tensor]:
    result = dict(batch)
    for name in (
        "future_factual_fine_action_values",
        "future_factual_coarse_action_values",
    ):
        if mode == "zero":
            result[name] = torch.zeros_like(batch[name])
        elif mode == "shuffle":
            result[name] = batch[name].roll(1, dims=1)
        else:
            raise ValueError(mode)
    return result


def make_two_group_variant(
    batch: Mapping[str, torch.Tensor], *, max_group_id: int
) -> dict[str, torch.Tensor]:
    """Duplicate the real group into a distinct second owner for runtime ABI checks."""

    result = {
        name: value.clone() if torch.is_tensor(value) else value
        for name, value in batch.items()
    }
    if result["action_group_mask"].shape[1] < 2:
        raise RuntimeError("multi-group runtime check needs group capacity >= 2")
    axis_one = (
        "action_semantic_ids",
        "state_semantic_ids",
        "composition_operator_ids",
        "current_state_values",
        "current_state_mask",
        "action_normalization_offset",
        "action_normalization_scale",
        "state_normalization_offset",
        "state_normalization_scale",
        "policy_query_dt",
        "policy_query_mask",
    )
    axis_two = (
        "history_coarse_action_values",
        "history_coarse_action_mask",
        "history_fine_action_values",
        "history_fine_action_mask",
        "history_fine_action_dt",
        "history_fine_sample_mask",
        "future_factual_coarse_action_values",
        "future_factual_coarse_action_mask",
        "future_factual_fine_action_values",
        "future_factual_fine_action_mask",
        "future_factual_fine_action_dt",
        "future_factual_fine_sample_mask",
    )
    for name in axis_one:
        value = result.get(name)
        if torch.is_tensor(value):
            value[:, 1].copy_(value[:, 0])
    for name in axis_two:
        value = result.get(name)
        if torch.is_tensor(value):
            value[:, :, 1].copy_(value[:, :, 0])
    # Make the second physical command observably different from the first.
    result["future_factual_fine_action_values"][:, :, 1].mul_(0.37).add_(0.11)
    result["future_factual_coarse_action_values"][:, :, 1].mul_(0.37).add_(0.11)
    result["action_group_mask"][:, 1] = True
    first_id = result["action_group_ids"][:, 0]
    result["action_group_ids"][:, 1] = (first_id + 1).clamp_max(max_group_id - 1)
    return result


def add_state_normalization(
    batch: dict[str, Any], config: Mapping[str, Any]
) -> None:
    if "state_normalization_offset" in batch:
        return
    from wm3d.data.grouped_normalization import GroupedRobotNormalizer
    from wm3d.data.manifest_contract import load_data_profile

    closure = config["data_closure"]
    profile = load_data_profile(Path(str(closure["data_profile_path"])))
    artifact = json.loads(
        Path(str(closure["grouped_normalization_path"])).read_text(
            encoding="utf-8"
        )
    )
    normalizer = GroupedRobotNormalizer(artifact, data_profile=profile)
    offsets = []
    scales = []
    for index in range(int(batch["source_id"].shape[0])):
        source = profile.source_order[int(batch["source_id"][index])]
        values = normalizer.tensors_for(
            source=source,
            embodiment_id=int(batch["embodiment_ids"][index]),
            group_ids=batch["action_group_ids"][index],
            action_semantic_ids=batch["action_semantic_ids"][index],
            state_semantic_ids=batch["state_semantic_ids"][index],
        )
        offsets.append(values.state_offset)
        scales.append(values.state_scale)
    batch["state_normalization_offset"] = torch.stack(offsets)
    batch["state_normalization_scale"] = torch.stack(scales)


def _module_list(model: nn.Module, name: str) -> list[nn.Module]:
    value = getattr(model, name, None)
    if not isinstance(value, (nn.ModuleList, list, tuple)):
        return []
    return [item for item in value if isinstance(item, nn.Module)]


@dataclass
class BlockCapture:
    role: str
    index: int
    sequence: int
    rank: int
    input_shape: list[int]
    output_shape: list[int] | None = None
    mask_shape: list[int] | None = None
    mask_semantics: str | None = None
    mask_matches_expected: bool = False
    invalid_input_max_abs: float | None = None
    invalid_output_max_abs: float | None = None
    attention_calls: int = 0
    attention_is_noncausal: bool = True
    attention_key_mask_matches: bool = True


class ExactRuntimeTrace:
    """Runtime-only hooks for the shared policy and exact factual paths."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.cfg = model.cfg
        self.state_blocks = _module_list(model, "state_blocks")
        self.action_blocks = _module_list(model, "action_blocks")
        self.bridges = _module_list(model, "bridges")
        self.dynamics = _module_list(model, "dynamics_blocks")
        self.handles: list[Any] = []
        self.current_label: str | None = None
        self.current_expected: torch.Tensor | None = None
        self.runs: dict[str, dict[str, Any]] = {}
        self._sequence = 0
        self._pending_blocks: dict[int, list[dict[str, Any]]] = defaultdict(list)
        self._pending_bridges: dict[int, list[dict[str, Any]]] = defaultdict(list)
        self._active_full_block: dict[tuple[str, int], dict[str, Any]] = {}
        self._active_full_bridge: dict[int, dict[str, Any]] = {}
        self.representatives: dict[str, tuple[nn.Module, tuple[Any, ...], dict[str, Any]]] = {}
        self._install()

    def _install(self) -> None:
        for role, modules in (
            ("state", self.state_blocks),
            ("action", self.action_blocks),
        ):
            for index, module in enumerate(modules):
                self.handles.append(
                    module.register_forward_pre_hook(
                        self._block_pre(role, index, module), with_kwargs=True
                    )
                )
                self.handles.append(
                    module.register_forward_hook(
                        self._block_post(role, index), with_kwargs=True
                    )
                )
                inner = _unwrap(module)
                attention = getattr(inner, "spatial" if role == "state" else "attn", None)
                if isinstance(attention, nn.Module):
                    self.handles.append(
                        attention.register_forward_pre_hook(
                            self._attention_pre(role, index), with_kwargs=True
                        )
                    )
        for index, module in enumerate(self.bridges):
            self.handles.append(
                module.register_forward_pre_hook(
                    self._bridge_pre(index, module), with_kwargs=True
                )
            )
            self.handles.append(
                module.register_forward_hook(
                    self._bridge_post(index), with_kwargs=True
                )
            )
            inner = _unwrap(module)
            for direction in ("action_reads_state", "state_reads_action"):
                cross = getattr(inner, direction, None)
                if isinstance(cross, nn.Module):
                    self.handles.append(
                        cross.register_forward_pre_hook(
                            self._bridge_cross_pre(index, direction), with_kwargs=True
                        )
                    )
        for index, module in enumerate(self.dynamics):
            self.handles.append(
                module.register_forward_pre_hook(
                    self._decoder_pre(index), with_kwargs=True
                )
            )
        state_norm = getattr(self.model, "state_norm", None)
        if isinstance(state_norm, nn.Module):
            self.handles.append(state_norm.register_forward_hook(self._state_norm_post))
        adapter = getattr(self.model, "original_v7_rgb_action", None)
        if isinstance(adapter, nn.Module):
            self.handles.append(
                adapter.register_forward_hook(self._adapter_post, with_kwargs=True)
            )
        group_cross = getattr(self.model, "factual_v7_group_query_cross", None)
        if isinstance(group_cross, nn.Module):
            self.handles.append(
                group_cross.register_forward_pre_hook(
                    self._group_cross_pre, with_kwargs=True
                )
            )
        query_action = getattr(self.model, "factual_v7_query_action", None)
        if isinstance(query_action, nn.Module):
            self.handles.append(
                query_action.register_forward_pre_hook(
                    self._query_action_pre, with_kwargs=True
                )
            )
        rgb_head = getattr(self.model, "rgb_head", None)
        if isinstance(rgb_head, nn.Module):
            self.handles.append(
                rgb_head.register_forward_pre_hook(
                    self._rgb_pre, with_kwargs=True
                )
            )

    def begin(self, label: str, expected_mask: torch.Tensor) -> None:
        self.current_label = label
        self.current_expected = expected_mask
        self._sequence = 0
        self.runs[label] = {
            "events": [],
            "blocks": {"state": [], "action": []},
            "bridges": [],
            "decoder_memories": [],
            "full_state_norm_outputs": [],
            "adapter_calls": [],
            "group_query_cross_calls": [],
            "query_action_inputs": [],
            "rgb_action_inputs": [],
        }

    def end(self) -> None:
        self.current_label = None
        self.current_expected = None

    def close(self) -> None:
        self.end()
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    @property
    def run(self) -> dict[str, Any] | None:
        if self.current_label is None:
            return None
        return self.runs[self.current_label]

    def _event(self, kind: str, index: int, rank: int) -> int:
        run = self.run
        assert run is not None
        sequence = self._sequence
        self._sequence += 1
        run["events"].append(
            {"sequence": sequence, "kind": kind, "index": index, "rank": rank}
        )
        return sequence

    def _block_pre(self, role: str, index: int, module: nn.Module):
        def hook(_module, args, kwargs):
            if self.run is None:
                return
            value = _tensor(args[0] if args else kwargs)
            if value is None:
                return
            capture = BlockCapture(
                role=role,
                index=index,
                sequence=self._event(f"{role}_block", index, value.ndim),
                rank=value.ndim,
                input_shape=list(value.shape),
            )
            item: dict[str, Any] = {"capture": capture}
            if value.ndim == 3 and self.current_expected is not None:
                sequence, valid, semantics = _extract_sequence_and_valid(
                    module,
                    args,
                    kwargs,
                    expected_length=self.current_expected.shape[1],
                )
                capture.mask_semantics = semantics
                if valid is not None and sequence is not None:
                    capture.mask_shape = list(valid.shape)
                    capture.mask_matches_expected = bool(
                        torch.equal(valid, self.current_expected)
                    )
                    capture.invalid_input_max_abs = _max_abs(
                        sequence.masked_select(~valid[..., None].expand_as(sequence))
                    )
                    item["valid"] = valid
                    if index == 0:
                        item["input"] = _cpu(sequence)
                        item["valid_cpu"] = _cpu(valid)
                        if self.current_label == "factual" and role not in self.representatives:
                            self.representatives[role] = (
                                module,
                                tuple(_cpu(v) if torch.is_tensor(v) else v for v in args),
                                {
                                    name: _cpu(v) if torch.is_tensor(v) else v
                                    for name, v in kwargs.items()
                                },
                            )
                self._active_full_block[(role, index)] = item
            self.run["blocks"][role].append(item)
            self._pending_blocks[id(module)].append(item)

        return hook

    def _block_post(self, role: str, index: int):
        def hook(module, _args, _kwargs, output):
            if self.run is None or not self._pending_blocks[id(module)]:
                return
            item = self._pending_blocks[id(module)].pop()
            value = _tensor(output)
            if value is None:
                return
            capture: BlockCapture = item["capture"]
            capture.output_shape = list(value.shape)
            valid = item.get("valid")
            if valid is not None:
                capture.invalid_output_max_abs = _max_abs(
                    value.masked_select(~valid[..., None].expand_as(value))
                )
            self._active_full_block.pop((role, index), None)

        return hook

    def _attention_pre(self, role: str, index: int):
        def hook(_module, _args, kwargs):
            item = self._active_full_block.get((role, index))
            if item is None:
                return
            capture: BlockCapture = item["capture"]
            capture.attention_calls += 1
            capture.attention_is_noncausal &= not bool(kwargs.get("is_causal", False))
            allowed = kwargs.get("allowed_mask")
            valid = item.get("valid")
            if not torch.is_tensor(allowed) or valid is None:
                capture.attention_key_mask_matches = False
                return
            expected = valid[:, None, None, :]
            capture.attention_key_mask_matches &= bool(torch.equal(allowed, expected))

        return hook

    def _bridge_pre(self, index: int, module: nn.Module):
        def hook(_module, args, _kwargs):
            if self.run is None or len(args) < 3:
                return
            state, action, mask = args[:3]
            if not all(torch.is_tensor(value) for value in (state, action, mask)):
                return
            item: dict[str, Any] = {
                "index": index,
                "rank": state.ndim,
                "sequence": self._event("bridge", index, state.ndim),
                "state_shape": list(state.shape),
                "action_shape": list(action.shape),
                "mask_shape": list(mask.shape),
            }
            if state.ndim == 3:
                item.update(
                    {
                        "mask_matches_expected": bool(
                            self.current_expected is not None
                            and torch.equal(mask, self.current_expected)
                        ),
                        "pre_state": state,
                        "pre_action": action,
                        "cross": {},
                    }
                )
                self._active_full_bridge[index] = item
            self.run["bridges"].append(item)
            self._pending_bridges[id(module)].append(item)

        return hook

    def _bridge_cross_pre(self, index: int, direction: str):
        def hook(_module, args, _kwargs):
            item = self._active_full_bridge.get(index)
            if item is None or len(args) < 2:
                return
            query, context = args[:2]
            if not torch.is_tensor(query) or not torch.is_tensor(context):
                return
            bridge = _unwrap(self.bridges[index])
            expected_state = bridge.state_norm(item["pre_state"])
            expected_action = bridge.action_norm(item["pre_action"])
            if direction == "action_reads_state":
                expected_query, expected_context = expected_action, expected_state
            else:
                expected_query, expected_context = expected_state, expected_action
            item["cross"][direction] = {
                "query_matches_pre": bool(torch.equal(query, expected_query)),
                "context_matches_pre": bool(torch.equal(context, expected_context)),
                "query_shape": list(query.shape),
                "context_shape": list(context.shape),
            }

        return hook

    def _bridge_post(self, index: int):
        def hook(module, _args, _kwargs, output):
            if self.run is None or not self._pending_bridges[id(module)]:
                return
            item = self._pending_bridges[id(module)].pop()
            if item["rank"] == 3 and isinstance(output, (tuple, list)) and len(output) == 2:
                state, action = output
                mask = self.current_expected
                item["state_output_shape"] = list(state.shape)
                item["action_output_shape"] = list(action.shape)
                if (
                    mask is not None
                    and tuple(state.shape[:2]) == tuple(mask.shape)
                    and tuple(action.shape[:2]) == tuple(mask.shape)
                ):
                    item["invalid_state_output_max_abs"] = _max_abs(
                        state.masked_select(~mask[..., None].expand_as(state))
                    )
                    item["invalid_action_output_max_abs"] = _max_abs(
                        action.masked_select(~mask[..., None].expand_as(action))
                    )
                else:
                    item["invalid_state_output_max_abs"] = None
                    item["invalid_action_output_max_abs"] = None
                item["state_update_rms"] = _rms(state - item["pre_state"])
                item["action_update_rms"] = _rms(action - item["pre_action"])
                # Drop live references after deriving the runtime evidence.
                item.pop("pre_state", None)
                item.pop("pre_action", None)
                self._active_full_bridge.pop(index, None)

        return hook

    def _decoder_pre(self, index: int):
        def hook(_module, args, _kwargs):
            if self.run is None or len(args) < 3:
                return
            memory, valid = args[1], args[2]
            if torch.is_tensor(memory) and torch.is_tensor(valid):
                self.run["decoder_memories"].append(
                    {
                        "index": index,
                        "shape": list(memory.shape),
                        "valid_shape": list(valid.shape),
                        "memory": _cpu(memory),
                        "valid": _cpu(valid),
                    }
                )

        return hook

    def _state_norm_post(self, _module, _args, output):
        if self.run is not None and torch.is_tensor(output) and output.ndim == 3:
            self.run["full_state_norm_outputs"].append(_cpu(output))

    def _adapter_post(self, _module, _args, kwargs, output):
        if self.run is None:
            return
        tensors = []
        if isinstance(output, (tuple, list)):
            tensors = [value for value in output if torch.is_tensor(value)]
        elif torch.is_tensor(output):
            tensors = [output]
        self.run["adapter_calls"].append(
            {
                "return_grouped": bool(kwargs.get("return_grouped", False)),
                "output_shapes": [list(value.shape) for value in tensors],
                "outputs": [_cpu(value) for value in tensors],
            }
        )

    def _group_cross_pre(self, _module, args, kwargs):
        if self.run is None or len(args) < 2:
            return
        query, context = args[:2]
        allowed = kwargs.get("allowed_mask")
        self.run["group_query_cross_calls"].append(
            {
                "query_shape": list(query.shape),
                "context_shape": list(context.shape),
                "allowed_shape": list(allowed.shape) if torch.is_tensor(allowed) else None,
                "context_rms": _rms(context),
            }
        )

    def _query_action_pre(self, _module, args, _kwargs):
        if self.run is not None and args and torch.is_tensor(args[0]):
            self.run["query_action_inputs"].append(
                {"shape": list(args[0].shape), "max_abs": _max_abs(args[0])}
            )

    def _rgb_pre(self, module, args, kwargs):
        if self.run is None:
            return
        values = dict(_named_arguments(module, args, kwargs))
        action = values.get("factual_action_summary")
        self.run["rgb_action_inputs"].append(
            {
                "shape": list(action.shape) if torch.is_tensor(action) else None,
                "max_abs": _max_abs(action) if torch.is_tensor(action) else 0.0,
            }
        )


def _block_to_dict(item: Mapping[str, Any]) -> dict[str, Any]:
    capture: BlockCapture = item["capture"]
    return {
        name: getattr(capture, name)
        for name in BlockCapture.__dataclass_fields__
    }


def _segment_checks(
    item: Mapping[str, Any], *, t: int, p: int, valid: torch.Tensor
) -> dict[str, Any]:
    sequence = item.get("input")
    if not torch.is_tensor(sequence):
        return {"observable": False}
    tp = t * p
    task = sequence[:, :1]
    state = sequence[:, 1 : 1 + tp]
    action = sequence[:, 1 + tp :]
    kg_valid = valid[:, 1 + tp :]
    valid_action = action.masked_select(kg_valid[..., None].expand_as(action))
    invalid_action = action.masked_select(~kg_valid[..., None].expand_as(action))
    return {
        "observable": True,
        "task_rms": _rms(task),
        "state_rms": _rms(state),
        "state_token_variance": float(state.float().var(dim=1, unbiased=False).mean()),
        "valid_action_rms": _rms(valid_action),
        "invalid_action_max_abs": _max_abs(invalid_action),
    }


def evaluate_runtime_trace(
    trace: ExactRuntimeTrace,
    outputs: Mapping[str, Mapping[str, torch.Tensor]],
    expected_masks: Mapping[str, torch.Tensor],
) -> tuple[dict[str, bool], dict[str, Any]]:
    cfg = trace.cfg
    expected_length = 1 + cfg.T * cfg.P + cfg.K * cfg.max_action_groups
    checks: dict[str, bool] = {
        "state_blocks_are_present": bool(trace.state_blocks),
        "action_blocks_are_present": bool(trace.action_blocks),
        "bridges_are_present": bool(trace.bridges),
    }
    details: dict[str, Any] = {"runs": {}}
    schedule = tuple(
        cfg.factual_v7_bridge_layers_state
        if cfg.factual_v7_bridge_layers_state
        else cfg.bridge_layers_state
    )
    checks["bridge_schedule_is_frozen_sorted_unique"] = (
        len(schedule) == len(trace.bridges)
        and tuple(sorted(set(schedule))) == schedule
        and all(0 <= value < len(trace.state_blocks) for value in schedule)
    )
    for label, run in trace.runs.items():
        expected = expected_masks[label].cpu()
        lane: dict[str, Any] = {
            "state_blocks": [_block_to_dict(item) for item in run["blocks"]["state"]],
            "action_blocks": [_block_to_dict(item) for item in run["blocks"]["action"]],
            "bridges": [
                {key: value for key, value in item.items() if not torch.is_tensor(value)}
                for item in run["bridges"]
            ],
            "adapter_calls": [
                {key: value for key, value in item.items() if key != "outputs"}
                for item in run["adapter_calls"]
            ],
            "group_query_cross_calls": run["group_query_cross_calls"],
            "query_action_inputs": run["query_action_inputs"],
            "rgb_action_inputs": run["rgb_action_inputs"],
        }
        full_by_role: dict[str, list[dict[str, Any]]] = {}
        for role, modules in (
            ("state", trace.state_blocks),
            ("action", trace.action_blocks),
        ):
            items = [item for item in run["blocks"][role] if item["capture"].rank == 3]
            factorized = [
                item for item in run["blocks"][role] if item["capture"].rank == 4
            ]
            full_by_role[role] = items
            checks[f"{label}_{role}_every_module_has_one_full_call"] = (
                len(items) == len(modules)
                and sorted(item["capture"].index for item in items)
                == list(range(len(modules)))
            )
            checks[f"{label}_{role}_policy_calls_stay_rank4_once"] = (
                len(factorized) == len(modules)
                and sorted(item["capture"].index for item in factorized)
                == list(range(len(modules)))
            )
            checks[f"{label}_{role}_all_full_lengths_and_masks_exact"] = bool(items) and all(
                item["capture"].input_shape[1] == expected_length
                and item["capture"].output_shape is not None
                and item["capture"].output_shape[1] == expected_length
                and item["capture"].mask_matches_expected
                for item in items
            )
            checks[f"{label}_{role}_all_full_blocks_noncausal_with_strict_key_mask"] = bool(items) and all(
                item["capture"].attention_calls == 1
                and item["capture"].attention_is_noncausal
                and item["capture"].attention_key_mask_matches
                for item in items
            )
            checks[f"{label}_{role}_invalid_queries_zero_after_every_block"] = bool(items) and all(
                item["capture"].invalid_output_max_abs == 0.0 for item in items
            )
            block0 = next(
                (item for item in items if item["capture"].index == 0), None
            )
            segment = (
                _segment_checks(block0, t=cfg.T, p=cfg.P, valid=expected)
                if block0 is not None
                else {"observable": False}
            )
            lane[f"{role}_block0_segments"] = segment
            checks[f"{label}_{role}_block0_has_real_task_TP_KG_segments"] = bool(
                segment.get("observable")
                and segment.get("task_rms", 0.0) > 0.0
                and segment.get("state_rms", 0.0) > 0.0
                and segment.get("state_token_variance", 0.0) > 0.0
                and segment.get("valid_action_rms", 0.0) > 0.0
                and segment.get("invalid_action_max_abs") == 0.0
            )

        full_bridges = [item for item in run["bridges"] if item["rank"] == 3]
        factorized_bridges = [item for item in run["bridges"] if item["rank"] == 4]
        counts = {index: 0 for index in range(len(trace.bridges))}
        for item in full_bridges:
            counts[item["index"]] += 1
        checks[f"{label}_each_factual_bridge_module_called_once"] = (
            counts == {index: 1 for index in range(len(trace.bridges))}
        )
        checks[f"{label}_policy_bridge_calls_stay_factorized_once"] = (
            len(factorized_bridges) == len(trace.bridges)
            and sorted(item["index"] for item in factorized_bridges)
            == list(range(len(trace.bridges)))
        )
        checks[f"{label}_bridges_are_synchronous_from_pre_streams"] = bool(full_bridges) and all(
            set(item.get("cross", {}))
            == {"action_reads_state", "state_reads_action"}
            and all(
                cross["query_matches_pre"] and cross["context_matches_pre"]
                for cross in item["cross"].values()
            )
            for item in full_bridges
        )
        checks[f"{label}_bridge_lengths_masks_updates_are_exact"] = bool(full_bridges) and all(
            item["state_shape"][1] == expected_length
            and item["action_shape"][1] == expected_length
            and item.get("state_output_shape", [0, 0])[1] == expected_length
            and item.get("action_output_shape", [0, 0])[1] == expected_length
            and item.get("mask_matches_expected", False)
            and item.get("invalid_state_output_max_abs") == 0.0
            and item.get("invalid_action_output_max_abs") == 0.0
            and item.get("state_update_rms", 0.0) > 0.0
            and item.get("action_update_rms", 0.0) > 0.0
            for item in full_bridges
        )
        ordered = True
        for bridge_index, state_index in enumerate(schedule):
            bridge = next(
                (item for item in full_bridges if item["index"] == bridge_index), None
            )
            if bridge is None:
                ordered = False
                continue
            prior_state = [
                event
                for event in run["events"]
                if event["kind"] == "state_block"
                and event["rank"] == 3
                and event["sequence"] < bridge["sequence"]
            ]
            ordered &= bool(prior_state) and prior_state[-1]["index"] == state_index
        checks[f"{label}_bridges_follow_only_frozen_schedule"] = ordered

        memories = run["decoder_memories"]
        post_state = run["full_state_norm_outputs"]
        checks[f"{label}_decoder_memory_is_complete_post_block_state_sequence"] = bool(
            memories
            and post_state
            and all(
                memory["shape"][1] == expected_length
                and torch.equal(memory["valid"], expected)
                and torch.equal(memory["memory"], post_state[-1])
                for memory in memories
            )
        )
        checks[f"{label}_adapter_preserves_group_axis"] = bool(
            run["adapter_calls"]
            and all(
                item["return_grouped"]
                and item["output_shapes"]
                and len(item["output_shapes"][0]) == 4
                and item["output_shapes"][0][1:3]
                == [cfg.K, cfg.max_action_groups]
                for item in run["adapter_calls"]
            )
        )
        checks[f"{label}_query_conditioner_reads_explicit_group_axis"] = bool(
            run["group_query_cross_calls"]
            and all(
                item["context_shape"][1] == cfg.max_action_groups
                and item["allowed_shape"][-1] == cfg.max_action_groups
                for item in run["group_query_cross_calls"]
            )
        )
        details["runs"][label] = lane

    if "multi_group" in trace.runs:
        multi = trace.runs["multi_group"]
        checks["multi_group_query_does_not_use_raw_mean"] = bool(
            multi["group_query_cross_calls"]
            and multi["query_action_inputs"]
            and any(
                len(item["shape"]) == 4
                and item["shape"][1:3]
                == [cfg.K, cfg.max_action_groups]
                for item in multi["query_action_inputs"]
            )
            and all(
                item["max_abs"] == 0.0
                for item in multi["query_action_inputs"]
                if len(item["shape"]) == 3
            )
        )
        checks["multi_group_rgb_direct_skip_is_disabled_not_meaned"] = bool(
            multi["rgb_action_inputs"]
            and all(item["max_abs"] == 0.0 for item in multi["rgb_action_inputs"])
        )
    else:
        checks["multi_group_query_does_not_use_raw_mean"] = False
        checks["multi_group_rgb_direct_skip_is_disabled_not_meaned"] = False

    for other in ("zero", "shuffle"):
        checks[f"policy_is_future_invariant_{other}"] = bool(
            torch.equal(outputs["factual"]["policy_action"], outputs[other]["policy_action"])
        )
        checks[f"action_free_is_future_invariant_{other}"] = bool(
            torch.equal(
                outputs["factual"]["action_free_pred_tokens"],
                outputs[other]["action_free_pred_tokens"],
            )
        )
    return checks, details


def direct_block_behavior(
    trace: ExactRuntimeTrace, *, device: torch.device
) -> tuple[dict[str, bool], dict[str, Any]]:
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    for role in ("state", "action"):
        representative = trace.representatives.get(role)
        if representative is None:
            checks[f"{role}_representative_bidirectional"] = False
            checks[f"{role}_representative_strict_invalid_key_query_mask"] = False
            continue
        module, cpu_args, cpu_kwargs = representative
        args = tuple(value.to(device) if torch.is_tensor(value) else value for value in cpu_args)
        kwargs = {
            name: value.to(device) if torch.is_tensor(value) else value
            for name, value in cpu_kwargs.items()
        }
        expected_length = 1 + trace.cfg.T * trace.cfg.P + trace.cfg.K * trace.cfg.max_action_groups
        sequence, valid, _ = _extract_sequence_and_valid(
            module, args, kwargs, expected_length=expected_length
        )
        if sequence is None or valid is None:
            checks[f"{role}_representative_bidirectional"] = False
            checks[f"{role}_representative_strict_invalid_key_query_mask"] = False
            continue

        def run(value: torch.Tensor) -> torch.Tensor:
            replaced = list(args)
            for index, item in enumerate(replaced):
                if item is sequence:
                    replaced[index] = value
                    break
            else:
                raise RuntimeError("representative sequence was not positional")
            output = module(*tuple(replaced), **kwargs)
            found = _tensor(output)
            if found is None:
                raise RuntimeError("representative block returned no tensor")
            return found

        valid_indices = torch.nonzero(valid[0], as_tuple=False).flatten()
        invalid_indices = torch.nonzero(~valid[0], as_tuple=False).flatten()
        early = int(valid_indices[0])
        late = int(valid_indices[-1])
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            baseline = run(sequence)
            late_value = sequence.clone()
            late_value[:, late, 0] += 4.0
            late_output = run(late_value)
            early_value = sequence.clone()
            early_value[:, early, 1] += 4.0
            early_output = run(early_value)
            early_reads_late = _rms(late_output[:, early] - baseline[:, early])
            late_reads_early = _rms(early_output[:, late] - baseline[:, late])
            invalid_valid_rms = None
            invalid_query_max = None
            if invalid_indices.numel():
                invalid_value = sequence.clone()
                invalid_value[:, invalid_indices, :] += 100.0
                invalid_output = run(invalid_value)
                invalid_valid_rms = _rms(
                    (invalid_output - baseline).masked_select(
                        valid[..., None].expand_as(baseline)
                    )
                )
                invalid_query_max = _max_abs(
                    invalid_output.masked_select(
                        ~valid[..., None].expand_as(invalid_output)
                    )
                )
        details[role] = {
            "early_reads_late_rms": early_reads_late,
            "late_reads_early_rms": late_reads_early,
            "invalid_key_valid_output_rms": invalid_valid_rms,
            "invalid_query_output_max_abs": invalid_query_max,
        }
        checks[f"{role}_representative_bidirectional"] = (
            early_reads_late > 0.0 and late_reads_early > 0.0
        )
        checks[f"{role}_representative_strict_invalid_key_query_mask"] = bool(
            invalid_valid_rms is not None
            and invalid_valid_rms == 0.0
            and invalid_query_max == 0.0
        )
    return checks, details


def _validate_input(batch: Mapping[str, torch.Tensor]) -> dict[str, bool]:
    times = batch["world_times_s"].float()
    return {
        "observed_tokens_are_T16_P64": tuple(batch["world_tokens"].shape[1:])
        == (16, 3, 64, 2048),
        "target_tokens_are_K8_P64": tuple(batch["target_tokens"].shape[1:])
        == (8, 64, 2048),
        "fine_action_is_K8_grouped": (
            batch["future_factual_fine_action_values"].shape[1] == 8
            and batch["future_factual_fine_action_values"].ndim == 5
        ),
        "timestamps_are_T16_plus_K8": tuple(times.shape[1:]) == (24,),
        "timestamps_are_finite_and_strict": bool(torch.isfinite(times).all())
        and bool((times[:, 1:] > times[:, :-1]).all()),
    }


def main() -> None:
    args = parse_args()
    started = time.monotonic()
    if args.output.exists():
        raise RuntimeError("exact full-sequence gate output must be new")
    code_root = args.v8_code_root.resolve(strict=True)
    sys.path.insert(0, str(code_root))
    from wm3d.models.model_factory import build_world_model
    from wm3d.training.pretrain import _batch_to_device, _forward
    from wm3d.training.runtime_contract import load_materialized_runtime

    config, _ = load_materialized_runtime(args.runtime.resolve(strict=True))
    cpu_batch = torch.load(
        args.batch.resolve(strict=True), map_location="cpu", weights_only=False
    )
    cpu_batch.pop("_fixed_validation_seed", None)
    cpu_batch.pop("_schema", None)
    add_state_normalization(cpu_batch, config)
    model_mapping = config["model_profile"]["model"]
    if not bool(model_mapping.get("appearance_enabled", False)):
        for name in (
            "appearance_context_tokens",
            "appearance_context_mask",
            "target_appearance_tokens",
            "target_appearance_mask",
        ):
            cpu_batch.pop(name, None)
    input_checks = _validate_input(cpu_batch)
    if not all(input_checks.values()):
        raise RuntimeError(f"exact full-sequence gate input failed: {input_checks}")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    profile = dict(config["model_profile"])
    profile.pop("expected_parameter_count", None)
    with torch.device(device):
        model = build_world_model(profile)
    model.eval()
    batch = _batch_to_device(cpu_batch, device)
    variants = {
        "factual": batch,
        "zero": action_variant(batch, "zero"),
        "shuffle": action_variant(batch, "shuffle"),
    }
    two_group = make_two_group_variant(
        cpu_batch, max_group_id=int(model.cfg.max_group_id)
    )
    variants["multi_group"] = _batch_to_device(two_group, device)
    expected_masks = {
        label: _expected_full_mask(value, t=model.cfg.T, p=model.cfg.P)
        for label, value in variants.items()
    }

    trace = ExactRuntimeTrace(model)
    outputs: dict[str, Mapping[str, torch.Tensor]] = {}
    forward_error: str | None = None
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for label, value in variants.items():
                trace.begin(label, expected_masks[label])
                outputs[label] = _forward(
                    model, value, appearance_teacher_ratio=0.0
                )
                trace.end()
    except Exception as exc:  # fail closed with a durable receipt
        forward_error = f"{type(exc).__name__}: {exc}"
        trace.end()

    if forward_error is None:
        runtime_checks, runtime_details = evaluate_runtime_trace(
            trace, outputs, expected_masks
        )
        behavior_checks, behavior_details = direct_block_behavior(
            trace, device=device
        )
    else:
        runtime_checks = {"complete_runtime_forward": False}
        runtime_details = {"runs": {}}
        behavior_checks = {
            "state_representative_bidirectional": False,
            "action_representative_bidirectional": False,
        }
        behavior_details = {}
    trace.close()
    checks = {**input_checks, **runtime_checks, **behavior_checks}
    passed = forward_error is None and all(checks.values())
    receipt = {
        "schema": SCHEMA,
        "purpose": "runtime-only exact v7-base full-sequence topology; no optimizer step",
        "candidate_code_root": str(code_root),
        "forward_error": forward_error,
        "checks": checks,
        "runtime_details": runtime_details,
        "direct_block_behavior": behavior_details,
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "elapsed_seconds": time.monotonic() - started,
        "passed": passed,
        "not_claimed": [
            "held-out learning quality",
            "long-run RGB quality",
            "closed-loop VLA success",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
