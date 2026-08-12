from __future__ import annotations

import copy
import subprocess
from types import SimpleNamespace

import pytest
import torch

from scripts.eval_wm3d_v8_stage1 import (
    _ACTION_FIELDS,
    _LEARNED_FIELDS,
    _action_shuffle_invariant,
    _auc,
)
from wm3d_v3.models.native_world_model import NativeWorldModel, NativeWorldModelConfig
from wm3d_v3.data.unified_cache_dataset import CacheDataError, _active_source_names
from wm3d_v3.stage1_planner.candidates import deterministic_action_cost
from wm3d_v3.stage1_planner.dataset import (
    BRANCH_INDEX_SCHEMA,
    BRANCH_SCHEMA,
    BRANCH_SEAL_SCHEMA,
    GENERATOR_RECEIPT_FIELDS,
    GENERATOR_RECEIPT_SCHEMA,
    Stage1BranchError,
    validate_rollout_audit_binding,
    _validate_candidate_action_shapes,
)
from wm3d_v3.stage1_planner.losses import planner_loss
from wm3d_v3.stage1_planner.planner_head import NativePlannerConfig, NativePlannerHead, planning_score
from wm3d_v3.stage1_planner.rollout import single_horizon_native_rollout
from wm3d_v3.stage1_planner.system import NativePlanningSystem, Stage1SystemConfig
from wm3d_v3.stage1_planner.train import (
    _expectations,
    _planner_contract_sha,
    _topology_sha,
    _verify_runtime_checkout,
)
from scripts.materialize_wm3d_v8_stage1_branches import (
    _validate_candidate_payload,
    _validate_stage0_window_clock,
)
from scripts.produce_wm3d_v8_robocasa_stage1_candidates import (
    AUDIT_SCHEMA,
    _validate_rollout_audit_authority,
)
from wm3d_v3.training.launch_qualification import LaunchQualificationError


def test_multisource_split_requires_all_train_sources_but_not_eval_sources() -> None:
    entries = (
        SimpleNamespace(source="blender", split="train"),
        SimpleNamespace(source="coffee", split="train"),
        SimpleNamespace(source="blender", split="val"),
        SimpleNamespace(source="blender", split="test"),
    )
    kwargs = {
        "source_order": ("blender", "coffee"),
        "selected_sources": ("blender", "coffee"),
        "entries": entries,
    }
    assert _active_source_names(**kwargs, split="train") == ("blender", "coffee")
    assert _active_source_names(**kwargs, split="val") == ("blender",)
    assert _active_source_names(**kwargs, split="test") == ("blender",)

    missing_train = tuple(entry for entry in entries if entry.source != "coffee")
    with pytest.raises(CacheDataError, match="training sources have no cache windows"):
        _active_source_names(
            source_order=("blender", "coffee"),
            selected_sources=("blender", "coffee"),
            entries=missing_train,
            split="train",
        )

    with pytest.raises(CacheDataError, match="cache selection produced no samples"):
        _active_source_names(**kwargs, split="missing")


def test_stage1_new_closure_requires_rollout_audit_binding() -> None:
    assert BRANCH_SCHEMA.endswith("_v3")
    assert BRANCH_INDEX_SCHEMA.endswith("_v3")
    assert BRANCH_SEAL_SCHEMA.endswith("_v3")
    assert GENERATOR_RECEIPT_SCHEMA.endswith("_v2")
    digest = "a" * 64
    row = {"schema": BRANCH_SCHEMA, "rollout_audit_sha256": digest}
    receipt = {
        name: False if name == "future_observation_leakage" else True
        for name in GENERATOR_RECEIPT_FIELDS
    }
    receipt.update(
        schema=GENERATOR_RECEIPT_SCHEMA,
        rollout_audit_sha256=digest,
    )
    assert validate_rollout_audit_binding(row, receipt) == digest
    missing = dict(row)
    missing.pop("rollout_audit_sha256")
    with pytest.raises(Stage1BranchError, match="binding is missing"):
        validate_rollout_audit_binding(missing, receipt)
    tampered = dict(receipt, rollout_audit_sha256="b" * 64)
    with pytest.raises(Stage1BranchError, match="binding mismatch"):
        validate_rollout_audit_binding(row, tampered)

    commit = "c" * 40
    _validate_rollout_audit_authority(
        {"schema": AUDIT_SCHEMA, "passed": True, "code_commit": commit},
        commit,
    )
    with pytest.raises(RuntimeError, match="code commit differs"):
        _validate_rollout_audit_authority(
            {"schema": AUDIT_SCHEMA, "passed": True}, commit
        )
    with pytest.raises(RuntimeError, match="code commit differs"):
        _validate_rollout_audit_authority(
            {
                "schema": AUDIT_SCHEMA,
                "passed": True,
                "code_commit": "d" * 40,
            },
            commit,
        )


def test_stage1_checkout_uses_shared_clean_runtime_authority(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    observed = {}

    def fake(repo, commit):
        observed.update(repo=repo, commit=commit)
        return commit

    monkeypatch.setattr(
        "wm3d_v3.stage1_planner.train.verify_clean_runtime_checkout", fake
    )
    runtime = {"run": {"code_commit": "c" * 40}}
    assert _verify_runtime_checkout(runtime, tmp_path) == "c" * 40
    assert observed == {"repo": tmp_path, "commit": "c" * 40}


def test_stage1_checkout_rejects_dirty_repository(tmp_path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.name", "Test"],
        check=True,
    )
    (tmp_path / "code.py").write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "code.py"], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-qm", "base"], check=True
    )
    head = subprocess.check_output(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"], text=True
    ).strip()
    assert _verify_runtime_checkout(
        {"run": {"code_commit": head}}, tmp_path
    ) == head
    (tmp_path / "untracked.py").write_text("value = 2\n", encoding="utf-8")
    with pytest.raises(LaunchQualificationError, match="dirty"):
        _verify_runtime_checkout({"run": {"code_commit": head}}, tmp_path)


def _planner() -> NativePlannerHead:
    return NativePlannerHead(NativePlannerConfig(
        token_dim=16, task_dim=12, hidden=32, spatial_layers=1,
        temporal_layers=1, heads=4, mlp_mult=2, dropout=0.0,
        max_horizon=4, patches=4, num_views=2, time_fourier_dim=8,
        time_min_period_s=0.01, time_max_period_s=10.0))


def _evidence() -> dict[str, torch.Tensor]:
    return {
        "tokens": torch.randn(2, 3, 4, 4, 16),
        "future_dt_s": torch.tensor([.1, .3, .8, 1.4]).view(1,1,4).expand(2,3,4),
        "token_mask": torch.ones(2,3,4,4,dtype=torch.bool),
        "task": torch.randn(2, 12),
        "depth": torch.rand(2, 3, 4, 2, 4),
        "depth_mask": torch.ones(2,3,4,2,4,dtype=torch.bool),
        "point": torch.randn(2, 3, 4, 2, 4, 3),
        "point_mask": torch.ones(2,3,4,2,4,dtype=torch.bool),
        "pose": torch.randn(2, 3, 4, 2, 9),
        "pose_mask": torch.ones(2,3,4,2,dtype=torch.bool),
        "confidence": torch.rand(2, 3, 4, 2, 4),
        "view_mask": torch.ones(2,3,4,2,dtype=torch.bool),
    }


def _forward(planner: NativePlannerHead, evidence: dict[str, torch.Tensor]):
    return planner(evidence["tokens"], evidence["task"],
        future_dt_s=evidence["future_dt_s"], token_mask=evidence["token_mask"],
        depth=evidence["depth"], depth_mask=evidence["depth_mask"],
        point=evidence["point"], point_mask=evidence["point_mask"],
        pose=evidence["pose"], pose_mask=evidence["pose_mask"],
        geometry_confidence=evidence["confidence"], view_mask=evidence["view_mask"])


def test_planner_action_shuffle_invariance() -> None:
    torch.manual_seed(7)
    planner = _planner().eval()
    evidence = _evidence()
    actions = torch.randn(2, 3, 4, 2, 5, 6)
    first = _forward(planner, evidence)
    shuffled_actions = actions[:, torch.tensor([2, 0, 1])]
    second = _forward(planner, evidence)
    assert not torch.equal(actions, shuffled_actions)
    for name in first:
        assert torch.equal(first[name], second[name])
    assert not any("action" in name for name, _ in planner.named_parameters())


def test_eval_action_shuffle_gate_uses_one_execution_mode() -> None:
    class ModeSensitiveActionBlindSystem:
        def score_observed_batch(self, batch):
            bias = float(torch.is_grad_enabled())
            return {
                name: batch["fixed_evidence"] + bias for name in _LEARNED_FIELDS
            }

    batch = {"fixed_evidence": torch.tensor([[3.0, 5.0, 7.0]])}
    for index, name in enumerate(_ACTION_FIELDS):
        batch[name] = torch.tensor([[index, index + 10, index + 20]])
    system = ModeSensitiveActionBlindSystem()
    grad_output = system.score_observed_batch(batch)
    with torch.no_grad():
        no_grad_output = system.score_observed_batch(batch)
    assert not torch.equal(
        grad_output["progress_logit"], no_grad_output["progress_logit"]
    )
    assert _action_shuffle_invariant(system, batch, torch.tensor([2, 0, 1]))


def test_eval_auc_uses_average_ranks_for_score_ties() -> None:
    assert _auc(
        labels=torch.tensor([True, False]).numpy(),
        scores=torch.tensor([4.0, 4.0]).numpy(),
    ) == 0.5
    labels = torch.tensor([False, True, True, False]).numpy()
    scores = torch.tensor([0.0, 1.0, 1.0, 1.0]).numpy()
    assert _auc(labels=labels, scores=scores) == 0.75
    permutation = [2, 3, 0, 1]
    assert _auc(labels=labels[permutation], scores=scores[permutation]) == 0.75


def test_masked_native_evidence_is_not_treated_as_a_real_zero() -> None:
    torch.manual_seed(19)
    planner = _planner().eval()
    evidence = _evidence()
    missing = copy.deepcopy(evidence)
    missing["depth"].zero_()
    missing["depth_mask"].zero_()
    missing["point"].zero_()
    missing["point_mask"].zero_()
    missing["pose"].zero_()
    missing["pose_mask"].zero_()
    invalid_values = copy.deepcopy(missing)
    invalid_values["depth"].fill_(1e6)
    invalid_values["point"].fill_(-1e6)
    invalid_values["pose"].fill_(1e6)
    missing_out = _forward(planner, missing)
    invalid_out = _forward(planner, invalid_values)
    assert all(torch.equal(missing_out[name], invalid_out[name]) for name in missing_out)
    measured_zero = copy.deepcopy(missing)
    measured_zero["depth_mask"].fill_(True)
    measured_zero["point_mask"].fill_(True)
    measured_zero["pose_mask"].fill_(True)
    measured_out = _forward(planner, measured_zero)
    assert any(not torch.equal(missing_out[name], measured_out[name]) for name in missing_out)


def test_label_shuffle_sensitivity_and_finite_planner_gradients() -> None:
    torch.manual_seed(11)
    planner = _planner().train()
    evidence = _evidence()
    evidence = {name: value[:1] for name, value in evidence.items()}
    labels = torch.tensor([[[0,0,0,0],[0,1,1,1],[0,0,0,0]]], dtype=torch.bool)
    def loss_for(current: torch.Tensor) -> torch.Tensor:
        output = _forward(planner, evidence)
        output["score"] = planning_score(output, torch.zeros(1,3))
        return planner_loss(output, branch_rewards=current.float(), branch_dones=torch.zeros_like(current),
            branch_success=current, branch_valid=torch.ones(1,3,dtype=torch.bool),
            uncertainty_target=torch.zeros(1,3))["loss"]
    original = loss_for(labels)
    shuffled = loss_for(labels[:, torch.tensor([1,0,2])])
    assert not torch.allclose(original, shuffled)
    original.backward()
    grads=[p.grad for p in planner.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(value).all() for value in grads)
    assert sum(float(value.abs().sum()) for value in grads) > 0


def test_grouped_action_cost_is_masked_and_not_fixed_7d() -> None:
    actions=torch.zeros(2,3,4,2,5,9)
    mask=torch.zeros_like(actions,dtype=torch.bool)
    samples=torch.zeros(actions.shape[:-1],dtype=torch.bool)
    mask[...,0,:3]=True
    samples[...,0]=True
    actions[...,0,0]=2
    cost=deterministic_action_cost(actions,mask,samples)
    assert cost.shape==(2,3) and bool((cost>0).all())
    coarse=torch.full((2,3,4,2,9),3.0)
    coarse_mask=torch.ones_like(coarse,dtype=torch.bool)
    coarse_only=deterministic_action_cost(
        torch.zeros_like(actions), torch.zeros_like(mask), torch.zeros_like(samples),
        coarse, coarse_mask)
    assert bool((coarse_only == 3).all())
    with pytest.raises(ValueError, match="grouped"):
        deterministic_action_cost(torch.zeros(2,3,4,7), torch.zeros(2,3,4,7,dtype=torch.bool), torch.zeros(2,3,4,dtype=torch.bool))


def test_branch_clocks_must_match_stage0_window_and_interval_ownership() -> None:
    world_times = torch.tensor(
        [0.0, 0.3, 0.9000001430511475, 1.700000047683716],
        dtype=torch.float64,
    )
    future_dt = world_times[2:] - world_times[1]
    candidate = {
        "branch_future_dt_s": future_dt[None].expand(2, -1).clone(),
        "candidate_fine_action_dt": torch.tensor(
            [[[[.0, .2]], [[.0, .3]]], [[[.1, .2]], [[.1, .3]]]]
        ),
        "candidate_fine_sample_mask": torch.ones(2, 2, 1, 2, dtype=torch.bool),
    }
    sample = {
        "world_times_s": world_times,
        "future_world_boundaries_dt": torch.tensor([0., .4, .9]),
    }
    _validate_stage0_window_clock(candidate, sample=sample, context=2, K=2, horizon=2)
    drifted = copy.deepcopy(candidate)
    drifted["branch_future_dt_s"][0, 0] += .01
    with pytest.raises(ValueError, match="differ from Stage0"):
        _validate_stage0_window_clock(drifted, sample=sample, context=2, K=2, horizon=2)
    escaped = copy.deepcopy(candidate)
    escaped["candidate_fine_action_dt"][0, 0, 0, 1] = .4
    with pytest.raises(ValueError, match="outside its world interval"):
        _validate_stage0_window_clock(escaped, sample=sample, context=2, K=2, horizon=2)
    rounded = copy.deepcopy(candidate)
    rounded["branch_future_dt_s"] = rounded["branch_future_dt_s"].to(torch.float32)
    with pytest.raises(ValueError, match="differ from Stage0"):
        _validate_stage0_window_clock(
            rounded, sample=sample, context=2, K=2, horizon=2
        )


def test_candidate_payload_accepts_exact_ckgsa_grouped_action_abi() -> None:
    C, K, H, G, S, A, V, P, D = 3, 2, 2, 2, 4, 5, 2, 4, 8
    value = {
        "candidate_fine_action_values": torch.ones(C, K, G, S, A),
        "candidate_fine_action_mask": torch.ones(C, K, G, S, A, dtype=torch.bool),
        "candidate_fine_action_dt": torch.zeros(C, K, G, S),
        "candidate_fine_sample_mask": torch.ones(C, K, G, S, dtype=torch.bool),
        "candidate_coarse_action_values": torch.zeros(C, K, G, A),
        "candidate_coarse_action_mask": torch.zeros(C, K, G, A, dtype=torch.bool),
        "branch_future_tokens": torch.randn(C, H, P, D),
        "branch_future_dt_s": torch.tensor([[0.2, 0.5]]).expand(C, -1).clone(),
        "branch_token_mask": torch.ones(C, H, P, dtype=torch.bool),
        "branch_depth": torch.ones(C, H, V, P),
        "branch_depth_mask": torch.ones(C, H, V, P, dtype=torch.bool),
        "branch_point": torch.ones(C, H, V, P, 3),
        "branch_point_mask": torch.ones(C, H, V, P, dtype=torch.bool),
        "branch_camera_pose": torch.ones(C, H, V, 9),
        "branch_camera_pose_mask": torch.ones(C, H, V, dtype=torch.bool),
        "branch_geometry_confidence": torch.ones(C, H, V, P),
        "branch_view_mask": torch.ones(C, H, V, dtype=torch.bool),
        "branch_rewards": torch.tensor([[0.0, 0.0], [0.0, 1.0], [0.0, 0.0]]),
        "branch_dones": torch.zeros(C, H, dtype=torch.bool),
        "branch_success": torch.tensor(
            [[False, False], [False, True], [False, False]]
        ),
        "branch_valid": torch.ones(C, dtype=torch.bool),
    }
    _validate_candidate_payload(
        value,
        model={
            "K": K,
            "P": P,
            "token_dim": D,
            "num_views": V,
            "max_action_groups": G,
            "max_action_substeps": S,
            "max_action_dim": A,
        },
        horizon=H,
    )
    malformed = copy.deepcopy(value)
    malformed["candidate_fine_action_values"] = malformed[
        "candidate_fine_action_values"
    ].unsqueeze(0)
    with pytest.raises(ValueError, match="same sealed K"):
        _validate_candidate_payload(
            malformed,
            model={
                "K": K,
                "P": P,
                "token_dim": D,
                "num_views": V,
                "max_action_groups": G,
                "max_action_substeps": S,
                "max_action_dim": A,
            },
            horizon=H,
        )


def test_stage1_dataset_accepts_ckgsa_and_rejects_extra_or_missing_axis() -> None:
    C, K, G, S, A = 11, 8, 8, 128, 16
    payload = {
        "candidate_fine_action_values": torch.zeros(C, K, G, S, A),
        "candidate_fine_action_mask": torch.zeros(
            C, K, G, S, A, dtype=torch.bool
        ),
        "candidate_fine_action_dt": torch.zeros(C, K, G, S),
        "candidate_fine_sample_mask": torch.zeros(C, K, G, S, dtype=torch.bool),
        "candidate_coarse_action_values": torch.zeros(C, K, G, A),
        "candidate_coarse_action_mask": torch.zeros(C, K, G, A, dtype=torch.bool),
    }
    model = {
        "max_action_groups": G,
        "max_action_substeps": S,
        "max_action_dim": A,
    }
    _validate_candidate_action_shapes(payload, candidates=C, K=K, model=model)

    for malformed in (
        payload["candidate_fine_action_values"].unsqueeze(0),
        payload["candidate_fine_action_values"][..., 0],
    ):
        broken = dict(payload)
        broken["candidate_fine_action_values"] = malformed
        with pytest.raises(Stage1BranchError, match="do not cover sealed K"):
            _validate_candidate_action_shapes(broken, candidates=C, K=K, model=model)


def test_stage1_dcp_exact_resume_contract_is_planner_ddp_and_branch_bound() -> None:
    planner=NativePlannerConfig(token_dim=0,task_dim=0,patches=0,num_views=0,max_horizon=0,
        time_fourier_dim=0,time_min_period_s=0,time_max_period_s=0)
    stage1={"run":{"lineage":"s1","global_batch_size":8},"branch":{"index_sha256":"a"*64},
        "planner":{"horizon":4,"model":planner.__dict__,"score":{
            "progress_weight":.5,"success_weight":1.,"risk_weight":.5,
            "uncertainty_weight":.25,"action_cost_weight":.05}}}
    runtime={"bindings":{"model_contract_sha256":"b"*64},"model_profile":{"model":{
        "token_dim":16,"task_dim":12,"P":4,"num_views":2,"time_fourier_dim":8,
        "time_min_period_s":.01,"time_max_period_s":10.0}}}
    expected=_expectations(step=10,stage1=stage1,stage1_sha="c"*64,runtime=runtime,world_size=4)
    assert expected.distributed_strategy == "ddp"
    assert expected.shard_degree == 1 and expected.world_size == 4
    assert expected.allow_topology_reshard is False
    assert expected.topology_contract_sha256 == _topology_sha(stage1,runtime)
    assert expected.model_contract_sha256 == _planner_contract_sha(stage1,runtime)


def _tiny_world() -> NativeWorldModel:
    cfg=NativeWorldModelConfig(T=2,P=4,K=2,token_dim=8,task_dim=6,num_views=2,
        state_hidden=16,state_layers=1,state_heads=4,state_ff_mult=2,
        action_hidden=16,action_layers=1,action_heads=4,action_ff_mult=2,
        bridge_layers_state=(0,),bridge_heads=4,dynamics_layers=1,view_hidden=16,view_heads=4,view_ff_mult=2,
        max_action_groups=2,max_action_dim=5,max_state_dim=6,max_action_substeps=3,max_policy_queries=4,
        max_group_id=8,max_embodiments=8,max_action_semantic_id=16,max_state_semantic_id=16,
        time_fourier_dim=8,max_aux_tokens=2,aux_dim=4,max_aux_type_id=4,rgb_hidden=8,rgb_size=8,
        rgb_decode_indices=(1,),geom_hidden=8,dropout=0,activation_checkpointing=False)
    return NativeWorldModel(cfg)


def _rollout_batch() -> dict[str, torch.Tensor]:
    B,C,T,K,V,P,D,G,S,A,Q=1,3,2,2,2,4,8,2,3,5,2
    batch={
        "world_tokens":torch.randn(B,T,V,P,D),"view_mask":torch.ones(B,T,V,dtype=torch.bool),
        "world_times_s":torch.tensor([[0.,.3,.7,1.1]]),"task_embedding":torch.randn(B,6),
        "history_fine_action_values":torch.randn(B,T,G,S,A),"history_fine_action_mask":torch.ones(B,T,G,S,A,dtype=torch.bool),
        "history_fine_action_dt":torch.tensor([[[[.05,.15,.25]]*G]*T]),"history_fine_sample_mask":torch.ones(B,T,G,S,dtype=torch.bool),
        "history_coarse_action_values":torch.zeros(B,T,G,A),"history_coarse_action_mask":torch.zeros(B,T,G,A,dtype=torch.bool),
        "action_group_ids":torch.tensor([[1,2]]),"action_group_mask":torch.ones(B,G,dtype=torch.bool),
        "action_semantic_ids":torch.ones(B,G,A,dtype=torch.long),"current_state_values":torch.randn(B,G,6),
        "current_state_mask":torch.ones(B,G,6,dtype=torch.bool),"state_semantic_ids":torch.ones(B,G,6,dtype=torch.long),
        "embodiment_ids":torch.ones(B,dtype=torch.long),"policy_query_dt":torch.tensor([[[0.,.2],[0.,.2]]]),
        "policy_query_mask":torch.ones(B,G,Q,dtype=torch.bool),"action_normalization_offset":torch.zeros(B,G,A),
        "action_normalization_scale":torch.ones(B,G,A),"aux_values":torch.zeros(B,T,2,4),
        "aux_mask":torch.zeros(B,T,2,dtype=torch.bool),"aux_type_ids":torch.zeros(B,T,2,dtype=torch.long),
        "candidate_fine_action_values":torch.randn(B,C,K,G,S,A),"candidate_fine_action_mask":torch.ones(B,C,K,G,S,A,dtype=torch.bool),
        "candidate_fine_action_dt":torch.tensor([[[[[.05,.15,.25]]*G]*K]*C]),"candidate_fine_sample_mask":torch.ones(B,C,K,G,S,dtype=torch.bool),
        "candidate_coarse_action_values":torch.zeros(B,C,K,G,A),"candidate_coarse_action_mask":torch.zeros(B,C,K,G,A,dtype=torch.bool),
    }
    return batch


def test_rollout_is_single_trained_horizon_and_gradient_owner_is_planner() -> None:
    torch.manual_seed(3)
    world=_tiny_world().eval()
    batch=_rollout_batch()
    with pytest.raises(ValueError, match="H <= K"):
        single_horizon_native_rollout(world,batch,horizon=3)
    rollout=single_horizon_native_rollout(world,batch,horizon=1,candidate_microbatch=2)
    assert rollout.tokens.shape==(1,3,1,4,8)
    assert rollout.future_dt_s.shape == (1,3,1)
    assert bool((rollout.future_dt_s > 0).all())
    system=NativePlanningSystem(world,Stage1SystemConfig(planner=NativePlannerConfig(
        token_dim=0,task_dim=0,hidden=16,spatial_layers=1,temporal_layers=1,heads=4,mlp_mult=2,
        dropout=0,max_horizon=0,patches=0,num_views=0,time_fourier_dim=0,
        time_min_period_s=0,time_max_period_s=0),horizon=1))
    for parameter in world.parameters():
        parameter.requires_grad_(False)
    cost=deterministic_action_cost(batch["candidate_fine_action_values"],batch["candidate_fine_action_mask"],batch["candidate_fine_sample_mask"])
    out=system.score_rollout(rollout,batch["task_embedding"],cost)
    out["score"].sum().backward()
    assert all(parameter.grad is None for parameter in world.parameters())
    assert any(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()) for parameter in system.planner.parameters())
