from __future__ import annotations

import copy

import pytest
import torch

from wm3d_v3.models.native_world_model import NativeWorldModel, NativeWorldModelConfig
from wm3d_v3.stage1_planner.candidates import deterministic_action_cost
from wm3d_v3.stage1_planner.losses import planner_loss
from wm3d_v3.stage1_planner.planner_head import NativePlannerConfig, NativePlannerHead, planning_score
from wm3d_v3.stage1_planner.rollout import single_horizon_native_rollout
from wm3d_v3.stage1_planner.system import NativePlanningSystem, Stage1SystemConfig
from wm3d_v3.stage1_planner.train import _expectations, _planner_contract_sha, _topology_sha
from scripts.materialize_wm3d_v8_stage1_branches import _validate_stage0_window_clock


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
    torch.manual_seed(7); planner = _planner().eval(); evidence = _evidence()
    actions = torch.randn(2, 3, 4, 2, 5, 6)
    first = _forward(planner, evidence)
    shuffled_actions = actions[:, torch.tensor([2, 0, 1])]
    second = _forward(planner, evidence)
    assert not torch.equal(actions, shuffled_actions)
    for name in first: assert torch.equal(first[name], second[name])
    assert not any("action" in name for name, _ in planner.named_parameters())


def test_masked_native_evidence_is_not_treated_as_a_real_zero() -> None:
    torch.manual_seed(19); planner = _planner().eval(); evidence = _evidence()
    missing = copy.deepcopy(evidence)
    missing["depth"].zero_(); missing["depth_mask"].zero_()
    missing["point"].zero_(); missing["point_mask"].zero_()
    missing["pose"].zero_(); missing["pose_mask"].zero_()
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
    torch.manual_seed(11); planner = _planner().train(); evidence = _evidence()
    evidence = {name: value[:1] for name, value in evidence.items()}
    labels = torch.tensor([[[0,0,0,0],[0,1,1,1],[0,0,0,0]]], dtype=torch.bool)
    def loss_for(current: torch.Tensor) -> torch.Tensor:
        output = _forward(planner, evidence); output["score"] = planning_score(output, torch.zeros(1,3))
        return planner_loss(output, branch_rewards=current.float(), branch_dones=torch.zeros_like(current),
            branch_success=current, branch_valid=torch.ones(1,3,dtype=torch.bool),
            uncertainty_target=torch.zeros(1,3))["loss"]
    original = loss_for(labels); shuffled = loss_for(labels[:, torch.tensor([1,0,2])])
    assert not torch.allclose(original, shuffled)
    original.backward(); grads=[p.grad for p in planner.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(value).all() for value in grads)
    assert sum(float(value.abs().sum()) for value in grads) > 0


def test_grouped_action_cost_is_masked_and_not_fixed_7d() -> None:
    actions=torch.zeros(2,3,4,2,5,9); mask=torch.zeros_like(actions,dtype=torch.bool)
    samples=torch.zeros(actions.shape[:-1],dtype=torch.bool)
    mask[...,0,:3]=True; samples[...,0]=True; actions[...,0,0]=2
    cost=deterministic_action_cost(actions,mask,samples)
    assert cost.shape==(2,3) and bool((cost>0).all())
    coarse=torch.full((2,3,4,2,9),3.0); coarse_mask=torch.ones_like(coarse,dtype=torch.bool)
    coarse_only=deterministic_action_cost(
        torch.zeros_like(actions), torch.zeros_like(mask), torch.zeros_like(samples),
        coarse, coarse_mask)
    assert bool((coarse_only == 3).all())
    with pytest.raises(ValueError, match="grouped"):
        deterministic_action_cost(torch.zeros(2,3,4,7), torch.zeros(2,3,4,7,dtype=torch.bool), torch.zeros(2,3,4,dtype=torch.bool))


def test_branch_clocks_must_match_stage0_window_and_interval_ownership() -> None:
    world_times = torch.tensor([0., .3, .7, 1.2])
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
    drifted = copy.deepcopy(candidate); drifted["branch_future_dt_s"][0, 0] += .01
    with pytest.raises(ValueError, match="differ from Stage0"):
        _validate_stage0_window_clock(drifted, sample=sample, context=2, K=2, horizon=2)
    escaped = copy.deepcopy(candidate); escaped["candidate_fine_action_dt"][0, 0, 0, 1] = .4
    with pytest.raises(ValueError, match="outside its world interval"):
        _validate_stage0_window_clock(escaped, sample=sample, context=2, K=2, horizon=2)


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
    torch.manual_seed(3); world=_tiny_world().eval(); batch=_rollout_batch()
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
    for parameter in world.parameters(): parameter.requires_grad_(False)
    cost=deterministic_action_cost(batch["candidate_fine_action_values"],batch["candidate_fine_action_mask"],batch["candidate_fine_sample_mask"])
    out=system.score_rollout(rollout,batch["task_embedding"],cost); out["score"].sum().backward()
    assert all(parameter.grad is None for parameter in world.parameters())
    assert any(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()) for parameter in system.planner.parameters())
