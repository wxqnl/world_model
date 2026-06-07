# wm3d_v5 Cleanup -- File Classification Manifest

Generated 2026-06-07. Action = KEEP (stays in active tree) or ARCHIVE (moves to `archive/<scripts|configs>/`, reversible).
Nothing is hard-deleted here; `.bak`/`__pycache__`/`egg-info` junk is handled separately in the de-clutter phase.

- Scripts: 54 keep / 41 archive (of 95)
- Configs: 47 keep / 103 archive (of 150)

## SCRIPTS

### [ARCHIVE] 500m experiments (6) -- superseded by 1B formal plan
- `run_oxe500m_staged_flow_after_stage_a_v1.sh`
- `run_oxe_140m_hunyuan_visual_proof_v1_8gpu.sh`
- `run_oxe_500m_full_hunyuan_joint_after_cache_v1.sh`
- `run_oxe_500m_full_hunyuan_joint_v1_2node.sh`
- `run_oxe_500m_stage_a_world_visual_hunyuan_oxeall_v1_8gpu.sh`
- `run_oxe_all_trainable_no_video_pipeline_v1.sh`

### [ARCHIVE] LIBERO policy ablation (5) -- phasecond/teacher/lowdim explosion
- `run_libero_remote_smoke.sh`
- `watch_libero_after_oxe_500m_full_hunyuan_joint_v1.sh`
- `watch_libero_after_oxe_all_v1.sh`
- `watch_train_lowdimhist_partial4.sh`
- `watch_train_lowdimhist_partial4_sharded.sh`

### [ARCHIVE] old manifest/util (4) -- superseded by build_oxe_*_v1 tooling
- `build_oxe_manifest.py`
- `compute_action_stats.py`
- `prepare_full_oxe_after_current_run.sh`
- `subsample_manifest.py`

### [ARCHIVE] p256/rgb1b era (5) -- abandoned 16x16 / rgb-1b branch
- `cache_oxe_p256.py`
- `cache_p256_4gpu.sh`
- `posttrain_p256_demos.sh`
- `watch_and_posttrain_p256_rgb1b.sh`
- `watch_and_posttrain_p256_rgb1b_actioncond.sh`

### [ARCHIVE] p64 demo heads (4) -- Jun-1 demo/head ablations
- `watch_and_demo_p64_context_motion.sh`
- `watch_and_demo_p64_context_renderer.sh`
- `watch_and_demo_p64_control_head.sh`
- `watch_and_demo_p64_control_head_v2_256.sh`

### [ARCHIVE] pre-DiT hunyuan adapter (4) -- superseded by dit_control stage3
- `run_hunyuan_backend_smoke.py`
- `run_v5_generation_stage_hunyuan_adapter_v1.sh`
- `train_hunyuan_flow_denoiser.py`
- `train_hunyuan_latent_adapter.py`

### [ARCHIVE] v3/v3.5/vla era (13) -- superseded by v5 native-3D
- `analyze_vla.py`
- `compare_cosmos_vs_v3.py`
- `cosmos_prep_v3.py`
- `depth_action_correlation.py`
- `replot_vla.py`
- `train_v3.sh`
- `train_v3_5.sh`
- `train_v3_p256.sh`
- `train_v3_vla.sh`
- `train_v3_vla_b.sh`
- `train_v3_vla_c.sh`
- `v3_v3_5_compare.py`
- `v3_v3_5_e2e_compare.py`

### [KEEP] (default) (1) -- unmatched -> kept (conservative)
- `watch_droid20k_cache_then_train_stage1_300m_v1.sh`

### [KEEP] cache/data (4) -- current caching + native3d validator
- `cache_geom_utils.py`
- `cache_lerobot_droid_wm3d.py`
- `cache_oxe.py`
- `validate_native3d_window_cache.py`

### [KEEP] cache/smoke-flow (3) -- cache + scaling smoke harness
- `cache_control_bundle.py`
- `run_cache_droid20k_stage1_then_train_300m_v1.sh`
- `run_scaling_staged_smoke_v1.sh`

### [KEEP] eval/benchmark (5) -- formal world-model/libero benchmarks
- `run_formal_native3d_benchmark_v1.sh`
- `run_formal_world_model_benchmark_v5_libero_long_v1.sh`
- `run_formal_world_model_benchmark_v5_stage0_v1.sh`
- `run_world3d_native_benchmark_v2.sh`
- `run_worldvla_libero_benchmark_v1.sh`

### [KEEP] manifest (4) -- current manifest builders
- `build_oxe_droid_balanced_manifest_v1.py`
- `build_oxe_trainable_manifest.py`
- `build_stage1_oxe_droid_manifest.py`
- `build_v5_native3d_experiment_manifest.py`

### [KEEP] setup/infra (6) -- env + distributed setup
- `download_world_model_hf_mirror.py`
- `nccl_smoke.py`
- `prepare_worker_172_27_0_7_sync_v1.sh`
- `run_dist_smoke_2node.sh`
- `run_dist_smoke_4node.sh`
- `setup_libero_micromamba_env.sh`

### [KEEP] sync (2) -- cache/manifest sync helpers
- `sync_manifest_cache_files_v1.sh`
- `sync_oxe_droid_cache_for_manifest_v1.sh`

### [KEEP] train/140m-diagnostic (9) -- recent diagnostic flow
- `run_140m_depth_stabilized_4node_test_flow_v1.sh`
- `run_140m_depthplus_4node_test_flow_v1.sh`
- `run_140m_mini10_stage0_to_stage2_flow_v1.sh`
- `run_140m_stage0_noreset_resume7500_to10000_eval_v1.sh`
- `run_140m_stage0_visual_depth_stabilized_4node_v1.sh`
- `watch_140m_stage0_continue20000_eval_v1.sh`
- `watch_140m_stage0_visual_depth_stabilized_4node_v1.sh`
- `watch_1b_stage0_smoke_eval_v1.sh`
- `watch_stage1_eval_then_stage2_300m_v1.sh`

### [KEEP] train/300m-flow (12) -- canonical staged flow
- `run_300m_run1_droid_smoke_fromscratch_2node_v1.sh`
- `run_300m_stage0_visual_geom_oxe_droid20k_balanced_2node_v1.sh`
- `run_300m_stage0_visual_geom_oxe_droid20k_balanced_4node_v1.sh`
- `run_300m_stage1_oxe_droid20k_fromscratch_2node_v1.sh`
- `run_300m_stage1_world_oxe_droid20k_balanced_from_stage0_2node_v2.sh`
- `run_300m_stage1p5_hunyuan_bridge_oxe_droid20k_balanced_from_stage1_2node_v1.sh`
- `run_300m_stage2_oxe_droid20k_balanced_joint_visual_proposer_from_stage1_2node_v2.sh`
- `run_300m_stage2_oxe_droid20k_joint_visual_proposer_from_stage1_2node_v1.sh`
- `run_300m_stage2p5_visual_depth_polish_oxe_droid20k_balanced_from_stage2_2node_v1.sh`
- `run_300m_stage_2node_v2.sh`
- `run_300m_stage_4node_v1.sh`
- `watch_300m_stage0_to_stage2_flow_v2.sh`

### [KEEP] train/generation-stage3 (6) -- Hunyuan DiT-control stage3
- `run_generation_canary_v1.sh`
- `run_stage3_generation_hunyuan_dit_control_v1.sh`
- `run_v5_generation_stage_hunyuan_dit_control_v1.sh`
- `train_hunyuan_dit_control_adapter.py`
- `watch_generation_canary_v1.sh`
- `world_prior_generation_canary.py`

### [KEEP] train/v5-native3d (2) -- current native-3D launchers
- `run_v5_stage_4node_v1.sh`
- `watch_native3d_v5_cache_then_train_1b_stage0_v1.sh`

## CONFIGS

### [ARCHIVE] 300m superseded v1 (3) -- replaced by balanced v2
- `_smoke_oxe_fullpolicy_cached_v4_8gpu.yaml`
- `v3_p64_300m_stage1_world_oxe_droid20k_fromscratch_2node_v1.yaml`
- `v3_p64_300m_stage2_oxe_droid20k_joint_visual_proposer_from_stage1_2node_v1.yaml`

### [ARCHIVE] 500m/policy experiments (12) -- superseded by 1B plan
- `v3_p64_140m_hunyuan_visual_proof_v1_8gpu.yaml`
- `v3_p64_140m_p0_action_policy_oxe_all_trainable_no_video_fullpolicy_v1_8gpu.yaml`
- `v3_p64_140m_p0_action_policy_oxe_fullpolicy_cached_v4_8gpu.yaml`
- `v3_p64_140m_p0_action_policy_oxe_prior_balanced_v3_8gpu.yaml`
- `v3_p64_140m_p0_action_policy_oxe_prior_bridge_fractal_v2_8gpu.yaml`
- `v3_p64_140m_p0_action_policy_oxe_prior_isolated_v1.yaml`
- `v3_p64_140m_p0_action_policy_oxe_v9warm_no_video_1gpu_bs128_v1.yaml`
- `v3_p64_140m_p0_action_policy_oxe_v9warm_no_video_v1.yaml`
- `v3_p64_500m_full_hunyuan_joint_v1_8gpu.yaml`
- `v3_p64_500m_stage_a_world_visual_hunyuan_oxeall_v1_8gpu.yaml`
- `v3_p64_500m_stage_b_oxe_joint_visual_proposer_v1_7gpu.yaml`
- `v3_p64_500m_stage_c_oxe_direct_policy_v1_8gpu.yaml`

### [ARCHIVE] LIBERO policy ablation (48) -- phasecond/teacher/recovery/success snapshots
- `libero_action_bc_anchor_partial4_v1.yaml`
- `libero_action_policy_direct_partial4_dense_v1.yaml`
- `libero_action_policy_lowdimhist_partial4_start_stride4_sharded_v1.yaml`
- `libero_action_policy_lowdimhist_partial4_start_stride4_v1.yaml`
- `libero_action_policy_lowdimhist_task1_demo0_overfit_v1.yaml`
- `libero_action_policy_lowdimhist_task1_phasecond_smoke.yaml`
- `libero_action_policy_lowdimhist_task1_phasecond_v1.yaml`
- `libero_action_policy_lowdimhist_task1_phasecond_v10_summary_stage_recovery.yaml`
- `libero_action_policy_lowdimhist_task1_phasecond_v11_object_state_stage.yaml`
- `libero_action_policy_lowdimhist_task1_phasecond_v12_object_state_failure_recovery.yaml`
- `libero_action_policy_lowdimhist_task1_phasecond_v13_plan_tail_recovery.yaml`
- `libero_action_policy_lowdimhist_task1_phasecond_v14_plan17_targetgeom.yaml`
- `libero_action_policy_lowdimhist_task1_phasecond_v15_plan17_dagger_tail.yaml`
- `libero_action_policy_lowdimhist_task1_phasecond_v16_plan17_localres_tail.yaml`
- `libero_action_policy_lowdimhist_task1_phasecond_v16b_plan17_localres_loww_tail.yaml`
- `libero_action_policy_lowdimhist_task1_phasecond_v17_plan17_localres_place.yaml`
- `libero_action_policy_lowdimhist_task1_phasecond_v18_plan17_localres_experttrace.yaml`
- `libero_action_policy_lowdimhist_task1_phasecond_v19_plan17_waypointres_experttrace.yaml`
- `libero_action_policy_lowdimhist_task1_phasecond_v2.yaml`
- `libero_action_policy_lowdimhist_task1_phasecond_v20_plan17_stage3waypoint_experttrace.yaml`
- `libero_action_policy_lowdimhist_task1_phasecond_v21_fullpolicy_experttrace.yaml`
- `libero_action_policy_lowdimhist_task1_phasecond_v3.yaml`
- `libero_action_policy_lowdimhist_task1_phasecond_v4_summary.yaml`
- `libero_action_policy_lowdimhist_task1_phasecond_v5_summary_griprefine.yaml`
- `libero_action_policy_lowdimhist_task1_phasecond_v6_summary_recovery_mono.yaml`
- `libero_action_policy_lowdimhist_task1_phasecond_v7_summary_object_recovery.yaml`
- `libero_action_policy_lowdimhist_task1_phasecond_v8_summary_mixed_recovery.yaml`
- `libero_action_policy_lowdimhist_task1_phasecond_v9_summary_direct_recovery.yaml`
- `libero_action_policy_lowdimhist_task1_recovery_step800_timealign_v2.yaml`
- `libero_action_policy_lowdimhist_task1_recovery_step800_v1.yaml`
- `libero_action_policy_lowdimhist_task1_start_stride4_from_partial4_v1.yaml`
- `libero_action_policy_lowdimhist_task1_start_stride4_griptransition_v1.yaml`
- `libero_action_policy_partial4_oxe500m_full_hunyuan_joint_v1_8gpu.yaml`
- `libero_action_policy_partial4_oxeall_fullpolicy_v1_8gpu.yaml`
- `libero_action_policy_partial4_oxeprior_v1_8gpu.yaml`
- `libero_action_policy_partial4_stage_d_from_oxe500m_stage_c_v1_8gpu.yaml`
- `libero_action_policy_task1_teacher_dagger_tail_v6.yaml`
- `libero_action_policy_task1_teacher_dagger_v5.yaml`
- `libero_action_policy_task1_teacher_intervention_success_v7.yaml`
- `libero_action_policy_task1_teacher_intervention_suffix_v8.yaml`
- `libero_action_policy_task1_teacher_intervention_suffix_v9.yaml`
- `libero_action_policy_task1_teacher_transformer50_rich_future_v2.yaml`
- `libero_action_policy_task1_teacher_transformer50_rich_future_v3_overfit.yaml`
- `libero_action_policy_task1_teacher_transformer50_rich_v1.yaml`
- `libero_action_policy_task1_teacher_transformer50_terminal_release_v4.yaml`
- `libero_success_p0_mixed_smoke.yaml`
- `libero_success_p0_partial4_mixed_v1.yaml`
- `libero_success_p0_smoke.yaml`

### [ARCHIVE] p64 demo/heads (20) -- Jun1-2 head ablations
- `v3_p64_140m_actioncond_context_motion.yaml`
- `v3_p64_140m_actioncond_context_motion_episode_eval.yaml`
- `v3_p64_140m_actioncond_context_motion_p0_heads_candrank.yaml`
- `v3_p64_140m_actioncond_context_motion_p0_heads_candrank_ce.yaml`
- `v3_p64_140m_actioncond_context_motion_p0_heads_direct_policy.yaml`
- `v3_p64_140m_actioncond_context_motion_p0_heads_direct_policy_lowdimhist.yaml`
- `v3_p64_140m_actioncond_context_motion_p0_heads_direct_policy_lowdimhist_phase.yaml`
- `v3_p64_140m_actioncond_context_motion_p0_heads_direct_policy_lowdimhist_phase_object_plan17_localres_stage3waypoint_summary.yaml`
- `v3_p64_140m_actioncond_context_motion_p0_heads_direct_policy_lowdimhist_phase_object_plan17_localres_summary.yaml`
- `v3_p64_140m_actioncond_context_motion_p0_heads_direct_policy_lowdimhist_phase_object_plan17_localres_waypoint_summary.yaml`
- `v3_p64_140m_actioncond_context_motion_p0_heads_direct_policy_lowdimhist_phase_object_plan17_summary.yaml`
- `v3_p64_140m_actioncond_context_motion_p0_heads_direct_policy_lowdimhist_phase_object_plan_summary.yaml`
- `v3_p64_140m_actioncond_context_motion_p0_heads_direct_policy_lowdimhist_phase_object_summary.yaml`
- `v3_p64_140m_actioncond_context_motion_p0_heads_direct_policy_lowdimhist_phase_summary.yaml`
- `v3_p64_140m_actioncond_context_motion_p0_heads_frozen.yaml`
- `v3_p64_140m_actioncond_context_motion_p0_heads_pairwise.yaml`
- `v3_p64_140m_actioncond_context_motion_p0_proposer.yaml`
- `v3_p64_140m_actioncond_context_renderer.yaml`
- `v3_p64_140m_actioncond_control_head.yaml`
- `v3_p64_140m_actioncond_control_head_v2_256.yaml`

### [ARCHIVE] v3/v3.5/p256 era (20) -- superseded
- `v3_5_e2e_oxe.yaml`
- `v3_5_oxe.yaml`
- `v3_oxe.yaml`
- `v3_p256_context_only_pixel_gt_smoke.yaml`
- `v3_p256_context_residual_pixel_gt_smoke.yaml`
- `v3_p256_oxe.yaml`
- `v3_p256_rgb1b_actioncond_oxe.yaml`
- `v3_p256_rgb1b_actioncond_stage1.yaml`
- `v3_p256_rgb1b_oxe.yaml`
- `v3_p256_token_only_pixel_gt_smoke.yaml`
- `v3_p64_138m_actioncond_full.yaml`
- `v3_p64_138m_actioncond_full_eval_depthplus_v1.yaml`
- `v3_p64_small_actioncond_ablate.yaml`
- `v3_p64_small_bs8_actioncond_ablate.yaml`
- `v3_p64_small_bs8_noaction_ablate.yaml`
- `v3_p64_small_noaction_ablate.yaml`
- `v3_smoke.yaml`
- `v3_vla.yaml`
- `v3_vla_b.yaml`
- `v3_vla_c.yaml`

### [KEEP] 140m-diagnostic (18) -- recent 140m staged diagnostic configs
- `v3_p64_140m_stage0_core3d_oxe_droid20k_balanced_fromscratch_2node_v1.yaml`
- `v3_p64_140m_stage0_visual_depth_mini10_2node_v1.yaml`
- `v3_p64_140m_stage0_visual_depth_oxe_droid20k_balanced_fromscratch_2node_v2.yaml`
- `v3_p64_140m_stage0_visual_depth_stabilized_4node_v1.yaml`
- `v3_p64_140m_stage0_visual_depth_stabilized_continue10000_to20000_v1.yaml`
- `v3_p64_140m_stage0_visual_depth_stabilized_current20k_eval_old_oxe_v1.yaml`
- `v3_p64_140m_stage0_visual_depth_stabilized_noreset_resume7500_to10000_eval_v1.yaml`
- `v3_p64_140m_stage0_visual_depth_stabilized_resume7500_to10000_eval_v1.yaml`
- `v3_p64_140m_stage0_visual_depthplus_4node_test_v1.yaml`
- `v3_p64_140m_stage1_dynamics_visual_replay_depth_stabilized_4node_v1.yaml`
- `v3_p64_140m_stage1_dynamics_visual_replay_depthplus_4node_test_v1.yaml`
- `v3_p64_140m_stage1_dynamics_visual_replay_mini10_2node_v1.yaml`
- `v3_p64_140m_stage1p5_hunyuan_bridge_depth_stabilized_4node_v1.yaml`
- `v3_p64_140m_stage1p5_hunyuan_bridge_depthplus_4node_test_v1.yaml`
- `v3_p64_140m_stage1p5_hunyuan_bridge_mini10_2node_v1.yaml`
- `v3_p64_140m_stage2_action_scaffold_depth_stabilized_4node_v1.yaml`
- `v3_p64_140m_stage2_action_scaffold_depthplus_4node_test_v1.yaml`
- `v3_p64_140m_stage2_action_scaffold_mini10_2node_v1.yaml`

### [KEEP] 1b-formal (7) -- 1B-class formal target configs
- `v3_p64_1b_stage0_visual_depth_wsd_4node_v1.yaml`
- `v3_p64_1b_stage0_visual_depth_wsd_bs3_4node_v1.yaml`
- `v3_p64_1b_stage0_visual_depth_wsd_bs3_fromscratch_4ep_4node_v1.yaml`
- `v3_p64_1b_stage0_visual_depth_wsd_bs3_fromscratch_4node_v1.yaml`
- `v3_p64_1b_stage0_visual_depth_wsd_bs4_4node_v1.yaml`
- `v3_p64_1b_stage0_visual_depth_wsd_smoke_4node_v1.yaml`
- `v3_p64_1b_stage1_dynamics_visual_replay_wsd_bs3_from_stage0_4node_v1.yaml`

### [KEEP] 300m-flow (7) -- canonical staged (balanced) configs
- `v3_p64_300m_run1_droid_smoke_fromscratch_2node_v1.yaml`
- `v3_p64_300m_runG_optional_text_world_prior_hunyuan_from_stage1_2node_v1.yaml`
- `v3_p64_300m_stage0_visual_geom_oxe_droid20k_balanced_fromscratch_2node_v1.yaml`
- `v3_p64_300m_stage1_world_oxe_droid20k_balanced_from_stage0_2node_v2.yaml`
- `v3_p64_300m_stage1p5_hunyuan_bridge_oxe_droid20k_balanced_from_stage1_2node_v1.yaml`
- `v3_p64_300m_stage2_oxe_droid20k_balanced_joint_visual_proposer_from_stage1_2node_v2.yaml`
- `v3_p64_300m_stage2p5_visual_depth_polish_oxe_droid20k_balanced_from_stage2_2node_v1.yaml`

### [KEEP] smoke-harness (8) -- staged smoke configs
- `scaling_smoke_stage_a_world_visual_v1.yaml`
- `scaling_smoke_stage_b_oxe_joint_v1.yaml`
- `scaling_smoke_stage_c_oxe_policy_v1.yaml`
- `scaling_smoke_stage_d_libero_policy_v1.yaml`
- `smoke_300m_revised_flow_v1_stage0.yaml`
- `smoke_300m_revised_flow_v1_stage1.yaml`
- `smoke_300m_revised_flow_v1_stage1p5.yaml`
- `smoke_300m_revised_flow_v1_stage2.yaml`

### [KEEP] v5-native3d (7) -- current native-3D configs
- `_eval_v5_p64_140m_stage0_native3d_exp8192_w2_loadgeom_v1.yaml`
- `_smoke_v5_p64_140m_stage0_native3d_exp8192_w2_1gpu.yaml`
- `v5_p64_140m_stage0_native3d_exp8192_w2_3node_v1.yaml`
- `v5_p64_140m_stage0_native3d_full_depthfocus_wsd_4node_v1.yaml`
- `v5_p64_1b_stage0_native3d_windowgeom_mini_smoke_v1.yaml`
- `v5_p64_1b_stage0_native3d_wsd_4node_v1.yaml`
- `v5_p64_300m_stage0_native3d_oxe_droid20k_balanced_2node_v1.yaml`
