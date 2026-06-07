# Cleanup Report: 300m Stage0 Prep v1

- Date: 2026-06-05T11:32:20
- Host: New-H100-3 (`/data/Minko`)
- Repo: `/data/Minko/world_model/wm3d_v3`
- Quarantine: `/data/Minko/cleanup_quarantine/20260605_112921`
- Quarantine manifest: `/data/Minko/cleanup_quarantine/20260605_112921/cleanup_manifest.json`
- Estimated active-tree space freed: 3.6 MB (3778360 bytes)

## Guardrails Applied

- Protected current Stage2 run skipped everywhere: `train_300m_stage2_oxe_droid20k_joint_visual_proposer_from_stage1_2node_v1`.
- Protected Stage1 result skipped everywhere: `wm3d_v3_p64_300m_stage1_world_oxe_droid20k_fromscratch_2node_v1`.
- No `.pt`, `.pth`, `.ckpt`, `.safetensors`, manifest, dataset, source, or real config files were deleted.
- Cleanup actions used move-to-quarantine, not permanent removal.
- `/data/Minko/logs` actions were limited to non-protected smoke/debug logs older than 48 hours.

## Processed Items

### AppleDouble local metadata sidecar
- Count: 58
- Size: 9.2 KB (9454 bytes)
- `world_model/wm3d_v3/._REPORT_WM3D_CLOSED_LOOP_STATUS_2026-06-02.md` -> `cleanup_quarantine/20260605_112921/world_model/wm3d_v3/._REPORT_WM3D_CLOSED_LOOP_STATUS_2026-06-02.md` (163 B)
- `world_model/wm3d_v3/wm3d_v3/._losses.py` -> `cleanup_quarantine/20260605_112921/world_model/wm3d_v3/wm3d_v3/._losses.py` (163 B)
- `world_model/wm3d_v3/wm3d_v3/models/._joint_model.py` -> `cleanup_quarantine/20260605_112921/world_model/wm3d_v3/wm3d_v3/models/._joint_model.py` (163 B)
- `world_model/wm3d_v3/wm3d_v3/models/._action_policy.py` -> `cleanup_quarantine/20260605_112921/world_model/wm3d_v3/wm3d_v3/models/._action_policy.py` (163 B)
- `world_model/wm3d_v3/wm3d_v3/models/._control_head.py` -> `cleanup_quarantine/20260605_112921/world_model/wm3d_v3/wm3d_v3/models/._control_head.py` (163 B)
- `world_model/wm3d_v3/wm3d_v3/policy/._world_model_policy.py` -> `cleanup_quarantine/20260605_112921/world_model/wm3d_v3/wm3d_v3/policy/._world_model_policy.py` (163 B)
- `world_model/wm3d_v3/wm3d_v3/policy/._token_policy.py` -> `cleanup_quarantine/20260605_112921/world_model/wm3d_v3/wm3d_v3/policy/._token_policy.py` (163 B)
- `world_model/wm3d_v3/wm3d_v3/policy/._http_policy_server.py` -> `cleanup_quarantine/20260605_112921/world_model/wm3d_v3/wm3d_v3/policy/._http_policy_server.py` (163 B)
- `world_model/wm3d_v3/wm3d_v3/data/._splits.py` -> `cleanup_quarantine/20260605_112921/world_model/wm3d_v3/wm3d_v3/data/._splits.py` (163 B)
- `world_model/wm3d_v3/wm3d_v3/eval/._make_long_rollout_gif.py` -> `cleanup_quarantine/20260605_112921/world_model/wm3d_v3/wm3d_v3/eval/._make_long_rollout_gif.py` (163 B)
- `world_model/wm3d_v3/wm3d_v3/eval/._action_sensitivity.py` -> `cleanup_quarantine/20260605_112921/world_model/wm3d_v3/wm3d_v3/eval/._action_sensitivity.py` (163 B)
- `world_model/wm3d_v3/wm3d_v3/eval/._make_demo_gif.py` -> `cleanup_quarantine/20260605_112921/world_model/wm3d_v3/wm3d_v3/eval/._make_demo_gif.py` (163 B)
- `world_model/wm3d_v3/wm3d_v3/eval/._run_eval.py` -> `cleanup_quarantine/20260605_112921/world_model/wm3d_v3/wm3d_v3/eval/._run_eval.py` (163 B)
- `world_model/wm3d_v3/wm3d_v3/eval/._system_harness.py` -> `cleanup_quarantine/20260605_112921/world_model/wm3d_v3/wm3d_v3/eval/._system_harness.py` (163 B)
- `world_model/wm3d_v3/wm3d_v3/video_backends/.___init__.py` -> `cleanup_quarantine/20260605_112921/world_model/wm3d_v3/wm3d_v3/video_backends/.___init__.py` (163 B)
- `world_model/wm3d_v3/wm3d_v3/video_backends/._hunyuan_video.py` -> `cleanup_quarantine/20260605_112921/world_model/wm3d_v3/wm3d_v3/video_backends/._hunyuan_video.py` (163 B)
- `world_model/wm3d_v3/wm3d_v3/training/._train_libero_action_policy.py` -> `cleanup_quarantine/20260605_112921/world_model/wm3d_v3/wm3d_v3/training/._train_libero_action_policy.py` (163 B)
- `world_model/wm3d_v3/wm3d_v3/training/._train_libero_success_p0.py` -> `cleanup_quarantine/20260605_112921/world_model/wm3d_v3/wm3d_v3/training/._train_libero_success_p0.py` (163 B)
- `world_model/wm3d_v3/wm3d_v3/training/._train.py` -> `cleanup_quarantine/20260605_112921/world_model/wm3d_v3/wm3d_v3/training/._train.py` (163 B)
- `world_model/wm3d_v3/wm3d_v3/benchmarks/._libero_rollout_cache.py` -> `cleanup_quarantine/20260605_112921/world_model/wm3d_v3/wm3d_v3/benchmarks/._libero_rollout_cache.py` (163 B)
- `world_model/wm3d_v3/wm3d_v3/benchmarks/._libero_object_state_reference.py` -> `cleanup_quarantine/20260605_112921/world_model/wm3d_v3/wm3d_v3/benchmarks/._libero_object_state_reference.py` (163 B)
- `world_model/wm3d_v3/wm3d_v3/benchmarks/._libero_expert_cache.py` -> `cleanup_quarantine/20260605_112921/world_model/wm3d_v3/wm3d_v3/benchmarks/._libero_expert_cache.py` (163 B)
- `world_model/wm3d_v3/wm3d_v3/benchmarks/._libero_remote_runner.py` -> `cleanup_quarantine/20260605_112921/world_model/wm3d_v3/wm3d_v3/benchmarks/._libero_remote_runner.py` (163 B)
- `world_model/wm3d_v3/wm3d_v3/benchmarks/._libero_start_windows.py` -> `cleanup_quarantine/20260605_112921/world_model/wm3d_v3/wm3d_v3/benchmarks/._libero_start_windows.py` (163 B)
- `world_model/wm3d_v3/wm3d_v3/benchmarks/._libero_rollout_recovery_cache.py` -> `cleanup_quarantine/20260605_112921/world_model/wm3d_v3/wm3d_v3/benchmarks/._libero_rollout_recovery_cache.py` (163 B)
- `world_model/wm3d_v3/wm3d_v3/benchmarks/._libero_trace_summary.py` -> `cleanup_quarantine/20260605_112921/world_model/wm3d_v3/wm3d_v3/benchmarks/._libero_trace_summary.py` (163 B)
- `world_model/wm3d_v3/wm3d_v3/benchmarks/._libero_demo_export.py` -> `cleanup_quarantine/20260605_112921/world_model/wm3d_v3/wm3d_v3/benchmarks/._libero_demo_export.py` (163 B)
- `world_model/wm3d_v3/scripts/._watch_and_demo_p64_control_head_v2_256.sh` -> `cleanup_quarantine/20260605_112921/world_model/wm3d_v3/scripts/._watch_and_demo_p64_control_head_v2_256.sh` (163 B)
- `world_model/wm3d_v3/scripts/._run_hunyuan_backend_smoke.py` -> `cleanup_quarantine/20260605_112921/world_model/wm3d_v3/scripts/._run_hunyuan_backend_smoke.py` (163 B)
- `world_model/wm3d_v3/scripts/._cache_control_bundle.py` -> `cleanup_quarantine/20260605_112921/world_model/wm3d_v3/scripts/._cache_control_bundle.py` (163 B)
- `world_model/wm3d_v3/scripts/._run_libero_remote_smoke.sh` -> `cleanup_quarantine/20260605_112921/world_model/wm3d_v3/scripts/._run_libero_remote_smoke.sh` (163 B)
- `world_model/wm3d_v3/scripts/._watch_and_demo_p64_control_head.sh` -> `cleanup_quarantine/20260605_112921/world_model/wm3d_v3/scripts/._watch_and_demo_p64_control_head.sh` (163 B)
- `world_model/wm3d_v3/configs/._v3_p64_140m_actioncond_control_head_v2_256.yaml` -> `cleanup_quarantine/20260605_112921/world_model/wm3d_v3/configs/._v3_p64_140m_actioncond_control_head_v2_256.yaml` (163 B)
- `world_model/wm3d_v3/configs/._libero_action_bc_anchor_partial4_v1.yaml` -> `cleanup_quarantine/20260605_112921/world_model/wm3d_v3/configs/._libero_action_bc_anchor_partial4_v1.yaml` (163 B)
- `world_model/wm3d_v3/configs/._v3_p64_140m_actioncond_context_motion_p0_heads_direct_policy_lowdimhist_phase_summary.yaml` -> `cleanup_quarantine/20260605_112921/world_model/wm3d_v3/configs/._v3_p64_140m_actioncond_context_motion_p0_heads_direct_policy_lowdimhist_phase_summary.yaml` (163 B)
- `world_model/wm3d_v3/configs/._libero_action_policy_lowdimhist_task1_phasecond_v8_summary_mixed_recovery.yaml` -> `cleanup_quarantine/20260605_112921/world_model/wm3d_v3/configs/._libero_action_policy_lowdimhist_task1_phasecond_v8_summary_mixed_recovery.yaml` (163 B)
- `world_model/wm3d_v3/configs/._libero_action_policy_lowdimhist_task1_phasecond_v4_summary.yaml` -> `cleanup_quarantine/20260605_112921/world_model/wm3d_v3/configs/._libero_action_policy_lowdimhist_task1_phasecond_v4_summary.yaml` (163 B)
- `world_model/wm3d_v3/configs/._libero_action_policy_lowdimhist_task1_phasecond_v2.yaml` -> `cleanup_quarantine/20260605_112921/world_model/wm3d_v3/configs/._libero_action_policy_lowdimhist_task1_phasecond_v2.yaml` (163 B)
- `world_model/wm3d_v3/configs/._libero_action_policy_lowdimhist_task1_phasecond_v5_summary_griprefine.yaml` -> `cleanup_quarantine/20260605_112921/world_model/wm3d_v3/configs/._libero_action_policy_lowdimhist_task1_phasecond_v5_summary_griprefine.yaml` (163 B)
- `world_model/wm3d_v3/configs/._libero_success_p0_smoke.yaml` -> `cleanup_quarantine/20260605_112921/world_model/wm3d_v3/configs/._libero_success_p0_smoke.yaml` (163 B)
- `world_model/wm3d_v3/configs/._libero_action_policy_lowdimhist_task1_demo0_overfit_v1.yaml` -> `cleanup_quarantine/20260605_112921/world_model/wm3d_v3/configs/._libero_action_policy_lowdimhist_task1_demo0_overfit_v1.yaml` (163 B)
- `world_model/wm3d_v3/configs/._v3_p64_140m_actioncond_context_motion_p0_heads_direct_policy.yaml` -> `cleanup_quarantine/20260605_112921/world_model/wm3d_v3/configs/._v3_p64_140m_actioncond_context_motion_p0_heads_direct_policy.yaml` (163 B)
- `world_model/wm3d_v3/configs/._libero_action_policy_lowdimhist_task1_phasecond_v1.yaml` -> `cleanup_quarantine/20260605_112921/world_model/wm3d_v3/configs/._libero_action_policy_lowdimhist_task1_phasecond_v1.yaml` (163 B)
- `world_model/wm3d_v3/configs/._libero_action_policy_lowdimhist_task1_phasecond_v3.yaml` -> `cleanup_quarantine/20260605_112921/world_model/wm3d_v3/configs/._libero_action_policy_lowdimhist_task1_phasecond_v3.yaml` (163 B)
- `world_model/wm3d_v3/configs/._v3_p64_140m_actioncond_context_motion_p0_heads_direct_policy_lowdimhist.yaml` -> `cleanup_quarantine/20260605_112921/world_model/wm3d_v3/configs/._v3_p64_140m_actioncond_context_motion_p0_heads_direct_policy_lowdimhist.yaml` (163 B)
- `world_model/wm3d_v3/configs/._libero_action_policy_lowdimhist_task1_phasecond_smoke.yaml` -> `cleanup_quarantine/20260605_112921/world_model/wm3d_v3/configs/._libero_action_policy_lowdimhist_task1_phasecond_smoke.yaml` (163 B)
- `world_model/wm3d_v3/configs/._libero_action_policy_lowdimhist_task1_phasecond_v7_summary_object_recovery.yaml` -> `cleanup_quarantine/20260605_112921/world_model/wm3d_v3/configs/._libero_action_policy_lowdimhist_task1_phasecond_v7_summary_object_recovery.yaml` (163 B)
- `world_model/wm3d_v3/configs/._v3_p64_140m_actioncond_context_motion_episode_eval.yaml` -> `cleanup_quarantine/20260605_112921/world_model/wm3d_v3/configs/._v3_p64_140m_actioncond_context_motion_episode_eval.yaml` (163 B)
- `world_model/wm3d_v3/configs/._libero_action_policy_lowdimhist_task1_phasecond_v6_summary_recovery_mono.yaml` -> `cleanup_quarantine/20260605_112921/world_model/wm3d_v3/configs/._libero_action_policy_lowdimhist_task1_phasecond_v6_summary_recovery_mono.yaml` (163 B)
- `world_model/wm3d_v3/configs/._v3_p64_140m_actioncond_context_motion_p0_heads_direct_policy_lowdimhist_phase.yaml` -> `cleanup_quarantine/20260605_112921/world_model/wm3d_v3/configs/._v3_p64_140m_actioncond_context_motion_p0_heads_direct_policy_lowdimhist_phase.yaml` (163 B)
- `world_model/wm3d_v3/configs/._libero_action_policy_direct_partial4_dense_v1.yaml` -> `cleanup_quarantine/20260605_112921/world_model/wm3d_v3/configs/._libero_action_policy_direct_partial4_dense_v1.yaml` (163 B)
- `world_model/wm3d_v3/configs/._v3_p64_140m_actioncond_control_head.yaml` -> `cleanup_quarantine/20260605_112921/world_model/wm3d_v3/configs/._v3_p64_140m_actioncond_control_head.yaml` (163 B)
- `world_model/wm3d_v3/tests/._test_context_renderer_integration.py` -> `cleanup_quarantine/20260605_112921/world_model/wm3d_v3/tests/._test_context_renderer_integration.py` (163 B)
- `world_model/wm3d_v3/tests/._test_control_progress_heads.py` -> `cleanup_quarantine/20260605_112921/world_model/wm3d_v3/tests/._test_control_progress_heads.py` (163 B)
- `world_model/wm3d_v3/tests/._test_action_sensitivity.py` -> `cleanup_quarantine/20260605_112921/world_model/wm3d_v3/tests/._test_action_sensitivity.py` (163 B)
- `world_model/wm3d_v3/tests/._test_control_bundle_video_backend.py` -> `cleanup_quarantine/20260605_112921/world_model/wm3d_v3/tests/._test_control_bundle_video_backend.py` (163 B)
- `world_model/wm3d_v3/tests/._test_eval_config.py` -> `cleanup_quarantine/20260605_112921/world_model/wm3d_v3/tests/._test_eval_config.py` (163 B)
- `world_model/wm3d_v3/tests/._test_episode_splits.py` -> `cleanup_quarantine/20260605_112921/world_model/wm3d_v3/tests/._test_episode_splits.py` (163 B)

### old smoke/debug log older than 48h
- Count: 15
- Size: 98.2 KB (100509 bytes)
- `logs/token_only_pixel_gt_p256_smoke_latest.log` -> `cleanup_quarantine/20260605_112921/logs/token_only_pixel_gt_p256_smoke_latest.log` (67 B)
- `logs/token_only_pixel_gt_p256_smoke_20260531_234209.log` -> `cleanup_quarantine/20260605_112921/logs/token_only_pixel_gt_p256_smoke_20260531_234209.log` (20.2 KB)
- `logs/policy_server_debug_terminal_linear_8796.log` -> `cleanup_quarantine/20260605_112921/logs/policy_server_debug_terminal_linear_8796.log` (283 B)
- `logs/context_only_pixel_gt_p256_smoke_latest.pid` -> `cleanup_quarantine/20260605_112921/logs/context_only_pixel_gt_p256_smoke_latest.pid` (8 B)
- `logs/context_residual_pixel_gt_p256_smoke_latest.log` -> `cleanup_quarantine/20260605_112921/logs/context_residual_pixel_gt_p256_smoke_latest.log` (73 B)
- `logs/context_only_pixel_gt_p256_smoke_20260531_231536.log` -> `cleanup_quarantine/20260605_112921/logs/context_only_pixel_gt_p256_smoke_20260531_231536.log` (20.1 KB)
- `logs/smoke_wm3d_p0_heads_candrank_ce.log` -> `cleanup_quarantine/20260605_112921/logs/smoke_wm3d_p0_heads_candrank_ce.log` (11.7 KB)
- `logs/policy_server_debug_terminal_linear_8796.pid` -> `cleanup_quarantine/20260605_112921/logs/policy_server_debug_terminal_linear_8796.pid` (8 B)
- `logs/mock_adapter_smoke.json` -> `cleanup_quarantine/20260605_112921/logs/mock_adapter_smoke.json` (731 B)
- `logs/smoke_wm3d_p0_heads_candrank.log` -> `cleanup_quarantine/20260605_112921/logs/smoke_wm3d_p0_heads_candrank.log` (9.1 KB)
- `logs/wm3d_policy_server_smoke.log` -> `cleanup_quarantine/20260605_112921/logs/wm3d_policy_server_smoke.log` (283 B)
- `logs/context_residual_pixel_gt_p256_smoke_20260531_223658.log` -> `cleanup_quarantine/20260605_112921/logs/context_residual_pixel_gt_p256_smoke_20260531_223658.log` (33.0 KB)
- `logs/context_residual_pixel_gt_p256_smoke_latest.pid` -> `cleanup_quarantine/20260605_112921/logs/context_residual_pixel_gt_p256_smoke_latest.pid` (8 B)
- `logs/token_only_pixel_gt_p256_smoke_latest.pid` -> `cleanup_quarantine/20260605_112921/logs/token_only_pixel_gt_p256_smoke_latest.pid` (8 B)
- `logs/policy_probe_smoke.json` -> `cleanup_quarantine/20260605_112921/logs/policy_probe_smoke.json` (2.5 KB)

### old smoke/debug results dir without checkpoint/manifest/config
- Count: 4
- Size: 190.3 KB (194858 bytes)
- `world_model/wm3d_v3/results/_smoke_no_video_harness_latest` -> `cleanup_quarantine/20260605_112921/world_model/wm3d_v3/results/_smoke_no_video_harness_latest` (55.7 KB)
- `world_model/wm3d_v3/results/_smoke_no_video_harness` -> `cleanup_quarantine/20260605_112921/world_model/wm3d_v3/results/_smoke_no_video_harness` (55.5 KB)
- `world_model/wm3d_v3/results/_smoke_no_video_harness_after_object_state` -> `cleanup_quarantine/20260605_112921/world_model/wm3d_v3/results/_smoke_no_video_harness_after_object_state` (56.1 KB)
- `world_model/wm3d_v3/results/hunyuan_smoke` -> `cleanup_quarantine/20260605_112921/world_model/wm3d_v3/results/hunyuan_smoke` (22.9 KB)

### python cache (pyc filenames only contained protected words)
- Count: 3
- Size: 82.6 KB (84601 bytes)
- `world_model/wm3d_v3/wm3d_v3/data/__pycache__` -> `cleanup_quarantine/20260605_112921/world_model/wm3d_v3/wm3d_v3/data/__pycache__` (28.3 KB)
- `world_model/wm3d_v3/scripts/__pycache__` -> `cleanup_quarantine/20260605_112921/world_model/wm3d_v3/scripts/__pycache__` (31.4 KB)
- `world_model/wm3d_v3/tests/__pycache__` -> `cleanup_quarantine/20260605_112921/world_model/wm3d_v3/tests/__pycache__` (22.9 KB)

### python/pytest cache
- Count: 8
- Size: 364.8 KB (373589 bytes)
- `world_model/wm3d_v3/.pytest_cache` -> `cleanup_quarantine/20260605_112921/world_model/wm3d_v3/.pytest_cache` (13.5 KB)
- `world_model/wm3d_v3/wm3d_v3/__pycache__` -> `cleanup_quarantine/20260605_112921/world_model/wm3d_v3/wm3d_v3/__pycache__` (14.4 KB)
- `world_model/wm3d_v3/wm3d_v3/models/__pycache__` -> `cleanup_quarantine/20260605_112921/world_model/wm3d_v3/wm3d_v3/models/__pycache__` (86.7 KB)
- `world_model/wm3d_v3/wm3d_v3/policy/__pycache__` -> `cleanup_quarantine/20260605_112921/world_model/wm3d_v3/wm3d_v3/policy/__pycache__` (27.5 KB)
- `world_model/wm3d_v3/wm3d_v3/encoders/__pycache__` -> `cleanup_quarantine/20260605_112921/world_model/wm3d_v3/wm3d_v3/encoders/__pycache__` (10.6 KB)
- `world_model/wm3d_v3/wm3d_v3/eval/__pycache__` -> `cleanup_quarantine/20260605_112921/world_model/wm3d_v3/wm3d_v3/eval/__pycache__` (106.2 KB)
- `world_model/wm3d_v3/wm3d_v3/training/__pycache__` -> `cleanup_quarantine/20260605_112921/world_model/wm3d_v3/wm3d_v3/training/__pycache__` (72.1 KB)
- `world_model/wm3d_v3/wm3d_v3/benchmarks/__pycache__` -> `cleanup_quarantine/20260605_112921/world_model/wm3d_v3/wm3d_v3/benchmarks/__pycache__` (33.8 KB)

### temporary contact sheet
- Count: 4
- Size: 2.9 MB (3015349 bytes)
- `world_model/wm3d_v3/results/context_residual_pixel_gt_p256_smoke/demo_best/contact_sheet_first_mid_last.jpg` -> `cleanup_quarantine/20260605_112921/world_model/wm3d_v3/results/context_residual_pixel_gt_p256_smoke/demo_best/contact_sheet_first_mid_last.jpg` (551.0 KB)
- `world_model/wm3d_v3/results/wm3d_v3_p64_140m_actioncond_context_renderer/demo_best_auto_latest_contact_sheet.jpg` -> `cleanup_quarantine/20260605_112921/world_model/wm3d_v3/results/wm3d_v3_p64_140m_actioncond_context_renderer/demo_best_auto_latest_contact_sheet.jpg` (799.1 KB)
- `world_model/wm3d_v3/results/wm3d_v3_p64_138m_actioncond_full/posttrain_p64_138m_actioncond_full_20260529_122534/demo_contact_sheet.jpg` -> `cleanup_quarantine/20260605_112921/world_model/wm3d_v3/results/wm3d_v3_p64_138m_actioncond_full/posttrain_p64_138m_actioncond_full_20260529_122534/demo_contact_sheet.jpg` (795.5 KB)
- `world_model/wm3d_v3/report_assets/demo_best_auto_latest_contact_sheet.jpg` -> `cleanup_quarantine/20260605_112921/world_model/wm3d_v3/report_assets/demo_best_auto_latest_contact_sheet.jpg` (799.1 KB)

## Skipped Items

### Protected training artifacts
- `world_model/wm3d_v3/results/wm3d_v3_p64_300m_stage1_world_oxe_droid20k_fromscratch_2node_v1`
- `world_model/wm3d_v3/results/wm3d_v3_p64_300m_stage1_world_oxe_droid20k_fromscratch_2node_v1/tb`
- `world_model/wm3d_v3/results/wm3d_v3_p64_300m_stage1_world_oxe_droid20k_fromscratch_2node_v1/ckpt`
- `world_model/wm3d_v3/results/wm3d_v3_p64_300m_stage1_world_oxe_droid20k_fromscratch_2node_v1/native3d_benchmark_v2_smoke`
- `world_model/wm3d_v3/results/wm3d_v3_p64_300m_stage1_world_oxe_droid20k_fromscratch_2node_v1/native3d_benchmark_v2_smoke2`
- `world_model/wm3d_v3/results/wm3d_v3_p64_300m_stage1_world_oxe_droid20k_fromscratch_2node_v1/basic_eval_after_stage1`
- `world_model/wm3d_v3/results/wm3d_v3_p64_300m_stage1_world_oxe_droid20k_fromscratch_2node_v1/world3d_smoke_20260605_004255`
- `world_model/wm3d_v3/results/wm3d_v3_p64_300m_stage1_world_oxe_droid20k_fromscratch_2node_v1/native3d_benchmark_v2_smoke3`
- `world_model/wm3d_v3/results/wm3d_v3_p64_300m_stage1_world_oxe_droid20k_fromscratch_2node_v1/tb/events.out.tfevents.1780586166.k8s-node1.980078.0`
- `world_model/wm3d_v3/results/wm3d_v3_p64_300m_stage1_world_oxe_droid20k_fromscratch_2node_v1/ckpt/step_00036000.pt`
- `world_model/wm3d_v3/results/wm3d_v3_p64_300m_stage1_world_oxe_droid20k_fromscratch_2node_v1/ckpt/best.pt`
- `world_model/wm3d_v3/results/wm3d_v3_p64_300m_stage1_world_oxe_droid20k_fromscratch_2node_v1/ckpt/step_00026000.pt`
- `world_model/wm3d_v3/results/wm3d_v3_p64_300m_stage1_world_oxe_droid20k_fromscratch_2node_v1/ckpt/step_00014000.pt`
- `world_model/wm3d_v3/results/wm3d_v3_p64_300m_stage1_world_oxe_droid20k_fromscratch_2node_v1/ckpt/step_00004000.pt`
- `world_model/wm3d_v3/results/wm3d_v3_p64_300m_stage1_world_oxe_droid20k_fromscratch_2node_v1/ckpt/step_00006000.pt`
- `world_model/wm3d_v3/results/wm3d_v3_p64_300m_stage1_world_oxe_droid20k_fromscratch_2node_v1/ckpt/epoch_000.pt`
- `world_model/wm3d_v3/results/wm3d_v3_p64_300m_stage1_world_oxe_droid20k_fromscratch_2node_v1/ckpt/step_00020000.pt`
- `world_model/wm3d_v3/results/wm3d_v3_p64_300m_stage1_world_oxe_droid20k_fromscratch_2node_v1/ckpt/step_00032000.pt`
- `world_model/wm3d_v3/results/wm3d_v3_p64_300m_stage1_world_oxe_droid20k_fromscratch_2node_v1/ckpt/step_00018000.pt`
- `world_model/wm3d_v3/results/wm3d_v3_p64_300m_stage1_world_oxe_droid20k_fromscratch_2node_v1/ckpt/step_00016000.pt`
- `world_model/wm3d_v3/results/wm3d_v3_p64_300m_stage1_world_oxe_droid20k_fromscratch_2node_v1/ckpt/step_00008000.pt`
- `world_model/wm3d_v3/results/wm3d_v3_p64_300m_stage1_world_oxe_droid20k_fromscratch_2node_v1/ckpt/step_00038000.pt`
- `world_model/wm3d_v3/results/wm3d_v3_p64_300m_stage1_world_oxe_droid20k_fromscratch_2node_v1/ckpt/epoch_002.pt`
- `world_model/wm3d_v3/results/wm3d_v3_p64_300m_stage1_world_oxe_droid20k_fromscratch_2node_v1/ckpt/step_00034000.pt`
- `world_model/wm3d_v3/results/wm3d_v3_p64_300m_stage1_world_oxe_droid20k_fromscratch_2node_v1/ckpt/step_00030000.pt`
- `world_model/wm3d_v3/results/wm3d_v3_p64_300m_stage1_world_oxe_droid20k_fromscratch_2node_v1/ckpt/step_00028000.pt`
- `world_model/wm3d_v3/results/wm3d_v3_p64_300m_stage1_world_oxe_droid20k_fromscratch_2node_v1/ckpt/step_00012000.pt`
- `world_model/wm3d_v3/results/wm3d_v3_p64_300m_stage1_world_oxe_droid20k_fromscratch_2node_v1/ckpt/step_00024000.pt`
- `world_model/wm3d_v3/results/wm3d_v3_p64_300m_stage1_world_oxe_droid20k_fromscratch_2node_v1/ckpt/step_00010000.pt`
- `world_model/wm3d_v3/results/wm3d_v3_p64_300m_stage1_world_oxe_droid20k_fromscratch_2node_v1/ckpt/epoch_001.pt`
- `world_model/wm3d_v3/results/wm3d_v3_p64_300m_stage1_world_oxe_droid20k_fromscratch_2node_v1/ckpt/latest.pt`
- `world_model/wm3d_v3/results/wm3d_v3_p64_300m_stage1_world_oxe_droid20k_fromscratch_2node_v1/ckpt/step_00022000.pt`
- `world_model/wm3d_v3/results/wm3d_v3_p64_300m_stage1_world_oxe_droid20k_fromscratch_2node_v1/ckpt/step_00002000.pt`
- `world_model/wm3d_v3/results/wm3d_v3_p64_300m_stage1_world_oxe_droid20k_fromscratch_2node_v1/native3d_benchmark_v2_smoke/visuals`
- `world_model/wm3d_v3/results/wm3d_v3_p64_300m_stage1_world_oxe_droid20k_fromscratch_2node_v1/native3d_benchmark_v2_smoke/world3d_claim_balanced.json`
- `world_model/wm3d_v3/results/wm3d_v3_p64_300m_stage1_world_oxe_droid20k_fromscratch_2node_v1/native3d_benchmark_v2_smoke/visuals/bridge_bridge_00027_sample_000000014095_s4_native3d_counterfactual.json`
- `world_model/wm3d_v3/results/wm3d_v3_p64_300m_stage1_world_oxe_droid20k_fromscratch_2node_v1/native3d_benchmark_v2_smoke/visuals/bridge_bridge_00005_sample_000000002824_s16_native3d_counterfactual.gif`
- `world_model/wm3d_v3/results/wm3d_v3_p64_300m_stage1_world_oxe_droid20k_fromscratch_2node_v1/native3d_benchmark_v2_smoke/visuals/bridge_bridge_00005_sample_000000002824_s16_native3d_counterfactual.json`
- `world_model/wm3d_v3/results/wm3d_v3_p64_300m_stage1_world_oxe_droid20k_fromscratch_2node_v1/native3d_benchmark_v2_smoke/visuals/bridge_bridge_00027_sample_000000014095_s4_native3d_counterfactual.gif`
- `world_model/wm3d_v3/results/wm3d_v3_p64_300m_stage1_world_oxe_droid20k_fromscratch_2node_v1/native3d_benchmark_v2_smoke2/visuals`
- ... 35 more protected paths omitted from report; left in place.

### Smoke/debug results not moved
- `world_model/wm3d_v3/results/_smoke_actioncond` (38.8 GB): contains checkpoint/manifest/config-like artifact: world_model/wm3d_v3/results/_smoke_actioncond/ckpt/best.pt
- `world_model/wm3d_v3/results/_smoke_hunyuan_flow_denoiser` (7.0 MB): contains checkpoint/manifest/config-like artifact: world_model/wm3d_v3/results/_smoke_hunyuan_flow_denoiser/ckpt/best.pt
- `world_model/wm3d_v3/results/_smoke_hunyuan_flow_rough` (7.0 MB): contains checkpoint/manifest/config-like artifact: world_model/wm3d_v3/results/_smoke_hunyuan_flow_rough/ckpt/best.pt
- `world_model/wm3d_v3/results/_smoke_hunyuan_latent_residual` (6.2 MB): contains checkpoint/manifest/config-like artifact: world_model/wm3d_v3/results/_smoke_hunyuan_latent_residual/ckpt/best.pt
- `world_model/wm3d_v3/results/_smoke_libero_bc_teacher_task1` (16.0 MB): contains checkpoint/manifest/config-like artifact: world_model/wm3d_v3/results/_smoke_libero_bc_teacher_task1/config.json
- `world_model/wm3d_v3/results/_smoke_libero_bc_teacher_task1_epoch1` (16.0 MB): contains checkpoint/manifest/config-like artifact: world_model/wm3d_v3/results/_smoke_libero_bc_teacher_task1_epoch1/config.json
- `world_model/wm3d_v3/results/_smoke_oxe_fullpolicy_cached_v4_8gpu` (2.7 GB): contains checkpoint/manifest/config-like artifact: world_model/wm3d_v3/results/_smoke_oxe_fullpolicy_cached_v4_8gpu/ckpt/best.pt
- `world_model/wm3d_v3/results/_smoke_phase_policy` (3.8 GB): contains checkpoint/manifest/config-like artifact: world_model/wm3d_v3/results/_smoke_phase_policy/ckpt/best.pt
- `world_model/wm3d_v3/results/context_only_pixel_gt_p256_smoke` (910.9 MB): contains checkpoint/manifest/config-like artifact: world_model/wm3d_v3/results/context_only_pixel_gt_p256_smoke/ckpt/best.pt
- `world_model/wm3d_v3/results/context_residual_pixel_gt_p256_smoke` (911.1 MB): contains checkpoint/manifest/config-like artifact: world_model/wm3d_v3/results/context_residual_pixel_gt_p256_smoke/ckpt/best.pt
- `world_model/wm3d_v3/results/scaling_smoke_eval_generation_v1` (2.2 MB): not processed because it is recent or outside narrow safe set
- `world_model/wm3d_v3/results/scaling_smoke_libero_policy_cache_v1` (28.8 KB): contains checkpoint/manifest/config-like artifact: world_model/wm3d_v3/results/scaling_smoke_libero_policy_cache_v1/manifest_32.jsonl
- `world_model/wm3d_v3/results/scaling_smoke_stage_a_world_visual_v1` (3.4 GB): contains checkpoint/manifest/config-like artifact: world_model/wm3d_v3/results/scaling_smoke_stage_a_world_visual_v1/ckpt/best.pt
- `world_model/wm3d_v3/results/scaling_smoke_stage_b_oxe_joint_v1` (3.5 GB): contains checkpoint/manifest/config-like artifact: world_model/wm3d_v3/results/scaling_smoke_stage_b_oxe_joint_v1/ckpt/best.pt
- `world_model/wm3d_v3/results/scaling_smoke_stage_c_oxe_policy_v1` (1.4 GB): contains checkpoint/manifest/config-like artifact: world_model/wm3d_v3/results/scaling_smoke_stage_c_oxe_policy_v1/ckpt/best.pt
- `world_model/wm3d_v3/results/scaling_smoke_stage_d_libero_policy_v1` (1.2 GB): contains checkpoint/manifest/config-like artifact: world_model/wm3d_v3/results/scaling_smoke_stage_d_libero_policy_v1/ckpt/best.pt
- `world_model/wm3d_v3/results/token_only_pixel_gt_p256_smoke` (910.7 MB): contains checkpoint/manifest/config-like artifact: world_model/wm3d_v3/results/token_only_pixel_gt_p256_smoke/ckpt/best.pt
- `world_model/wm3d_v3/results/wm3d_libero_success_p0_mixed_smoke_v1` (1.3 GB): contains checkpoint/manifest/config-like artifact: world_model/wm3d_v3/results/wm3d_libero_success_p0_mixed_smoke_v1/ckpt/best.pt
- `world_model/wm3d_v3/results/wm3d_libero_success_p0_smoke_v1` (1.3 GB): contains checkpoint/manifest/config-like artifact: world_model/wm3d_v3/results/wm3d_libero_success_p0_smoke_v1/ckpt/best.pt
- `world_model/wm3d_v3/results/wm3d_v3_p64_300m_run1_droid_smoke_fromscratch_2node_v1` (42.2 GB): contains checkpoint/manifest/config-like artifact: world_model/wm3d_v3/results/wm3d_v3_p64_300m_run1_droid_smoke_fromscratch_2node_v1/ckpt/best.pt

### Empty directories left in place
These were only audited. Most are checkpoint placeholders or recent run directories, so they were not removed in this pass.
- `world_model/wm3d_v3/__incoming_sync_tmp__`
- `world_model/wm3d_v3/bert`
- `world_model/wm3d_v3/results/wm3d_v21_ckpt_sweep_cam128_warm0/step_000400`
- `world_model/wm3d_v3/results/wm3d_v3_p256_rgb1b_actioncond/ckpt`
- `world_model/wm3d_v3/results/wm3d_v3_p256_rgb1b_actioncond_stage1/live_demo_ep1_20260531_202500/short`
- `world_model/wm3d_v3/results/wm3d_v3_p256_rgb1b_actioncond_stage1_failed_eof_20260530_002655/ckpt`
- `world_model/wm3d_v3/results/wm3d_v3_p64_140m_actioncond_context_motion_p0_proposer/ckpt`
- `world_model/wm3d_v3/results/wm3d_v3_p64_140m_p0_action_policy_oxe_v9warm_no_video_1gpu_bs128_baseonly_v1/ckpt`
- `world_model/wm3d_v3/results/wm3d_v3_p64_140m_p0_action_policy_oxe_v9warm_no_video_1gpu_bs128_v1/ckpt`
- `world_model/wm3d_v3/results/wm3d_v3_p64_500m_stage_a_world_visual_hunyuan_oxeall_v1_8gpu/manual_epoch0_eval_20260604_170721`

## Source/Config Obsolescence Candidates

These are report-only candidates. They were not modified because training/config updates are owned by the main agent.
- `configs/v3_p64_300m_run1_droid_smoke_fromscratch_2node_v1.yaml`: 300m smoke run config; keep unless main agent confirms obsolete
- `scripts/run_300m_run1_droid_smoke_fromscratch_2node_v1.sh`: 300m smoke launcher; not removed
- `configs/_smoke_oxe_fullpolicy_cached_v4_8gpu.yaml`: smoke config referencing smoke manifest/results; not removed
- `scripts/run_scaling_staged_smoke_v1.sh`: creates scaling_smoke manifests/results; recent and not removed
- `configs/scaling_smoke_stage_a_world_visual_v1.yaml`: recent scaling smoke config; not removed
- `configs/scaling_smoke_stage_b_oxe_joint_v1.yaml`: recent scaling smoke config; not removed
- `configs/scaling_smoke_stage_c_oxe_policy_v1.yaml`: recent scaling smoke config; not removed
- `configs/scaling_smoke_stage_d_libero_policy_v1.yaml`: recent scaling smoke config; not removed
- `configs/libero_success_p0_smoke.yaml`: LIBERO smoke training config; not removed
- `configs/libero_success_p0_mixed_smoke.yaml`: LIBERO mixed smoke training config; not removed
- `scripts/run_libero_remote_smoke.sh`: remote smoke runner; not removed
- `scripts/run_dist_smoke_2node.sh`: distributed smoke runner; not removed

## Risk Notes

- Quarantined files can be restored by moving them back from the matching path under the quarantine root.
- The active tree space reduction is small because large smoke directories containing checkpoints/configs/manifests were intentionally skipped.
- Python cache removal may cause harmless recompilation on the next import/test run.
- AppleDouble `._*` files were classified as local macOS metadata sidecars after `file` reported AppleDouble encoding; their real counterpart files were left untouched.
- Contact sheets were treated as temporary generated media; no model outputs required for training or checkpoints were moved.

## Disk Snapshot

```text
Filesystem                   Size  Used Avail Use% Mounted on
/dev/mapper/data_vg-data_lv  7.0T  5.5T  1.2T  83% /data
```
