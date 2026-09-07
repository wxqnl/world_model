# Model guide

- `NativeWorldModel` remains the shared implementation for native 1B/5B profiles.
- `FoundationWorldModel` is the V10 WSA-Large adapter. Keep the official Wan2.2 video,
  ActionDiT, Future3D and MoT owners intact. The native-3D extension may zero-gate the observed
  per-view P144 state into Future3D and causally read K8xP64 state deltas from the shared tokens;
  it must not expand or replace the MoT owners.
- WSA-Large does not use Qwen3-VL. Do not add WSA-Base/Qwen components to the foundation profile.
- Keep the foundation token hierarchy explicit: P144 joint current state, K8xP64 temporal rollout,
  and the released WSA P1296 final output. Do not globally increase `P` without an isolated
  quality/throughput comparison.
- Clean future action/time/mask may condition video and native-3D world prediction, but never the
  ActionDiT policy output. Future VGGT targets are loss-only and never model inputs.
- Candidate-conditioned video uses the actual ActionDiT input at each flow-matching step,
  with a fixed noise trajectory and the upstream action scheduler. The released WSA-Large
  VideoDiT ignores its optional clean action argument; passing that argument is not conditioning.
- Keep policy prediction in predict_action, independent of predict_world. Never expose the
  action result from a candidate-clamped world pass as a policy prediction.
- Keep policy state action-free and factual world refinement explicitly action-conditioned.
- Preserve continuous time, masks, grouped embodiment identity, and explicit RGB/depth/point/pose
  heads.
- A capacity change belongs in a model profile, not a forked model class.
