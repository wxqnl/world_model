# Model guide

- `NativeWorldModel` remains the shared implementation for native 1B/5B profiles.
- `FoundationWorldModel` supports two explicit initialization contracts. `wsa_checkpoint`
  is the historical released-checkpoint adapter; `wan_video` is the current own-model
  pretraining route and must reject WSA checkpoints and old adapter warm starts.
- Reuse upstream Wan video, ActionDiT, MoT, flow schedulers and samplers. The native route
  replaces only Future3D's state contract: per-view K8xP64 queries participate inside MoT;
  one-to-one VGGT readout supervises every horizon. Do not bolt on a second posthoc dynamics head.
- Use the base WSALarge mask, not WSALargeJoint: native 3D/action cannot read noised future
  RGB. Future native labels are loss-only; the observed native state is the only 3D input.
- WSA-Large does not use Qwen3-VL. Do not add WSA-Base/Qwen components to the foundation profile.
- Source-native future commands keep their K-interval slots and masks. ActionTimeConditioner
  must return exact zero on empty slots, including after training; masked time values cannot
  affect the command mean. The delivered WSA action MSE averages only valid times/dimensions.
  Native3D rollout likewise divides by the valid command count within each interval.
- Keep the two layouts explicit. Historical adapter: P144 observation, auxiliary K8xP64
  rollout and released P1296 output. Native pretraining: P64 observation and K8xP64 future
  states per view inside MoT, without a second rollout loss or P1296 resampler.
  Preserve view/horizon identity; capacity changes require a real quality/throughput comparison.
- Clean future action/time/mask may condition video and native-3D world prediction, but never the
  ActionDiT policy output. Future VGGT targets are loss-only and never model inputs.
- Candidate-conditioned video holds the actual ActionDiT input at the clean, masked candidate
  with action noise timestep zero throughout video denoising. Only unknown video is sampled;
  do not progressively noise a known command or return it as policy output. WSA trains the
  two noise levels independently. Its released VideoDiT ignores the optional clean action
  argument; passing that argument alone is not conditioning.
- The current foundation phase trains video/action/3D jointly. Default evaluation uses
  predict_joint from observation, without future labels, and separately measures predict_action.
  Candidate intervention is an optional later-phase diagnostic, not the primary quality gate.
- Keep policy prediction in predict_action, independent of predict_world. Joint generation may
  return its generated action, explicitly distinguished from the independent policy. Never expose
  the action result from a candidate-clamped world pass as a policy prediction.
- Keep policy state action-free and factual world refinement explicitly action-conditioned.
- Preserve continuous time, masks, grouped embodiment identity, and explicit RGB/depth/point/pose
  heads.
- A capacity change belongs in a model profile, not a forked model class.
