# Training guide

- Stage0 training and offline evaluation use the same model factory and objective.
- DDP and FSDP2 are runtime choices, not different trainers.
- Checkpoints are complete numbered DCP directories with `COMMITTED.json`; never restore from an
  unverified `latest` alias.
- Required gradient owners must be covered exactly once and have finite, nonzero gradients after
  the first real accumulated backward.
- Resource qualification is launch-specific; stable run contracts must remain resumable.
- Historical foundation profiles use the released WSA joint objective. Native Wan pretraining
  reuses its video/action flow objectives and gives WM3D the in-MoT trajectory supervision.
  Do not mix old V7 RGB/action losses into either route or load WSA weights in the native route.
- Native Wan uses FP32 master parameters/moments and BF16 compute. Root FSDP preserves physical
  timestamps and supervision dtypes; it must not blanket-cast the input batch to BF16.
- MoT calls block internals directly. Until the upstream call boundaries change, use root-only
  FSDP for this route, with within-node sharding and cross-node replication. Do not wrap blocks
  whose forward is bypassed. GPU forward/backward/checkpoint qualification remains mandatory.
- Lower LR for the pretrained video backbone is an optimizer group, not a second training loop.
  New action/3D/interface parameters use the normal LR. The pilot does not freeze random experts.
- Only launch on idle, explicitly assigned GPUs. Do not stop another task to qualify this model.
