# Factual motion qualification ladder

This ladder prevents a structural RGB/action regression from consuming a
500-step or 2500-step 1B run before it is detected.

## Gate A: implementation and gradient invariant

Run one micro-probe optimizer step. It uses the production NativeWorldModel
implementation with reduced width/depth and checks:

- future factual action changes factual P64 and RGB;
- the action-free native prior and policy are bitwise invariant;
- the physical action encoder, pre-trunk factual path, two-layer factual
  decoder, and RGB decoder all receive finite non-zero gradients;
- factual and zero branches use the same differentiable encoder contract.

Any failure is a code failure. Do not start a distributed qualification.

Run Gate A with:

    PYTHONPATH=wm3d \
    python wm3d/scripts/tools/run_factual_motion_microprobe.py \
      --mode structural --steps 1 \
      --output /data/Minko/wm3d_factual_motion_microprobe_runs/structural_001

## Gate B: real high-motion learnability

The fixed seed-7340 K8 high-motion window is converted into eight K1 examples.
All eight examples have identical observation, task, history, and future
timestamp. Only their real physical future action and target differ. The probe
then trains the small model with the existing V7 token, RGB, perceptual,
gradient, and motion objectives. It introduces no new loss.

The receipt compares factual action against both zero action and a horizon-
shuffled physical action. It also reports temporal delta direction, amplitude,
and error for P64 and RGB. Passing requires:

- loss reduction within the wall-time budget;
- factual target error below zero and shuffled controls;
- P64 and RGB temporal direction improving;
- policy/action-free bitwise invariance;
- every required factual/RGB gradient remaining finite and non-zero.

This catches a disconnected action path, a late homogeneous action bias, a
copy-last shortcut, and a model that learns time while ignoring action.

Example:

    cd /data/Minko/wm3d_v8_v7_base_factual_microprobe_20260831
    PYTHONPATH=wm3d \
    /data/Minko/.venvs/wm3d_direct_v8_20260821/bin/python \
      wm3d/scripts/tools/run_factual_motion_microprobe.py \
      --output /data/Minko/wm3d_factual_motion_microprobe_runs/run_001 \
      --steps 80

The expected runtime is minutes on one H100. receipt.json, before.gif, and
after.gif make the result machine-checkable and visually inspectable.

## Gate C: short full-scale qualification

Only after A and B pass, run the unchanged 1B model for 20 to 100 fresh steps:

- step 1 proves distributed gradient ownership and FSDP materialization;
- step 20 checks finite learning trends and throughput;
- step 100 runs the fixed-seed high-motion audit.

Gate C validates scale/distribution behavior. It is not used to rediscover
basic action conditioning errors that Gate A/B can expose in minutes.

Passing this ladder proves implementation connectivity and short-horizon
learnability. It does not replace later checkpoints or downstream robot-task
evaluation as evidence of final model quality.
