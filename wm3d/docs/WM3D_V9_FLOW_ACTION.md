# WM3D V9 flow-action contract

## Scope

V9 is an isolated alternative to V8. It replaces only the executable policy's
continuous one-shot regression owner with a flow-matching decoder. The world,
RGB/P256, geometry, grouped-robot representation, action normalization and
public serving output remain WM3D contracts. V8 profiles are fail-closed to
`policy_action_mode: regression`; V9 profiles are fail-closed to
`policy_action_mode: flow_matching`.

No V9 training run is authorized or implied by this implementation.

## WSA source mapping

The implementation follows the concrete action path in official WSA code:

- WSA Base trains `x_t = (1-t) action + t noise`, predicts the velocity
  `noise - action`, and integrates from Gaussian noise at inference.
- WSA Large uses a dedicated ActionDiT with time modulation and context
  conditioning in every block.
- WSA Large's continuous scheduler uses
  `phi(u) = shift*u / (1 + (shift-1)*u)`, shifted training weights, and Euler
  integration over a descending sigma schedule. Its default shift is 5 and
  its action sampler uses 10 inference steps.

Primary references:

- <https://github.com/zaleni/WSA>
- <https://arxiv.org/abs/2607.03941>
- <https://github.com/zaleni/WSA/blob/main/src/lerobot/policies/WSA_Base/modeling_wsa_base.py>
- <https://github.com/zaleni/WSA/blob/main/src/lerobot/policies/WSA_Large/core/models/wan22/action_dit.py>
- <https://github.com/zaleni/WSA/blob/main/src/lerobot/policies/WSA_Large/core/models/wan22/schedulers/scheduler_continuous.py>

## WM3D adaptation

WSA assumes a padded single action vector and task-specific policy packaging.
WM3D must support heterogeneous robots without changing physical meaning. V9
therefore keeps the public group-major tensor `[B,G,C,A]`, semantic IDs,
per-source normalization, real query timestamps, variable group/query masks
and deterministic physical decoding.

The flow decoder receives the action-free policy query produced from observed
native 3D/vision, task, exact current state and real action history. The noisy
continuous trajectory is embedded separately. Every flow block has:

1. time-modulated factorized self-attention over trajectory time and robot
   group;
2. factorized cross-attention back to the policy condition at every layer;
3. time-modulated SwiGLU refinement.

This is the WM3D equivalent of WSA ActionDiT's persistent context path. It is
not an auxiliary loss and cannot be optimized away by balancing a new scalar
objective.

Future candidate actions remain forbidden from the action-free/policy trunk.
They continue to condition only the factual world branch. Thus a fixed flow
noise sample must give bitwise-identical policy output when only the factual
future candidate changes.

## Training objective

For continuous, semantically valid action dimensions:

```text
u ~ Uniform(0,1)
sigma = shift*u / (1 + (shift-1)*u)
epsilon ~ Normal(0,1)
x_sigma = (1-sigma)*action + sigma*epsilon
velocity_target = epsilon - action
```

The optimized continuous fine-action term is WSA's shifted-schedule weighted
velocity MSE. It **replaces** V8 continuous SmoothL1. It is not added beside
it. The old composed coarse regression is retained as an evaluation metric but
has zero V9 optimization contribution. This avoids competing action losses.

Absolute gripper and other binary controller fields retain BCE logits and are
excluded from Gaussian diffusion. Their normalization must remain identity.
This preserves exact discrete semantics across heterogeneous action schemas.

## Inference and serving

Inference starts from masked Gaussian noise and applies the sealed 10-step
shifted Euler schedule. The final normalized continuous sample and sigmoid
binary values pass through the same offset/scale and grouped-action decoder as
V8. Serving continues to consume `policy_action` and `policy_action_mask`; no
V9-only robot adapter is required.

Validation batches may carry labels through the shared data adapter, but the
flow sampler never reads them. Supplying a fixed initial noise tensor makes
sampling deterministic for tests and A/B evaluation.

## Profiles and verification

- `configs/model/native_1b_v9_flow.yaml`
- `configs/model/native_5b_v9_flow.yaml`
- `configs/objective/stage0_native_v9_flow.yaml`
- `tests/test_flow_action_policy.py`

The tests cover scheduler math, velocity targets, binary semantics, physical
decoding, deterministic iterative sampling, future-candidate isolation,
context dependence, non-stacked losses, gradients and meta-device profile
construction. These tests establish code and ABI correctness; they do not
establish downstream robot success. A future V9 experiment still needs action
regression/trajectory metrics across sources and closed-loop evaluations that
are broader than any one benchmark such as LIBERO.
