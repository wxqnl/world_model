# WM3D V9 flow-action contract

## 工程状态

- GitHub branch: `v9`
- development branch: `codex/v9-flow-action-20260826`
- isolated worktree: `/data/Minko/wm3d_v9_flow_action_20260826`
- model profiles: `native_1b_v9_flow.yaml`、`native_5b_v9_flow.yaml`
- objective profile: `stage0_native_v9_flow.yaml`
- implementation: `wm3d/models/flow_action.py`
- contract tests: `tests/test_flow_action_policy.py`

当前交付是代码和 ABI 合同，不包含 V9 已训练 checkpoint，也没有启动 V9/5B 实验。V9 从
当前 V8 RGB/P256 修正之后独立分支，禁止把 V8/旧 canary checkpoint 作为初始化或 resume。
未来实验必须重新 materialize V9 model/data/runtime seal。

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

### 不与 WSA 生搬硬套的部分

WSA 的公开 action 包装主要面向单个 padded action vector；WM3D 同时服务不同 embodiment、
多个 physical group、不同 semantic dimensions 和 source-native timestamps。V9 因此不复制
WSA 的 task-specific adapter，而只复用已经被其代码验证的 flow 核心：continuous interpolation、
velocity target、ActionDiT 式 persistent conditioning、shifted weighting 和 Euler sampling。

以下 WM3D 合同不能为了复刻 WSA 跑分而改变：

- group-major `[B,G,C,A]`、semantic/group mask 和真实 query time；
- adapter 审计后的米/弧度/absolute gripper 物理定义；
- 每个 source 的 normalization/calibration 与 serving 对称性；
- future factual candidate 与 executable policy 的严格隔离；
- RGB/P256/world/geometry 的独立 gradient ownership。

## Inference and serving

Inference starts from masked Gaussian noise and applies the sealed 10-step
shifted Euler schedule. The final normalized continuous sample and sigmoid
binary values pass through the same offset/scale and grouped-action decoder as
V8. Serving continues to consume `policy_action` and `policy_action_mask`; no
V9-only robot adapter is required.

Validation batches may carry labels through the shared data adapter, but the
flow sampler never reads them. Supplying a fixed initial noise tensor makes
sampling deterministic for tests and A/B evaluation.

## 未来实验顺序

V9 不应因为代码已合并就直接扩大到 5B。资源可用时按以下顺序独立验证：

1. fresh 1B 小步 canary，检查所有 loss/gradient 有限、flow block 与 policy context 都有梯度、
   fixed-noise future-candidate isolation 严格成立；
2. 固定 validation seed 比较 V8 regression 与 V9 flow 的 normalized/physical trajectory error、
   gripper accuracy、task/vision/state/history conditioning sensitivity 和跨 source 表现；
3. 用相同 observation/task 和多组初始 noise 检查轨迹分布不是单点坍缩或随机噪声；
4. 多种机器人/模拟器闭环任务验证真实执行成功率，LIBERO 只能是其中一项，不能反向定义
   grouped action ABI；
5. 只有 1B 证据同时通过代码、离线 action、serving 和闭环门槛，才考虑单独启动 V9 5B。

RGB/P256 指标沿用 V8 门槛，但它不是 V9 action 路线的替代证据。任何 V9 action 改动都不得
以牺牲 static/motion RGB、world dynamics 或 policy/action-free 不变量换取单一 benchmark 分数。

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
