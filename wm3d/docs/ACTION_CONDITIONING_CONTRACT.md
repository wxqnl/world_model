# Action conditioning contract

## 目标

Stage0 的 action policy 必须从 observation/native 3D、task、当前物理状态和真实 action
history 预测可执行 grouped action。future candidate action 只属于 world/planning 分支，不能进入
action-free state 或 policy trunk。

本实现补强 task 在 policy 深层主链中的持续作用，但不增加新的 task loss，也不改已有 action
loss 的权重。现有 fine/coarse action supervision 是唯一训练信号，避免多目标权重竞争。

同一 embodiment 可以来自多个数据源，而每个来源的 action/state z-score 坐标可能不同。
`embodiment/group/semantic` 不能唯一说明这套数值坐标。policy query 因此还接收与训练、
反归一化完全同源的 calibration descriptor（action/state offset、log-scale 和有效维），让模型
显式知道自己正在预测哪套控制坐标；descriptor 不含 source id 或未来 action，也没有新增 loss。

## 实现

- task bank 的 pooled embedding 仍只经过一次共享 `task_action` 投影。
- 每个 `ActionBlock` 的 attention 与 feed-forward 输入，以及最终 visual spatial read 的
  policy query，使用独立的 feature-wise task modulation。
- modulation 只作用于 future policy query；history token 继续表示真实物理证据，不在每层被
  task 重新标注。
- scale/shift gate 全零初始化，并经 `tanh` 有界化。初始化时前向逐元素等同于未启用该路径的
  模型；训练只会从已有 action objective 学习需要多少 task 调制。
- calibration 使用 query-only 的零初始化线性投影。初始化时逐元素保持旧 policy；第一步即可
  从已有 action objective 获得梯度。它不写回 history/world/RGB，不引入数据集专用 head。
- 该路径属于 `policy_action_trunk` 的 gradient owner，不参与 world、geometry、appearance 或
  RGB 的参数所有权。

## 保持不变的合同

- 统一模型按真实 timestamp/mask 接受动态 history 与 query 数，不声明全机器人统一频率或
  horizon；当前 Panda V8 policy profile 使用 H16 的真实 20 Hz history、K8 candidate、H1
  执行，这只是经审计的 Panda controller profile，不是由 LIBERO 定义的全局结构。
- 每个 adapter 保持自身经审计的物理 action ABI；Panda 仍为 delta position（米）、
  base-frame SO(3) rotvec（弧度）和 absolute close01。训练与 serving 必须选择同一封存
  calibration profile，并同时应用 action 与 current-state normalization。
- grouped action owner、mask、time、semantic/group id 均不改变。
- future factual/zero candidate 对 policy 和 action-free state 的输出差异必须逐元素为零。
- task/calibration 改动不改变 world/RGB、数据闭包、batch 或已有 Action objective；当前 RGB 路径见 [Native RGB](WM3D_NATIVE_RGB.md)。
- 不启用随机 history 截短。当前 history/state bridge 与 world/RGB 共享，在没有联合实证前
  截短会重新耦合 action 修复与 RGB 路径；history 依赖只作为评测消融观察。

## 代码门槛

必须同时满足：

1. 启用 modulation 且 gate 为零时，policy、world 和 RGB 与旧前向逐元素一致。
2. gate 学习后，只能改变 policy query/policy 输出，不能改变 history、world 或 RGB。
3. 每层 attention/FF gate 和最终 spatial gate 都能从现有 action loss 获得有限非零梯度。
4. 1B/5B profile 可 materialize，参数合同与 gradient ownership 一致。
5. serving 的 H16/K8/H1、物理单位和 normalization 与训练一致。

## 实证门槛

代码门槛只能证明实现正确，不能证明真实 VLA 能力。fresh canary 至少需要固定同 seed 比较：

- 正确 task 相对 shuffled/neutral task 的 action error 改善；
- 正确 observation 相对 mismatched/neutral observation 的改善；
- state/history 消融有意义，但不能成为唯一有效条件；
- baseline action 优于 neutral，token/action gain 正向；
- policy/action-free 不变量逐元素成立，RGB 指标不退化。

最终能力仍以独立 action regression 和多种闭环机器人/模拟器任务成功率为准；LIBERO 只是
其中一个可选下游验收器，不能反向定义统一 action head、group 布局或训练目标。
