# WM3D 代码导读

这份文档说明 `wm3d` 包的调用关系。安装、下载、cache 和启动命令仍以项目根目录及各目录的
`README.md` 为准；这里关注模型为什么这样拆分，以及代码实际执行了什么。

## 从哪里开始读

按一次训练样本的流向阅读：

1. [`encoders/CODING.md`](encoders/CODING.md)：三路 RGB 如何经冻结 VGGT 变成逐视角 token、
   depth、point、camera 和 confidence；
2. [`data/CODING.md`](data/CODING.md)：动作与传感器怎样对齐，cache 怎样封存，Dataset 怎样组成
   T24/K16 窗口，sampler 怎样精确恢复；
3. [`models/CODING.md`](models/CODING.md)：多视角融合、state/action 两条主干、bridge、防泄漏约束和
   显式输出头；
4. [`training/CODING.md`](training/CODING.md)：loss、FSDP2、训练循环、checkpoint 和 eval。

## 一次样本的实际路径

```text
公开数据 episode
  ├─ 三路 RGB ──► 冻结 VGGT ──► view token + depth/point/pose/confidence
  ├─ 任务文本 ──► 冻结 FLAN-T5-XL task bank
  ├─ 原始动作 ──► grouped action 对齐与 robust normalization
  └─ 力/触觉/状态 ──► context-only auxiliary token
                              │
                              ▼
              immutable frame shard + window index
                              │
                              ▼
                     WM3D.forward(batch)
  ┌──────────────────────────────────────────────────────────────┐
  │ view token ─► MultiViewTokenFuser ─► state trunk             │
  │ context action + future query ─────► action trunk            │
  │ state trunk ◄──── constrained bidirectional bridges ────► action trunk
  └──────────────────────────────────────────────────────────────┘
         │                                             │
         ├─ future VGGT token                          └─ grouped action distribution
         ├─ RGB
         ├─ depth / point
         └─ camera / geometry confidence
                              │
                              ▼
                   native losses + FSDP2 + DCP
```

训练代码不会在线调用 VGGT 或 FLAN-T5。两者只在数据准备阶段生成冻结输入或监督；正式优化的主体是
WM3D state/action 主干和显式输出头。

## 核心张量契约

下面是 `configs/train/5b_h200.yaml` 这个训练特例的形状。`WM3DConfig` 本身没有写死“5B”名称。

| 名称 | 形状 | 含义 |
|---|---|---|
| `world_tokens` | `[B,24,3,144,2048]` | 24 帧、三视角、12×12 VGGT patch token |
| `view_mask` | `[B,24,3]` | 每帧各相机是否存在 |
| `state` | `[B,40,144,2560]` | 24 帧 context 与 16 帧 future query |
| `action` | `[B,40,8,2048]` | 最多 8 个动作组的 context/query latent |
| action value | `[B,40,8,6,16]` | 每视觉帧 6 个动作 substep、每组最多 16 维 |
| `pred_tokens` | `[B,16,144,2048]` | 全部未来帧的世界 token |
| `depth` | `[B,16,3,144]` | 每视角显式深度 |
| `point` | `[B,16,3,144,3]` | 每视角显式 3D point |

`world_tokens` 这个参数名仍指逐视角 context token。多视角融合后的 `context_state` 才会送入 state trunk。
阅读 `WM3D.forward()` 时应留意这个命名差异。

## 三条不可破坏的边界

### 1. 世界状态归 state trunk 所有

`state trunk` 产生 future token、RGB、depth、point、camera 和 confidence。Action trunk 不生成世界，
VGGT 也不参与在线 rollout。这里的 native 3D 指显式几何状态、监督与输出，并不表示内部 Transformer
使用 voxel 或 point-cloud token；内部 state 仍是 12×12 patch lattice。

### 2. 策略动作归 action trunk 所有

Action trunk 只读取 context action 和 learned future queries。不同机器人通过 group ID、embodiment ID、
维度 mask 和组 mask 共用一条动作主干。

### 3. 真实未来动作不能泄漏给策略

真实未来动作只加到 future state query，作为世界动力学条件。Action trunk 不读取这些标签；bridge 从
state 写入 action 时也只读取前 T 帧 context state。`tests/test_model.py` 会修改未来动作并检查 action
输出逐位不变，同时检查对应梯度为零。

## 配置与实现的关系

模型结构来自 `WM3DConfig`，YAML 只提供一组参数。正式配置必须经过
`scripts/tools/materialize_config.py` 绑定：

- sealed dataset 与 receipt；
- code receipt；
- environment contract 与 receipt；
- world size、shard degree 和 run lineage；
- checkpoint 硬停位置。

基础 YAML 故意包含 `__MATERIALIZE_REQUIRED__`，不能直接启动。这样可以避免同一个文件在不同数据、代码或
集群拓扑下被误认为同一次实验。

## 当前验证能说明什么

仓库单测覆盖形状、forward/backward、loss、参数预算、采样恢复和未来动作防泄漏。真实两卡 smoke 已经跑通
一轮训练、验证、显式 checkpoint 和 eval。这些证据说明调用链、数值和恢复协议可运行，不证明 5B 配置的
最终 RGB、几何或机器人成功率。具体尚未验证的设计选择写在四份目录级文档中，避免把工程假设写成实验结论。
