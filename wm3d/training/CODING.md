# 训练实现导读

本目录负责把物化配置、sealed dataset、WM3D 和分布式运行时绑定成可恢复训练。这里不定义模型结构，也不
修改数据；任何 code/data/environment receipt 不一致都会在创建正式 optimizer step 前失败。

## 训练系统全貌

训练层把已经固定的模型与数据契约接入分布式优化、checkpoint 和评测：

```text
基础 YAML + code/env/data/topology receipt
  └─ materialize_config.py ─► 单次 run 的物化配置与 lineage
      └─ preflight ─────────► shape、SHA、硬件、磁盘、batch、checkpoint 计划
          └─ WindowDataset + StepAddressedBatchSampler
              └─ WM3D + compute_native_loss
                  └─ bottom-up FSDP2/HSDP + AdamW + WSD schedule
                      ├─ train / validation metrics
                      └─ atomic DCP checkpoint + manifest + COMPLETE
                                      │
                                      ▼
                         显式 checkpoint eval / compare
```

| 部分 | 代码入口 | 职责 |
|---|---|---|
| 配置与准入 | `materialize_config.py`、preflight helpers | 固定本次 run 的代码、环境、数据、拓扑和 lineage |
| 分布式运行时 | `runtime.py`、`fsdp.py` | 初始化 NCCL mesh，bottom-up 包装模型并约束设备拓扑 |
| 优化目标 | `losses.py` | 汇总 token、RGB、depth、point、camera 和 grouped action loss |
| 训练循环 | `train.py` | accumulation、optimizer、scheduler、validation 与硬停 |
| 持久化 | `checkpoint.py` | 原子写入分片模型、optimizer、sampler、RNG 和 manifest |
| 结果检查 | `eval.py`、`compare_eval.py` | 对指定编号 checkpoint 做绝对检查与相邻里程碑对比 |

训练层负责优化与可恢复性，并沿用 Dataset 字段和 `WM3D.forward()` 的所有权边界。正式运行只接受物化配置和
显式编号 checkpoint；代码拒绝 `latest`、隐式单卡回退、缺 receipt 的数据和不完整 checkpoint。

## 1. 一次正式启动的顺序

`train.py::main()` 的主要阶段如下：

```text
torchrun/NCCL 初始化
  → 读取物化 YAML
  → code、environment、dataset receipt preflight
  → 核验 shape、batch、磁盘、硬件和 checkpoint 计划
  → 构造 Dataset 与 step-addressed sampler
  → 构造 WM3D 并核验精确参数量
  → bottom-up FSDP2/HSDP
  → AdamW + eager optimizer state
  → 显式编号 checkpoint resume（可选）
  → train / validation / checkpoint 硬停循环
```

基础 YAML 不能直接启动。`materialize_config.py` 会把 dataset、代码、环境、拓扑与 run lineage 的路径和 SHA
写入新配置，训练只接受该物化结果。

## 2. 分布式初始化

`initialize_distributed()` 强制从 `torchrun` 启动：

```python
required = ("RANK", "LOCAL_RANK", "WORLD_SIZE")
if missing:
    raise RuntimeContractError(
        "torchrun environment ... direct python launch is forbidden"
    )

torch.cuda.set_device(local_rank)
dist.init_process_group(
    backend="nccl",
    init_method="env://",
    device_id=device,
)
```

正式训练不自动退回 CPU、Gloo 或单进程。这样可以避免启动命令写错后无声运行在另一种拓扑。

## 3. HSDP device mesh

```python
replicate_degree = world_size // shard_degree
mesh = init_device_mesh(
    "cuda",
    (replicate_degree, shard_degree),
    mesh_dim_names=("replicate", "shard"),
)
```

128 张卡、`shard_degree=8` 时，mesh 为 `[replicate=16, shard=8]`：

- 一个 8 卡节点内切分参数、梯度与 optimizer state；
- 16 个 shard group 之间做数据并行复制；
- 节点内依赖 NVLink，跨节点依赖高速网络。

64 卡同理得到 `[8,8]`。配置默认不允许任意 shard degree，避免 checkpoint 与通信模型漂移。

## 4. Bottom-up FSDP2

模型通过 `iter_transformer_units()` 暴露通信粒度：

```python
yield self.view_fuser
yield from self.state_blocks
yield from self.action_blocks
yield from self.bridges
```

运行时逐个包装，再包装根模型：

```python
for unit in units:
    fully_shard(
        unit,
        mesh=mesh,
        mp_policy=policy,
        reshard_after_forward=True,
    )
fully_shard(model, mesh=mesh, mp_policy=policy, ...)
```

Mixed precision 为：

```text
parameter/output: BF16
gradient reduce:  FP32
```

Bottom-up wrapping 让每个 Transformer block 形成可控的 all-gather/reshard 单元，避免整个 5B 根模块同时常驻。
实现使用 PyTorch 官方 [`fully_shard`](https://docs.pytorch.org/docs/main/distributed.fsdp.fully_shard.html) API。

## 5. 为什么提前创建 AdamW state

AdamW 通常在某参数第一次获得梯度时才创建 `exp_avg` 和 `exp_avg_sq`。可选模态若早期 batch 没出现，DCP
加载时可能遇到 optimizer state 缺项。代码在训练前为所有参数创建等价的零状态：

```python
state["step"] = torch.tensor(0.0, dtype=torch.float32)
state["exp_avg"] = torch.zeros_like(parameter)
state["exp_avg_sq"] = torch.zeros_like(parameter)
```

这不是预先做一次 optimizer update；step 仍为零，动量也为零。它固定 checkpoint schema，使有无早期可选
模态不影响 optimizer state 文件集。

## 6. Gradient accumulation

Sampler 每个 optimizer step 产生 `gradient_accumulation` 个 local micro-batch。非最后一个 micro-step 关闭
FSDP gradient sync：

```python
model.set_requires_gradient_sync(enabled, recurse=True)
model.set_reshard_after_backward(enabled, recurse=True)
```

最后一个 micro-step 才跨 rank 同步并执行：

```text
unscale/finite check → gradient clip → optimizer.step → scheduler step
```

Global batch 必须严格满足：

```text
world_size × micro_batch_size × gradient_accumulation
```

配置校验不允许 YAML 声明值与这个乘积不同。

## 7. Native loss 总入口

`wm3d_loss()` 先要求所有显式输出和 target 存在，再检查 output 全部有限：

```python
missing_output = required_output.difference(output)
missing_batch = required_batch.difference(batch)
if missing_output or missing_batch:
    raise KeyError(...)

for name in required_output:
    _validate_finite(name, output[name])
```

Loss 不允许某个 head 临时缺失后只训练剩余项。目标是让 token、RGB、几何和 action 从 step 0 共同优化。

## 8. Future token loss

```python
token_mse = F.mse_loss(pred_tokens.float(), target_tokens.float())
token_cosine = (
    1.0 - F.cosine_similarity(
        pred_tokens.float(), target_tokens.float(), dim=-1
    )
).mean()
```

MSE 约束向量的绝对值，cosine 约束方向。Target 是冻结 VGGT 2048D token，而不是另一个可训练 encoder 的
moving target。

## 9. RGB loss

只选择模型实际解码的 future frame，并用 `target_view_mask` 屏蔽缺失相机：

```python
rgb_view_mask = target_view_mask.index_select(1, rgb_frame_indices)
```

三项损失：

```python
charbonnier = sqrt((prediction - target) ** 2 + epsilon)
gradient = |dx_pred - dx_target| + |dy_pred - dy_target|
laplacian = |L(prediction) - L(target)|
```

Charbonnier 是平滑 L1 型重建项；gradient 与 Laplacian 增加边缘和多尺度高频约束。它们不能单独解决像素
回归导致的多模态平均问题。默认只解码 K16 中四帧，若 RGB 清晰度不足，应同时评估监督密度和生成分布，而
不是只调 loss 权重。

## 10. Depth、point 与 camera loss

Depth 使用 log-domain 误差：

```python
depth_log_error = depth.log() - target_depth.log()
depth_log = mean(depth_log_error ** 2) - 0.5 * mean(depth_log_error) ** 2
```

该形式参考 scale-invariant depth objective，例如
[Depth Map Prediction from a Single Image](https://papers.nips.cc/paper_files/paper/2014/hash/91c56ce4a249fae5419b90cba831e303-Abstract.html)。
代码中的系数、confidence mask 和 pseudo-label target 是本项目版本。

`depth_gradient` 实际比较 future horizon 相邻帧：

```python
depth_gradient = (
    depth_log_error[:, 1:] - depth_log_error[:, :-1]
).abs()
```

所以它是时间一致性项，不是图像空间 Sobel gradient。

Point 与 camera 使用 Smooth L1：

```python
F.smooth_l1_loss(prediction, target, beta=0.01, reduction="none")
```

Geometry confidence 使用 BCE。Point/depth 只在 target confidence 大于零且 view 有效的位置计算。

## 11. Grouped action loss

Action head 预测 mean 与 log scale。对角 Gaussian NLL：

```python
inverse_variance = torch.exp(-2.0 * action_log_scale)
action_nll_values = (
    0.5 * (target_action - action_mean).square() * inverse_variance
    + action_log_scale
)
```

只在 `action_dim_mask & action_group_mask` 的维度求均值。额外约束：

```python
velocity_error = abs(
    (mean[:, 1:] - mean[:, :-1])
    - (target[:, 1:] - target[:, :-1])
)

contact_loss = binary_cross_entropy_with_logits(...)
scale_reg = masked_mean(action_log_scale.square())
```

Velocity 项比较视觉 horizon 相邻帧对应动作块的变化；contact 只监督离散 grip/contact proxy；scale
regularization 防止模型仅靠放大方差降低 NLL。

## 12. Loss 权重

代码返回 raw loss，权重来自物化 YAML：

```python
total = sum(raw[name] * float(weights[name]) for name in raw)
return {"total": total, **raw}
```

日志因此同时保留未加权各项，便于发现某一 head 退化。现有权重是训练计划的一部分，不是经过完整 5B
Pareto 消融得到的通用最优值。正式 run 之间比较时必须绑定同一配置 SHA。

## 13. WSD learning-rate schedule

学习率由 optimizer step 直接计算：

```python
if step < warmup_steps:
    return peak_lr * (step + 1) / warmup_steps
if step < decay_start:
    return peak_lr
progress = (step - decay_start) / (total_steps - decay_start)
return min_lr + (peak_lr - min_lr) * 0.5 * (1 + cos(pi * progress))
```

Warmup-stable-decay 的训练策略可参考
[MiniCPM](https://arxiv.org/abs/2404.06395)。当前实现的总步数、stable fraction 和 cosine 尾段由本项目
配置固定；函数无内部状态，恢复后相同步数得到相同学习率。

## 14. Checkpoint 写入事务

Checkpoint 目录只接受显式名称：

```text
step_00001000
```

保存先创建不可见临时目录：

```python
temporary = root / f".step_XXXXXXXX.incomplete.{uuid}"
```

各 rank 使用 PyTorch Distributed Checkpoint 写 model/optimizer shard，同时单独保存本 rank 的：

```python
{
    "python": random.getstate(),
    "numpy": np.random.get_state(),
    "torch_cpu": torch.get_rng_state(),
    "torch_cuda": torch.cuda.get_rng_state(),
}
```

Rank 0 完成以下发布序列：

```text
fsync DCP 与 RNG payload
  → 写 metadata.json
  → 计算每个文件 size/SHA，写 MANIFEST.json
  → 写 COMMITTED.json
  → 再次 fsync 文件与目录
  → os.replace(incomplete, step_XXXXXXXX)
  → fsync checkpoint root
```

任一 rank 文件系统失败都会通过 `all_gather_object` 传播，避免其他 rank 永久等待。DCP 实现来自 PyTorch
[Distributed Checkpoint](https://docs.pytorch.org/docs/main/distributed.checkpoint.html)。额外的 manifest、fsync 和
原子发布是本项目的持久化协议。

## 15. Checkpoint 验证与精确恢复

`verify()` 检查：

- 目录不是 symlink，名称严格匹配 `step_[0-9]{8}`；
- metadata、manifest、commit schema 与 step 一致；
- control-file SHA 与 canonical manifest SHA；
- 实际文件集与 manifest 完全一致；
- 每个 payload 的 size 和 SHA；
- 没有 symlink、非常规文件或不安全相对路径。

`load()` 还把 metadata 与 `ResumeExpectations` 比较：

```text
step
run_lineage
config_sha256
dataset_receipt_sha256
world_size
shard_degree
```

正式配置默认不允许 topology reshard。Model、optimizer 和每 rank RNG 全部恢复后，sampler 根据 optimizer step
重建下一个样本地址。代码从不搜索 `latest`。

## 16. Eval 证明什么

`eval.py` 只加载含 `COMMITTED.json` 的显式 checkpoint，并使用固定 validation sampler。它汇总：

- 全部训练 loss；
- token MAE/MSE；
- RGB MAE/MSE/RMSE/PSNR 与 target/prediction 拼图；
- depth、point、confidence、camera 指标；
- action error、contact accuracy、输出标准差和监督覆盖数。

输出标准差和覆盖数用于识别常数输出或“没有有效 target 却得到正常均值”的假结果。Eval report 绑定 checkpoint、
数据、配置、代码和 lineage。

这些指标是数值正确性和训练回退门禁，不等于机器人闭环成功率。跨模型结论需要相同数据协议的 rollout 或公开
benchmark，不能只比较两个 run 的训练 loss。

## 17. 正式训练的失败边界

运行时会停止而不是自动修复：

- CUDA/NCCL/torchrun 拓扑不符合配置；
- code、environment 或 dataset receipt 不一致；
- 模型参数量偏离 YAML 中冻结值；
- batch 算术不一致；
- loss、梯度或模型输出出现 NaN/Inf；
- checkpoint 缺 shard、哈希错误或 lineage 不符；
- 配置或 import 引入 V8、Qwen、Wan、A2 路径。

训练器不会自动换卡、改单机、降低 batch 或跳过损坏样本。集群调度层可以重新提交，但必须从已经验证的显式
checkpoint 恢复。
