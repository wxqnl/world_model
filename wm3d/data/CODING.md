# 数据实现导读

本目录把来源不同的机器人 episode 变成同一种训练样本。核心任务包括：定义可哈希契约、对齐异构动作和
传感器、压缩并随机读取 cache、组成 T24/K16 窗口，以及让多机采样可以由 optimizer step 精确重建。

## 数据 pipeline 全貌

从公开源数据到 `WM3D.forward()` 的 batch，完整路径如下：

```text
公开数据源 + immutable source lock
  └─ sources.py：扫描 episode，生成统一 episode plan
      ├─ action.py：动作统计、时间对齐、group/mask、aux token
      ├─ encoders：VGGT 视觉/几何 cache 与 task embedding bank
      └─ codec.py：token int8、RGB JPEG pack、逐帧 shard
                    │
                    ▼
       window parquet/index + dataset seal/receipt
                    │
                    ▼
       WindowDataset 按地址读取 T24 context + K16 future
                    │
                    ▼
       StepAddressedBatchSampler 决定 source/rank/sample
                    │
                    ▼
       model batch：context 条件 + future world/action target
```

| 阶段 | 主要代码 | 输入与输出 |
|---|---|---|
| 契约与来源 | `contracts.py`、`sources.py` | source lock → 统一 episode plan、稳定 ID 与 shape contract |
| 动作与传感器 | `action.py` | 原始控制流 → grouped action、dim/group mask、aux token |
| 离线表示 | `encoders`、task bank | RGB/文本 → 冻结 VGGT cache 与 task embedding |
| 封存与随机读 | `codec.py` | 逐帧数组/RGB → frame shard、JPEG pack、SHA receipt |
| 窗口组装 | `dataset.py` | window address → T/K 模型输入和显式监督 target |
| 分布式取样 | `sampler.py` | optimizer step + rank → 可重建的 source mix 与样本地址 |

样本在 Dataset 边界分成两类字段。context 侧包含三视角 token、历史 grouped action、task、aux 和可选低频
memory；future 侧包含 token、RGB、depth、point、camera 与动作 target。训练时不重新扫描原始数据，也不根据
文件内容猜测 shape。source lock、contract、index、cache 与 seal 的 SHA 必须完全一致，Dataset 才会返回样本。

## 1. 文件与调用顺序

| 文件 | 入口职责 |
|---|---|
| `contracts.py` | Dataset、embodiment、action group、aux modality、seal 和安全路径 |
| `sources.py` | 扫描公开数据并形成统一 episode plan |
| `action.py` | 动作统计、时间插值、group/mask 和辅助传感器 token |
| `codec.py` | D2048 token int8 编码、JPEG pack、frame shard 随机读 |
| `dataset.py` | 按 window index 读取连续帧，组成模型输入和 target |
| `sampler.py` | 确定 source mix、rank 样本地址和精确恢复 |
| `assets.py` | 核验 VGGT 等离线资产 receipt |

## 2. Dataset contract 先于数据

`DatasetContract` 固定以下内容：

- T、K、P、token D、RGB 分辨率和视觉频率；
- 最大动作组数、每组维度和 substep 数；
- task bank、embodiment 表与 source 表；
- shard/index/receipt 的相对路径和 SHA；
- 每个 source 的正式窗口范围。

训练不是根据文件内容猜 shape。Dataset、模型 YAML 和 checkpoint metadata 都绑定同一 contract digest；字段或
shape 漂移会在 preflight 阶段失败。

### Embodiment 与 action group

每种机器人通过 `EmbodimentSpec` 描述，例如：

```text
embodiment: bimanual_mobile
action groups:
  - left_arm: 7D continuous pose
  - left_gripper: 1D discrete
  - right_arm: 7D continuous pose
  - right_gripper: 1D discrete
  - base: 3D velocity
```

每组具有稳定 `group_id`、维度名称、控制模式和原始频率。模型看到统一上限 `[G=8,S=6,A=16]`，实际存在
部分由 `action_group_mask` 和 `action_dim_mask` 标明。

这种显式异构接口服务于跨机器人数据。它解决的问题与
[Open X-Embodiment](https://arxiv.org/abs/2310.08864) 相同，但 group taxonomy 和 padding contract 是本项目定义。

## 3. Robust action normalization

`robust_action_normalization()` 按动作维度统计：

```python
median = np.median(selected)
low, high = np.quantile(selected, [0.01, 0.99])
center[dimension] = median
scale[dimension] = max((high - low) / 2.0, 1.0e-6)
```

随后：

```python
normalized = (aligned - center) / scale
normalized = np.clip(normalized, -5.0, 5.0)
```

Median 和 1%–99% 范围比均值/标准差更不容易被少量异常遥操作值拉偏。每维少于 32 个有效样本时直接失败，
不产生不可靠统计。

该归一化用于数值稳定，不会抹掉单位差异的语义；group ID、embodiment ID 和维度 mask 仍告诉模型输入属于哪种
控制接口。

## 4. 连续与离散动作如何插值

`align_grouped_actions()` 为每个 5Hz 视觉时间创建 6 个 substep：

```python
substep_offsets = np.arange(action_substeps) / (
    feature_fps * action_substeps
)
query = (visual_timestamps[:, None] + substep_offsets[None]).reshape(-1)
```

在 5Hz、S=6 时，query 间隔为 1/30 秒。连续控制使用线性插值：

```python
output[:, dimension] = np.interp(
    query, series.timestamps, series.values[:, dimension]
)
```

离散 grip/control 使用最近邻：

```python
choose_right = abs(ts[right] - query) < abs(ts[left] - query)
indices = np.where(choose_right, right, left)
output[:] = series.values[indices]
```

线性插值适合连续位置/速度，最近邻避免把开合状态插成不存在的中间值。Query 超出原始时间范围或邻接样本无效时，
对应 `dim_mask=False`。

最终写入：

```python
values[:, slot, :, :dimension] = normalized
dim_mask[:, slot, :, :dimension] = valid
group_ids[slot] = group.group_id
group_mask[slot] = True
```

离散组最后一维的正负号还会形成 `contact` 标签：

```python
contact[:, slot] = (normalized[..., -1] > 0.0).astype(np.float32)
```

它更准确的名称是 gripper/contact proxy，不等于真实力传感接触。

## 5. Auxiliary sensor token

Force、tactile、proprioception 等可选传感器被编码到固定 D256 token：

```text
[modality one-hot | normalized values | per-dimension validity bits | padding]
```

真实代码：

```python
tokens[:, slot, modality.type_id] = 1.0
start = max_aux_type_id
tokens[:, slot, start:start + dimension] = normalized
tokens[:, slot, start + dimension:start + 2 * dimension] = valid.astype(np.float32)
mask[:, slot] = valid.any(axis=-1)
```

只在视觉时间戳上取样，不把视觉帧之后的高频值聚合回来，因此 auxiliary token 不能暴露未来信息。

## 6. Token 压缩

VGGT D2048 token 占存储的主要部分。`quantize_per_vector()` 对每个 2048D 向量单独做对称 int8 量化：

```python
maximum = value.float().abs().amax(dim=-1, keepdim=True)
scale = (maximum / 127.0).clamp_min(torch.finfo(torch.float16).tiny)
quantized = torch.round(value.float() / scale).clamp(-127, 127).to(torch.int8)
```

读取时：

```python
restored = (quantized.float() * scale.float()).to(torch.bfloat16)
```

每个向量存 2048 个 int8 和一个 FP16 scale。它比整 shard 共用 scale 更能保留不同 token 的动态范围。
这是有损存储；seal 保证的是量化后文件内容不变，不表示 token 与原始 FP16 逐位一致。

## 7. RGB JPEG pack

每帧各视角被编码成独立 JPEG record，再顺序追加到 pack：

```python
encoded = encode_jpeg(image.contiguous(), quality=92)
os.write(fd, payload)
offsets.append(current_offset)
lengths.append(len(payload))
```

Window index 保存每个 record 的 offset/length。读取器使用 `os.pread()`：

```python
payload = os.pread(self._fd, length, offset)
image = decode_jpeg(torch.frombuffer(bytearray(payload), dtype=torch.uint8))
```

随机窗口只解码需要直接 RGB 监督的 future frame，不解码整个 episode。Writer 使用 `O_EXCL`、`O_NOFOLLOW`
和 `fsync`，避免覆盖已有 pack、跟随符号链接或发布未落盘文件。

## 8. WindowDataset 怎样组成样本

Window index 指向 immutable frame shard 的连续区间。一个样本读取 `T+K=40` 个视觉 frame 和对应动作区间。

### Context 输入

```python
result = {
    "world_tokens": view_tokens[:T],
    "view_mask": feature["view_mask"][:T].bool(),
    "task_embedding": self.tasks[task_id],
    "context_action_values": action["action_values"][:T],
    "context_action_dim_mask": action["action_dim_mask"][:T].bool(),
}
```

这里 `world_tokens` 实际保留 `[T,V,P,D]` 的逐视角 token，由模型在线 fuser 学习融合。

### Future token target

Future target 使用 geometry confidence 融合三视角：

```python
confidence = feature["geometry_confidence"].float().clamp_min(0.0)
confidence = confidence * view_mask[..., None]
weights = confidence[..., None]
world_tokens = (view_tokens.float() * weights).sum(dim=1) / (
    confidence.sum(dim=1).clamp_min(1.0e-6)[..., None]
)

result["target_tokens"] = world_tokens[T:T + K].to(torch.bfloat16)
```

如果任一 future patch 的 confidence 总和为零，Dataset 报 `DataIntegrityError`，不提供无几何证据的监督。

### Future action 的两个用途

```python
result["future_factual_action_values"] = action_values[T:T + K]
result["target_action_values"] = action_values[T:T + K]
```

它们数值相同但进入不同代码路径：

- `future_factual_action_values` 只条件化模型的 future state query；
- `target_action_values` 只在 loss 中监督 action head；
- action trunk 的 future 输入始终是 learned query。

防泄漏由模型结构和单测保证，不依赖字段名约定。

### RGB target

Dataset 只按 `rgb_decode_indices` 读取默认四帧 `[3,7,11,15]` 的 JPEG records。Depth、point、camera 和
confidence 则保留全部 K16 future frame。

## 9. 低频 memory

Dataset 从同一 episode、当前窗口之前采样低频 frame summary。默认每 25 个 5Hz frame 取一个，即约每 5 秒
一个；最多 12 个 slot，可覆盖约 60 秒历史。

Memory 不允许跨 episode，也不读取当前 window 未来。没有足够历史的位置以零填充并由 `memory_mask` 屏蔽。

## 10. Source schedule

多来源训练先把整数权重展开。例如：

```text
source_order   = [A, B, C]
source_weights = {A: 3, B: 1, C: 1}
expanded       = [A, A, A, B, C]
```

每个五步 cycle 使用确定性 Fisher-Yates 打乱，但数量严格保持 3:1:1：

```python
cycle, position = divmod(optimizer_step, cycle_length)
cycle_sources = self._cycle_sources(cycle)
source_name = cycle_sources[position]
```

一个 optimizer step 的所有 rank 和 accumulation micro-batch 使用同一个 source，避免在一个 batch 中混合不同
shape/contract 的来源。

## 11. 样本地址怎样精确恢复

每个 source 内使用 affine permutation：

```python
epoch, position = divmod(ordinal, source_length)
multiplier = coprime_multiplier(source_length, epoch_seed)
offset = splitmix64(epoch_seed) % source_length
index = (multiplier * position + offset) % source_length
```

乘数与长度互质，因此一个 pass 内不会重复。每个 rank 的 ordinal 由以下量直接计算：

```python
step_base = source_occurrence * global_batch
micro_base = step_base + micro_step * global_micro_batch
rank_base = micro_base + rank * micro_batch_size
```

恢复时只要 optimizer step、seed、world size、rank、batch 参数和 sealed source span 相同，就能重建下一批样本。
Sampler 没有需要 pickle 的隐式 iterator 状态。

正式训练默认禁止改变 world size/shard topology 后继续同一 lineage，因为样本地址和优化状态都与拓扑绑定。

## 12. Fail-closed 条件

数据层会拒绝：

- 非有限 RGB、action、token、scale 或 target；
- 时间戳不严格递增；
- 一帧三路视角全缺失；
- future token 没有任何几何 confidence；
- action group、维度、频率或 normalization 与 embodiment contract 不符；
- shard/index 路径逃出 dataset root、符号链接或 seal 外文件；
- source 窗口数小于一个 global batch，无法组成无放回 step。

这些检查会牺牲部分“尽量读下去”的容错性，但能防止数千卡训练在错误数据上继续消耗。
