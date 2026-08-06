# 模型实现导读

`wm3d.py` 定义完整的在线模型。它没有调用 VGGT、语言模型或视频生成器；输入已经是封存后的视觉 token、
task embedding、grouped action 和可选传感器 token。

本文按 `WM3D.forward()` 的执行顺序说明代码。示例形状来自 `configs/train/5b_h200.yaml`：T24、P144、
K16、三视角、state hidden 2560、action hidden 2048。

## 先看完整模型

一次 forward 中，观测、世界状态和动作按下面的路径流动：

```text
三视角 context token [B,T,V,P,2048]
  └─ MultiViewTokenFuser ───────────────────────► context state [B,T,P,2560]
                                                   + K 个 learned future state query
task embedding ───────────────────────────────────► state / action 两条主干
context-only aux / low-frequency memory ─────────► state trunk
future factual action ───────────────────────────► 只条件化 future state

context grouped action + K 个 learned action query
  └─ GroupedActionTokenizer ─────────────────────► action latent [B,T+K,G,2048]

        32 层 FactorizedStateBlock              24 层 ActionBlock
                  │                                      │
                  └──── 10 个 StateActionBridge ─────────┘
                         分层双向交换信息
                  │                                      │
                  ▼                                      ▼
     token / RGB / depth / point / camera       grouped action distribution
```

模型由以下部分组成：

| 部分 | 使用的表示 | 主要职责 |
|---|---|---|
| 多视角入口 | 冻结 VGGT cache，三视角 2048D patch token | 在同一空间位置上融合可用视角，生成 state trunk 的 context |
| state trunk | `[B,T+K,P,2560]` | 建模时间与空间动力学，持有未来世界状态 |
| action trunk | `[B,T+K,G,2048]` | 建模 embodiment-aware grouped action，输出连续动作分布与 contact |
| state-action bridge | state patch summary 与 action group latent | 在指定深度双向交换世界与动作信息，同时执行防泄漏 mask |
| 条件输入 | task、context-only aux、低频 memory、future factual action | 给两条主干提供任务、传感器、长时信息和动作条件 |
| 显式输出头 | future state / future action latent | 直接预测 VGGT token、RGB、depth、point、camera、confidence 和动作 |

所有权按输出划分：state trunk 负责未来世界，action trunk 负责策略动作。VGGT 只提供离线观测表示和
几何监督；在线模型中没有 VGGT、语言模型、VLA action head 或视频生成器。真实未来动作只进入 future state，
不会写入 context，也不会作为 action trunk 的答案输入。

`WM3D.forward()` 依次执行六步：

1. `MultiViewTokenFuser` 把三视角 context cache 融合成单一 patch lattice；
2. 在 context 后拼接 K 个未来 state query，并加入时间、空间、task 与合法的条件输入；
3. `GroupedActionTokenizer` 把历史动作编码成 group latent，并为未来动作放置 learned query；
4. state/action block 按层交错执行，十个 bridge 在固定深度交换信息；
5. 取最后 K 帧 latent，分别送入世界、几何、RGB 和动作输出头；
6. training loss 对显式 target 计算监督，梯度回到两条主干与 bridge。

当前 5B 配置约 49.57 亿参数，其中 state trunk 约占 65.6%，action trunk 约占 24.1%，bridge 约占
8.6%。`WM3D` 类本身不绑定 5B；其他规模通过同一组配置字段改变深度和宽度。

## 1. 配置与基础层

`WM3DConfig.validate()` 在构造模型前检查几项结构约束：

```python
grid = isqrt(self.P)
if grid * grid != self.P:
    raise ValueError(f"P must be a square token grid, got {self.P}")

if hidden % heads:
    raise ValueError(...)

if any(i < 0 or i >= self.K for i in self.rgb_decode_indices):
    raise ValueError("rgb_decode_indices must refer to future steps")
```

P 必须是平方数，因为 RGB decoder 会把 patch 序列还原成二维网格。`rgb_size / sqrt(P)` 必须是 2 的
幂，才能用多级 2× 上采样精确到达目标分辨率。

### RMSNorm

```python
def forward(self, x):
    x_fp32 = x.float()
    x_norm = x_fp32 * torch.rsqrt(
        x_fp32.square().mean(-1, keepdim=True) + self.eps
    )
    return x_norm.to(dtype=x.dtype) * self.weight
```

归一化统计在 FP32 中计算，再转回 BF16。这样做减少 BF16 平方和倒平方根的误差。结构来自
[RMSNorm](https://arxiv.org/abs/1910.07467)；FP32 统计是训练稳定性的实现选择。

### SwiGLU

```python
inner = _round_multiple(dim * mult)
self.gate_up = nn.Linear(dim, inner * 2, bias=False)
self.down = nn.Linear(inner, dim, bias=False)

gate, value = self.gate_up(x).chunk(2, dim=-1)
return self.down(F.silu(gate) * value)
```

中间维度向 256 的倍数取整，方便大矩阵使用规则 shape。门控 FFN 参考
[GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)；具体倍率 2.5 和 8/3 是本配置的
参数预算，不是论文结论。

### Attention

`SelfAttention` 和 `CrossAttention` 都调用 `torch.nn.functional.scaled_dot_product_attention`。代码只依赖
PyTorch SDPA 语义，实际使用哪一种 CUDA kernel 由 PyTorch 和输入 shape 决定。

布尔 `allowed_mask` 中 `True` 表示对应 key 可以被读取。该约定用于缺失视角和 grouped action；state
时间 attention 则使用 `is_causal=True`。

## 2. MultiViewTokenFuser

调用位置：

```python
context_state = self.view_fuser(world_tokens, view_mask)
context_state = self.fused_input_norm(context_state)
```

输入为 `[B,24,3,144,2048]`，输出为 `[B,24,144,2560]`。

核心 reshape：

```python
x = self.in_proj(tokens) + self.view_embed
x = x.permute(0, 1, 3, 2, 4).reshape(
    batch * frames * patches, views, -1
)
```

它把输入变成 `[B*T*P,V,1024]`。每个 attention 序列只含同一时间、同一 patch 的三个相机 token，
因此这个模块不混合时间和空间位置。

缺失相机处理：

```python
valid = view_mask[:, :, None, :].expand(batch, frames, patches, views)
valid = valid.reshape(batch * frames * patches, views)
allowed = valid[:, None, None, :]

x = x + self.attn(self.attn_norm(x), allowed_mask=allowed)
logits = self.gate(x).squeeze(-1).masked_fill(~valid, float("-inf"))
fused = (x * logits.softmax(dim=-1)[..., None]).sum(dim=1)
```

无效视角不能作为 attention key，最终 gate 权重也严格为零。所有相机都缺失的帧直接报错。融合不是固定
平均：每个有效视角先读取其他有效视角，再由 learned gate 决定该时间和 patch 的加权比例。

当前实现不读取 camera pose 或逐 patch geometry confidence。它依赖离线 VGGT token 已经包含跨视角几何
关系。是否应改成 camera-aware 或 confidence-aware fusion 仍需消融；现有验证只证明 mask、shape、梯度和
端到端数值正确。

## 3. Context state 与 future query

```python
future_state = self.future_state_queries.expand(batch, -1, -1, -1)
state = torch.cat((context_state, future_state), dim=1)
state = state + self.state_time + self.state_space
state = state + self.task_state(task_embedding)[:, None, None]
```

形状变化：

```text
context_state       [B,24,144,2560]
future_state_queries[B,16,144,2560]
state               [B,40,144,2560]
```

Future query 是可学习参数，不包含未来图像。`state_time` 区分 40 个时间位置，`state_space` 区分 144 个
patch 位置，task embedding 对所有时间和 patch 广播。

## 4. FactorizedStateBlock

32 个 state block 是世界模型的主要容量，约占总参数的 65.6%。每层代码只有三步：

```python
batch, frames, patches, dim = x.shape

spatial = x.reshape(batch * frames, patches, dim)
spatial = spatial + self.spatial(self.spatial_norm(spatial))
x = spatial.view(batch, frames, patches, dim)

temporal = x.transpose(1, 2).reshape(batch * patches, frames, dim)
temporal = temporal + self.temporal(
    self.temporal_norm(temporal), is_causal=True
)
x = temporal.view(batch, patches, frames, dim).transpose(1, 2)

return x + self.ff(self.ff_norm(x))
```

### 空间 attention

`[B,40,144,2560]` 被看成 `[B*40,144,2560]`。每一帧内部的 144 个 patch 可以全局交换信息，不跨帧。

### 因果时间 attention

随后转成 `[B*144,40,2560]`。同一个 patch 编号沿时间读取当前和过去帧，不能读取未来帧。因此 future
query 可以使用 context 和更早的 future latent，context 不能被 future query 反向改写。

时空分解避免在 5760 个 state 位置上做一次全局 attention。思路与
[TimeSformer](https://proceedings.mlr.press/v139/bertasius21a.html) 的 divided space-time attention 接近，
但这里加入 causal 时间约束并用于未来预测。

限制也很明确：同一 patch 编号跨时间并不总对应同一物体。相机运动或物体跨格移动主要依靠空间层和 VGGT
特征补偿，代码没有显式光流或 token warp。

## 5. GroupedActionTokenizer

动作张量后缀固定为 `[G,S,A]`：最多 8 组、每视觉帧 6 个 substep、每组最多 16 维。不同机器人使用
`group_mask` 和 `dim_mask` 表示真实存在的组与维度。

### Context action token

```python
masked = values * dim_mask.to(dtype=values.dtype)
pair = torch.cat(
    (masked.flatten(-2), dim_mask.to(values.dtype).flatten(-2)), dim=-1
)
context = self.value_proj(pair)
```

输入同时包含数值和维度 mask。单独把 padding 置零不够，因为真实动作也可能等于零；显式 mask 让模型
区分“有效的零动作”和“不存在的维度”。

### Future policy query

```python
future = self.future_queries.expand(batch, -1, -1, -1)
tokens = torch.cat((context, future), dim=1)
tokens = tokens + self.group_embed(group_ids)[:, None]
tokens = tokens + self.embodiment_embed(embodiment_ids)[:, None, None]
```

Action trunk 的未来输入是 learned query，不是未来动作标签。Group embedding 表示 arm、gripper、base
等控制组，embodiment embedding 表示机器人本体。

这种接口用于处理跨机器人异构动作；动机与
[Open X-Embodiment](https://arxiv.org/abs/2310.08864) 和
[Octo](https://arxiv.org/abs/2405.12213) 面对的 action-space 问题一致，但当前 grouped tokenizer、固定上限
和 mask 规则是本项目实现。

## 6. ActionBlock 与因果 mask

Action trunk 有 24 层，约占总参数的 24.1%。每层把 `[B,40,G,2048]` 展平为 `[B,40*G,2048]`：

```python
flat = x.reshape(batch, frames * groups, dim)
flat = flat + self.attn(self.attn_norm(flat), allowed_mask=allowed_mask)
flat = flat + self.ff(self.ff_norm(flat))
x = flat.view(batch, frames, groups, dim)
return x * group_mask[:, None, :, None].to(dtype=x.dtype)
```

Mask 根据 frame ID 构造：

```python
time_ids = torch.arange(frames).repeat_interleave(groups)
causal = time_ids[None, :] <= time_ids[:, None]
allowed = causal[None, None] & valid[:, None, None, :]
```

同一帧的动作组可以互相读取，未来帧不能作为过去帧的 key；不存在的组不能作为 key，并在每层末尾重新归零。

## 7. 真实未来动作怎样只条件化世界

训练 world dynamics 时已知实际执行的未来动作。代码将它投影到 state hidden：

```python
group_features = self.factual_proj(
    self._flat_pair(future_values, future_dim_mask)
)
weights = group_mask[:, None, :, None].to(dtype=group_features.dtype)
factual = (group_features * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1.0)
```

随后只加到 future state：

```python
state = torch.cat(
    (state[:, :cfg.T], state[:, cfg.T:] + factual[:, :, None, :]),
    dim=1,
)
```

这实现的是动作条件化未来模型：

```text
future world = f(context world, planned/factual future action, task)
```

动作条件化预测可追溯到机器人交互中的视频预测工作，例如
[Unsupervised Learning for Physical Interaction through Video Prediction](https://arxiv.org/abs/1605.07157)。
WM3D 的差别是同时预测显式 token、RGB 和几何。

## 8. StateActionBridge 与防泄漏约束

State/action 两条主干在 10 个 state 层位交换信息。Bridge 先把 patch state 求均值：

```python
state_summary = state.mean(dim=2)             # [B,40,2560]
action_flat = action.reshape(batch, frames * groups, -1)
```

Action 只能读取前 T 帧：

```python
action_update = self.action_reads_state(
    self.action_norm(action_flat),
    self.state_norm(state_summary[:, :self.T]),
)
```

State 可以读取 action latent：

```python
state_update = self.state_reads_action(
    self.state_norm(state_summary),
    self.action_norm(action.reshape(batch, frames * groups, -1)),
)
state = state + state_update[:, :, None, :] / patches**0.5
```

这条不对称读取规则很重要。Future factual action 已进入 future state；如果 action branch 可以读取 future
state，它会间接看到自己的标签。代码只把 `state_summary[:,:T]` 写回 action，从结构上切断该路径。

`tests/test_model.py::test_future_factual_actions_cannot_leak_into_policy` 做两项检查：修改未来真实动作后，
`pred_tokens` 必须变化而 `action_mean` 必须逐位相同；action 输出对未来真实动作的梯度必须为零。

Bridge 使用 cross-attention 的异构 latent 交互方式与
[Perceiver](https://proceedings.mlr.press/v139/jaegle21a.html) 有结构上的共同点。10 个层位及 patch-summary 广播
公式属于工程设计，目前没有 bridge 数量消融。

## 9. Auxiliary token 与低频 memory

Auxiliary token 只加到 context state：

```python
aux = self.aux_proj(aux_tokens)
weights = aux_mask[..., None].to(dtype=aux.dtype)
aux = (aux * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1.0)
state = torch.cat(
    (state[:, :cfg.T] + aux[:, :, None, :], state[:, cfg.T:]), dim=1
)
```

未来 force、tactile 或 proprioception 不会进入模型。

Memory 通过逐帧 state summary 读取过去低频 token：

```python
summary = state.mean(dim=2)
update = self.memory_cross(
    self.memory_norm(summary), self.memory_norm(memory), allowed_mask=allowed
)
return state + update[:, :, None, :] / state.shape[2] ** 0.5
```

它每 4 个 state block 注入一次。Memory 保留长时任务信息，但只有帧级 summary，没有完整空间网格。

## 10. 主干如何交错执行

32 个 state block 和 24 个 action block 不是先后运行。构造函数按比例计算每个 state 层后应执行几个 action
层：

```python
self._action_steps = [
    (action_layers * (i + 1) // state_layers)
    - (action_layers * i // state_layers)
    for i in range(state_layers)
]
```

主循环：

```python
for state_i, state_block in enumerate(self.state_blocks):
    state = self._run(state_block, state, enabled=checkpointing)
    for _ in range(self._action_steps[state_i]):
        action = self._run(self.action_blocks[action_i], ...)
        action_i += 1
    if state_i in self._bridge_by_state_layer:
        state, action = self._run(self.bridges[bridge_i], ...)
    if (state_i + 1) % memory_every == 0:
        state = self._add_memory(state, memory_tokens, memory_mask)
```

`_run()` 使用 non-reentrant activation checkpoint。该选择节省激活显存，以反向时重算 block 为代价。

## 11. 显式输出头

### Future token

```python
future_state = state[:, cfg.T:]
pred_tokens = self.state_out(future_state)
```

全部 K16 帧、全部 P144 patch 都预测 2048D VGGT-space token。

### Geometry head

```python
x = self.in_proj(future_state)[:, :, None] + self.view_embed
camera = self.camera(future_state.mean(dim=2)).view(B, K, V, 9)

depth = F.softplus(self.depth(x).squeeze(-1))
point = self.point(x)
confidence = torch.sigmoid(self.confidence(x).squeeze(-1))
```

Depth 保证为正，confidence 位于 `[0,1]`。Point 是每 patch 的 XYZ。Camera 由帧级 state summary 预测 9D
VGGT pose encoding；代码没有显式 SE(3) 正交约束，这是一项已知限制。

### RGB decoder

选定 future frame 被还原为 12×12 feature map，经过 residual convolution 和五级 2× nearest-neighbor
upsampling，最终产生三视角 384×384 RGB：

```python
selected = future_state.index_select(1, index_tensor)
x = self.in_proj(selected)[:, :, None] + self.view_embed
x = x.permute(...).reshape(B * frames * views, channels, 12, 12)
rgb = torch.sigmoid(self.out(self.decoder(x)))
```

默认只直接解码 future index `[3,7,11,15]`，用于限制激活显存。所有 future frame 都有 token/geometry
监督，但只有四帧有直接 RGB loss；若最终 RGB 仍模糊，这个监督密度是优先消融项。

### Action head

```python
action_mean = self.mean(x).view(B, K, G, S, A)
action_log_scale = self.log_scale(x).clamp(-7.0, 3.0).view(...)
contact_logit = self.contact(x)
```

当前连续动作使用对角 Gaussian，简单且可表达异方差，但不能充分表示多峰动作。需要更强多模态策略时，可在
不改变 state ownership 的前提下替换为 mixture、flow 或 diffusion head。

## 12. 参数归属与当前证据

5B 配置精确参数量为 4,956,589,929：

| 组件 | 参数 | 比例 |
|---|---:|---:|
| state trunk | 3,250,831,360 | 65.59% |
| action trunk | 1,195,474,944 | 24.12% |
| 10 个 bridge | 424,719,360 | 8.57% |
| multiview fuser | 16,783,360 | 0.34% |
| RGB、geometry、action head | 13,725,033 | 0.28% |

参数主要在 world/action dynamics，而不在 encoder 或输出头。单测冻结了精确参数预算，并覆盖完整
forward/backward、native losses、activation checkpoint 和防泄漏。两卡 smoke 证明代码可执行，不代表当前层数、
bridge 位置、K16 或输出质量已经达到最优。
