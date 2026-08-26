# WM3D 分组机器人归一化合同

## 为什么必须有这一层

WM3D 同时训练不同来源、不同 embodiment 与不同 action group。米、弧度、关节
位置、底盘速度等连续量的数值尺度不同；若直接把物理量送入共享 encoder 和
policy loss，大尺度通道会压制小尺度通道。

归一化只是一层训练坐标变换，不改变机器人 ABI：

- episode cache 永远保存 adapter 审计后的物理单位；
- dynamics target 的 coarse effect 先在物理单位按
  `sum / SO(3) / last / time_weighted_mean` 完成组合；policy head 先从训练
  坐标反变换回物理单位，再用同一组 operator 组合；
- history、factual dynamics conditioner、current-state 与连续 fine target 在
  runtime 视图中归一化；
- policy head 的连续输出在 normalized coordinate 中受监督，随后立即反变换为
  物理量；serving 和 coarse composition 只读取物理量 `policy_action`；
- policy query 显式接收同一 action/state transform 的 calibration descriptor。这样同一
  embodiment 的多个 source 不需要靠视觉去猜归一化键，serving 也不能只给 action stats
  而漏掉 current-state stats；
- gripper、`binary_contact` 与已审计为 `[0,1]` 的离散 `controller_mode`
  只使用 identity transform，不做 z-score；它们由统一 head 的 binary logit
  路径训练和解码。

## 统计键与防泄漏

统计键为：

```text
(source, source_id, embodiment, embodiment_id,
 group, group_id, kind, lane, dimension, semantic, semantic_id)
```

它不依赖 source 名称分支，来源和机器人布局均由 data profile 驱动。每个真实
维度必须恰有一行，缺行、重复、semantic 漂移或 group 漂移都会 fail-closed；
padding 维不进入统计。`lane` 只允许 `fine_command`、`coarse_effect` 与
`current_state`：每个 source/group 只封存其数据 profile 和 robot payload
实际声明的 action lane。fine-command 不派生 coarse 监督，coarse-effect 也不伪造
fine stats；任一有效 mask 落到非匹配 lane 都会阻断。

`time_weighted_mean` 采用真实 query 时间做 zero-order hold；无效 padding query
不是物理时钟事件，不能截断最后一个真实 command 到 world interval 末端的持续时间。

统计仅遍历 sealed window index 中的 `train` 窗口。这样既不会读取 val/test，
也能确保 coarse effect 使用和训练完全相同的窗口组合尺度。artifact 必须绑定：

- data profile SHA；
- model profile SHA；
- sealed window index SHA；
- artifact 自身 SHA（由 materialized runtime data closure 绑定）；
- rows 的 canonical SHA。

因此 1B/5B 共用一套实现，但由于 T/K/采样窗口不同，各自从对应 sealed window
index 生成统计 artifact；昂贵的 episode cache 仍完全共享，不需要重做 VGGT
或 RGB/depth/point cache。

## 生成顺序

完成 episode cache seal 和目标 model profile 的 window seal 后执行：

```bash
./run_wm3d.sh normalization \
  --data-profile /abs/data_profile.yaml \
  --model-profile configs/model/native_1b.yaml \
  --window-index /abs/window_index.jsonl \
  --window-index-sha256 <WINDOW_INDEX_SHA256> \
  --cache-root /abs/cache_root \
  --output /abs/grouped_normalization.json
```

构建器逐文件校验承载真实 action/state 数值的 robot shard SHA，采用 float64、
mask-aware 的 population mean/std；低方差连续维使用相对量级 floor，避免除以
近零数。feature payload 已由 episode/window seal 与 window index SHA 完整绑定；
构建统计只读取很小的 `source_observation_row` 边界列，不为每个 1B/5B profile
重复哈希 TB 级视觉 payload。训练和 eval 在实际采样 feature shard 首次打开时仍按
index SHA 校验完整 payload。输出采用 no-clobber 发布。
robot 解码结果与 feature 边界列只保留小型 episode LRU，不会随完整语料的
episode 数量线性占用内存。

封存 runtime 时必须额外传入：

```bash
./run_wm3d.sh runtime \
  ... \
  --grouped-normalization /abs/grouped_normalization.json
```

训练和离线 eval 都由同一 materialized runtime 加载同一个 SHA-bound artifact；
没有 zero/fallback、插值或按数据集硬编码。

部署 adapter 必须显式选择与目标数据/控制器对应的封存 normalization profile，并把同一组
action offset/scale 用于 history 归一化、policy query calibration 与输出反归一化，同时把
state offset/scale 用于 current state 归一化与同一 calibration。未知 profile 必须 fail-closed，
不能用单位矩阵或靠模型猜 source。这个约束对所有下游成立，不是 LIBERO 专用分支。

## action velocity 决策

`action_velocity` 默认且强制为 `0`。原实现对相邻 query 的 raw difference 求均值，
它随 query cadence 改变；更重要的是，Cartesian delta、joint absolute、base
velocity 之间不存在统一的物理平滑量。简单除以 `dt` 也不能修复 delta command
的语义差异。因此非零配置当前 fail-closed，直到将来为具体 semantic 明确定义并
验证 cadence-invariant 的正则合同。
