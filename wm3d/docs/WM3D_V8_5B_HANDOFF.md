# WM3D V8 5B 训练交接

本文件对应 GitHub `v8` 分支。5B 与当前 1B 长训使用同一份 V8 模型语义，不是独立维护的旧
dual-path/P256 变体。

## 冻结合同

- 唯一模型配置：`configs/model/native_5b_v8_action_owned_transport.yaml`
- 参数量：`5,087,822,644`
- encoder：`configs/encoder/vggt_native_p144.yaml`
- objective：`configs/objective/stage0_v8_action_owned_transport.yaml`
- runtime：先用 `configs/runtime/h200_64_fsdp2_canary1k.yaml`
- 时空合同：`T=24`、P144、`K=16`，RGB 监督全部 K16

future physical action 经过 source normalization 后，在 factual StateStream 的 block 0 之前按
horizon 因果注入，并由 group-preserving conditioner 持续作用于 P144 future state。
policy/action-free trunk 不读取 future candidate。P144 factual future state 是唯一运动所有者，
直接预测 renderer 实际使用的 backward flow；最后观测 RGB 只能通过该 flow 被传输到未来帧。
motion head 只作辅助监督，不门控 flow，也不存在 unwarped copy 或 full-frequency redraw。
高频 refiner 只补充有界高通细节。训练期 RAFT 仅产生同一 backward-flow 字段的监督 target，
不进入模型、optimizer 或 serving。absolute future P256、P256 AR/teacher forcing、独立
appearance lane 和旧 renderer-only action 通路均禁用。

## 新鲜启动

不要使用旧 `v8` checkout 中已经生成的 site/runtime，也不要从任何旧 5B checkpoint 初始化。
拉取后重新生成 canary site：

```bash
git clone --branch v8 --single-branch https://github.com/wxqnl/world_model.git
cd world_model/wm3d

SITE=/data/wm3d/control/5b_v8_canary1k.env
./run_wm3d.sh 5b init canary1k "$SITE" direct_raw
# 编辑数据路径、模型 snapshot、许可、8 节点地址与 rendezvous
./run_wm3d.sh 5b env "$SITE"
./run_wm3d.sh 5b data-template "$SITE"
./run_wm3d.sh 5b doctor "$SITE"
./run_wm3d.sh 5b task-bank "$SITE"
./run_wm3d.sh 5b cache-plan "$SITE"
./run_wm3d.sh 5b streaming-prepare "$SITE"
./run_wm3d.sh 5b runtime "$SITE"
```

`doctor` 与 `runtime` 会执行 5B V8 语义门禁。只要 model/encoder/objective 仍指向旧
V7-context/P256 路线、参数量不匹配、factual action 顺序不对或 appearance teacher ratio
非零，就会在大作业前失败。

### 复用已经下载的 AgiBotWorld2026

如果原始数据已经由 Hugging Face 下载到本地，不要重新搬运或重新下载。令
`RAW_ROOT` 的下一层正好是 `agibot_world_2026/`，其中保留官方三个目录：
`ImitationLearning/`、`RichInteraction/`、`ReinforcementLearning/`。先做快速只读检查：

```bash
./run_wm3d.sh agibot-existing-check \
  --snapshot-root "$RAW_ROOT/agibot_world_2026"
```

该命令只枚举 archive，并对每个前缀抽查一个 archive 的 LeRobot `meta/info.json`、parquet 与
视频/图像结构；不解压、不改写、不哈希整个多 TB 数据。之后仍需执行 `lock` 和 `download`：
把 site 的 `RAW_ROOT` 指向上述父目录，`download` 会按冻结 revision/file list 校验已有文件并只补
真正缺失的文件，然后生成后续 archive materialization 所需的 receipt。不要绕过 receipt，也不要
把已经解压但来源不明的目录直接写进正式 data profile。

当前 1B 训练默认 `INCLUDE_AGIBOT_2026=NO`，不代表 5B 不支持它；5B site 默认是 `YES`。
如同事明确不想使用该数据，必须在生成 data template 之前设为 `NO`，不能只把目录留空。

64 卡启动方式见 `docs/WM3D_5B_SCALING.md`。先完成同拓扑 1K canary，至少覆盖真实前向、
反向、梯度所有权、编号 checkpoint、独立进程 exact resume 和固定评测；通过后再用新 site
从 step 0 启动正式训练，不能把 canary checkpoint 当正式初始化。

## 启动前核对

- runtime 中模型名为 `native_5b_v8_action_owned_rgb_transport`
- 封存参数量为 `5,087,822,644`
- appearance teacher start/end ratio 都为 0
- encoder 没有 P256 appearance feature
- objective 的 `rgb_flow_teacher` 为 `0.20`，且 disocclusion loss 为 0
- renderer 使用未被 motion gate 衰减的 factual backward flow
- future action 对 policy/action-free 输出逐元素无影响
- factual conditioner、RGB transport、action head 和 policy 都有有限非零梯度
- 所有 rank 使用同一份 runtime/data seal/normalization 和训练-serving action/state calibration

这里的 1K 只验证 5B 实现、分布式状态和学习方向；真实 VLA 能力仍要由独立 action regression
与多类闭环任务评测证明，不能只靠 Stage0 token/RGB 指标下结论。
