# WM3D V10：WSA-Large foundation 路线

## 结论与边界

V10 是一条独立的 foundation 实验路线，不是对当前 V8 长训目录的原地修改。它以官方
WSA-Large 为模型基座，保留 Wan2.2 视频专家、ActionDiT、Future3D 与 Mixture-of-Transformers
联合训练主干，再接入 WM3D 已封存的数据、机器人物理语义和原生 3D 监督。

WSA-Large 的基座是 Wan2.2-TI2V-5B，不使用 Qwen3-VL。Qwen3-VL 属于 WSA-Base，V10
因此不加载 Qwen、不创建 Qwen LoRA，也不要求 Qwen image processor。任务文本沿用
WSA-Large 的 UMT5 上下文，并预先写入只读 context store，训练时不常驻文本编码器。

V10 在独立分支中增加了受限的 native-3D causal expert。它不替换 WSA 的视频、action 或
Future3D 主链，而是在同一组 Future3D token 上显式接入当前原生 3D 状态，并解码 K8 原生 3D
轨迹。替代正式训练前仍需做短程受控训练，同时检查视频、物理 action、K8 轨迹和最终 Future3D；
不能直接套用 WSA 官方 benchmark 数字。

## 当前训练优先级

当前先训练基础 WM 与可执行 action：从观测、任务、状态和历史，学会生成未来视频、
原生 3D 状态和动作。保留 WSA 现有 video/action/3D 联合目标；不删除真实动作标签，
不把 action loss 关闭，也不为候选动作控制新增损失、decoder 或训练分支。

当前验收分别检查视频运动与清晰度、Future3D/K8 预测、独立 policy 的物理误差及
task/vision/state/history 依赖。主 demo 不提供真实未来 action，而由官方联合 sampler
自己生成视频与动作。联合样本中的 action 和部署用独立 policy 都要报告，不能互相替代。

给定候选动作的反事实预测、动作可控性与候选排序留待后续专项训练/验收。修好的接口保留，
但 real/zero 视频响应不再是当前阶段的主要门槛。训练稳定、没有目标泄漏仍是必要条件；
q20 只证明接口可运行，学习趋势应在短适配中验证，不要求 20 个样本就达到最终质量。

## 冻结架构

```text
三视角视频 + 任务/机器人条件 ───────────→ Wan2.2 video expert ─────┐
noisy action + 任务/状态/历史 ──────────→ ActionDiT policy ────────┤
最后观测帧 → frozen VGGT P144 ─zero-gate→ Future3D P144 expert ───┤→ WSA MoT
                                                       │          │
真实 future action/time/mask ───────────────────────────┴→ causal K8×P64 rollout
抽样未来帧 → frozen VGGT P64 ─────────────────────────────→ rollout delta loss
最后未来帧 → frozen VGGT P144 ────────────────────────────→ 官方 P1296 final 3D loss
```

这里的 `P` 是每个视角的空间 token 数，不是 RGB 分辨率。当前采用分层预算：

- `P144`（12×12）用于 WSA 已有的联合 Future3D expert 和当前状态；
- `K8×P64`（每时刻 8×8）用于完整时间轨迹，保持 causal rollout 计算可控；
- `P1296`（36×36）保留 WSA 官方最终高分辨率 3D 输出。

不把 30 层 MoT 扩成 `K8×P144`。那会显著增加序列注意力、activation 和训练时间，而当前状态
与最终细节已经分别由 P144/P1296 覆盖。只有短训证明 P64 是明确空间瓶颈后，才单独做 P64/P144
rollout A/B，不能在首个版本里盲目扩大。

必须保持以下合同：

- 视频沿用 WSA-Large 的三视角画布和完整 K8 未来监督；不引入 copy-last、P256 teacher forcing
  或另一套 RGB decoder。
- 最后观测三视角只作为 `P144` current native state，经 zero-gated、逐视角 projection 注入
  Future3D query。gate 初始严格为零，只保证这个新增支路初始不扰动 Future3D 输入；动作和
  状态边界投影因尺寸变化而重置，整模不能宣称与原 WSA 等价。缺失视角始终 mask 为零，
  视角之间不池化。
- native-3D rollout 是单一轻量 causal recurrence。它读取 shared Future3D、当前 P144 状态和
  同一个 source-native future action/time/mask；每个 world interval 内的 fine command 不插值，
  按真实时间编码后聚合。rollout 只能写 world 分支，不能写回 ActionDiT policy。
- rollout 预测相对最后观测状态的 P64 delta，而不是复制绝对未来外观；一次只抽样一个未来
  horizon 作 teacher target，horizon 由封存 sample index 确定，跨样本均匀覆盖 K8。当前、抽样
  horizon 和最终帧在一次 frozen VGGT 调用中编码，future target 从不进入模型输入或 serving。
- action 保留 WM3D 的 grouped ABI。容量为 3 组 × 14 维，但有效组、有效维和原生命令数都由
  mask 决定；18 个现有 source 的实际未来命令数为 8、16、24 或 32，不做插值。
- action 时间等于 world interval 起点加该 interval 内的真实 fine-command offset。训练和推理
  使用同一个 `ActionTimeConditioner`。
- 当前状态为 3 组 × 10 维；offset、scale、semantic、embodiment 与 action history 通过
  zero-gated 条件融合进入 WSA 文本上下文，不传 source ID。
- WSA-Large 发布权重的 action 输入/输出和 proprio 边界原为 24 维。V10 改为 42/30 维，因此
  只重置这些尺寸不兼容的边界投影；视频、ActionDiT、Future3D 的 MoT 主干权重继续加载。
- Future3D 仍是 WSA-Large 主干；最终监督教师从官方 DA3 改成当前项目冻结的 VGGT token。
  WSA 最终 3D 与 K8 rollout 共用原来的总 3D 权重，分别为 0.05/0.05，避免额外 objective
  压过 video/action。rollout 只用一个按 target-delta 尺度归一化的 robust regression；方向余弦
  与幅值比仅作验收指标，不作为第二项竞争 loss，也不增加第二套 3D owner。
- 物理反归一化严格使用同一 source/group profile：`physical = normalized * scale + offset`，
  padded 维保持零。

## 目录与模型资产

建议固定四个目录：

```bash
export WSA_SOURCE=/data/Minko/third_party/WSA
export WSA_MODEL=/data/Minko/models/WSA-Large
export WAN_MODEL=/data/Minko/models/Wan2.2-TI2V-5B
export WAN_CONTEXT=/data/Minko/wm3d_foundation_context
```

先获取官方源码：

```bash
git clone https://github.com/zaleni/WSA.git "$WSA_SOURCE"
```

## 从空白环境开始

基础 WM3D 与 WSA 增量依赖分开封存：

```bash
./run_wm3d.sh env
WM3D_WSA_SOURCE_ROOT="$WSA_SOURCE" ./run_wm3d.sh foundation-env
```

`foundation-env` 会复用已验证的 CUDA/PyTorch 基础环境，只在 `.venv-foundation` 安装固定的
WSA 增量依赖。第一次执行创建 receipt，之后执行只验证，不覆盖已有 receipt。完成后可运行：

```bash
FOUNDATION_PYTHON_BIN=.venv-foundation/bin/python \
  ./run_wm3d.sh check
```

再下载并检查 WSA-Large 与 Wan2.2 权重：

```bash
./run_wm3d.sh foundation-fetch \
  --output "$WSA_MODEL" \
  --wan-output "$WAN_MODEL"
```

网络受限时可额外传入兼容 Hugging Face Hub 的 `--endpoint`。已有本地资产时不需要重复下载；
配置文件中的四个路径必须指向实际目录。

## 构建任务 context store

context store 只需为当前 sealed data closure 创建一次：

```bash
./run_wm3d.sh foundation-context \
  --metadata-root /path/to/streaming_metadata \
  --wan-model-root "$WAN_MODEL" \
  --upstream-source-root "$WSA_SOURCE" \
  --output "$WAN_CONTEXT"
```

脚本要求每个任务文本非空并唯一封存；未知任务不会在训练时临时编码或静默回退。

## 生成 runtime

先确认 `configs/model/foundation_large.yaml` 中的四个资产路径与本机一致，然后用现有
direct-raw closure 物化 runtime：

```bash
export DATA_ROOT=/path/to/sealed_data
export RUN_ROOT=/path/to/wm3d_v10_run

PYTHON_BIN=.venv-foundation/bin/python ./run_wm3d.sh runtime \
  --model configs/model/foundation_large.yaml \
  --data "$DATA_ROOT/data_profile.yaml" \
  --runtime configs/runtime/h100_8_foundation.yaml \
  --objective configs/objective/foundation_joint.yaml \
  --data-mode direct_raw \
  --streaming-metadata-seal "$DATA_ROOT/streaming_metadata_seal.json" \
  --environment-lock .venv-foundation/environment_receipt.json \
  --run-name wm3d-v10-foundation \
  --run-lineage fresh-foundation \
  --output-root "$RUN_ROOT" \
  --output "$RUN_ROOT/runtime.yaml"
```

正式物化要求 Git 工作树干净。不要使用 `--allow-dirty-smoke` 生成正式 runtime。

## 单机 8×H100 预检与训练

当前 profile 使用 DDP、BF16、micro batch 1、累积 8 次、global batch 64。先跑 preflight：

```bash
./run_wm3d.sh foundation-preflight \
  --standalone --nproc_per_node=8 -- \
  --runtime "$RUN_ROOT/runtime.yaml"
```

通过后才启动短程资格训练：

```bash
./run_wm3d.sh foundation-train \
  --standalone --nproc_per_node=8 -- \
  --runtime "$RUN_ROOT/runtime.yaml" \
  --stop-after-step 20
```

不要直接从 20 步跳到正式长训。至少继续验证 step100/500 的视频运动、清晰度、ActionDiT
物理误差、task/state/history 消融与 Future3D 指标；只有三条能力同时向好才扩大训练。

## Checkpoint 视频与 Action 对照

默认用完整 checkpoint 导出观测条件下的联合视频/动作，以及独立 policy 预测：

```bash
FOUNDATION_PYTHON_BIN=.venv-foundation/bin/python \
  ./run_wm3d.sh foundation-demo \
  --runtime "$RUN_ROOT/runtime.yaml" \
  --checkpoint "$RUN_ROOT/checkpoints/step_00000020" \
  --output "$RUN_ROOT/eval/joint_demo" \
  --seed 7340 --num-inference-steps 20
```

输出目录必须尚不存在。入口使用指定 Python 的 torch.distributed.run，不依赖旁边是否有
torchrun 可执行文件。

默认 mode 为 joint。导出器先移除未来 action 和 3D 标签，视频输入仅保留观测帧；
predict_joint 调用官方 WSA 联合 sampler，从随机噪声同时生成未来视频和 action。
面板标为 joint prediction。报告联合视频相对 copy-last 的误差、逐帧变化，以及联合 action
和独立 policy 各自相对常量零基线的误差、逐 query 误差和分维度物理误差。
分维度物理误差必须按 source 语义解释，不能混合米、弧度和夹爪单位当成单一质量分数。

只有传入 --mode candidate 才运行 real/zero 反事实诊断；该模式不是当前默认验收。
predict_action 只根据观测、任务、状态和历史预测动作。predict_world 则根据给定候选动作
预测视频：每个采样步始终把 clean、masked 候选送入 ActionDiT，并固定 action noise timestep
为 0；仅未知的未来视频走原 WSA sampler。已知动作不能再随视频加噪。WSA 的训练本来就
独立采样两个噪声级别；这种条件推理与 [UWM 的 forward-dynamics 定义](https://arxiv.org/html/2504.02792v1#S3.SS2)
一致。这里的噪声 timestep 不是机器人真实 action timestamp。该分支的 action 输出不作为
policy 预测返回。
候选对照调用结束后会恢复原采样函数。两个模式都检查真实未来 action 标签不会改变独立
policy 输出；joint 模式不需要该标签作为输入，也不会启用 candidate clamp。

发布的 WSA-Large 关闭了 VideoDiT 的 action_conditioned 选项。旧导出器仅向那个未启用的
入口传 real/zero action，实际上生成了两次相同的联合样本；旧对照不能说明模型的动作因果
能力。修正后的真实响应仍需用 checkpoint 测量，不能凭接口修复认定方向或任务能力已经正确。

面板中的 normalized zero action 指归一化坐标零值，即物理坐标中的 source offset，
不等于机械臂不动。时序指标仅统计 h1 至 h8 的有效视角，不把输入首帧的 VAE 重建差计作
预测运动。所有 GIF 共用同一调色板，各帧另存为无损 PNG；定量分析应使用 PNG 或
receipt 中量化前的指标。GIF 播放速度只用于展示，真实动作时间仍由原数据时间戳决定。

此前 q20 的逐步加噪候选对照（seed 7340、sample 139、20 次采样、两个有效视角）：
RGB 响应 RMS 为 0.00217，未启用入口的旧调用为 0；独立 policy 前后差异为 0。未来帧运动
幅度保持率约 50.1%，方向一致性约 0.017。这些是旧采样方式的结果，不是 clean-action
修复后的成绩，也不足以证明动作方向或任务能力。此次仅修正推理条件，不改参数或训练
objective。全仓检查 610 项通过、4 项按环境条件跳过。

q20 的 global batch 为 1、累积为 1，合计仅 20 次训练样本曝光。实际加载审计确认视频、
ActionDiT、Future3D 主干没有遗漏；尺寸不兼容而重置的张量仅有 action_encoder.weight、
action head.weight/bias 和 proprio_encoder.weight。它们与新增机器人条件模块需要在当前
物理语义和归一化坐标上适配。不能把 q20 等同于已适配好的 WSA 官方模型；也不能仅凭训练步数
就排除残余实现问题。是否学到动作跟随，应分别检查条件视频和独立 policy 的真实样本结果。

同 q20、同 sample 139/seed 7340 的 clean-action 对照已完成：视频 factual/zero 响应 RMS
从 0.00217 升至 0.02303，但方向一致性从 0.0171 变为 0.0121，帧间变化保持率从 50.1%
变为 48.0%。factual L1 为 0.119612，zero L1 为 0.119583，没有正确动作优于 zero 的证据。
独立 policy MAE 为 0.82532，而该样本的归一化常量零基线为 0.63345；candidate 对 policy
差异仍为 0。这里只确认已知动作条件不再被采样器人为加噪削弱，不能宣称动作预测质量已经解决。
单个高运动样本不代表全部 source；正式长训之前仍需短适配与多样本动作/视频趋势验收。

## 已完成的实现证据

- 现有 sealed direct-raw 数据的 18 个 source 均能无插值映射到 WSA action ABI。
- 聚焦测试与全仓检查均通过：606 项通过、1 项因可选依赖跳过。覆盖 P144/P64 形状、逐视角
  隔离、zero-gate 恒等、future target 隔离、action 时间分组和 rollout loss 反向传播。
- 移除 Qwen 后，带 native-3D causal expert 的模型共有 7,254,131,334 个参数，其中
  6,549,442,666 个可训练；新增 current-state conditioner 与 K8 rollout 合计 5,202,304 个
  参数，不到整模的 0.1%。
- 真实 WSA-Large/Wan2.2/VGGT 权重已在 8×H100 上完成资源预检，并用 18 个真实 source、
  11,682 个训练窗口和 292 个验证窗口完成一次真实 AdamW 更新。该步 `video=0.20438`、
  `action=0.88790`、官方 final-3D 加权项 `0.15110`、K8 rollout 加权项 `0.01726`、
  `total=1.26063`、`grad_norm=3.53067`，全部有限。
- current、确定性抽样 horizon 与 final 三个时间点在一次 frozen VGGT 调用中编码；实测每个
  sample 为 3 帧、约 0.77 秒。输出在退出 inference context 后转成普通无梯度 tensor，已验证
  可安全参与可微 loss。
- rollout objective 只保留归一化 delta Smooth-L1。零初始化时方向余弦不进入 loss，避免零向量
  附近梯度奇异；direction 与 amplitude 仅作为 detached 验收指标。真实更新的整体梯度范数稳定，
  不再出现该竞争项导致的异常放大。
- 单卡真实优化器峰值分配显存约 51.35 GiB；真实一阶 ActionDiT 推理峰值约 13.90 GiB。
- 真实推理输出保持 8×42 normalized action，并还原为 8×3×14 physical action；mask 与原生
  时间戳保持一致。
- 干净 foundation 环境已经完成首次创建和第二次只读复验。

这些证据证明“实现接通、可训练、可推理”，不证明收敛后的任务能力。完整 8 卡吞吐和稳定性
仍需在不占用现有训练资源的窗口做连续短训；至少完成 step20/100 的 video、ActionDiT、K8
rollout 与 final-3D 联合趋势验收后，才能决定是否扩大 P64 或进入长训。

## 参考基座

- WSA 官方仓库：https://github.com/zaleni/WSA
- WSA-Large 权重：https://huggingface.co/zaleni/WSA-Large
- Wan2.2-TI2V-5B：https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B
- WSA 论文：https://arxiv.org/abs/2607.03941

代码仓库为 Apache-2.0；发布或分发 WSA-Large 权重前，还应单独确认模型仓库的权重许可元数据。
