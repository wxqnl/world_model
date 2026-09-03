# WM3D V10：WSA-Large foundation 路线

## 结论与边界

V10 是一条独立的 foundation 实验路线，不是对当前 V8 长训目录的原地修改。它以官方
WSA-Large 为模型基座，保留 Wan2.2 视频专家、ActionDiT、Future3D 与 Mixture-of-Transformers
联合训练主干，再接入 WM3D 已封存的数据、机器人物理语义和原生 3D 监督。

WSA-Large 的基座是 Wan2.2-TI2V-5B，不使用 Qwen3-VL。Qwen3-VL 属于 WSA-Base，V10
因此不加载 Qwen、不创建 Qwen LoRA，也不要求 Qwen image processor。任务文本沿用
WSA-Large 的 UMT5 上下文，并预先写入只读 context store，训练时不常驻文本编码器。

这条路线已经证明可以装载、训练反传和执行 ActionDiT 推理，但尚未证明在 WM3D 数据上优于
当前 V8。替代正式训练前仍需做短程受控训练，并同时检查视频、物理 action 和 Future3D，不能
直接套用 WSA 官方 benchmark 数字。

## 冻结架构

```text
三视角最后观测帧 ───────────────→ Wan2.2 video expert ─┐
任务文本 → frozen UMT5 context ──────────────────────────┤
当前状态 + 历史动作 + 机器人语义/归一化 ─→ gated fusion ├→ WSA MoT joint loss
源数据原生未来动作 + 精确时间戳 ─────────→ ActionDiT ────┤
最后未来帧 frozen VGGT token target ─────→ Future3D ─────┘
```

必须保持以下合同：

- 视频沿用 WSA-Large 的三视角画布和完整 K8 未来监督；不引入 copy-last、P256 teacher forcing
  或另一套 RGB decoder。
- action 保留 WM3D 的 grouped ABI。容量为 3 组 × 14 维，但有效组、有效维和原生命令数都由
  mask 决定；18 个现有 source 的实际未来命令数为 8、16、24 或 32，不做插值。
- action 时间等于 world interval 起点加该 interval 内的真实 fine-command offset。训练和推理
  使用同一个 `ActionTimeConditioner`。
- 当前状态为 3 组 × 10 维；offset、scale、semantic、embodiment 与 action history 通过
  zero-gated 条件融合进入 WSA 文本上下文，不传 source ID。
- WSA-Large 发布权重的 action 输入/输出和 proprio 边界原为 24 维。V10 改为 42/30 维，因此
  只重置这些尺寸不兼容的边界投影；视频、ActionDiT、Future3D 的 MoT 主干权重继续加载。
- Future3D 仍是 WSA-Large 主干；监督教师从官方 DA3 改成当前项目已经冻结的 VGGT 最后未来帧
  token。教师 target 不进入推理输入，也不参与优化器。
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

## 已完成的实现证据

- 现有 sealed direct-raw 数据的 18 个 source 均能无插值映射到 WSA action ABI。
- 移除 Qwen 后，模型共约 7.25B 参数；其中官方 WSA backend 约 7.16B，WM3D 新增机器人/
  时间条件约 0.09B。
- 真实 WSA-Large 权重上的两次 AdamW 前后向已完成；视频、ActionDiT、Future3D、proprio、
  action-time 和 robot-conditioning 梯度均有限非零。
- 单卡真实优化器峰值分配显存约 51.35 GiB；真实一阶 ActionDiT 推理峰值约 13.90 GiB。
- 真实推理输出保持 8×42 normalized action，并还原为 8×3×14 physical action；mask 与原生
  时间戳保持一致。
- 干净 foundation 环境已经完成首次创建和第二次只读复验。

这些证据证明“实现接通、可训练、可推理”，不证明收敛后的任务能力。完整 8 卡吞吐和稳定性
仍需在不占用现有训练资源的窗口做正式 preflight。

## 参考基座

- WSA 官方仓库：https://github.com/zaleni/WSA
- WSA-Large 权重：https://huggingface.co/zaleni/WSA-Large
- Wan2.2-TI2V-5B：https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B
- WSA 论文：https://arxiv.org/abs/2607.03941

代码仓库为 Apache-2.0；发布或分发 WSA-Large 权重前，还应单独确认模型仓库的权重许可元数据。
