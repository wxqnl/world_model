# WM3D V8 5B 训练

使用本次交付代码包。不要沿用旧 checkout 生成的 site 或 runtime。
当前入口是原生直接 RGB；`5b doctor` 必须输出 `native_direct_5b` 和
`5,245,128,313` 参数。本文不表示 64 卡新配方已经完成实机资格。

## 1. 当前配方

- 模型：`configs/model/native_5b_v8_native_direct_rgb.yaml`
- 目标：`configs/objective/stage0_v8_native_direct_rgb.yaml`
- 8 节点 × 8 H200，FSDP2，节点内 8 卡分片；每卡 batch 4，总 batch 256。
- T24、P144、K16、384×384 RGB，监督全部 K16，保留真实时间戳和 grouped action。
- 先运行独立 1K 资格训练，再 fresh 启动 600K 正式训练。资格权重不能用于正式初始化。

与当前 1B 共用物理 factual pass、Action/Policy、原生直接 RGB 和数据 ABI。
直接 RGB、context residual、blend/motion 能生成新露出区域；高频 refiner 只补晚期细节。
不加载 P256 appearance、自回归外观、RAFT/flow teacher 或预训练视频 decoder。
Future candidate 不得影响 policy/action-free 输出。

RGB 与 Action objective 使用当前 1B 同一份文件。AdamW 从 1e-6 warmup 到 1e-5，
warmup 500 步、weight decay 0.02；没有按参数量或卡数自动放大学习率。
保留 5B 的序列、分辨率和宽度，实际显存与吞吐必须由 H200 资格训练确认。

## 2. 填写本地模型和数据路径

把交付代码放到 `/data/world_model`，所有节点使用同一份代码和共享路径：

```bash
cd /data/world_model/wm3d

MODEL_ROOT=/共享目录/模型
DATA_ROOT=/共享目录/已下载数据

./run_wm3d.sh 5b configure "$MODEL_ROOT" "$DATA_ROOT"
SITE=/data/wm3d/control/5b_canary1k.env
```

模型目录需包含合同要求的 VGGT 源码、VGGT-1B 权重及 Qwen3-VL-Embedding-2B 权重。
数据可以来自魔搭、Hugging Face 或内部存储；不要求重新下载。

`configure` 只检查本地输入并生成 site，不申请 GPU、不启动训练：

- `RAW_COMPATIBLE`：能识别原始数据，但缺已审计 data profile，不能直接训练。
- `PROFILE_PATH_MISMATCH`：control 包里的路径在当前机器不存在，需要按当前路径重新物化。
- `PROFILE_READY`：可以准备任务编码、窗口和归一化统计。
- `TRAIN_METADATA_READY`：所需 metadata 文件齐全，还需 runtime 和集群 preflight 验证。

只有原始 AGI 2026 下载目录时，仍需与该目录匹配的已审计 adapter、manifest 和
data profile。目录识别不能证明动作单位、坐标系、夹爪极性或训练划分正确。
不能自行猜测这些定义，也不能用空文件绕过检查。

1B 的原始数据和经过审计的 adapter 可以复用。5B 的 T24/P144/K16 与 1B 不同，
必须生成对应的窗口、统计和 runtime，不能直接复制 1B 的 metadata seal。

## 3. 准备环境与 metadata

在每个节点创建同一环境；下面的数据准备命令只在一个节点执行：

```bash
cd /data/world_model/wm3d
SITE=/data/wm3d/control/5b_canary1k.env

./run_wm3d.sh 5b env "$SITE"
./run_wm3d.sh 5b doctor "$SITE"
./run_wm3d.sh 5b task-bank "$SITE"
./run_wm3d.sh 5b cache-plan "$SITE"
./run_wm3d.sh 5b streaming-prepare "$SITE"
./run_wm3d.sh 5b runtime "$SITE"
./run_wm3d.sh 5b status "$SITE"
./run_wm3d.sh 5b configure "$MODEL_ROOT" "$DATA_ROOT"
```

这是 direct_raw 路径，只生成任务编码、索引和统计，不预先缓存全部 RGB/VGGT 特征。
metadata 按确定顺序并行生成；不改变来源权重、episode split 或物理转换。
现成数据不需要 Hugging Face token。

最后一次检查应显示 `ready_for_preflight: true`。
`ready_to_train` 不再因文件存在就变为 true；输入扫描不能替代真实集群 preflight。
遇到 `runtime_issues`，先修复环境或重新生成匹配的新 runtime，不能手改封存文件。

## 4. 运行 64 卡资格训练

申请 8 个完整 H200 节点，进入 Slurm allocation。脚本自动取 master 节点和 rank。
当前 H200 配方要求 NVLink、400Gb/s InfiniBand 和相应资源检查；不能拿当前
1B 的双节点 TCP 测试冒充这一拓扑已经通过。

```bash
cd /data/world_model/wm3d
SITE=/data/wm3d/control/5b_canary1k.env

./run_wm3d.sh 5b slurm "$SITE" preflight
./run_wm3d.sh 5b slurm "$SITE" train 20
```

先检查这 20 步：真实前后向正常、必要梯度有限非零、没有 CUDA/NCCL/ECC 错误，
进程在提交 step20 checkpoint 后正常结束。然后验证独立进程恢复并完成 1K：

```bash
./run_wm3d.sh 5b slurm "$SITE" preflight
./run_wm3d.sh 5b slurm "$SITE" resume 20

./run_wm3d.sh 5b slurm "$SITE" preflight
./run_wm3d.sh 5b slurm "$SITE" eval 1000
./run_wm3d.sh 5b verify "$SITE" 1000
```

恢复入口会校验完整 checkpoint。资格需要 64 个 rank、真实梯度、保存/恢复/评测全部通过，
并确认 future action 不泄漏到 policy/action-free。固定多来源样本还要检查运动方向、
静态/运动区误差和真实/错配 action 对照；总 loss 有限不等于图像和 VLA 质量已通过。

## 5. Fresh 启动正式训练

资格通过后创建独立正式 site：

```bash
cd /data/world_model/wm3d
CANARY_SITE=/data/wm3d/control/5b_canary1k.env
SITE=/data/wm3d/control/5b_formal600k.env

test ! -e "$SITE"
install -m 600 "$CANARY_SITE" "$SITE"
sed -i 's/^WM3D_5B_PRESET=canary1k$/WM3D_5B_PRESET=formal600k/' "$SITE"
sed -i 's/^WM3D_5B_RUN_ID=.*/WM3D_5B_RUN_ID=formal_native_direct/' "$SITE"

./run_wm3d.sh 5b doctor "$SITE"
./run_wm3d.sh 5b runtime "$SITE"
./run_wm3d.sh 5b slurm "$SITE" preflight
./run_wm3d.sh 5b slurm "$SITE" train
```

不得从资格、旧 5B、1B 或其他版本初始化。正式训练中断后，只恢复本次正式 run
最新的完整 checkpoint：

```bash
./run_wm3d.sh 5b slurm "$SITE" preflight
./run_wm3d.sh 5b slurm "$SITE" resume 完整checkpoint的step号
```

不修改运行中的代码、runtime、数据权重和归一化统计。若出现非有限、future 泄漏、
缺失梯度、明确通信/存储错误，保留证据并修复首个根因，不通过降分辨率、减少 K16
或换旧 checkpoint 绕过资格。
