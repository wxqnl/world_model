# WM3D 1B direct_raw 站点模板

当前默认模型为 `configs/model/native_1b_v8_native_direct_rgb.yaml`，
目标为 `configs/objective/stage0_v8_native_direct_rgb.yaml`。
原生 factual 状态直接驱动 RGB，晚期有界高频 refiner 只补细节；无绝对 P256、
外观自回归或 RAFT 训练。模型 1,262,837,817 参数，T16/P64/K8，256px，全 K8 RGB 监督。

本文描述单机 8×H100 的交付模板，不是当前服务器双节点 16 卡 50K run 的恢复配置。
正在运行的作业以自己的 sealed runtime 为准，不因文档或模板更新而原地修改。

## 数据与存储

公开 OXE 模板包含 60 个 source；它是待审计的来源清单，不代表每次正式训练都已使用
全部 60 个。实际来源、数量和权重由封存 data profile 决定，不为实验效果更改权重或 split。

完整 OXE 原始数据约 15–20TB，下载临时文件和 checkpoint 另需余量。direct_raw 不生成
旧的 2048→384 PCA、视觉 latent 或 episode LRU，但训练节点必须能稳定读取原始视频。
不需要为已下载数据重复提供下载 token。

已审计 data profile 必须包含真实 camera、任务文本、action/state 物理语义与时间戳。
准备流程见 [数据准备](WM3D_FROM_ZERO.md)。不要从模板直接跳到训练。

## 新站点运行顺序

在交付代码包的 wm3d 目录执行，先建立环境，再填写真实模型和数据路径：

```bash
./run_wm3d.sh env
SITE=/data/wm3d_1b/control/1b_canary1k.env
./run_wm3d.sh 1b init canary1k "$SITE"
# 编辑 site 中的数据、模型权重、工作目录与 rendezvous。
./run_wm3d.sh 1b doctor "$SITE"

./run_wm3d.sh 1b task-bank "$SITE"
./run_wm3d.sh 1b cache-plan "$SITE"
./run_wm3d.sh 1b streaming-prepare "$SITE"
./run_wm3d.sh 1b runtime "$SITE"
./run_wm3d.sh 1b preflight "$SITE"
./run_wm3d.sh 1b train "$SITE" 100
```

确认 step100 完整并正常退出后，以独立进程恢复：

```bash
./run_wm3d.sh 1b preflight "$SITE"
./run_wm3d.sh 1b resume "$SITE" 100
./run_wm3d.sh 1b preflight "$SITE"
./run_wm3d.sh 1b eval "$SITE" 1000
./run_wm3d.sh 1b verify "$SITE" 1000
```

模板 batch 为 micro8/global64、8 ranks。host cache 8GiB/rank，prefetch workers4/windows32，
不启用 CUDA async pipeline；这些是新站点默认值，不覆盖已有作业设置。

## 验收与正式运行

检查必要梯度有限非零、checkpoint 完整可恢复、未来 action 不影响 policy/action-free；
同一真实样本比较 factual、no-op、wrong-action 与 copy-last，分别观察运动方向、幅值、
静态/运动区误差。有限 loss 或几张训练样本不能证明所有来源的最终质量和闭环能力。

资格通过后，`1b init formal100k` 创建独立正式 site，重复 runtime/preflight/train。
正式训练 fresh 初始化；故障后只从本 run 最新完整 checkpoint exact resume。
更多数据路径和缓存说明见 [direct_raw](WM3D_DIRECT_RAW.md)。
