# WM3D

当前实现使用 Wan2.2-TI2V-5B 视频骨干、动作分支和 native 3D 轨迹分支。生产模型使用 T16 / K8 / P64、最多三个视角、slot-noise query 和 WSA perceiver readout；冻结 VGGT-Omega 提供逐时刻几何监督。

正式预训练已完成 167,446 次 optimizer 更新。LIBERO、WorldArena、RoboNet 的下游训练各使用单机 8×H100。参数、数据协议和完成条件分别见[预训练说明](docs/WM3D_NATIVE_PRETRAIN.md)与[benchmark 说明](docs/BENCHMARKS.md)。

## 使用

`run_wm3d.sh` 是统一入口。先绑定本地模型、数据、环境与输出路径，再物化 runtime；runtime 包含数据身份、初始化、优化器、采样器和资源约束。

```bash
./run_wm3d.sh --help
PYTHON_BIN=/path/to/venv/bin/python ./run_wm3d.sh check

FOUNDATION_PYTHON_BIN=/path/to/venv/bin/python ./run_wm3d.sh foundation-preflight \
  --standalone --nproc_per_node=8 -- --runtime /path/to/runtime.json

FOUNDATION_PYTHON_BIN=/path/to/venv/bin/python ./run_wm3d.sh foundation-train \
  --standalone --nproc_per_node=8 -- --runtime /path/to/runtime.json
```

WorldArena 与 RoboNet 使用各自已部署的代码差异补丁。应用方法、来源和初始化约束见[复现说明](docs/REPRODUCIBILITY.md)。补丁只用于新的独立 checkout。

## 目录

| 目录 | 内容 |
| --- | --- |
| `wm3d/data/` | 原生数据读取、动作语义、归一化、窗口和确定性采样 |
| `wm3d/models/` | 模型工厂、视频／动作／3D 分支及损失 |
| `wm3d/encoders/` | 冻结视觉与任务编码器 |
| `wm3d/training/` | 统一训练循环、FSDP2、完整 DCP 保存／恢复、资源与初始化约束 |
| `wm3d/serving/` | 机器人策略输入输出接口 |
| `configs/` | 模型、数据、runtime、objective 配置；`benchmarks/` 保存六份数值配方 |
| `variants/` | WorldArena 与 RoboNet 运行版本相对公共代码的差异 |
| `environments/` | 依赖锁、环境引导和上游必要补丁 |
| `tests/` | 数据、模型、checkpoint 与分布式接口回归 |
| `docs/` | 当前使用说明；历史路线集中在 `archive/` |

## 历史兼容

旧 native RGB、1B／H200 扩展和 Stage1 代码仍服务已有资产与回归测试，文档集中在历史目录：
[1B streaming](docs/archive/WM3D_1B_STREAMING.md)、
[H200 5B 扩展](docs/archive/WM3D_5B_SCALING.md)、
[Stage1](docs/archive/WM3D_STAGE1_UNIFIED.md)。

Stage1 原有入口 `stage1-seal-selection`、`stage1-replay-authority`、`stage1-audit-rollouts`、`stage1-produce` 保留；它们不属于当前三项 benchmark 接续。已有 schema 名称保持磁盘兼容。
