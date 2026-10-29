# WM3D · 5B

WM3D 基于 Wan2.2-TI2V-5B 视频骨干，联合建模视频、动作和多视角 native 3D 未来轨迹。几何监督来自冻结的 VGGT-Omega，机器人输入保留动作分组、有效掩码和数据源语义。

本分支整理了完成 167,446 次 optimizer 更新的正式预训练代码，以及 LIBERO、WorldArena、RoboNet 下游训练的运行版本。代码统一放在 [wm3d/](wm3d/README.md)，入口为 `wm3d/run_wm3d.sh`。

```bash
git clone --branch 5B --single-branch https://github.com/wxqnl/world_model.git
cd world_model/wm3d
./run_wm3d.sh --help
```

- [模型与预训练](wm3d/docs/WM3D_NATIVE_PRETRAIN.md)
- [三项下游训练与评测](wm3d/docs/BENCHMARKS.md)
- [代码来源与运行版本](wm3d/docs/REPRODUCIBILITY.md)
- [文档索引](wm3d/docs/README.md)
- [整理范围](wm3d/docs/CLEANUP.md)

checkpoint、原始数据、训练日志、运行凭证和机器环境保存在训练服务器，未打包进 Git。本次整理不修改正在运行的三项训练。
