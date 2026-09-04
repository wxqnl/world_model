# WM3D

WM3D 使用真实时间戳，在原生 3D 状态上联合预测未来 RGB、geometry 和 grouped robot action。
项目代码位于 [wm3d/](wm3d/README.md)。1B 和 5B 共用实现，当前 RGB 使用原生直接 decoder。

- [当前实现与证据](wm3d/docs/NATIVE_DIRECT_RGB.md)
- [5B 操作手册](wm3d/docs/WM3D_5B_SCALING.md)
- [1B / 5B 扩展](wm3d/docs/WM3D_SCALING.md)
- [诊断与轻量预实验](wm3d/scripts/tools/README.md)

在当前交付 checkout 中配置环境并检查：

```bash
cd world_model/wm3d
./run_wm3d.sh env
source .venv/bin/activate
PYTHON_BIN=.venv/bin/python ./run_wm3d.sh check
```

历史配置为旧 checkpoint 和归因对照保留。新训练只使用当前手册中的模型与目标，
不要沿用旧 site/runtime。训练数据、日志、模型权重和填写后的 site 不提交到仓库。
