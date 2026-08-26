# WM3D

WM3D 是动作条件的原生 3D 世界模型。模型使用真实时间戳，在显式 3D lattice 上联合
预测未来 native token、RGB、depth、point、camera pose 和 grouped robot action。

正式项目位于 [`wm3d/`](wm3d/README.md)，包含公开数据准备、缓存、Stage0 预训练、
Stage1 规划、分布式 checkpoint、离线评测和真实双卡验收。仓库不再携带上一代模型的
训练代码；需要复用旧数据时，只能通过当前项目的 legacy importer 转换到 WM3D ABI。

> 当前 `v9` 分支是与 V8 隔离的 flow-action 研究路径。它保留 V8 的 world、RGB/P256、
> grouped physical-action ABI 和数据语义，只把 executable policy 的连续动作 owner 替换为
> WSA 风格条件 flow matching。V8 checkpoint 不能用于 V9 初始化，V9 目前也不包含已经训练
> 完成的能力结论。完整边界见
> [V9 flow-action contract](wm3d/docs/WM3D_V9_FLOW_ACTION.md)。

快速开始：

```bash
git clone --branch v9 --single-branch https://github.com/wxqnl/world_model.git
cd world_model/wm3d
./run_wm3d.sh env
source .venv/bin/activate
PYTHON_BIN=.venv/bin/python ./run_wm3d.sh check
```

完整的数据、训练和评测流程见 [WM3D README](wm3d/README.md)。需要复现实验时必须先
重新 materialize V9 model/data/runtime seal，不能把 V8 runtime 或 checkpoint 改名复用。
