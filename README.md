# WM3D V8

V8 的预训练由两个连续阶段组成：

1. **Stage0：原生 3D 世界动力学与可迁移 action policy 联合预训练**
2. **Stage1-P：冻结执行 action owner 后，训练多候选原生 3D 推演与选择**

模型始终在显式 3D lattice 上预测未来 token、RGB、depth、point 和 pose。Stage1-P
不会把模型改成 VLA，也不会用 latent 3D 取代显式几何输出。

当前实现、配置、启动脚本和测试位于 [`wm3d_v8/`](wm3d_v8/README.md)。
`wm3d_v7/` 保留上一版本的公开数据与 5B 配方，便于核对数据合同和版本差异；V8
训练入口以 `wm3d_v8/` 为准。
