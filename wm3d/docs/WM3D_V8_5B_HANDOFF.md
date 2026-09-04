# V8 5B 交接入口

唯一操作流程见 [WM3D 5B 训练](WM3D_5B_SCALING.md)，本页不再维护第二套启动命令。

当前交付使用 `native_5b_v8_native_direct_rgb.yaml` 和
`stage0_v8_native_direct_rgb.yaml`。参数量 5,245,128,313，T24/P144/K16、384px，
与当前 1B 共用物理 factual pass、Action/Policy 和原生直接 RGB。
1B 的训练资格不能替代 5B 的实机前后向、显存、checkpoint 和多机通信验证。

旧 transport/RAFT/P256 配置及旧 site/runtime 不用于新训练。
`doctor`、runtime 物化和启动前检查会拒绝旧配方。
先在目标 64×H200 集群运行资格训练，通过后才 fresh 启动正式训练。
