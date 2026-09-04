# Runtime profiles

本目录声明拓扑、batch、精度、分块、optimizer/schedule、checkpoint 和资源门槛。
当前 1B 16 卡正式 recipe 为 h100_16_fsdp2_v8_native_direct_rgb_50k.yaml。
5B 的当前操作只看 ../../docs/WM3D_5B_SCALING.md；64 卡 canary 与 formal 使用
相同的 AdamW 起点，不能仍沿用旧 transport 的高学习率默认值。

新配方须在真实拓扑验证前后向、显存、checkpoint/resume 和 eval。
Meta 构建不验证资源容量。旧 profile 保留为历史运行兼容，不是可互换的新训练选项。
不要修改已运行或已封存的 runtime，也不要为通过资格临时减 K、分辨率或模型宽度。
