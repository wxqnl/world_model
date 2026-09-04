# 训练目标

当前 1B/5B 共用 `stage0_v8_native_direct_rgb.yaml`。
原生 RGB 使用 L1/perceptual/gradient/motion 监督和已有真实错配 action 排序。
Flow/disocclusion、appearance/P256 teacher/AR 目标关闭；Policy objective 不随 renderer 另起一套。

变更目标必须同时更新训练 coverage、评测入口、启动检查和相关回归。
不能把关闭 lane 的零指标当成断路，也不能用 world/RGB action gain 代替 policy/VLA 能力。
旧 profile 保留为历史 checkpoint 与受控对照兼容；不要从旧 q300、transport、
V7 parity 或 core profile 启动新正式训练。运行中的 sealed runtime 不随文件更新而改变。

未经同配置真实验证，不增加分离 loss、改变源权重或把诊断结果声明为长期泛化通过。
