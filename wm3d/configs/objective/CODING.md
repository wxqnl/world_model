# Objective profiles

本目录声明 Stage0 各监督项及权重。新增或关闭 lane 时，必须同步 coverage 门禁、离线评测
receipt 和对应测试，不能让零 coverage 的 masked loss 被当作有效结果。

V8 的 RGB 动作因果课程必须保持 verified V7 60K 合同：使用 source-homogeneous
local batch 的 shuffled future action，factual 与 wrong-action RGB 两个 forward 都可导；
30K 开始、10K 线性 ramp、每 8 步运行一次，rank/separation 最大权重分别为
2.0/0.5。不得用 detached zero-action telemetry 代替，也不得在 validation 中隐式运行
额外 forward。诊断预实验可以显式 force fully-ramped 权重，但不能改变正式 profile。
