# WM3D 原生 RGB

当前实现和诊断证据见 [Native direct RGB](NATIVE_DIRECT_RGB.md)。
1B 与 5B 使用同一个原生直接 decoder，分别监督完整 K8/256px 与 K16/384px。

## 当前路径

Future physical action 进入独立 factual pass，形成预测的未来 native state。
RGB decoder 读取 factual state、当前 RGB 金字塔、task 和同一中心化物理 action，
输出 direct RGB、context residual 及 learned blend/motion。它可以生成新显露像素，
不要求未来画面完全由当前画面搬运得到。

晚期有界高通 refiner 用于细节。当前路径不使用 absolute P256、appearance AR/
teacher forcing、RAFT/flow teacher 或外部视频 decoder。
未来真值图像只用于监督；future candidate 不得进入 policy/action-free。

## 监督与验收

1B/5B 共用 `stage0_v8_native_direct_rgb.yaml`。
RGB L1 1.2、perceptual 0.55、gradient 0.08、motion L1 1.0，
motion BCE/Dice 各 0.03；保留已有真实错配 action 排序，不新增另一套 Action head。
Flow/disocclusion 和 appearance 目标关闭。

必须分开报告：
- 未来帧之间的方向、幅值和误差，以及 context 到首个预测帧的跳变；
- overall、静态区、运动区清晰度与真实/错配 action 差异；
- policy 的物理动作误差与条件依赖，不能用 RGB/token gain 代替 VLA 能力。

RGB/Action 轻量 probe、全尺寸生产 A/B 和 checkpoint 审计入口保留，见
[scripts/tools](../scripts/tools/README.md)。旧 transport/V7 对照工具仅用于诊断，
不能作为当前默认模型或长训放行依据。
