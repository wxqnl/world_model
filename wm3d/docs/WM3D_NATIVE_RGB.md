# WM3D 原生 RGB 解码器

WM3D 的正式 RGB 路径不依赖 Wan 或其他外部视频生成器。世界状态主干先预测
2048 维 native tokens，RGB decoder 再将这些 token 直接还原为未来图像。

## 当前正式结构

| 项目 | native-1b | native-5b |
|---|---:|---:|
| token grid | 8×8 | 12×12 |
| 输出分辨率 | 256×256 | 384×384 |
| decoder hidden | 1280 | 1536 |
| 每个上采样层 residual blocks | 2 | 2 |
| RGB 参数量 | 129,748,803 | 186,197,379 |
| 监督未来帧 | 全部 8 帧 | 全部 16 帧 |

该实现恢复经过 V7 验证的 token-to-pixel 主路径，并做两项受控增强：1B hidden
从 1152 增至 1280；5B 扩至 1536。上采样仍采用可学习的 stride-2
transpose convolution，没有引入额外视频模型。训练时只解码 batch 中实际带 RGB
监督的相机，推理时仍可显式请求全部相机，避免在缺失视角上浪费计算。图像按固定小块
执行 decoder（1B 每次 4 张、5B 每次 2 张），计算结果不变，但显存峰值不随
`micro_batch × K × view` 线性增长。

## RGB 目标

正式目标同时包含：

- L1：保持颜色和绝对像素结构；
- Charbonnier：对少量异常像素保持稳健；
- spatial gradient：约束边缘；
- VGG LPIPS：约束人眼感知的纹理与结构清晰度。

LPIPS 网络被冻结，不属于世界模型参数，也不进入 optimizer 或 checkpoint；梯度只从
LPIPS 输入传回 native RGB decoder 与 token 输出层。LPIPS 必须来自封存的运行环境，
缺少依赖时训练直接失败，不能静默退回纯像素损失。

## 验收原则

旧 lightweight RGB head 的 checkpoint 只能作为 backbone 初始化来源，不能作为新结构的
exact resume。新 decoder 必须经过独立 canary，至少报告 RGB L1、LPIPS、PSNR、边缘保持率、
时序变化保持率和固定样本可视化。仅有 finite loss 或 loss 下降不能证明图像质量达标。
