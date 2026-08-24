# WM3D 原生 RGB 解码器

WM3D 的正式 RGB 路径不依赖 Wan 或其他外部视频生成器。模型保留原生 3D
世界状态作为动力学主干，同时使用一条逐视角 appearance 通路保存渲染所需的高频信息。
两条通路来自同一次冻结 VGGT encoder 前向，不会重复执行视觉编码。

## 当前正式结构

| 项目 | 1B dual path | 5B dual path |
|---|---:|---:|
| 3D geometry grid | 8×8（P64） | 12×12（P144） |
| per-view appearance grid | 16×16（P256） | 16×16（P256） |
| appearance context | 最近 4 帧 | 最近 6 帧 |
| appearance dynamics | 2 层、hidden 512 | 4 层、hidden 768 |
| 输出分辨率 | 256×256 | 384×384 |
| decoder hidden | 1280 | 1536 |
| 每个上采样层 residual blocks | 2 | 2 |
| 监督未来帧 | 全部 8 帧 | 全部 16 帧 |

3D 主干仍只消费融合后的 geometry tokens，继续负责几何、动作、状态和世界动力学。
appearance tokens 不在相机之间提前融合；appearance dynamics 分别预测每个视角的未来
P256 latent，并接受 3D future state 作为条件。RGB decoder 同时读取对应视角的 appearance
latent 和 3D 条件，因此纹理不会被迫先穿过 P64/P144 的融合瓶颈，几何一致性也没有被绕开。

1B 在原有主干上只增加约 850 万参数；总参数量约 13.28 亿。5B 仍保持原有 P144
几何容量，appearance 通路固定为 P256，避免为了 RGB 纹理把整个 3D 主干扩大四倍。

## 数据表示

冻结 VGGT 的一次 forward 同时产出：

- `view_tokens`：取最深的第 23 层特征，逐视角池化到 geometry grid，随后进入原有多视角融合与 3D 主干；
- `appearance_tokens`：取较浅且仍保持 2048 维 ABI 的第 4 层特征，保持逐视角 P256，不做 PCA、不做跨视角平均，只供 appearance
  dynamics、appearance loss 和 RGB decoder 使用。

浅层 appearance 保留更多颜色、边缘和局部纹理；深层 geometry 仍保留 VGGT 的完整几何推理。两者来自同一次 forward，不会增加第二次 VGGT 编码。

当前正式 `direct_raw` 路径只为当前 `T+K` window 在线生成两组 token，不量化、不落盘，
也没有 episode LRU 或 sidecar。`streaming_raw` 仅作为兼容旧运行的可选路径；它会将两组
token 量化后写入有容量上限的 episode LRU。若显式使用完整 episode cache，则必须使用
双通路 encoder 合同：

- 1B：`configs/encoder/vggt_native_p64_appearance_p256.yaml`；
- 5B：`configs/encoder/vggt_native_p144_appearance_p256.yaml`。

旧的 geometry-only cache 没有 P256 appearance latent，不能直接用于 dual-path 正式训练；此前从 VGGT 最深层生成的 P256 appearance cache 也不能与新的第 4 层 appearance 合同混用。它们仍可用于旧结构 A/B，不会被删除或伪装成新 cache。

## Teacher 到预测 latent 的切换

训练开始时，RGB decoder 主要读取真值 future appearance latent，先学会稳定的
appearance-to-RGB 重建；随后按照 runtime 中的
`appearance_teacher_start_ratio`、`appearance_teacher_end_ratio` 和
`appearance_teacher_decay_steps` 线性切换到模型预测 latent。无论 teacher ratio 是多少，
appearance dynamics 都持续接受 MSE 与 cosine 监督，不会因为 teacher forcing 而没有梯度。
验证在同一批固定样本上同时记录 teacher0、当前线性 schedule 和 teacher1，直接量化
appearance inference gap。step5000 的同样本探针没有支持“latent lerp 导致模糊”的假设：
ratio=0.5 的结果优于按样本随机选两个 endpoint 的期望，因此正式 schedule 保留 lerp。
另一方面，step5000 固定样本的 teacher0 短程训练能快速改善真实推理路径，所以 1B 正式
profile 在前 500 step、5B 正式 profile 在前 1,000 step 完成 teacher1 到 teacher0 的
切换，不再让 decoder 到 step10000 才完全暴露于预测 appearance latent。

正式模型与目标配置为：

- `configs/model/native_1b_dual_path.yaml`；
- `configs/model/native_5b_dual_path.yaml`；
- `configs/objective/stage0_native_dual_path.yaml`。

旧 `native_1b.yaml` / `native_5b.yaml` 保留为 geometry-only 对照，不是新训练的默认选择。

## RGB 目标

正式目标包含：

- Charbonnier（权重 1.5、epsilon 1e-3）：保持颜色和绝对像素结构；
- spatial gradient：约束边缘；
- VGG LPIPS：约束人眼感知的纹理与结构清晰度。

RGB L1 仍逐步报告用于可比性，但优化权重为 0；旧的 `0.5*L1 + 1.0*Charbonnier`
在 epsilon 1e-6 下几乎把同一绝对残差计算两遍，现配置用一个独立 epsilon 的
Charbonnier 保留近似相同的像素梯度总尺度。

LPIPS 网络被冻结，不属于世界模型参数，也不进入 optimizer 或 checkpoint；梯度只从
LPIPS 输入传回 native RGB decoder 与 token 输出层。LPIPS 必须来自封存的运行环境，
缺少依赖时训练直接失败，不能静默退回纯像素损失。

图像按 runtime 的固定小块执行 decoder（当前 1B H100 正式配置每次 16 张、5B H200
配置每次 2 张）；冻结感知网络的输入单独按每次 8 张分块。两种分块都只调整执行粒度和
峰值显存，不改变数学结果。训练只解码实际带 RGB 监督的相机，推理可以显式请求全部
可用相机。

## 已完成的真实链路验证

`direct_raw` 已在真实 RoboCasa 与 OXE 视频上验证：RoboCasa 的 direct/full decode RGB
逐像素一致；OXE 的大 PTS offset 半开区间与旧 sealed decoder 选择完全相同；AV1 长 episode
会走有界 PyAV keyframe fallback。真实 VGGT 输出的 P64 geometry、P256 appearance、RGB、
depth、point 和 camera 全部有限。node42 的两卡 1.327B FSDP2 pilot 已完成完整 objective、
backward、optimizer、两分片 COMMITTED checkpoint 和独立 exact-resume 读取。该短跑验证实现、
分布式保存与恢复链路，不用于宣称最终图像质量；清晰度结论必须来自相同
数据预算下 geometry-only 与 dual-path 的固定样本对比。

## 验收原则

旧结构 checkpoint 只能作为显式转换后的 backbone 初始化来源，不能作为 dual-path exact
resume。正式训练前先跑小规模 canary；质量评估必须同时报告 RGB L1、LPIPS、PSNR、边缘
保持率、时序变化保持率，以及同一批固定样本上的真值、teacher-latent 重建和 predicted-latent
预测。仅有 finite loss 或 loss 下降不能证明图像质量达标。
