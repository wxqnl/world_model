# WM3D 原生 RGB 训练

本文对应 `native_1b_v8_action_owned_transport.yaml` 与
`stage0_v8_action_owned_transport.yaml`。旧 P256 dual-path、appearance teacher/AR
说明不再描述这条路径。当前修改仍需全量数据与分布式资格验证，本文不构成放行凭据。

## 当前计算路径

未来 physical action 只进入独立 factual pass。每个 future state slot 在 state block 0
之前读取同 horizon 命令，后续 block 保留按 horizon 的 action modulation。
RGB 和 token/geometry 监督读取同一 factual P64 输出。policy/action-free pass 不读取
future candidate，换候选动作时其输出必须逐元素不变。

原生 transport decoder 读取 factual tokens、task 与 view embedding，预测 32×32
backward flow，再上采样为完整图像的像素位移。最后观测 RGB 仅通过此 flow 搬运到
未来位置。motion head 是辅助监督，不能乘到 flow 上把运动压回零。晚期有界高通
refiner 只补细节，没有完整频率的重绘分支。

当前 1B 使用 P64、完整 K8、256×256 RGB、decoder hidden 1024；
不提取绝对 P256，不使用 appearance 自回归、future target latent 或外部视频生成器。
RAFT 只在数据准备侧生成训练监督，不是可训练 decoder，也不进入 serving。

纯搬运不能生成最后观测中未出现的新内容。必须单独报告遮挡/新显露区域的质量；
flow oracle 有效不等于所有未来图像均能精确重建。

## 训练目标与单位

RGB 使用现有 L1 1.2、perceptual 0.55、gradient 0.08、motion-reweighted L1 1.0
及 motion BCE/Dice 各 0.03。flow teacher 权重 0.20，移动/静态有效区域按样本均衡。

预测和标签已是完整图像的像素位移。默认目标将 EPE 除以半图像尺寸；这属于损失
归一化，不是输出 flow 单位错误。诊断开关 `rgb_flow_teacher_pixel_units: true` 可取消
这项归一化，旧 runtime 默认语义不变。256px 下它会把该项的有效权重放大 128 倍，
不能只凭单位测试或运动幅度增长将它当作已验证修复。

默认动作对照仍保留 no-op token 项，并按每 8 步、1000 步 ramp 启用现有 compatible
real-action negative 排序；正误分支的行为遵循各自目标定义。每步真实负例替代 no-op
的方案仅在诊断命令中启用，尚未发布为默认训练方案。候选负例须保持同物理 layout/
normalization，并满足最小动作距离。全量数据的 source 权重不因诊断对照而改变。

2026-09-04 的完整 1B 同样本 384 步对照发现：取消 flow 归一化并同时替换动作排序后，
运动幅度及部分方向指标提高，但三个 source 的 motion/static RGB 与 flow EPE 都比
原配置差，且位移过冲。因此该组合未通过；默认配置已撤回这两项实验改动。需要区分
动作分离、正确位移、重建误差，不能用错误动作分数变差冒充正确动作预测变好。

拆开后的 384 步对照中，只改变真实动作排序、保持 flow 原权重的一路改善了三个 source
的 RGB 帧间方向一致性；Droid、RoboCasa 的运动区 L1 分别下降约 12% 和 27%，Bridge
基本持平。只放大 flow 项主要改善 flow EPE，但 RGB 运动区质量并不一致改善。
因此下一步资格使用 `stage0_v8_real_action_rank_qualification.yaml`，不包含 flow 放大。
它仍需实际分布式资格；同样本拟合不能保证长训或未见数据有效。5B 默认配置未更新。
全量索引按原顺序流式写入与读取，窗口选择和逐窗口归一化累积顺序不变。

## 快速检查的用途

`scripts/tools/run_factual_motion_microprobe.py` 保留，用于发现断路、mask、梯度和
隔离错误。小模型在单一重复轨迹上学会运动，不足以放行正式训练。

`scripts/tools/run_production_flow_loss_ab.py` 使用完整 1B、K8、256px、真实多 source
已物化 batch 与生产 optimizer/LR，比较旧目标和修正目标。它输出 normal、
physical-noop、real-mismatch 的 motion/static RGB、flow、时序变化与隔离指标。
重复同一批样本的结果仍属于优化诊断，不能称为独立泛化或 VLA 成功率。

正式启动前还要验证相同代码与目标的真实分布式前后向、checkpoint 保存/读取和完整
数据闭包。不能把 182 episode 的资格数据标为全量。训练后继续在固定且跨 episode 的
样本上跟踪运动方向、误差、清晰度与 action 对照；总 loss 下降或单独变“更动”
都不能替代这些检查。
