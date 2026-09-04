# Native direct RGB

## 实现边界

V8 的物理 action、task、state、history 和 policy 主链不变。Future candidate 只进入独立 factual pass 和 RGB，不得进入 action-free / policy 输入。

这次只替换 RGB 的输出限制：`rgb_action_owned_direct` 使用已有的 `NativeOriginalV7ContextRGBImageDecoder`；物理 factual pass 与原 transport 配置共用实现。P64、当前 RGB 金字塔和同一中心化物理 action 进入 decoder，输出 direct RGB、context residual 与 learned blend/motion。它能够生成新露出区域的像素，不要求所有未来像素都由当前画面 warp 得到。

参照的是 [V5 实际训练调用](https://github.com/wxqnl/world_model/blob/wm3d-v5-626/wm3d_v5/wm3d_v3/models/joint_model.py)，不是仓库中的其他同名历史目录。V5 在 native 路径上把 **预测的** future tokens 和同一 action 传给 context pixel decoder，不用 future target 代替模型预测。这里恢复的是 RGB 生成能力，并非宣称 V8 整个 StateStream 已逐行复制 V5。

保留现有 33,280 参数的晚期有界高通 refiner。关闭绝对 P256、P256 AR / teacher forcing、flow / RAFT 训练和 disocclusion loss。不加载预训练视频模型。所有 K8 RGB horizon 都用实际预测路径训练。

## 配置

- 1B：`configs/model/native_1b_v8_native_direct_rgb.yaml`，1,262,837,817 参数。
- 5B：`configs/model/native_5b_v8_native_direct_rgb.yaml`，5,245,128,313 参数；参数构建检查不等于已经做过 5B 训练资格验证。
- Objective：`configs/objective/stage0_v8_native_direct_rgb.yaml`。
- 16 卡 runtime：`configs/runtime/h100_16_fsdp2_v8_native_direct_rgb_50k.yaml`。

Direct、transport 和 legacy V7 renderer 互斥。`physical_factual_pass` 决定现代物理 action 的计算路径，不能因换 renderer 而退回旧的 action 编码。正式启动必须用封存的新 profile，不得只在诊断脚本中改开关。

RGB L1、perceptual、gradient、motion 权重以及 policy action loss 保持原值。只移除不再存在的 flow/disocclusion 监督。已存在的真实错配 action 排序项保持不变；它衡量 world/RGB action 因果性，不能代替 policy 的物理动作精度。

## 如何判断改动有效

1. 对齐非 RGB 权重后，切换 renderer 不得改变 P64、policy 和 action-free 输出。
2. 改变 future action 时，policy/action-free 必须逐元素不变；factual/RGB 路径必须获得有限非零梯度。
3. 使用同一真实样本、同一模型容量、optimizer 和学习率比较，不用缩小版网络替代生产实现。
4. 同时报告 RGB / 静态 / 运动区误差、正确与错配 action 差异，以及未来帧之间的方向和幅值。必须将 context→第一个预测帧的跳变单独看，不能把它当成后续视频运动。
5. 固定小批次拟合只证明可学性。生产资格还需检查真实采样、16-rank FSDP、显存、吞吐、checkpoint，以及完整数据可用性。正式训练必须 fresh，全量数据保持原 source 权重、split 与物理语义。

## 当前证据与限制

384 次全尺寸真实批次拟合中，Bridge 的未来帧间变化保持率从 transport 的 18.4% 提高到 direct 的 51.7%，方向余弦从 0.091 提高到 0.510，运动区 L1 从 0.1773 降到 0.0704。Droid 的运动区误差和方向也改善，但变化幅度仍弱；MG 的 direct 结果仍不如 transport。不能只选 Bridge 宣布所有来源通过，更不能将拟合样本当作 held-out 泛化结果。

生产双机资格已 fresh 完成 20 步，全部关键梯度有限非零，16 个模型分片与 16 个 rank state 完整且双机读取通过；最后 10 步约 16.8 秒/步。这个检查证明生产调用和运行合同成立，不是最终画质验收。

全量离线数据准备支持有界、多进程的窗口生成与统计：窗口记录、顺序、计数、source 权重和物理转换不变；总体矩的合并使用原 float64 均值/方差公式，存在机器精度级舍入差异。18 个来源的真实数据对照已验证窗口逐项相同，11,682 个训练窗口的统计最大相对差异约 2e-15。训练和 serving 仍读取同一份封存统计，不存在两套归一化坐标。

所有来源的 future-candidate / policy 隔离均通过。早期画质与最终质量应分开报告，不因某一个阈值未达到就无证据增加新的结构、loss 或改动数据分布。

`scripts/tools/export_production_rgb_ab.py` 可对诊断快照导出按真实 K8 时间排序的 target / direct / transport / copy-last / no-op / wrong-action 面板与 GIF。诊断快照只用于评估，不能作为正式初始化。
