# Objective profiles

本目录声明 Stage0 各监督项及权重。新增或关闭 lane 时，必须同步 coverage 门禁、离线评测
receipt 和对应测试，不能让零 coverage 的 masked loss 被当作有效结果。

历史 V7 配置的后期稀疏排序调度不能约束当前 action-owned transport 配置。
旧 materialized runtime 继续按自身封存的配置执行，不随 YAML 更新改变。

`stage0_v8_action_owned_transport.yaml` 保留已存在的默认目标；暂停的旧正式训练
不能因此自动恢复。2026-09-04 的两项候选修改仍只用于受控诊断：

- `rgb_flow_teacher_pixel_units=true` 取消半图像尺寸归一化；在 256px 下相当于将
  flow 项有效权重放大 128 倍，不是只改变日志单位。
- 每步真实错配排序替代 no-op token 项，是独立的目标变化，不能和 flow 改动捆绑后
  只根据运动幅度评价。384 步组合对照已显示三个 source 重建误差变差，未通过。
- 默认仍使用原 no-op 项及每 8 步、1000 步 ramp 的真实错配排序；
  target-agnostic separation 为零。
- 物理标签、source 权重与 policy 输入没有改变。未通过的实验不得同步为 5B 默认。

后续单独真实错配排序的 384 步同样本对照显示 RGB 运动方向与运动区误差改善；
资格配置 `stage0_v8_real_action_rank_qualification.yaml` 仅包含这项改动，
flow 仍为原半图像尺寸归一化、权重 0.20。该配置尚未成为默认正式或 5B 方案。
它复用现有唯一 action 对照目标，不新增模型参数、policy 输入或 source 重采样。

配置修改、梯度接通与模型效果必须分开报告。全尺寸同样本拟合只能诊断优化是否可行，
不能冒充跨 episode 泛化、正式资格完成或真实 VLA 成功率。长训前还需同代码分布式资格
及全量数据闭包；不得把旧的小规模资格闭包作为正式全量数据。
