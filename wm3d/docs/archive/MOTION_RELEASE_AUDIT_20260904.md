> 历史诊断，不是当前训练或发布门槛。当前入口：../NATIVE_DIRECT_RGB.md。

# 2026-09-04 RGB 运动修复：未通过正式发布

截至 13:38 CST，没有启动新的全量正式训练。当前实验分支不能作为“RGB 已解决”的 V8 发布。

## 已证实的问题与证据边界

当前 transport decoder 用最后观测图像按预测 backward flow 搬运，再叠加有界高频残差。它没有生成新显露颜色和低频内容的通路。这是可由代码证明的表达能力缺口，但不能据此断言它是运动幅度偏小的唯一原因。

T3VIP 的机器人视频预测实现明确组合搬运图像、遮挡区域和新生成图像；DPG 同样区分传播区域与需要生成的区域。本次参考这一分工，不使用外部预训练视频 decoder。

- [T3VIP 官方实现](https://github.com/nematoli/t3vip/blob/main/t3vip/models/t3vip.py)
- [DPG，ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Gao_Disentangling_Propagation_and_Generation_for_Video_Prediction_ICCV_2019_paper.html)

T3VIP 使用 3D 变换计算遮挡，本实验使用学习得到的 2D 软可见性；二者不是完全复刻关系。参考文献证明这种分工已有先例，不证明本实验已成功。

旧预实验还存在验证覆盖问题：重复优化六个真实样本可以证明可拟合，但不能证明在生产采样中会学会运动；182-episode 闭包的 292 个 validation windows 实际全部来自 oxe_austin_sailor。不能把这一验证集称为跨 source 泛化评测。

另外，flow prediction 和 target 原本均为像素单位，loss 除以半图像尺寸是归一化。取消除法相当于放大权重；组合 A/B 失败，不能再称它为已修正的输出单位 bug。

## 本次实现

代码根：`/data/Minko/wm3d_transport_occlusion_complete_20260904/wm3d`。
分支：`codex/transport-occlusion-complete-20260904`。

可选配置 `rgb_transport_occlusion_completion` 默认关闭，不改变现有默认模型或 5B 发布配置。开启后：

- 新增 39,009 参数，总参数量 1,190,555,925。
- 只读取 factual decoder feature 和已搬运图像，预测 tanh 有界颜色残差。
- 复用原有 disocclusion BCE/Dice，各 0.03；不改变 flow 权重和单位。
- RGB 梯度不能扩大可见性 mask；mask 接收原有遮挡标签监督。
- 软 mask 并不严格为零，因此不能宣称可见区域绝对不受补全影响。
- 零初始化输出逐元素等于旧 transport 输出；visibility 不乘到 flow 上。
- 不读取 future target 作为模型输入；future candidate 对 policy/action-free 隔离保持。

611 项测试通过、1 项跳过。完整 1B、BF16、activation checkpointing 的真实前后向中，factual action/token、flow、completion、visibility 与 policy 梯度均有限非零。

这些结果证明线路和不变量；下面的优化对照没有通过质量发布要求。

## 完整 1B 同条件对照

两组均 fresh seed7340、完整生产模型、RGB256/K8、生产 optimizer 和 500-step LR warmup，反复训练同一批 6 个真实样本、3 个 source，共 384 步。不是正式分布式训练，也不是 held-out 泛化评测。

基线已使用真实错配 action 排序；新组只增加上述补全分支及已有遮挡监督。耗时约 617 秒，基线约 584 秒；单卡诊断不能替代正式 16 卡吞吐基准。

| 指标（基线 → 新补全） | Bridge | Droid | RoboCasa MG |
| --- | --- | --- | --- |
| RGB L1 | 0.070923 → 0.070317 | 0.014199 → 0.015009 | 0.011770 → 0.016081 |
| 运动区 L1 | 0.153780 → 0.141202 | 0.132461 → 0.130801 | 0.030847 → 0.038698 |
| 静态区 L1 | 0.019207 → 0.026074 | 0.005957 → 0.006940 | 0.006499 → 0.009832 |
| 帧间变化保持率 | 37.8% → 46.3% | 51.1% → 59.1% | 78.6% → 99.1% |
| 帧间变化方向 cosine | 0.2127 → 0.2264 | 0.3312 → 0.2845 | 0.6277 → 0.4453 |
| Flow EPE，像素 | 1.6421 → 2.5898 | 0.5236 → 0.6205 | 0.5727 → 1.0061 |

三个 source 的静态误差和 flow EPE 均上升；RoboCasa 的运动误差和方向也退化。变化幅度增大不能替代方向及质量。Bridge 改善不构成跨来源一致改善。

所有 factual/physical-noop/真实错配的 policy_action_raw、action-free tokens 仍逐元素不变。

**结论：此补全候选未通过，不得自动启用为新正式或 5B 默认配置。不得从中再抽一帧较好的结果称“已解决”。**

结果路径：

- `/data/Minko/wm3d_motion_production_probe_20260904/ab_real384.json`
- `/data/Minko/wm3d_transport_occlusion_probe_20260904/completion384.json`
- `/data/Minko/wm3d_transport_occlusion_probe_20260904/completion384_real.log`

## 真实动作排序 q100 状态

根：`/data/Minko/wm3d_v8_real_action_rank_qualification_1b_2node16_step100_20260904`。
双机正常完成 100 步并停止；step100 COMMITTED、16 distcp、16 rank_state、梯度所有权和双机读取通过。

但 train-split 跨来源审计仍显示弱运动：Bridge 帧间保持率约 1.7%，Droid 约 7.3%，BC_Z 约 3.1%；五 source 的平均正常动作对错配 RGB gain 约 -0.00000070。单 source held-out Sailor 保持率约 3.2%，RGB L1 仍略差于 copy-last。只改排序的候选也没有足够证据正式放行。

- `eval/crosssource_seed7340.json`：五 source、十对样本，明确是 train diagnostic。
- `eval/limitedval_seed7340.json`：一个 source、四对 held-out 样本。
- `logs/heldout_audit.log`：原五 source val 审计因该闭包只有一个 source 而在评测前失败，不是 checkpoint 加载失败。

100 步本身不足以证明长期学习必然失败；这里的决定是“不宣称已通过”，不是证明任何长训都不可能收敛。

## 全量数据准备

根：`/data/Minko/wm3d_full_robot_data_20260904`。
服务：`wm3d-full-robot-closure-ready-tasks`，截至本记录仍在生成 metadata。

18 个原已审计 source 的原始权重、物理 adapter、camera 和 episode split 不变。658,278 个原始 episode 中，Bridge 6,840 个确实无可恢复任务文本的 episode 按用户授权排除，原数据保留。保留 651,438 个原始 episode；Droid 合法分段后 manifest 含 670,760 条 episode/segment，不能将其全称为独立原始 episode。

56,400 条唯一任务文本特征已完成。metadata/window index/归一化/seal 尚未全部完成，不得称全量正式已具备启动条件；18 源外原未通过 adapter 的 8 源没有悄悄加入。

采样权重未改，未重新缓存 RGB/VGGT。没有使用 node41 计算；原始数据可能通过既有 `/shared` 挂载读取。

## 下一步边界

保留本次代码和失败证据，不覆盖正在准备数据的工作树，不修改已暂停正式的 runtime，不恢复旧 checkpoint，不启动 5B。

下一次修正必须先用当前退化证据定位梯度/区域归属问题，不能仅因纯 warp 的表达缺口就假定增加补全一定会促进运动。任何新的模型动作都必须与同 seed 的现有基线对照，并分开报告静态、运动、方向和可执行 policy 能力。

全量 closure 完成不等于模型放行。模型必要验证与全量数据/NFS/16-rank preflight 都完成后，用户已授权 fresh 启动正式 1B；不能把默认 flag 打开或资格正常退出当成自动启动条件。
