# WM3D 1B / 5B 扩展

1B 与 5B 共用模型、数据 ABI、训练器和 checkpoint 实现。当前模型 profile 为
`native_1b_v8_native_direct_rgb.yaml` 与 `native_5b_v8_native_direct_rgb.yaml`；
objective 共用 `stage0_v8_native_direct_rgb.yaml`。5B 操作流程只维护在
[5B 训练](WM3D_5B_SCALING.md)。

## 容量与序列

| 项目 | 1B | 5B |
|---|---:|---:|
| 实际参数 | 1,262,837,817 | 5,245,128,313 |
| 观测 T / future K | 16 / 8 | 24 / 16 |
| native token P / D | 64 / 2048 | 144 / 2048 |
| state hidden / layers | 1600 / 18 | 2560 / 32 |
| action hidden / layers | 1280 / 14 | 2048 / 24 |
| RGB | 256×256，全 K8 | 384×384，全 K16 |
| RGB decoder 参数 | 106,091,016 | 235,709,000 |
| 有界高频 refiner 参数 | 33,280 | 33,344 |

参数来自实际 profile 的模型构建。Meta 构建只确认参数和 shape，不证明 CUDA 前后向、
显存容量或训练吞吐。5B 保留其序列长度、分辨率和宽度，不能用缩小模型替代资格。

## 两种容量共用的语义

物理 future action 只进入 factual world/RGB 分支。Policy 使用 observation、task、
exact current state 和真实 action history；future candidate 改变时，policy/action-free
输出必须逐元素不变。Task modulation 与 query-only calibration 继续保留。

Grouped action、mask、semantic、embodiment 和时间戳支持不同机器人，不把所有来源
压成统一频率。Panda 的执行协议只是一个 deployment profile，不限制其他控制器。
动作单位、坐标系、夹爪定义由经过审计的 adapter 负责，训练和 serving 使用同份
action/state 归一化统计。详见 [Action 合同](ACTION_CONDITIONING_CONTRACT.md) 和
[归一化](WM3D_GROUPED_NORMALIZATION.md)。

RGB 使用原生直接输出、context residual 和 blend/motion。高频 refiner 只补晚期细节。
不存在 absolute P256/appearance AR/teacher forcing、RAFT 训练或预训练视频 decoder。
完整路径和效果边界见 [Native direct RGB](NATIVE_DIRECT_RGB.md)。

## 数据和运行时

四份输入分别负责 model、data、runtime 和 objective，物化后形成不可修改的运行配置。
现有原始视频、动作及已审计 adapter 可以共用；T/P/K 不同的 5B 必须重新准备对应窗口
和统计，不能复用 1B 的训练 metadata 或 checkpoint 初始化。

当前本地 1B 是 16×H100、micro8/global128、50K，采用 TCP。5B 交付是
64×H200、micro4/global256、600K，节点内 8 卡分片，集群要求独立 preflight。
1B 的 TCP 运行证据不能证明同事的 H200/IB 集群已经通过。

5B 的 optimizer 起点与当前 1B 一致：AdamW，start 1e-6、peak 1e-5、min 1e-6，
warmup 500，betas 0.9/0.95，weight decay 0.02。没有未经验证的自动学习率放大。
保留 5B 的 activation checkpointing、RGB 分块和 reshard 策略；实际吞吐需要实测。

只允许从同一正式 run 完整、合同一致的 COMMITTED checkpoint 恢复。
旧 profile 保留为历史 checkpoint/对照实验兼容配置，不是新训练的候选菜单。

## 质量边界

单批拟合、meta 构建、梯度接通、实际分布式训练、跨 episode 泛化和闭环 VLA 是不同证据。
5B 先做目标拓扑的真实资格，再启动 fresh 正式训练。后续仍需跨来源运动/清晰度审计，
以及 policy 物理动作回归和多任务闭环验证。

Stage1 继续使用独立的冻结 Stage0 规划流程，见
[Stage1 统一规划](WM3D_STAGE1_UNIFIED.md)。
历史扩展说明保存在 [archive](archive/CODING.md)，不作为当前启动依据。
