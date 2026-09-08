# V10：保留预训练 Action 接口

本分支只修正 fresh 初始化，不改变模型拓扑、物理动作标签、归一化、数据比例或 loss。

此前把 WSA ActionDiT 的输入／输出从 24 维扩到 WM3D 的 42 维时，上游加载器会跳过形状不匹配的整个矩阵。冻结 Transformer 主体后，这相当于重新学习进出基座的动作接口。

现在 `pretrained_interface_init: pretrained_action_prefix` 保留 WSA action encoder 和 head 的已有行／列及 bias。新增输入列、输出行和 bias 置零，仍由原 action objective 训练。参数量和前后向计算量不变。

这只是参数初始化先验，不是把所有机器人的物理语义假定为一致，也不把 WSA 的归一化套到 WM3D 数据上。proprio 的新物理状态布局不按索引复制；对照没有支持把旧 state 权重整体搬过来。

旧 materialized runtime 没有这个字段时仍使用 `legacy_skip`，保证旧 checkpoint 的评测／恢复行为不变。新模式只用于独立 fresh 资格实验。已经训练完的 1000 步实验不改写、不自动续训。

## 2026-09-08 的证据

对已有验证集的 17 个 source、34 个真实窗口，同 seed、20 步去噪、零 optimizer 更新：

- 随机重建不匹配接口：Action 归一化 MAE 1.026895。
- 只保留 pretrained action 权重：0.920570，降低约 10.4%。
- 只复制旧 state 权重：1.024325，没有同样收益。
- action 和 state 都复制：0.922283，没有超过 action-only。

这证明初始化可以减少不必要的重学，不证明 34 个样本覆盖了所有机器人，更不证明闭环任务能力。部分 source 仍退化，新的语义适配仍需训练。

另一组 step1000 对照显示，直接关闭 robot conditioning 会让 Action MAE 从 0.723485 上升到 0.901057；只给视频保留原始文本也会破坏部分场景。因此没有删除 native-3D、history 或 robot conditioning，没有修改文本路由。

RGB 问题尚未验收通过。保持当前联合生成路径，用与旧实验完全相同的完整训练索引、micro16、8 卡、accum1、global128、学习率和采样 seed，先跑到 step100 后停止并比较。不得把此初始化修正表述为 RGB 和 Action 已全部解决。

实验记录：
/data/Minko/wm3d_v10_formal_adapter_8h100_accum1_1000steps_20260908/eval/adapter_diagnosis_20260908
/data/Minko/wm3d_v10_formal_adapter_8h100_accum1_1000steps_20260908/eval/adapter_routing_20260908
/data/Minko/wm3d_v10_formal_adapter_8h100_accum1_1000steps_20260908/eval/interface_initialization_20260908

注：第一组中的 native33 命名只是排查时的窗口长度探针，不能作为官方训练窗口基线。官方 dataset 的 33 个原始帧经 stride 4 后实际输入 9 帧；正式对照使用 9 帧版本。官方参考没有提供 proprio，不能作为相同物理 ABI 下的 Action 能力对照。
