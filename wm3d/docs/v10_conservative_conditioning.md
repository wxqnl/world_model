# V10：conditioning 保守更新对照

本分支是资格实验，不是正式训练推荐配置。尚未证明 RGB 或真实机器人任务已经达标。

## 已完成的对照

保留 WSA action encoder/head 的预训练参数前缀后，同数据、同 seed 的 fresh step100：

- 验证 action loss 从 1.987800 降到 1.008654，但 video loss 从 0.127337 升到 0.142672。
- 17 个 source、68 个真实窗口的 H1 归一化 MAE 从 0.979754 降到 0.758529，但物理平移 MAE 从 0.108577 米升到 0.117221 米；没有整体通过。
- Furniture 同样本的帧间变化保持率从 47.49% 降到 8.61%。原始 PNG 同样静止，不是 GIF 导出问题。
- 旧对照的 47.49% 也不代表运动方向正确，不能把恢复这个数值称为解决 RGB。

固定 step100、固定去噪 seed，仅在内存中互换参数所有者的只读对照显示：

- 新模型恢复旧 action 接口，Furniture 保持率仍为 8.54%。
- 新模型恢复旧 robot conditioning，保持率恢复到 47.35%。
- 旧模型换入新 robot conditioning，保持率降到 8.70%。
- 单独互换 proprio、native 3D 或 action time 没有同样效果。

因此，这次新增退化随 robot conditioner/fusion 的训练结果转移。该证据只定位本轮退化，不证明所有历史 RGB 问题都已解释。

## 本次唯一训练配置改动

在 runtime profile 的 optimizer 中增加可选字段：

```yaml
robot_conditioning_lr_scale: 0.1
```

只有现有 `foundation_robot_conditioning` 所有者使用原学习率的十分之一；其他接口仍使用原学习率。每一步调度都保留比例，不新增 loss、head、模型参数或输入路径，不改变数据、采样、batch、标签、归一化和随机状态。

缺省值 1.0 返回原来的平坦 optimizer 参数列表，保留旧 checkpoint 的 optimizer 布局。显式分组的实验必须 fresh 启动；不要把旧单组 optimizer checkpoint 当作新两组 checkpoint 恢复。

仍使用 action-prefix 初始化，与刚完成的 qualification 直接比较，不把两个改动同时与旧实验混成一个结论。WSA/Wan 主体继续冻结；native 3D、history、task、state 都保留。

## 验收

使用完整原训练索引，8 H100、micro16、accum1、global128，原 1000 步学习率日程，先在 step100 停止。复用既有 68 个 held-out 窗口和四个多视角场景，比较：

- 归一化及物理动作误差，尤其首个实际执行动作；gripper 切换样本不足时明确不作结论。
- 原始全部九帧的运动方向、帧间变化、运动区误差及物体稳定性；不只看整体 L1。
- task/vision/state/history/calibration 依赖、future 隔离、serving 一致性。

只有功能改善才可进一步训练；当前降低 conditioning LR 仍是假设，不是已经验证的修复。

证据目录：

- /data/Minko/wm3d_v10_action_prior_qualification_8h100_g128_20260908/eval/paired_step100
- /data/Minko/wm3d_v10_action_prior_qualification_8h100_g128_20260908/eval/module_swaps_step100
