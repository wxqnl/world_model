# FP32 小接口更新修复

本分支独立于 expert-LoRA 对照，没有新增 LoRA 参数，也没有全量解冻 WSA。
这项修复处理优化器数值问题，不宣称单独解决了全部 RGB/action 质量问题。

## 已观测证据

step100 checkpoint 中，action encoder、action head、proprio encoder 的参数和
Adam 一二阶状态都是 BF16。按当时真实保存的矩估算更新，再舍入写回 BF16：
action encoder/head/proprio 约57.8%/70.8%/98.6%的非零更新被舍入掉。
这是实际状态上的数值估算，不是下一批真实梯度回放，也不是任务质量评测。

因此，finite非零梯度不足以证明这些接口在低学习率下有效学习。
新建的机器人条件模块原本是 FP32，而重新初始化的状态投影仍是 BF16，
这种不一致会限制小接口的适配能力。

## 最小修复

显式配置：

```yaml
finetune_mode: adapters
trainable_interface_dtype: fp32
```

只将现有可训练 action encoder/head、proprio encoder、native-3D输出norm/proj
转成FP32参数，Adam状态随参数为FP32。大基座继续冻结并保持BF16；
接口前向在训练和独立serving中都使用BF16 autocast。
原层对象和已有action-time hook保留，不改变标签、输入、采样、loss或参数数量。

旧runtime缺省值model维持旧布局/行为。新的包装层有新的state-dict路径，
必须fresh资格；不能伪装为旧checkpoint的exact resume。

## 验证

- 真实Linear/LayerNorm的BF16前向在零更新时与旧层逐元素一致。
- 原forward hook保留；activation-checkpoint反传为FP32梯度。
- 连续低于BF16量化单位的Adam更新可以在FP32参数中累积；
  旧BF16参数在该数值回归中不变。
- optimizer保存/恢复保持FP32状态。
- CUDA生产设备验证须在当前GPU作业结束后运行，不能用CPU单测替代。

后续使用完整原索引、原采样、micro16/accum1/global128、原1000步日程，
fresh到100停止，与conservative-conditioning step100直接比较。
本分支不混入LoRA，以区分数值修复与额外适配容量的效果。
