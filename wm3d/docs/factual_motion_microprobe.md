# Factual motion 快速验证

保留 `scripts/tools/run_factual_motion_microprobe.py`。它适合查 action 接线、mask、
future/policy 隔离与梯度问题；其小模型、单轨迹重复拟合不能决定正式 1B 是否能长训。

## 结构检查

用 micro-probe 的 structural 模式跑一次真实样本前后向，确认 factual/RGB 路径有
有限非零梯度，换 future candidate 不改变 policy/action-free 输出。
这只验证计算接通，不验证运动泛化。

## 生产尺寸对照

`scripts/tools/run_production_flow_loss_ab.py` 使用实际 1B/K8/256px 模型，fresh 初始化，
读取生产数据准备器保存的真实 batch。每 source 至少两个样本才能构造兼容错配动作。
不读 checkpoint，不缩模型，不改标签、时间戳和采样权重。

例如在服务器项目目录内，用同一 runtime 与同一批真实输入分别运行：

```bash
python scripts/tools/run_production_flow_loss_ab.py \
  --runtime /path/to/runtime.yaml --batches /path/to/real_batches \
  --output /path/to/baseline.json --steps 384

python scripts/tools/run_production_flow_loss_ab.py \
  --runtime /path/to/runtime.yaml --batches /path/to/real_batches \
  --output /path/to/corrected.json --steps 384 \
  --pixel-units --real-negative-every-step
```

对照保持相同初始化、生产 LR/warmup、optimizer 和梯度裁剪。报告各 source 的
normal/no-op/mismatch、motion/static 误差、flow 方向/幅值、帧间变化和隔离不变量。
样本在训练与检查中重复使用，因此结果只能说明局部可学习性与修改方向。

## 分布式资格与长训

生产尺寸诊断没有结构失败后，用同一代码和目标运行真实 16-rank 短资格，检查资源、
梯度所有权、完整 COMMITTED checkpoint 与跨机读取。资格所用数据范围应明确标注。
正式训练只允许使用已完成的全量可用数据闭包，不能自动沿用资格子集。

长期效果仍需跨 episode、跨 source checkpoint 评估。policy/VLA action 与
world candidate action 必须分开报告，token gain 不能替代可执行动作质量。
