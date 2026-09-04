# 训练诊断与报告

保留现有轻量预实验和真实生产验证，不以删除测试代替修复。

| 入口 | 用途与边界 |
|---|---|
| `run_factual_motion_microprobe.py` | 历史 transport 的轻量断路、mask、梯度与隔离检查；缩小模型的结果不能放行当前 native-direct 正式训练 |
| `run_production_flow_loss_ab.py` | 完整生产模型、真实批次、optimizer/LR 的 RGB/Action 对照；保留历史文件名，支持 native direct；重复样本拟合不代表泛化 |
| `export_production_rgb_ab.py` | 按真实 K 顺序导出 target、prediction、copy-last、noop、wrong-action；不插帧或伪造运动 |
| `audit_action_owned_transport_checkpoint.py` | 对真实 checkpoint 做同样本审计；支持 native direct，不强制加载 RAFT |
| `run_action_owned_transport_gate.py` / `run_exact_v7_fullsequence_gate.py` | 历史归因对照，不是新训练入口 |
| `report_5b_run.py` | 汇总数据、运行、吞吐、梯度、checkpoint 和评测；不能用文件存在替代运行证据 |

当前模型与目标见 ../../docs/NATIVE_DIRECT_RGB.md；5B 操作见 ../../docs/WM3D_5B_SCALING.md。
使用工具时传入对应 runtime/batch，不能沿用脚本中的旧机器路径当作当前正式数据。
运行报告默认只读；测试和评测不能占用正在正式训练的 GPU。

移除的 `test_v7_aligned_rgb_single_step.py` 是无引用的一次性旧实验入口。
其生产 RGB/Action 检查由上表中的真实 A/B 和 checkpoint 审计覆盖。
