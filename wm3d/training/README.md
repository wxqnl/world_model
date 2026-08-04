# 训练运行时

Loss、FSDP2、训练循环、原子 checkpoint 与 eval 的代码说明见 [`CODING.md`](CODING.md)。

| 文件 | 职责 |
|---|---|
| `config.py` | 加载物化 YAML，核验 schema、receipt 和 training contract SHA |
| `environment.py` | 环境 receipt 与当前 runtime 绑定 |
| `runtime.py` | torchrun/NCCL、HSDP device mesh、FSDP2、WSD LR 和依赖边界 |
| `loss.py` | token/RGB/depth/point/camera/grouped-action 的 native objective |
| `checkpoint.py` | DCP 分片、optimizer/scheduler/sampler/RNG 状态与原子 commit |
| `train.py` | 训练/验证循环、精确恢复、指标和硬停 |
| `eval.py` | 完整 checkpoint 的数值与可视化评测 |

5B 正式训练采用 BF16 参数、FP32 reduce、逐层 activation checkpoint 与 FSDP2/HSDP。global
batch 必须能由 world size × micro batch × accumulation 精确得到。sampler 由 optimizer step
寻址，所以恢复后 source mix 和样本地址不依赖 worker 启动时序。

checkpoint 只有在所有 shard、metadata、manifest 写完并原子发布 `COMMITTED.json` 后才有效；
恢复绝不读取 `latest` 或未提交目录。loss 只监督显式 native 输出，任何 NaN/Inf、零监督覆盖或
receipt 不一致都会停止训练。

## Checkpoint 评测

单 checkpoint 正确性门禁：

```bash
./wm3d.sh eval site.env /abs/run/checkpoints/step_XXXXXXXX
```

`eval.py` 只接受含 `COMMITTED.json` 的显式编号目录，加载同一 run 的物化配置，并验证
checkpoint metadata、dataset seal、代码/环境 receipt、参数量和 run lineage。固定 validation
sampler 后汇总以下 native 输出：

- token 与所有训练 loss；
- RGB MAE/MSE/RMSE/PSNR；
- depth、point、geometry confidence、9D camera pose；
- grouped-action mean/NLL/velocity 与 contact probability/accuracy。

任何指标非有限、监督覆盖为零、RGB/action 退化为常数，或绑定不一致都会失败。rank 0 原子
发布 `report.json` 和 `rgb_target_top_prediction_bottom.png`，已有输出不会被覆盖。

同一 run 两个 checkpoint 的回退检查：

```bash
./wm3d.sh compare-eval site.env \
  /abs/eval/step_00001000/report.json \
  /abs/eval/step_00005000/report.json \
  /abs/eval/compare_00001000_to_00005000.json
```

比较器只接受数据、代码、模型、lineage 和评测规模完全一致的报告。它用于训练健康与明显回退
检测，不替代机器人闭环成功率或跨模型 benchmark。

先用 canary 验证 loss、吞吐、梯度和恢复，再提交正式任务；不要在运行中修改物化配置。
