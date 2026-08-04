# 训练运行时

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

评测入口：

```bash
./wm3d.sh eval site.env /abs/run/checkpoints/step_XXXXXXXX
```

先用 canary 验证 loss、吞吐、梯度和恢复，再提交正式任务；不要在运行中修改物化配置。
