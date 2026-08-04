# Slurm 作业模板

| 文件 | 作业类型 |
|---|---|
| `sbatch_action_stats_array.sh` | 分布式 action 统计 |
| `sbatch_encode_array.sh` | VGGT cache array |
| `sbatch_task_bank.sh` | task bank 构建 |
| `sbatch_train.sh` | canary/正式 FSDP2 训练 |
| `sbatch_eval.sh` | checkpoint 评测 |

这些文件不是站点配置；partition、account、节点数、路径和 receipt 由 `pipeline.py` 传入。
每节点只启动一个 launcher，再由 torchrun 创建 8 个主 worker。修改集群参数应优先编辑
`site.env`，不要复制出一批站点专属 sbatch 文件。

训练退出后只从带 `COMMITTED.json` 的最高编号 checkpoint 恢复。Slurm requeue 或节点故障
不会赋予未提交目录合法性。
