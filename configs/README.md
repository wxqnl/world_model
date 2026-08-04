# 配置目录

这里存放声明式 YAML、JSON 和站点配置样例，不放训练实现。正常入口仍是仓库根目录的
`./wm3d.sh`。

| 目录 | 内容 |
|---|---|
| `cluster/` | 新集群需要填写的路径、Slurm、网络和存储参数 |
| `data/` | 正式公开数据清单、上游 revision lock 和字段映射 |
| `smoke/` | 91 MB ALOHA 小样本的端到端验证配置 |
| `train/` | 模型规模与训练超参数预设；5B 只是其中一个预设 |

训练 YAML 是模板，包含 `__MATERIALIZE_REQUIRED__` 的文件不能直接交给 `torchrun`。
`scripts/cluster/materialize_config.py` 会把 dataset seal、代码/环境 receipt、world size、batch
和 run lineage 写入新的不可变配置，然后 Slurm 才会提交训练。

常用检查：

```bash
./wm3d.sh plan site.env
./wm3d.sh audit site.env
./wm3d.sh params site.env configs/train/5b_h200.yaml
```

各子目录的输入、输出和修改边界见其 README。
