# 训练预设

| 文件 | 用途 |
|---|---|
| `5b_h200.yaml` | 64/128 张 H200 的正式 5B 预训练模板 |
| `5b_h200_canary.yaml` | 与正式架构同构的短 canary |
| `5b_smoke.yaml` | 双卡软件链验证，不代表正式吞吐 |

5B 是一次训练规模，不是项目名，也不是独立代码分支。模型类、数据加载器、FSDP2、恢复和
评测均是通用 WM3D 实现；定义其他规模时复制一个 YAML，修改宽度/层数/时空参数，并同步
冻结 `model_budget`。

正式 5B 模型段来自 V7 native 配置：T24、P144、K16、D2048，state trunk
2560×32，grouped-action trunk 2048×24，10 个双向 bridge。当前文件与清理前 V7 anchor 的
`model/data/distributed/optimizer/schedule/train/loss` 七段逐项相同；新增内容只有通用 schema、
精确参数预算和 H200 硬件门禁。可重复核验：

```bash
./wm3d.sh audit site.env
./wm3d.sh params site.env configs/train/5b_h200.yaml
```

模板不能直接启动。`materialize_config.py` 会绑定 dataset/code/environment receipt、拓扑、batch
和 run lineage。任何形状变化都必须重新运行参数审计、单测、smoke 和同构 canary。
