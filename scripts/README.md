# 脚本目录

正常使用只需要仓库根目录的 `./wm3d.sh`。下面这些目录是它调用的流水线实现，按职责拆开，
便于排查某个阶段的问题。

| 目录 | 职责 | 由谁调用 |
|---|---|---|
| `pipeline.py` | 串联下载、处理、缓存、训练和评测 | `wm3d.sh` |
| `data/` | source lock、公开数据下载、格式转换、action 统计、VGGT cache、dataset seal | `pipeline.py`、smoke、Slurm array |
| `assets/` | 下载、封存和核验固定 revision 的编码器资产 | `pipeline.py`、smoke |
| `cluster/` | 物化训练配置、代码 receipt、集群 preflight、节点启动 | `pipeline.py`、Slurm |
| `slurm/` | action/cache 数组任务、训练和评测作业定义 | `pipeline.py` |
| `smoke/` | 91 MB ALOHA 样本的双卡端到端验证与报告 | `wm3d.sh smoke` |
| `tools/` | 模型参数组成报告 | `wm3d.sh params` |

主要调用链：

```text
wm3d.sh
├── data    → pipeline.py → data/ + assets/ + slurm/
├── train   → pipeline.py → cluster/ + slurm/
├── eval    → pipeline.py → cluster/ + slurm/
├── smoke   → smoke/run.sh → data/ + assets/ + cluster/ + smoke/
└── params  → tools/report_parameters.py
```

正式运行不要绕过 `wm3d.sh` 直接拼接这些命令；统一入口负责加载 `site.env`、固定路径、
验证 receipt，并保证恢复和封存语义一致。

各目录的输入、输出和失败恢复方式：

- [`data/README.md`](data/README.md)
- [`assets/README.md`](assets/README.md)
- [`cluster/README.md`](cluster/README.md)
- [`slurm/README.md`](slurm/README.md)
- [`smoke/README.md`](smoke/README.md)
- [`tools/README.md`](tools/README.md)

如果只是使用项目，从 `./wm3d.sh help` 开始即可；只有定位某一阶段或扩展新数据源时才需要
进入这些实现目录。
