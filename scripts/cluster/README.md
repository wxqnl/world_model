# 集群控制脚本

| 脚本 | 作用 |
|---|---|
| `seal_code.py` | 封存本次训练使用的代码和配置文件集 |
| `materialize_config.py` | 把模板、dataset seal、环境/代码 receipt 和拓扑绑定成最终配置 |
| `preflight_cluster.py` | 核验 GPU、NVLink、IB、磁盘、共享内存、receipt 和数据 |
| `launch_torchrun_node.sh` | 每节点启动一个 torchrun launcher |
| `launch_eval_node.sh` | 对完整编号 checkpoint 启动评测 |

调用顺序是 `seal_code → materialize_config → preflight_cluster → launch`。正常操作由
`./wm3d.sh train site.env` 完成，不手工跳过前置门禁。preflight 失败时先修复报告中的具体
条件，再重新运行同一阶段；不要改低阈值绕过。
