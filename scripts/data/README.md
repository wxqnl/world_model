# 数据处理脚本

本目录实现 `lock → download → prepare → cache → seal`，由 `scripts/pipeline.py`、smoke 和
Slurm array 调用。

| 阶段 | 主要脚本 | 输出 |
|---|---|---|
| 固定来源 | `resolve_source_lock.py` | 每个公开仓库的 40 位 commit 与文件白名单 |
| 下载 | `download_raw_snapshots.py` | 可断点续传的固定 revision 快照和下载 receipt |
| 安全展开 | `safe_extract_lerobot_collection.py`、`safe_materialize_agibot_beta.py` | 无路径逃逸的 materialized 数据 |
| Schema 审计 | `inspect_lerobot_schema.py`、`scan_sources.py` | RGB/action/timestamp/episode 实测证据 |
| Beta 转换 | `list_agibot_beta_tasks.py`、`convert_agibot_beta_task.py` | 官方工具转换后的标准任务分片 |
| 统一契约 | `compile_dataset_contract.py` | episode plan、split 与 embodiment contract |
| Action | `build_action_stats.py` | grouped-action robust normalization |
| Task | `build_task_bank.py` | 固定 task embedding bank；`--backend smoke` 只供小样本 |
| 3D cache | `cache_vggt_shard.py` | 三视角 VGGT token、depth、point、pose、confidence |
| 发布 | `seal_dataset.py`、`verify_dataset.py` | 无缺失/重复且全哈希绑定的 dataset seal |

正式训练只读取 seal 覆盖的文件。worker 可以按相同 shard 参数安全重试；merge/seal 不能并发
启动两份。失败时保留 worker receipt 和临时文件用于定位，不要手工拼 manifest 或删除坏片后
继续训练。
