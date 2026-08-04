# 新服务器 Smoke

| 文件 | 作用 |
|---|---|
| `run.sh` | 串联小数据下载、处理、VGGT cache、双卡训练、checkpoint 和 eval |
| `verify_resources.py` | 检查 GPU0–1 空闲、ECC、磁盘和运行环境 |
| `report.py` | 汇总 revision、seal、参数量、指标和 checkpoint 哈希 |

运行 `./wm3d.sh smoke /abs/work-root`。这是交接后第一条 GPU 命令，使用真实公开数据和真实
VGGT，不使用随机假数据替代 pipeline。资源不满足会在启动 worker 前停止。成功必须同时满足
训练/验证 finite、原子 checkpoint 可恢复、显式 RGB/depth/point/action 有监督，并生成
`smoke_report.json`。
