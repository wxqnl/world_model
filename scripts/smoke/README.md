# 新服务器 Smoke

| 文件 | 作用 |
|---|---|
| `run.sh` | 串联小数据下载、处理、VGGT cache、双卡训练、checkpoint 和 eval |
| `verify_resources.py` | 检查 GPU0–1 空闲、ECC、磁盘和运行环境 |
| `report.py` | 汇总 revision、seal、参数量、指标和 checkpoint 哈希 |

运行 `./wm3d.sh smoke /abs/work-root`。这是交接后第一条 GPU 命令，使用真实公开数据和真实
VGGT，不使用随机假数据替代 pipeline。环境与数据阶段本身不占 GPU；资源不满足会在任一
GPU worker 启动前停止。成功必须同时满足训练/验证 finite、原子 checkpoint 可恢复、显式
RGB/depth/point/action 有监督，并生成 `smoke_report.json`。

无法直连 huggingface.co 时，使用标准 Hugging Face 镜像环境变量原地重试：

```bash
HF_ENDPOINT=https://hf-mirror.com ./wm3d.sh smoke /abs/work-root
```

全过程追加到 `WORK_ROOT/logs/smoke.log`，最近一次尝试的阶段和退出码原子写入
`WORK_ROOT/smoke_status.json`。重复执行会核验已完成 receipt 并从未完成阶段继续，不要删除
下载缓存或中间目录。

VGGT 源码从 GitHub 官方 codeload 获取，并同时校验固定 commit 对应的 archive SHA256 与
解包后 tree SHA256；安全解包到唯一临时目录后才原子发布，网络中断不会留下一个被误当成
正式资产的半成品目录。
