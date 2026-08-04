# Smoke 配置

这里的 ALOHA 配置只用于在新服务器验证软件和数据链，不参与正式数据统计：

- `aloha_sources.lock.yaml`：固定 91 MB 样本 revision；
- `aloha_dataset.yaml`：最小 dataset/embodiment 契约；
- `aloha_layouts.json`：样本字段映射。

运行：

```bash
./wm3d.sh smoke /abs/work-root
```

它会真实下载、处理、缓存 VGGT、用 GPU0–1 跑一步 FSDP2、保存原子 checkpoint 并评测。
若 GPU 已被其他进程占用会 fail-closed，不会抢卡。通过标志为
`/abs/work-root/smoke_report.json` 中 `pass=true`。
