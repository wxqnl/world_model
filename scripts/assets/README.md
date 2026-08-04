# 编码器资产脚本

VGGT 源码、VGGT 权重和 task encoder 权重都是离线特征生产依赖，正式训练只读取其 sealed
cache，不在线更新这些编码器。

| 脚本 | 作用 |
|---|---|
| `download_encoder_assets.py` | 下载固定 revision 的源码和模型快照 |
| `seal_encoder_assets.py` | 记录完整文件集、大小和 SHA，原子发布 receipt |
| `verify_encoder_assets.py` | cache 前复核路径、revision 与内容哈希 |

这些脚本由 `scripts/pipeline.py` 和 smoke 调用。资产目录必须是普通目录，不能通过 symlink
偷换；文件集变化后旧 receipt 会失效，必须重新封存和重建对应 cache。
