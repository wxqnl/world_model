# 3D 编码器

VGGT 是离线几何观测器，不是世界模型主干，也不在正式预训练中更新。

| 文件 | 作用 |
|---|---|
| `vggt_encoder.py` | 从固定 revision 的本地 VGGT 源码/权重提取 patch、depth、point、pose 与 confidence |
| `vggt_features.py` | 三视角融合、12×12 pooling、缺失相机 mask 和 WM3D cache 格式 |

输入 `[B,T,V,3,H,W]` 会重排成 `[B*T,V,3,H,W]`：同一时刻的 head/left-hand/right-hand
可以联合建立几何，但不同时刻永远不会在 VGGT 内互相注意，因此 cache 不含未来信息。输出
保留每个视角的 token、depth、point、camera pose 和 confidence，同时生成置信度融合的
`world_tokens`。

正式 cache 要求 VGGT depth/camera/point 三个 head 都存在，token D 必须为 2048，模型和源码
revision 必须与 asset receipt 完全一致。
