# Python 包

按一次样本的真实调用顺序阅读代码，见 [`CODING.md`](CODING.md)。四个子目录的实现细节分别写在各自的
`CODING.md` 中。

`wm3d` 是正式训练会封存进 code receipt 的运行时包：

```text
公开 RGB/action/state
        │
        ▼
data ──► encoders（离线 VGGT cache）
        │
        ▼
models（原生显式 3D world state ↔ grouped action）
        │
        ▼
training（native loss、FSDP2、精确恢复、eval）
```

| 目录 | 职责 |
|---|---|
| `data/` | 数据契约、字段映射、grouped action、数据集和可恢复 sampler |
| `encoders/` | 固定 VGGT 资产到显式 3D cache 的无未来泄漏编码 |
| `models/` | WM3D 原生 world/action 主干与显式输出头 |
| `training/` | loss、配置校验、FSDP2、checkpoint、训练和评测 |

包内代码不读取 `site.env`，也不写死集群路径；这些由外层 pipeline 物化后传入。修改任何目录
都必须重新 seal code，并至少运行单测、参数审计和 smoke。
