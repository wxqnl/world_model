# 数据运行时

| 文件 | 职责 |
|---|---|
| `contracts.py` | dataset、embodiment、seal、哈希和安全路径契约 |
| `sources.py` | 公开来源扫描和统一 episode plan |
| `action.py` | 15–30 Hz grouped action 对齐、mask 与 robust normalization |
| `codec.py` | cache 张量的类型、形状和无损读写契约 |
| `dataset.py` | 读取 sealed row groups，构造 T24 上下文与 K16 未来监督 |
| `sampler.py` | 按 optimizer step 寻址的确定性 source mix 与精确恢复 |
| `assets.py` | 编码器资产 receipt 的加载与深度核验 |

正式样本的核心形状：

- `world_tokens`: `[T, V, P, 2048]`，5 Hz；
- future RGB/depth/point/camera/confidence：显式监督；
- action: `[T+K, G, S, A]`，其中 G/A 可变并由 mask 标明，S=6 对应 30 Hz；
- auxiliary token：proprio/force/tactile 等按 modality type 和 validity 编码。

未来 factual action 只用于条件化 future world query；policy/action trunk 输入只包含 context
action，防止目标动作泄漏。dataset seal 之外的文件、缺字段样本或非 finite 值均 fail-closed。
