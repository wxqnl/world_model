# 模型：V7 native-3D 血统与 5B 预设

逐模块代码、张量形状、防泄漏路径和设计取舍见 [`CODING.md`](CODING.md)。

## 可核验血统

当前 `wm3d.py` 不是从 V8、Qwen 或 Wan 改写而来。它来自本分支清理前的 V7 native 5B
实现，anchor 为：

- commit：`7241146891a61225a1c38947c57193967a9c11e9`；
- source：`wm3d_v7/wm3d_v3/models/native5b.py`；
- Git blob：`225b34e9ef65895398ceb97e1c57ba164929cb77`。

当前 855 行 core 在把 `Native5BConfig/NativeWM3D5B` 改成通用
`WM3DConfig/WM3D`、去掉文件名中的规模标签后逐字一致。VGGT features、grouped-action 对齐、
native loss 与正式 YAML 也由同一 anchor 核验。运行：

```bash
./wm3d.sh audit site.env
```

需要明确：这是 **V7 原生设计的 5B 扩展**，不是旧 V7 1B checkpoint 的二进制同形复制。
T/P/K、hidden、层数与 grouped-action 接口已按 5B 计划扩大，因此旧 1B 权重不能假装
shape-compatible 直接加载；继承的是 V7 的 native-3D ownership、因果语义和训练目标。

## 模型所有权

```text
三视角显式 3D cache ──► MultiView fuser ──► State trunk（T+K, 12×12 lattice）
                                                  │
context grouped action ──► Action trunk ◄──10×──►│
        ▲                                         │
        │ learned future query                    ├─► future 3D tokens
        └─────────────────────────────────────────┼─► RGB
future factual action ──► future world query only ├─► depth / point / camera / confidence
                                                  └─► grouped action distribution
```

- state trunk 持有未来世界，使用帧内空间 attention 和同 patch 因果时间 attention；
- action trunk 从 step 0 独立训练，拥有 grouped robot action；
- 10 个双向 bridge 只让 action 读取 context state，避免 future factual action 泄漏到 policy；
- RGB、depth、point、camera、confidence 都由 native future state 显式预测；
- 没有语言模型、VLA action head、latent-3D owner 或视频生成器依赖。

5B 默认精确参数量为 4,956,589,929，其中 state trunk 65.5860%、action trunk
24.1189%、bridge 8.5688%。详细配置见 `configs/train/README.md`。
