# 数据配置

本目录把“下载什么”和“如何解释字段”分开保存：

| 文件 | 作用 |
|---|---|
| `raw_sources.lock.yaml` | 上游 Hugging Face 仓库、固定 commit 与下载白名单 |
| `public_6106h.yaml` | 数据源权重、embodiment、视角、grouped action 和辅助传感器契约 |
| `public_6106h_layouts.json` | 各公开数据实际字段到统一 WM3D 字段的映射 |

`6106.4 h` 是容量规划值；正式训练只认 source scan 与 dataset seal 中的实测帧数、时长和
哈希。默认来源包括 DROID、Bridge V2、RoboCasa365 Atomic/Composite/MG、
AgiBotWorld2026 真机部分与 AgiBotWorld Beta。许可证或 gated 数据未获授权时必须停止，不能
用空目录占位。

执行顺序：

```bash
./wm3d.sh lock site.env
./wm3d.sh download site.env
./wm3d.sh prepare site.env
./wm3d.sh cache site.env
```

前三个文件的字段必须一起修改。新增来源后先补 source lock，再补 dataset source 与
embodiment，最后补 layout；随后运行测试和 smoke，不能直接进入正式训练。
