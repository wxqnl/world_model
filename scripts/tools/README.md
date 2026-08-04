# 审计工具

| 工具 | 入口 | 作用 |
|---|---|---|
| `report_parameters.py` | `./wm3d.sh params site.env [YAML]` | 在 meta device 上计算精确参数组成，不分配 5B 权重 |
| `audit_v7_lineage.py` | `./wm3d.sh audit site.env` | 对照 Git 历史中的 V7 anchor，核验模型、VGGT、action、loss、正式配置和依赖边界 |

`audit` 不是文字声明：它检查 anchor commit/blob、允许的重命名后逐字一致性、V7 YAML 七个
继承段、显式输出 owner、禁用后续 Qwen/Wan/A2 依赖以及 4,956,589,929 参数预算。必须从
完整 Git clone 运行；浅克隆需要先取得仓库历史。
