# 审计工具

| 工具 | 入口 | 作用 |
|---|---|---|
| `report_parameters.py` | `./wm3d.sh params site.env [YAML]` | 在 meta device 上计算精确参数组成，不分配 5B 权重 |
| `audit_v7_lineage.py` | `./wm3d.sh audit site.env` | 对照 Git 历史中的 V7 anchor，核验模型、VGGT、action、loss、正式配置和依赖边界 |
| `compare_eval_reports.py` | `./wm3d.sh compare-eval site.env BASE CAND [OUTPUT]` | 核验两个 eval 报告可比，并拦截 native 指标明显回退 |

`audit` 不是文字声明：它检查 anchor commit/blob、允许的重命名后逐字一致性、V7 YAML 七个
继承段、显式输出 owner、禁用后续 Qwen/Wan/A2 依赖以及 4,956,589,929 参数预算。必须从
完整 Git clone 运行；浅克隆需要先取得仓库历史。

`compare-eval` 只比较 `wm3d_v7_checkpoint_eval_v1` 报告。它先要求两边的 dataset seal、
training contract、代码 receipt、参数量、run lineage、world size 和 eval steps 完全一致，
然后比较 total loss、RGB、depth、point、geometry confidence、camera、action 与 contact。
不满足绑定关系或超过回退阈值时返回非零退出码；若给出 OUTPUT，则以原子、拒绝覆盖方式写入
可哈希 JSON。
