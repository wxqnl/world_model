# WM3D 数据准备与训练入口

5B 按 [5B 训练](WM3D_5B_SCALING.md) 操作；1B 站点模板见
[1B direct_raw](WM3D_1B_STREAMING.md)。当前不预先生成全量视觉 cache。

## 已下载数据

现有原始视频和模型权重可以复用，来源不限于 Hugging Face。需要区分两种状态：

- 已审计 data profile：检查真实路径后，生成本模型的 task bank、窗口、归一化统计和 runtime。
- 只有原始下载目录：先核对数据结构与 adapter 语义；目录可识别不代表动作字段已正确转换。

5B 的本地检查通过 `./run_wm3d.sh 5b configure MODEL_ROOT DATA_ROOT` 完成。
它不会下载、训练，也不会替未审计的数据猜单位或坐标系。

## 数据准备在做什么

1. `schema-audit` 读取真实字段、shape、camera 和时间戳。
2. `adapter-audit` 核实 action/state 的单位、坐标系、夹爪语义、group 和时钟。
3. `inventory` / `collection-inventory` 绑定 episode 与正确 camera 视频分片。
4. `data-profile` 固定来源、split、权重和 adapter。
5. `task-bank → cache-plan → streaming-prepare` 生成任务编码、窗口索引和训练集归一化统计。
6. `runtime → preflight` 检查模型、数据、环境和真实分布式资源是否一致。

命令参数通过对应入口的 `--help` 查看；5B 的完整命令统一放在交付文档中。
模板不等于已审计数据，不能把模板中的占位路径直接用于训练。

这不是为了让 demo 更好看而重采样，也不修改原始 action/RGB。
已有来源权重、episode split 和物理转换保持不变。空任务文本、错误视频绑定、
缺失或损坏资产必须明确修复或记录排除，不能生成虚假标签。

## 哪些产物不能直接复用

原始数据与经审计的 adapter 可以复用；窗口和统计必须匹配模型的 T/P/K、时间戳、
有效维与训练 split。1B 的 T16/P64/K8 不等于 5B 的 T24/P144/K16，不能只改模型路径
却继续使用旧 metadata seal。更新配方后生成新 runtime，不手改运行中的封存文件。

旧 episode-cache、Beta converter 和手工数据导入命令保存在
[历史数据流程](archive/WM3D_FROM_ZERO_LEGACY.md)，仅供维护旧数据使用，不是默认训练路线。
