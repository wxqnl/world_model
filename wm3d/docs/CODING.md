# 文档维护

- README.md 给出当前项目和入口。
- NATIVE_DIRECT_RGB.md 记录当前 RGB 实现与诊断证据。
- WM3D_NATIVE_RGB.md 为原生 RGB 概览；WM3D_SCALING.md 对照 1B/5B 容量。
- WM3D_5B_SCALING.md 是唯一 5B 操作手册；WM3D_V8_5B_HANDOFF.md 只指向该手册。
- WM3D_FROM_ZERO.md、WM3D_DIRECT_RAW.md 说明数据与训练基础流程。
- ACTION_CONDITIONING_CONTRACT.md、WM3D_GROUPED_NORMALIZATION.md 保留 Action/serving 合同。
- WM3D_STAGE1_UNIFIED.md 保留独立规划阶段。
- archive/ 只保留历史诊断，不参与当前默认启动。

命令必须与实际入口一致。不能将 meta 参数检查、本地文件识别或单批拟合写成实机资格/
长训质量通过。文档不要求读者复制摘要值，也不声称未发布的 GitHub 分支已经更新。
