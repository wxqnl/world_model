# WM3D V8 LIBERO 清理方案

日期：2026-08-10

## 目标

`wm3d_v8/libero` 是唯一维护入口。目录只保留从环境、数据下载、缓存、训练到官方闭环评测所需的代码、配置、测试和中文说明。

## 不变量

- 不停止或改动 node41 正在运行的 step30000 正式评测。
- 不删除或移动 checkpoint、数据、cache index、正式 episode、summary、binding、smoke receipt 和 launch receipt。
- 训练与评测保持四套统一模型、双视角、direct 20 Hz、H1、7D proprio、H4 action history 和 absolute gripper。
- 不恢复 delta gripper、flow serving、inverse adapter、task routing、teacher replay 或旧 V6/V7 LIBERO 路径。

## 第一阶段：评测运行期间

1. 审核 `wm3d_v8/libero` 与当前已验证实现的差异。
2. 把 checkpoint overlay loader 修复、absolute-gripper 合同和必要回归测试放入 V8 的唯一实现。
3. 将 `06_eval.sh` 改为显式 checkpoint/SHA 绑定、8 GPU、2,000 个 official episode、原子结果和断点续跑；不再默认使用不确定的 `best.pt`。
4. 修复 EGL 启动流程：逐卡预检、错峰启动 runner，并让基础设施失败与任务失败分开记录。
5. 删除 V8 LIBERO 目录内被新入口替代且无引用的文件；保留 `00_setup_env.sh` 到 `06_eval.sh` 的顺序入口。
6. 运行静态合同、单元测试和 shell/Python 语法检查。GPU canary 等当前正式评测结束后执行。

## 第二阶段：正式评测结束后

1. 确认 node41 没有进程以旧 V7 目录为 cwd，也没有从该目录启动新的 LIBERO runner。
2. 将最终 summary、checkpoint SHA 和评测 receipt 写入 V8 的结果说明。
3. 删除旧 V7 LIBERO 训练/评测脚本、旧配置、`.pre_*`、历史 smoke 和已被替代的合同测试。
4. 保留正式 checkpoint、cache 和评测结果；这些制品不属于代码清理范围。
5. 从 V8 唯一入口完成环境检查、缓存校验、CPU 合同测试、单 episode smoke 和 8 GPU 小规模 canary。

## 验收标准

- README 只描述一条 00→06 流程。
- 正式训练配置只有一个，正式 eval 入口只有一个。
- 仓库搜索不到旧 delta-gripper、旧硬编码 run root 或旧 `best.pt` 默认评测入口。
- 回归测试通过，eval 启动 receipt 精确绑定 checkpoint/config/stats SHA。
- 8 GPU canary 不依赖首次 EGL 失败重试，且每个 episode 只产生一个原子结果。
