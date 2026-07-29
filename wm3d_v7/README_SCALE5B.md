# WM3D-V7 Native 5B 预训练交付包

这是 **WM3D-V7** 的原生 3D 世界模型扩展版，不是 V8、A2、Qwen、Wan 或 VLA
改造。模型本体直接负责未来 RGB、depth、3D point、camera/confidence 和机器人
动作；文本只在离线阶段编码为 V7 既有的 2048 维 task embedding。

默认配置的精确参数量是 **4,956,589,929**，正式目标是 64/128 张 H200：

- `T=24`：5 Hz 下 4.8 秒视觉上下文；
- `P=144`：每帧显式 `12×12` 原生空间 token；
- `K=16`：5 Hz 下预测未来 3.2 秒；
- 外部 token `D=2048`，world state trunk 内部宽度 `2560`；
- 32 层 state trunk、24 层 grouped-action trunk、10 个双向桥；
- action 保留 30 Hz，即每个视觉帧 6 个动作 substep；
- 三路 RGB，支持有 mask 的 proprio/force/tactile/LiDAR 辅助观测；
- FSDP2/HSDP、BF16、逐层 activation checkpoint、事务化 DCP 精确恢复；
- **不接 Wan，不使用 `latest`，不允许隐式下载或静默换数据/拓扑。**

## 先读什么

1. [架构与参数组成](docs/scale5b/ARCHITECTURE.md)
2. [数据下载、转换与 cache](docs/scale5b/DATA_PIPELINE.md)
3. [环境、集群启动与恢复](docs/scale5b/CLUSTER_RUNBOOK.md)
4. [交付验收清单](docs/scale5b/HANDOFF_CHECKLIST.md)

## 同事拿到代码后的最短路径

```bash
cd /workspace/wm3d_v7
export PYTHONPATH=/workspace/wm3d_v7

# 1. 生成并审阅精确参数预算
/opt/wm3d/bin/python scripts/scale5b/report_parameter_budget.py \
  --config configs/scale5b/wm3d_v7_native5b_h200.template.yaml

# 2. 复制 raw-source lock，填入三个数据仓库和官方转换器仓库的 40 位提交 SHA
cp configs/scale5b/raw_sources.lock.template.yaml \
  /releases/wm3d_v7_native5b/raw_sources.lock.yaml

# 3. 按 CLUSTER_RUNBOOK.md 构建独立 AgiBot-v2 转换镜像和正式训练镜像
# 4. 按 DATA_PIPELINE.md 下载、转换、schema 审计、cache、merge、seal
# 5. 按 CLUSTER_RUNBOOK.md 跑 1k canary，再启动 600k formal
```

正式训练只接受五个不可变输入：encoder asset receipt、dataset seal、code
receipt、environment receipt 和 materialized training YAML。任何一个 digest、路径、
world size、shard degree、sampler/RNG lineage 不一致都会 fail-closed。

## 已内置的关键门禁

- 精确总参数断言与未来动作 no-leak 梯度测试；
- LeRobot 单根与 collection 扫描，多包重复 episode index 自动命名空间化；
- RoboCasa 12D 与 AgiBot G2 22D grouped-action 合同；
- Alpha 官方 converter、LeRobot 0.1.0 提交和独立转换环境 receipt 的四重绑定；
- 最终 handoff manifest 同时绑定训练容器与 AgiBot converter 容器/runtime bundle；
- 数据列宽、episode 行区间、视频路径和符号链接深度检查；
- action/aux 统计、task bank、VGGT cache、索引、payload 全链路 receipt；
- 64/128 卡 HSDP 拓扑、H200 HBM、NVLink、IB、ECC、磁盘、all-reduce preflight；
- 带 optimizer、sampler、schedule、Python/NumPy/CPU/CUDA RNG 的编号 checkpoint 精确恢复。

当前无重叠规划总量约 `5,649.4h`：现有约 495h V7 formal 中约 98h 的旧
RoboCasa MG 40k 必须先剔除，再由完整 1,615h MG 替换。最终权威值只能来自冻结
快照完成后的 `source_scan.json`，不能拿规划小时数冒充实测数据量。
