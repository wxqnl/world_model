# WM3D-V7 Native 5B 正式交接清单

这是一份发布闸门，不是建议列表。任何空白、猜测、仅口头确认或无法复现的项目都视为
**NO-GO**。接收方应按顺序执行并把所有 receipt、SHA 和审批记录归档到同一个 release
目录。

## A. 项目边界与模型

- [ ] 代码来自 `world_model/wm3d_v7` 的审定 `v7` commit/tag。
- [ ] 明确这是 **WM3D-V7 Native 5B**，不是 V8/A2/Qwen/Wan/VLA。
- [ ] `scripts/scale5b/qualify_release.sh` 在最终容器内全绿。
- [ ] V7-only dependency guard 对全部 sealed Python/YAML/JSON 通过。
- [ ] meta-device 精确参数量为 `4,956,589,929`。
- [ ] `T/P/K/token-D = 24/144/16/2048`，state hidden=2560。
- [ ] 显式 RGB、depth、point、confidence、camera 输出开启，未退化为 latent-3D。
- [ ] grouped native action trunk 从 step 0 训练。
- [ ] future-action no-leak 梯度测试通过。
- [ ] 缺相机、缺辅助传感器和可变 action 维度 mask 的测试通过。
- [ ] 没有任何 Wan/video generator 权重、下载或训练依赖。

## B. 原始数据与许可

- [ ] 三个外部数据仓库和一个官方转换器仓库均固定为40位不可变 commit SHA：
  - `ember-lab-berkeley/robocasa365-pretrain-mg`
  - `agibot-world/AgiBotWorld2026`
  - `agibot-world/AgiBotWorld-Beta`
  - `agibot-world/AgiBotWorld-Alpha`（只下载官方 converter）
- [ ] AgiBot Beta 与 Alpha gated 许可均已由数据负责人批准。
- [ ] 内部 V7 formal 数据具备合法使用记录和冻结 manifest。
- [ ] 内部 full manifest 每行有非空、已审计的 `provenance_dataset`。
- [ ] `prepare_legacy_residual_manifest.py` 已精确剔除 `robocasa365_mg`。
- [ ] residual receipt 显示剔除/保留 episode 均非空，output SHA 与输入目录一致。
- [ ] 旧约98h MG 40k 未与完整1,615h MG重复进入合同。
- [ ] `raw_sources.lock.yaml` 不含 branch、tag、短 SHA 或占位符。
- [ ] `HF_TOKEN` 仅由 secret manager 注入，未写入 Git、YAML、命令历史或日志。
- [ ] 每个下载目录都有 `.wm3d_v7_download_receipt.json`。
- [ ] 下载 receipt 的 repo/revision/文件数/总字节与 lock 一致。
- [ ] 原始快照已做独立存储快照，并在后续阶段只读挂载。
- [ ] `simulation/` 未混入真实机器人 formal 数据。

## C. 解包、转换与 schema

- [ ] AgiBotWorld2026 三类 tar 仅通过
      `safe_extract_lerobot_collection.py` 展开。
- [ ] 所有 archive 都通过路径逃逸、link/device 和 `meta/info.json` 检查。
- [ ] 每个 archive receipt 都绑定原始归档 SHA-256；重跑时无同大小内容漂移。
- [ ] AgiBotWorld2026 的 Imitation/Rich/RL 三个 collection 均完成。
- [ ] Beta task 列表由 `list_agibot_beta_tasks.py` 从冻结快照生成。
- [ ] Beta TAR 只通过 `safe_materialize_agibot_beta.py` 分片解包。
- [ ] Beta 三类目录的 task/episode 集合与 `task_info` 精确相等。
- [ ] Beta 最终存在且核验
      `.wm3d_v7_beta_materialization_receipt.json`；无残留临时文件。
- [ ] 三个 AgiBotWorld2026 collection 均有
      `.wm3d_v7_collection_materialization_receipt.json`，并精确绑定 download
      receipt、archive 集合和所有 LeRobot root。
- [ ] Beta 数据与 Alpha 官方 converter 分别来自冻结 revision；未使用浮动 `main`。
- [ ] converter download receipt 的 repo/revision/target 与 converter 路径一致。
- [ ] AgiBot converter 使用独立、digest 固定的 OCI/SIF 镜像，不借用训练镜像。
- [ ] converter 环境为 Python 3.10.15、LeRobot 0.1.0，源码提交精确为
      `8e7d6970eaf5a64b8af6ec45586d201b8ca9ef16`。
- [ ] converter 源码归档、`pyproject.toml`、`poetry.lock` SHA 与
      `agibot_converter_environment_contract.json` 一致。
- [ ] 镜像内 `/opt/agibot-converter/environment_receipt.json` 复核通过。
- [ ] `environment_contract.json`、`environment_receipt.json` 和
      `LEROBOT_REVISION` 已作为同目录三文件 bundle 原样归档；复制后静态复核通过。
- [ ] converter 输入是完成 receipt 绑定的 Beta materialized root，不是 TAR snapshot。
- [ ] 每个 Beta task 都有转换完成 receipt；Beta materialization receipt、Alpha
      converter download receipt、converter environment receipt 和 converter SHA
      均一致，无遗失或重复 task。
- [ ] RoboCasa 与四个 collection 都运行 `inspect_lerobot_schema.py`。
- [ ] schema 报告覆盖所有 root，而不是抽样子集。
- [ ] 相机键、action/state 列、vector width、robot type 由真实 parquet 确认。
- [ ] 异构 robot/schema 已拆成独立 source/layout；没有“公共 schema 猜测”。
- [ ] RoboCasa common layout 已验证为12D分组动作。
- [ ] AgiBot G2 common layout 已验证为22D分组动作。
- [ ] force/tactile/LiDAR 只在真实存在时启用，并有逐维 validity mask。
- [ ] Beta 转换根有 `.wm3d_v7_beta_conversion_collection_receipt.json`，task 集合与
      冻结 task list 精确相等。

## D. 数据扫描、cache 与 seal

- [ ] inventory/source-layout 中所有环境变量都指向冻结、只读路径。
- [ ] `compile_dataset_contract.py` 已生成 dataset contract。
- [ ] `scan_sources.py` 已完成全部 episode 扫描。
- [ ] `source_scan.json` 中每源 train/val 非空、实测小时为正。
- [ ] 全部 parquet row range、视频路径、required column 和 vector width 通过。
- [ ] episode ID 在同源 collection 和跨源范围都唯一。
- [ ] train/val/test 以 episode 为单位、seed 固定且无交叉。
- [ ] 跨版本、跨源视频与轨迹去重报告完成。
- [ ] 规划的约5,649.4h已被实测、去重后的小时数替代并签字。
- [ ] 256个 action-stat shard 全部存在且 lineage 相同。
- [ ] action statistics merge 通过，连续维度 finite，离散维度未被错误归一化。
- [ ] task bank 已从冻结 T5 资产离线构建。
- [ ] 1,024个 VGGT/cache shard 全部有 commit receipt。
- [ ] cache 包含三视角 token、RGB、depth、point、confidence、camera、
      30Hz grouped action、contact/gripper 和全部 validity mask。
- [ ] 缺相机只发布 invalid mask，没有黑图伪造观测。
- [ ] `merge_and_seal.py` 完成且未覆盖已有正式目录。
- [ ] `verify_dataset.py --deep` 通过。
- [ ] dataset receipt、selection、manifest、index 和所有 cache SHA 已归档。
- [ ] dataset 根目录已切换为训练只读。

## E. Encoder 资产

- [ ] VGGT 源码固定 commit。
- [ ] `facebook/VGGT-1B` 和 `google/flan-t5-xl` 固定不可变 revision。
- [ ] `prepare_encoder_assets.py` 生成 portable bundle，无 symlink。
- [ ] `verify_encoder_assets.py --deep` 通过。
- [ ] 训练节点不访问 Hugging Face，也不临时下载模型。
- [ ] T5 仅用于离线 task bank，不进入正式训练图。

## F. 环境与代码 release

- [ ] AgiBot dataset-v2 converter 镜像和正式训练镜像是两个独立、digest 固定的 artifact。
- [ ] wheelhouse 在 x86_64 Linux 上按 lock 离线构建。
- [ ] CUDA base image 由 digest 固定，不使用浮动 tag。
- [ ] 容器内 Python/PyTorch/CUDA/NCCL/FSDP2/DCP 版本与 contract 一致。
- [ ] `/opt/wm3d/environment_receipt.json` 校验通过。
- [ ] 2-GPU FSDP2+DCP 保存/精确恢复 smoke 通过。
- [ ] 单节点8-GPU smoke 通过。
- [ ] 当前 V7 Native5B scope 无 dirty/untracked 文件。
- [ ] `seal_code.py` 生成 code receipt，绑定 commit 与逐文件 SHA。
- [ ] 最终 SIF/Enroot/OCI artifact 已哈希并归档。

## G. 集群与配置 materialization

- [ ] 正式拓扑为16×8 H200（推荐）或8×8 H200（最低可行）。
- [ ] 每节点8卡为完整 NVLink/NVSwitch clique。
- [ ] IB、ECC、GPU UUID、HBM、`/dev/shm`、memlock、fd limit 均通过。
- [ ] dataset 与 output 文件系统启动时各有至少10TB可用余量。
- [ ] canary 与 formal 使用不同 run name、output root 和 run lineage。
- [ ] materialized YAML 内不存在 `__MATERIALIZE_REQUIRED__`。
- [ ] config 同时绑定 code/environment/dataset/asset receipt。
- [ ] world size、shard degree、micro/global batch 和 accumulation 算术一致。
- [ ] `create_handoff_manifest.py` 生成原子 handoff manifest。
- [ ] handoff manifest 同时绑定训练容器、converter 容器和 converter 三文件
      runtime bundle。
- [ ] 每节点只启动1个 torchrun launcher，并由其派生8个 worker。
- [ ] `max_restarts=0`；禁止 elastic 自动掩盖 formal 故障。

## H. 1k 全规模 canary

- [ ] canary 使用和 formal 完全相同的模型、数据、loss 与拓扑。
- [ ] canary 从独立初始化开始，不从 smoke 或其他实验 checkpoint 恢复。
- [ ] 全集群 preflight PASS 后才进入 step 0。
- [ ] step 持续、无 OOM/CUDA/NCCL/nonfinite/Traceback/No-space/I/O/data 错误。
- [ ] peak HBM 至少留15%余量。
- [ ] source mix 每100步精确满足 `10/15/10/8/12/45`。
- [ ] token/RGB/depth/point/camera/action/contact loss 和梯度均 finite。
- [ ] 每个启用 action group 在梯度采样点都有非零 finite 梯度。
- [ ] RGB 边缘/频谱、运动区域与多视角一致性检查通过。
- [ ] step1000 DCP checkpoint 有 `COMMITTED.json`，深验可加载。
- [ ] DCP 恢复后 model/optimizer/scheduler/sampler/RNG 连续。
- [ ] steady-state seconds/step、存储吞吐和3–5周预算已由负责人签字。

## I. 600k 正式训练

- [ ] formal 采用独立初始化和独立 lineage，**不从 canary 恢复**。
- [ ] `sbatch_native5b_h200.sh` 使用已审 formal config。
- [ ] 监控覆盖 loss、梯度、source mix、吞吐、HBM、ECC、IB、磁盘和 DCP。
- [ ] 只信带 `COMMITTED.json` 的完整编号 DCP checkpoint。
- [ ] 不信 `latest` symlink，不直接编辑 checkpoint 或 sampler state。
- [ ] 未发生隐式 world-size、数据版本、代码、环境或配置变化。
- [ ] 训练预算默认600,000 optimizer steps；任何变更均形成新签字记录。

## J. 故障与精确恢复

- [ ] 先记录故障证据，禁止盲目重启或删除半成品。
- [ ] 只选择最新完整编号、深验通过的 DCP checkpoint。
- [ ] resume 前 code/environment/dataset/config/run-lineage 全部匹配。
- [ ] 恢复后首个 step 与 checkpoint 的 optimizer/scheduler/sampler/RNG 连续。
- [ ] 部分写入目录没有被误当成可恢复 checkpoint。
- [ ] world-size 或 shard-degree 变化已经过独立 qualification。

## K. 最终交给训练同事的文件

- [ ] 审定的 Git commit/tag 与源码仓库地址。
- [ ] 本清单、中文数据手册、中文架构说明、中文集群手册。
- [ ] `raw_sources.lock.yaml` 和全部许可记录。
- [ ] 全部 schema、下载、转换、scan、cache、dataset seal receipt。
- [ ] encoder-asset receipt。
- [ ] code receipt、训练 environment contract/receipt、训练容器 SHA。
- [ ] converter 容器 SHA 和 converter 三文件 runtime bundle。
- [ ] canary/formal materialized YAML。
- [ ] `handoff_manifest.json`。
- [ ] canary 审查结论、实测吞吐、预计训练时长和负责人签字。
- [ ] 值班人、升级路径、checkpoint 保留策略和存储配额。

只有 A–K 全部勾选后，才允许提交正式 600k allocation。
