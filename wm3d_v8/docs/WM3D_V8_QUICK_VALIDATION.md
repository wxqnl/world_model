# WM3D V8 快速完整验证方案

> 历史记录（旧 Stage1 ABI）：本页中的 Stage1 quick 命令与 `[*,384]` branch
> 资产不属于当前 unified Stage1 发布合同，不能复跑或提升为发布证据。当前
> 入口见 [Stage1 统一规划手册](WM3D_V8_STAGE1_UNIFIED.md)。本页仅保留当时的
> Stage0/实验审计事实，避免把历史结果改写成新实现的证据。

## 目标

本次不继续把 Stage0 训练到 100K。验证门槛收敛为：

1. V8 Stage0 从真实多源数据训练到 step1000；
2. 从完整编号 checkpoint 独立进程精确恢复到 step1020；
3. 在同一组固定真实验证样本上分别完成两类对照：formal 同轨迹 step1000→1020 用于验证精确恢复连续性；独立短 canary step100→formal step1020 用作早期能力基线，但不冒充同轨迹消融；
4. Stage0 checkpoint 严格进入 Stage1，冻结原生 3D core 和统一 action owner，只训练 action-blind planner；
5. Stage1 用同一真实 simulator outcome 同时监督 observed native future 和冻结 V8 world imagined future，先运行 0→25，再独立恢复到 step100；
6. ranking loss 直接监督最终 serving score，而不是内部近似 utility，最后在 val/test 的两类 future 上评测。

这套验证回答的是“V8 全流程是否正确、可训练、可恢复并能进入规划阶段”。它不是 100K 收敛结论，也不是 LIBERO 闭环最终分数。

## 已完成的 Stage0 证据

- 完整 checkpoint：`step_00001000.pt`；
- 独立 exact resume checkpoint：`step_00001020.pt`；
- step1020 SHA256：`ab2c6b6760197eae45f3fdc3f09e7070744121c52bcfb820db227feed476ac84`；
- V8 v3 合同：5Hz native 3D、20Hz `[B,8,7]` policy、36D causal action condition、严格 10D current-state proprio；
- formal 同轨迹固定验证报告：`v8_fixed_validation_formal_step1000_vs_step1020_v1.json`，SHA256 `e150e4db945381ee0c1c0328a4315a9c6921199f80039a01b3de37b23a007969`；
- 跨运行早期基线报告：`v8_fixed_validation_step100_vs_step1020_v1.json`，SHA256 `241ecc7dbeac446c52e15413c22c169c18f01e42d400610a816a55ef4d5df6b6`。

formal 同一训练轨迹在独立进程 exact resume 后的全源聚合结果：

| 指标 | step1000 | step1020 | 变化 |
|---|---:|---:|---:|
| unified direct-policy objective | 0.666687 | 0.659781 | -1.04% |
| coarse pose normalized L1 | 0.729704 | 0.728736 | -0.13% |
| coarse gripper accuracy | 0.950 | 1.000 | +0.050 |
| composed physical pose L1 | 0.020549 | 0.020350 | -0.97% |
| RGB L1 | 0.054823 | 0.054516 | -0.56% |
| RGB PSNR | 18.965 | 19.055 | +0.090 dB |
| pose rotation | 6.692° | 4.141° | -38.11% |
| pose translation normalized L1 | 0.528203 | 0.483168 | -8.53% |

20 个 resume step 很短，depth `+0.72%`、point `+4.64%`、native token MSE `+0.33%` 出现小幅波动。因此这份同轨迹证据支持“恢复后状态连续、action/RGB/pose 继续学习”，不声称每个世界指标在 20 步内单调下降。

下面是独立短 canary step100 与 formal step1020 的早期能力对比：

| 指标 | step100 | step1020 | 变化 |
|---|---:|---:|---:|
| unified direct-policy objective | 0.779146 | 0.659750 | -15.32% |
| coarse pose normalized L1 | 0.812406 | 0.728717 | -10.30% |
| RGB L1 | 0.063351 | 0.054517 | -13.95% |
| RGB PSNR | 18.461 | 19.055 | +0.594 dB |
| depth | 0.181337 | 0.164777 | -9.13% |
| point | 0.351804 | 0.307883 | -12.48% |
| pose rotation | 7.687° | 4.155° | -45.95% |
| native token MSE | 0.671292 | 0.599989 | -10.62% |

两者模型结构、V8 v3 action/proprio ABI 与固定样本相同，但 warmup 配方和 RoboCasa fine-stat 资产不同。因此该表只说明从短 canary 到更充分 formal checkpoint 的整体能力提升，不能解释为单一变量或严格同一轨迹的收益。

所有固定样本的 policy 输出对 teacher action 扰动保持严格零差异，而 world dynamics 对真实 action 条件保持非零响应。因此 action head 没有偷看 teacher action，world core 也没有丢掉 action-conditioned dynamics。

## Stage1 的边界

Stage1 是规划能力阶段，不改变 Stage0：

- V8 Stage0 全量参数冻结；
- planner 只读显式 future token、depth、point、pose 和任务语义；
- candidate action 只进入 V8 world dynamics，planner head 不读 action；
- action cost 使用物理 `[C,32,7]` 在 planner 外部计算；
- 真实 simulator 的 20Hz 动作经已审计 adapter 转成 V8 canonical action，每四步复合与封存的 5Hz 物理动作逐元素误差为 0，再打包为 `[C,32,36]`；
- factual-teacher 分支只提供基准证据，严格不参与候选选择。

当前 20 个真实同根 rollout 是历史 V7 候选生成器产生并在固定 RoboCasa simulator 中精确执行的结果。它们可以验证 V8 planner、V8 dynamics ABI 和 Stage0→Stage1 transition，但不能冒充“V8 action head 重新生成的候选集合”。正式大规模 Stage1 应用 V8 统一 action owner 重新收集更大的候选数据。

## 已完成的 Stage1 证据

Stage1 在 node43 的 8 张 GPU 上完成了 `0→25`，随后从独立进程严格恢复到 step100。恢复时绑定了 Stage0 checkpoint、配置、world size、run lineage、sampler step 和每个 rank 的 RNG 状态；Stage0 action owner 的两个模块哈希在训练前后保持一致。

完整 overlay：

| step | SHA256 |
|---:|---|
| 25 | `2bc77c4ce026e377d997de7a0df30cf866094937b8c2810f67cd4a52b4120be2` |
| 50 | `c61cffd890630ddc16e8d864a781afbfeb5afca25d6bc5d413e4ed78da3a38f8` |
| 75 | `203c7171c90fd0684d16a5d6fa5ab3a8a8092b3d17f7fdc28526e361904acdfa` |
| 100 | `0d141a23b1df98346a9497a833f7a543e7b6f05e07540981c203a4a26798ef29` |

所有训练 loss、gradient norm、observed/imagined 子项均为有限值，planner 梯度非零；冻结的 Stage0 没有梯度。step100 是按固定 val 上“真实 future 与 imagined future 两者中较低的 serving-score AUC 最大”选出的，test 没有参与 checkpoint 选择。20 个 branch payload 和 20 个 simulator runtime payload 均逐文件核验 SHA256；branch SHA 闭包清单的 SHA256 为 `647769deb3bdf4ee969b23690e6953b958f854040b8a9f9c9c3b34ad61d8c254`。

| split / evidence | 随机初始化 serving AUC | step100 serving AUC | 变化 |
|---|---:|---:|---:|
| train / observed true future | 0.266 | 0.939 | +0.673 |
| train / V8 imagined future | 0.321 | 0.943 | +0.622 |
| val / observed true future | 0.634 | 0.829 | +0.194 |
| val / V8 imagined future | 0.280 | 0.840 | +0.560 |
| test / observed true future | 0.346 | 0.563 | +0.216 |
| test / V8 imagined future | 0.364 | 0.623 | +0.260 |

未参与选择的 test 同时在两类 evidence 上改善，证明 planner 学习和 Stage0 imagined-future 接口都有效。test 只有 4 个 root，每个 top-1 样本相当于 25 个百分点，因此本验收以候选级 AUC 为主，不把 top-1 success 当作最终闭环分数。step100 的 imagined token fidelity 为 MSE `0.587759`、cosine `0.941314`；这说明短 Stage0 已提供可消费的预测，但尚不能替代长训练后的世界模型质量。

关键报告及 SHA256：

| 报告 | SHA256 |
|---|---|
| Stage0 formal step1000 vs step1020 fixed validation | `e150e4db945381ee0c1c0328a4315a9c6921199f80039a01b3de37b23a007969` |
| Stage0 canary step100 vs formal step1020 early baseline | `241ecc7dbeac446c52e15413c22c169c18f01e42d400610a816a55ef4d5df6b6` |
| Stage1 val random init | `87daa4d531d42e24545077b0fd46478a29be63057a9d62bf4285df7485ac1769` |
| Stage1 val step100 | `d45fe37dbe8abd98f91cf7f5d07f3bf4e623a842f108d4fe054a729fe7cea4db` |
| Stage1 test random init | `53ab482ad0c396579d3a3f9c566389dfba21a9c7f8985772f1d40e98091f51d1` |
| Stage1 test step100 | `19a699539efa3478b0ae6325784847cf6308ce7ae30b3b23bb5192945d435656` |

与本页对应的机器可读交付凭证为
[`WM3D_V8_QUICK_VALIDATION_RECEIPT.json`](WM3D_V8_QUICK_VALIDATION_RECEIPT.json)。发布前最终代码自检编译了 102 个 Python 文件，测试结果为 `140 passed`；v3 未封存模板的 static preflight 会结构化列出待填 SHA，正式 full preflight 仍会拒绝任何 `PENDING_*`。

## node43 执行

环境变量：

```bash
export NCCL_NVLS_ENABLE=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

训练第一段：

```bash
torchrun --standalone --nproc_per_node=8 \
  -m wm3d_v3.stage1_planner.train \
  --cfg configs/wm3d_v8_stage1_native_planner_quick.yaml \
  --stop-after-step 25
```

独立进程精确恢复：

```bash
torchrun --standalone --nproc_per_node=8 \
  -m wm3d_v3.stage1_planner.train \
  --cfg configs/wm3d_v8_stage1_native_planner_quick.yaml \
  --resume /data/Minko/world_model/wm3d_v8_stage0_causal_dual_view_20260809/results/stage1_native_planner_mixed_quick_v3/ckpt/step_00000025.pt
```

评测脚本：

```bash
python scripts/eval_wm3d_v8_stage1.py \
  --cfg configs/wm3d_v8_stage1_native_planner_quick.yaml \
  --overlay <完整编号overlay> --overlay-sha256 <SHA256> \
  --split test --mode both --device cuda:0 \
  --output <report.json>
```

## 交付判断

本次上述条件已经全部满足，因此代码可以作为 V8 集群扩展基线交付。这里的“通过”只表示实现、数据合同、恢复和 Stage0→Stage1 学习链路通过，不表示 1020-step Stage0 或 100-step Stage1 已达到最终性能。

验收项：

- Stage0 formal 同轨迹 fixed validation 全部 finite，恢复后 action/RGB/pose 继续改善；短 canary 到 formal checkpoint 的跨运行多任务基线总体改善；
- Stage0 exact resume 通过；
- Stage1 planner loss、梯度全部 finite 且非零；
- Stage1 exact resume 通过；
- Stage1 overlay 只含 planner 参数，Stage0 action owner 哈希不变；
- val/test 的真实 future 排序优于随机初始化，并报告 imagined-future 迁移结果；
- 明确保留“20 条候选来源于历史 proposal”的限制，不将 quick validation 当正式规模结果。
