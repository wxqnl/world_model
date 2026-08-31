# WM3D V8 5B 训练交接

本文件对应 GitHub `v8` 分支。5B 与当前 1B 长训使用同一份 V8 模型语义，不是独立维护的旧
dual-path/P256 变体。

## 冻结合同

- 模型配置：`configs/model/native_5b_v8_core.yaml`
- 兼容入口：`configs/model/native_5b_dual_path.yaml`，内容必须与上述安全配置等价；新任务不要再选它
- 参数量：`5,440,933,496`
- encoder：`configs/encoder/vggt_native_p144.yaml`
- objective：`configs/objective/stage0_v8_core.yaml`
- runtime：先用 `configs/runtime/h200_64_fsdp2_canary1k.yaml`
- 时空合同：`T=24`、P144、`K=16`，RGB 监督全部 K16

future physical action 在 state encoder 之前进入独立 factual pass，并在两层独立 factual
decoder 的 query/memory 中再次注入。policy/action-free trunk 不读取 future candidate。
P144 factual future state 是运动与低频 RGB 的唯一所有者；原始 V7
`ContextResidualPixelDecoder` 直接消费该状态。高频 refiner 仅做有界晚期细节，不读取
absolute future P256、copy-last 或 future target。V8 5B 禁用 P256 AR、appearance teacher
forcing、RAFT/flow 和旧 renderer-only action 通路。

## 新鲜启动

不要使用旧 `v8` checkout 中已经生成的 site/runtime，也不要从任何旧 5B checkpoint 初始化。
拉取后重新生成 canary site：

```bash
git clone --branch v8 --single-branch https://github.com/wxqnl/world_model.git
cd world_model/wm3d

SITE=/data/wm3d/control/5b_v8_canary1k.env
./run_wm3d.sh 5b init canary1k "$SITE" direct_raw
# 编辑数据路径、模型 snapshot、许可、8 节点地址与 rendezvous
./run_wm3d.sh 5b env "$SITE"
./run_wm3d.sh 5b data-template "$SITE"
./run_wm3d.sh 5b doctor "$SITE"
./run_wm3d.sh 5b task-bank "$SITE"
./run_wm3d.sh 5b cache-plan "$SITE"
./run_wm3d.sh 5b streaming-prepare "$SITE"
./run_wm3d.sh 5b runtime "$SITE"
```

`doctor` 与 `runtime` 会执行 5B V8 语义门禁。只要 model/encoder/objective 仍指向旧
absolute-P256/teacher 路线、参数量不匹配、factual action 顺序不对或 appearance teacher ratio
非零，就会在大作业前失败。

64 卡启动方式见 `docs/WM3D_5B_SCALING.md`。先完成同拓扑 1K canary，至少覆盖真实前向、
反向、梯度所有权、编号 checkpoint、独立进程 exact resume 和固定评测；通过后再用新 site
从 step 0 启动正式训练，不能把 canary checkpoint 当正式初始化。

## 启动前核对

- runtime 中模型名为 `native_5b_v8_exact_v7_factual_high_frequency_refiner`
- 封存参数量为 `5,440,933,496`
- appearance teacher start/end ratio 都为 0
- encoder 没有 P256 appearance feature
- future action 对 policy/action-free 输出逐元素无影响
- factual decoder、RGB decoder、action head 和 policy 都有有限非零梯度
- 所有 rank 使用同一份 runtime/data seal/normalization 和训练-serving action/state calibration

这里的 1K 只验证 5B 实现、分布式状态和学习方向；真实 VLA 能力仍要由独立 action regression
与多类闭环任务评测证明，不能只靠 Stage0 token/RGB 指标下结论。
