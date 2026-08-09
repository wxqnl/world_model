# V8 Stage0 因果双视图操作说明

## 训练内容

该配置只修改 V8 Stage0。Stage1 不在本次变更范围内。

每个窗口包含 16 帧观测和 8 帧预测目标。缓存执行两次独立 VGGT
前向：

1. context forward 只读取 16 帧观测，结果只能进入模型输入；
2. target forward 读取 24 帧，丢弃前 16 帧结果，只保留最后 8 帧作为监督。

两次前向都使用第一帧观测相机坐标系。模型仍显式预测 RGB、depth、
point 和 pose。缓存不得把 target token 写回 context。

固定合同如下：

~~~text
schema                    wm3d_v8_stage0_causal_dual_view_v1
representation            wm3d_v8_vggt_observed_context_target_split_v1
context_future_leakage    false
target_usage              supervision_only
geometry_coordinate_frame first_observed_camera
T / P / D / K             16 / 64 / 2048 / 8
~~~

## 数据源和采样

训练混合五个事实动作源：

| source | cycle count |
|---|---:|
| OXE DROID | 35 |
| OXE Bridge | 15 |
| RoboCasa atomic | 10 |
| RoboCasa composite | 20 |
| RoboCasa MG | 20 |

每个源必须同时有 train 和 validation 数据。每个 split 的窗口数不得少于
一个分布式全局 batch：

~~~text
batch_size_per_gpu × gpus_per_node × num_nodes
~~~

preflight 会读取五源 dataset，并在样本不足、张量缺失、shape 错误或出现
非有限值时拒绝启动。

## 生成 OXE 缓存

以下命令需要分别对 DROID/Bridge 和 train/val 执行。正式数据应使用分片
参数覆盖全部窗口；--max-windows 只用于短 canary。
短 canary 还必须设置 `--max-windows-per-clip 1`，使每个 index 至少
包含 16 个不同 clip；这与训练侧 `max_windows_per_episode: 1` 保持
一致，不会把同一 episode 的多个窗口误当成一个可填满 global batch
的独立样本。

~~~bash
export WM3D_VGGT_SOURCE_ROOT=/root/wm3d_v8_runtime/vggt_a288dd0f_v1
export HF_HOME=/data/Minko/.cache/huggingface
export HF_HUB_OFFLINE=1

python scripts/cache_wm3d_v8_stage0_causal_dual_view_oxe.py \
  --manifest <source_manifest.jsonl> \
  --cache-root <sealed_oxe_cache_root> \
  --output-root <new_v8_cache_root> \
  --index <new_index.jsonl> \
  --codec <pca384_int8_strict_v2.pt> \
  --codec-downstream-report <codec_downstream_report.json> \
  --source <oxe_droid_action|oxe_bridge_action> \
  --split <train|val> \
  --num-shards <N> --shard-id <ID> --device cuda:0
~~~

## 生成 RoboCasa 缓存

Atomic、composite 和 MG 必须各自生成，--v7-source 必须与输入分区一致。
输出必须使用新目录，不能覆盖 V7 compact cache。

~~~bash
export QWEN3_VL_EMBEDDING_PATH=<local_qwen3_vl_embedding_2b_snapshot>
export WM3D_VGGT_SOURCE_ROOT=/root/wm3d_v8_runtime/vggt_a288dd0f_v1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python scripts/cache_robocasa365_v7_compact.py \
  --root <robocasa_partition_root> \
  --manifest <audited_factual_manifest.jsonl> \
  --action-audit <factual_action_audit.json> \
  --rgb-sidecar-index <partition_rgb_sidecar_index.jsonl> \
  --codec <pca384_int8_strict_v2.pt> \
  --codec-downstream-report <codec_downstream_report.json> \
  --output-root <new_v8_partition_cache_root> \
  --index <new_partition_index.jsonl> \
  --causal-dual-view --v7-source <atomic|composite|mg> \
  --num-shards <N> --shard-id <ID> --device cuda:0
~~~

生产器采用原子发布和 no-clobber 语义。目标文件已存在但内容不一致时，
命令直接失败。

## Seal 和 full preflight

seal_wm3d_v8_stage0_causal_dual_view_canary.py 合并三个 RoboCasa index，
写入只引用新缓存的 runtime config，并运行 full preflight。短 canary 还需
显式传入：

~~~text
--max-steps 20
--out-root <fresh_result_root>
--run-lineage <fresh_lineage>
~~~

seal 报告只有同时满足以下条件才可交给 launcher：

~~~text
passed=true
launch_ready=true
errors=[]
blockers=[]
~~~

启动前记录报告文件 SHA256，并执行：

~~~bash
WM3D_V8_CANARY_SEAL_SHA256=<seal_report_sha256> \
  scripts/launch_wm3d_v8_stage0_causal_dual_view_canary.sh check
~~~

check 会重新核对 runtime config、resolved config、seal SHA、输出目录、
GPU/ECC、磁盘和 full preflight。通过后把 check 改为 launch。

## Canary 验收

20-step canary 必须自然硬停，并满足：

- 五源 step 持续前进，loss 和梯度为有限值；
- direct action、flow auxiliary、native no-teacher action 和 future anchor
  都参与训练；
- RGB、depth、point、pose loss 为有限值；
- 没有 OOM、CUDA、NCCL、Traceback、I/O 或数据错误；
- step_00000020.pt 稳定、ZIP 可读，并含 model、optimizer、scheduler、
  sampler 和 RNG 状态。

Canary 通过后仍保持暂停。正式长训需要完整缓存和单独的 authority review。
任何任务都只从完整编号 checkpoint 恢复，不使用 latest。
