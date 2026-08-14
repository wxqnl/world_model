# WM3D 从零数据、训练与评测

本文给出空服务器到训练 checkpoint 的正式顺序。所有步骤都通过 `run_wm3d.sh` 进入；1B 与 5B 只替换 model/runtime profile 和 window/runtime 输出目录。

## 1. 环境

```bash
export ROOT=/cluster/project/wm3d
export WORK=/cluster/datasets/wm3d
export RAW=$WORK/raw
export AUDIT=$WORK/audit
export INVENTORY=$WORK/inventory
export CACHE=$WORK/episode_cache
export RUNS=$WORK/runs
cd "$ROOT"
./run_wm3d.sh env
source .venv/bin/activate
install -m 600 /dev/null "$WORK/hf.token"
```

在编辑器中把 Hugging Face token 写入 `hf.token`；不要把 token 放入 YAML、仓库或 shell history。

## 2. 锁定并下载

先在每个上游页面确认许可，然后解析 mutable ref，冻结 40 位 commit 和完整 file list：

```bash
./run_wm3d.sh lock-resolve \
  --template configs/sources/public_sources_5649h_legacy_compatible.template.yaml \
  --output "$WORK/public_sources.lock.yaml" \
  --token-file "$WORK/hf.token" \
  --confirm-licenses YES_I_HAVE_ACCEPTED_THE_UPSTREAM_LICENSES

./run_wm3d.sh download \
  --lock "$WORK/public_sources.lock.yaml" --raw-root "$RAW" \
  --token-file "$WORK/hf.token" --max-workers 32
```

网络中断可原命令重跑。下载 receipt 绑定 lock SHA、revision、file-list SHA、文件数和总字节；未知 revision、无权限或缺文件都会阻断。

## 3. AgiBotWorld2026：一个快照拆成三个 source

冻结快照必须按 `ImitationLearning`、`RichInteraction`、`ReinforcementLearning` 分别生成 collection。下面命令对 imitation 执行 32 路并行；另外两路只需替换 prefix/output：

```bash
OUT=$WORK/materialized/agibot_2026_imitation
mkdir -p "$OUT"
for WORKER in $(seq 0 31); do
  ./run_wm3d.sh archive-collection \
    --snapshot-root "$RAW/agibot_world_2026" \
    --download-receipt "$RAW/receipts/agibot_world_2026.json" \
    --download-source agibot_world_2026 --source-prefix ImitationLearning \
    --output-root "$OUT" --worker-index "$WORKER" --worker-count 32 &
done
wait
./run_wm3d.sh archive-collection \
  --snapshot-root "$RAW/agibot_world_2026" \
  --download-receipt "$RAW/receipts/agibot_world_2026.json" \
  --download-source agibot_world_2026 --source-prefix ImitationLearning \
  --output-root "$OUT" --finalize
```

archive 第一次读取时只哈希一次；续跑/finalize 校验 receipt、精确 archive closure 和大小，不重复扫描多 TB payload。三个 `collection_receipt.json` 分别绑定下载 receipt 和 prefix，后续对应三个训练 source。

## 4. AgiBot Beta 的版本锁定转换

Beta 上游要求使用 AgiBot Alpha 的 `scripts/convert_to_lerobot.py`。该 converter 已在 source lock 中单独冻结。正式 conversion receipt 必须同时绑定 Beta download receipt、converter commit/文件 SHA、独立 converter venv receipt 和转换后 LeRobot root closure。

仓库不会在没有真实 Beta 快照时猜 action/state 语义，但下载→官方转换本身有可执行 runner。先从冻结 `task_info/task_*.json` 生成去重 task-id 文件，再为官方 converter 建立独立 Python 3.10/LeRobot-v2 环境并写出 environment receipt。converter 环境的精确 LeRobot commit/package 版本必须来自冻结官方 converter 的 import/API 审计，不能沿用当前训练 venv。把 receipt SHA 填入一份新的 converter contract（禁止直接改模板）：

```bash
./run_wm3d.sh beta-task-list \
  --raw-root "$RAW/agibot_beta" \
  --download-receipt "$RAW/receipts/agibot_beta.json" \
  --output "$WORK/agibot_beta_task_ids.txt" \
  --receipt "$WORK/agibot_beta_task_ids.receipt.json"

cp configs/converters/agibot_beta_official.template.yaml \
  "$WORK/agibot_beta_official.yaml"
# 将 environment_receipt_sha256 替换为实际 64 位 SHA 后执行：
./run_wm3d.sh external-convert \
  --contract "$WORK/agibot_beta_official.yaml" \
  --input-root "$RAW/agibot_beta" \
  --input-download-receipt "$RAW/receipts/agibot_beta.json" \
  --converter-root "$RAW/agibot_alpha_converter" \
  --converter-download-receipt "$RAW/receipts/agibot_alpha_converter.json" \
  --environment-receipt "$WORK/agibot_converter_env/environment_receipt.json" \
  --python-bin "$WORK/agibot_converter_env/bin/python" \
  --binding-file task_id="$WORK/agibot_beta_task_ids.txt" \
  --output-root "$WORK/materialized/agibot_beta"
```

runner 对每个 task 调用冻结官方 CLI：`--src_path INPUT --task_id ID --tgt_path OUTPUT/jobs/N`，并在全部成功后验证每个输出的 `meta/info.json` closure，再原子发布 conversion/collection receipt。converter commit、文件 SHA、输入/转换器 download receipt、环境 receipt、task-id 文件 SHA、精确 argv 和输出 root 全部进入 receipt。若上游 revision/CLI/env 不兼容，转换会明确失败且不发布正式 collection；不得改成猜测 adapter。

## 5. Schema audit 与第一处人工确认

普通 LeRobot source 使用 `--root "$RAW/droid"`；collection 增加 `--collection`。示例：

```bash
mkdir -p "$AUDIT"
./run_wm3d.sh schema-audit \
  --root "$WORK/materialized/agibot_2026_imitation" --collection \
  --max-roots 1000000 --max-data-files 2 --max-video-files 8 \
  --require-homogeneous \
  --upstream-receipt "$WORK/materialized/agibot_2026_imitation/collection_receipt.json" \
  --candidate-output "$AUDIT/agibot_2026_imitation.candidate.json" \
  --output "$AUDIT/agibot_2026_imitation.schema.json"
```

命令只列出实际字段、Arrow shape、view 候选、上游 receipt 和每个 root 的 schema signature，不生成正式 adapter。

这是全流程第一处必须由人确认的位置。根据真实样本和数据文档填写 adapter YAML，逐项确认 canonical view、group/action/state column、单位、scale/offset、坐标系、旋转组合、gripper 极性、真实时钟以及 fine/coarse supervision。然后执行：

```bash
ADAPTER=$WORK/adapters/agibot_world2026.yaml
ADAPTER_SHA=$(sha256sum "$ADAPTER" | awk '{print $1}')
./run_wm3d.sh adapter-audit \
  --schema-audit "$AUDIT/agibot_2026_imitation.schema.json" \
  --adapter-candidate "$AUDIT/agibot_2026_imitation.candidate.json" \
  --adapter-contract "$ADAPTER" --adapter-contract-sha256 "$ADAPTER_SHA" \
  --data-template configs/data/public_robot_6106h.template.yaml \
  --source agibot_2026_imitation --operator "$USER" \
  --confirm I_VERIFIED_FIELDS_UNITS_FRAMES_GRIPPER_GROUPS_AND_NATIVE_CLOCKS \
  --output "$AUDIT/agibot_2026_imitation.adapter_receipt.json"
```

adapter 引用的字段、列、view 或 group 与任一 root 不一致即失败。rich/RL/Beta 各自需要 receipt；可复用经过确认的 adapter YAML，但不能复用别的 source receipt。

## 6. Inventory 与 data profile

### 6.1 默认交付：legacy-compatible 5649.4h

默认正式模板是 `configs/data/public_robot_5649h_legacy_compatible.template.yaml`。它保留
V7 已交付的六个数据家庭、小时预算和整数采样周期，但不复用 V7 action/state cache。
其中 397 小时 residual 必须先运行：

```bash
./run_wm3d.sh legacy-residual-import \
  --legacy-plan "$WORK/v7/legacy_residual_plan.jsonl" \
  --raw-root "$WORK/v7/raw_relocated" \
  --data-template configs/data/public_robot_5649h_legacy_compatible.template.yaml \
  --source legacy_v7_formal \
  --adapter-contract "$WORK/adapters/legacy_v7_residual.yaml" \
  --adapter-contract-sha256 "$LEGACY_ADAPTER_SHA" \
  --adapter-audit-receipt "$AUDIT/legacy_v7_residual.adapter_receipt.json" \
  --output-manifest "$INVENTORY/legacy_v7_formal.jsonl" \
  --output-receipt "$INVENTORY/legacy_v7_formal.receipt.json"
```

importer 会重新读取真实 Parquet/video，证明 arm6+gripper1 的全部 action 列被 WM3D arm7
adapter 恰好覆盖，重新审计 10D current state 与 action 首帧时钟，并拒绝 MG 重复来源、
symlink/path escape、损坏视频或缺失字段。旧 plan 的 split、fps 和 duration 不被信任。

`public_robot_6106h.template.yaml` 是额外加入 DROID、Bridge 和拆分 RoboCasa 的可选扩展
profile。只有在希望改变数据家庭/权重时才选它；不能把它的结果写成 5649.4h 兼容运行。

### 6.2 普通 inventory

单 root：

```bash
./run_wm3d.sh inventory \
  --data-template configs/data/public_robot_6106h.template.yaml --source droid \
  --raw-root "$RAW/droid" --adapter-contract "$WORK/adapters/droid.yaml" \
  --adapter-contract-sha256 "$DROID_ADAPTER_SHA" \
  --adapter-audit-receipt "$AUDIT/droid.adapter_receipt.json" \
  --output-manifest "$INVENTORY/droid.jsonl" \
  --output-receipt "$INVENTORY/droid.receipt.json"
```

collection：

```bash
./run_wm3d.sh collection-inventory \
  --data-template configs/data/public_robot_6106h.template.yaml \
  --source agibot_2026_imitation \
  --collection-root "$WORK/materialized/agibot_2026_imitation" \
  --collection-receipt "$WORK/materialized/agibot_2026_imitation/collection_receipt.json" \
  --adapter-contract "$WORK/adapters/agibot_world2026.yaml" \
  --adapter-contract-sha256 "$AGIBOT_ADAPTER_SHA" \
  --adapter-audit-receipt "$AUDIT/agibot_2026_imitation.adapter_receipt.json" \
  --output-manifest "$INVENTORY/agibot_2026_imitation.jsonl" \
  --output-receipt "$INVENTORY/agibot_2026_imitation.receipt.json"
```

collection inventory 给每个 child root 增加 content namespace，并把 payload/video path 改为 collection-root 相对路径，避免 archive 间局部 episode index 冲突。

全部 source 完成后运行 `data-profile`；`--inventory SOURCE=RECEIPT` 可重复，集合必须与模板 source 完全相等：

```bash
./run_wm3d.sh data-profile \
  --template configs/data/public_robot_6106h.template.yaml \
  --inventory droid="$INVENTORY/droid.receipt.json" \
  --inventory bridge="$INVENTORY/bridge.receipt.json" \
  --inventory atomic="$INVENTORY/atomic.receipt.json" \
  --inventory composite="$INVENTORY/composite.receipt.json" \
  --inventory mg="$INVENTORY/mg.receipt.json" \
  --inventory agibot_2026_imitation="$INVENTORY/agibot_2026_imitation.receipt.json" \
  --inventory agibot_2026_rich="$INVENTORY/agibot_2026_rich.receipt.json" \
  --inventory agibot_2026_reinforcement="$INVENTORY/agibot_2026_reinforcement.receipt.json" \
  --inventory agibot_beta="$INVENTORY/agibot_beta.receipt.json" \
  --output "$WORK/public_robot_6106h.yaml" \
  --receipt "$WORK/public_robot_6106h.receipt.json"
```

## 7. Task bank 必须先于 cache plan

```bash
./run_wm3d.sh task-bank \
  --data-profile "$WORK/public_robot_6106h.yaml" \
  --encoder-contract configs/encoder/task_qwen3_vl_embedding_2b.yaml \
  --output-root "$WORK/task_bank" --device cuda

TASK_BANK_SHA=$(sha256sum "$WORK/task_bank/index.jsonl" | awk '{print $1}')
./run_wm3d.sh cache-plan \
  --data-profile "$WORK/public_robot_6106h.yaml" \
  --encoder-contract configs/encoder/vggt_native_p144.yaml \
  --task-encoder-contract configs/encoder/task_qwen3_vl_embedding_2b.yaml \
  --task-bank-index "$WORK/task_bank/index.jsonl" \
  --output "$WORK/cache_tasks.jsonl"
```

不能反过来。task-bank receipt 绑定 data profile、全部 source manifest、task encoder 和 index SHA；每个 cache task identity 同时绑定 task encoder/bank SHA。

## 8. 并行 cache、seal 与 1B/5B window

每张 GPU 运行一个 worker；同一 task manifest 可跨节点作业数组执行：

```bash
./run_wm3d.sh cache-worker \
  --task-manifest "$WORK/cache_tasks.jsonl" \
  --data-profile "$WORK/public_robot_6106h.yaml" \
  --encoder-contract configs/encoder/vggt_native_p144.yaml \
  --task-bank-root "$WORK/task_bank" --task-bank-index-sha256 "$TASK_BANK_SHA" \
  --cache-root "$CACHE" --worker-index "$GLOBAL_WORKER_INDEX" \
  --worker-count "$GLOBAL_WORKER_COUNT" --device "cuda:$LOCAL_RANK" --batch-frames 8

./run_wm3d.sh cache-seal \
  --task-manifest "$WORK/cache_tasks.jsonl" --receipt-root "$CACHE/receipts" \
  --episode-index-fragment-root "$CACHE/episode_index_fragments" \
  --output-index "$CACHE/episode_index.jsonl" --output-seal "$CACHE/episode_seal.json"

MODEL_PROFILE=configs/model/native_5b.yaml
./run_wm3d.sh window \
  --episode-index "$CACHE/episode_index.jsonl" --episode-seal "$CACHE/episode_seal.json" \
  --cache-root "$CACHE" --data-profile "$WORK/public_robot_6106h.yaml" \
  --model-profile "$MODEL_PROFILE" --output-index "$WORK/windows/native_5b.jsonl" \
  --output-seal "$WORK/windows/native_5b.seal.json"

WINDOW_SHA=$(sha256sum "$WORK/windows/native_5b.jsonl" | awk '{print $1}')
./run_wm3d.sh normalization \
  --data-profile "$WORK/public_robot_6106h.yaml" \
  --model-profile "$MODEL_PROFILE" \
  --window-index "$WORK/windows/native_5b.jsonl" \
  --window-index-sha256 "$WINDOW_SHA" --cache-root "$CACHE" \
  --output "$WORK/windows/native_5b.grouped_normalization.json"
```

昂贵 episode cache 绑定 raw manifest row、adapter、视觉 encoder、task encoder/bank 和 representation SHA，不绑定 T/K、训练步数或 1B/5B profile。view token 为 int8 per-vector + scale，depth/point 为 fp16，RGB 为 JPEG pack。改成 `native_1b` 只需重建便宜 window index。

grouped normalization 只读 train split 的真实 window，按 embodiment/group/语义封存统计，并绑定 data profile、model profile 与 window index SHA。它属于便宜的 per-profile 产物，不会触发 episode cache 重算。

## 9. Runtime、canary、恢复和 eval

```bash
RUNTIME_PROFILE=configs/runtime/h200_64_fsdp2_canary1k.yaml
./run_wm3d.sh runtime \
  --model "$MODEL_PROFILE" --data "$WORK/public_robot_6106h.yaml" \
  --runtime "$RUNTIME_PROFILE" --objective configs/objective/stage0_native.yaml \
  --cache-root "$CACHE" --episode-cache-index "$CACHE/episode_index.jsonl" \
  --episode-cache-seal "$CACHE/episode_seal.json" \
  --cache-index "$WORK/windows/native_5b.jsonl" --cache-seal "$WORK/windows/native_5b.seal.json" \
  --grouped-normalization "$WORK/windows/native_5b.grouped_normalization.json" \
  --environment-lock .venv/environment_receipt.json \
  --run-name wm3d_native_5b --run-lineage public_robot_6106h_native_5b_v1 \
  --output-root "$RUNS/wm3d_native_5b" \
  --output "$RUNS/wm3d_native_5b/runtime.yaml"
./run_wm3d.sh preflight \
  --nnodes="$NNODES" --nproc_per_node="$GPUS_PER_NODE" --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" --master_port="$PREFLIGHT_MASTER_PORT" -- \
  --runtime "$RUNS/wm3d_native_5b/runtime.yaml"
```

正式大作业前，用相同链路配 `native_1b + smoke_2gpu_fsdp2` 跑真实两卡 canary：0→编号 checkpoint，退出进程，再从该目录 exact resume；随后运行统一 offline eval。训练/eval 的 torchrun 命令见根 README。checkpoint authority 只能是原子提交的 `step_XXXXXXXX/`。

默认 64×H200 集群不能从模板直接启动 600K。先用
`h200_64_fsdp2_canary1k.yaml` 完成 1K、编号 DCP、独立进程 exact resume 和固定验证；
再用 `h200_64_fsdp2_validation100k.yaml` 做 100K 扩展验证；两者通过后才物化
`h200_64_fsdp2.yaml` 的 600K 正式 runtime。三个 profile 都要求新鲜 resource receipt：
GPU 型号/HBM、ECC、GPU 空闲、节点内 NVLink clique、IB、ulimit、`/dev/shm`、数据/输出
磁盘余量和真实 all-reduce 带宽任一不达标即拒绝启动。

## 10. Fail-closed 清单

以下任一项阻断：mutable revision、未接受许可、缺 source file、legacy/heterogeneous schema、未确认 adapter 语义、字段宽度不符、伪造 action/state 时间戳、coarse→fine 插值、collection child 缺失、manifest/adapter/task-bank/encoder SHA 漂移、cache receipt 缺失、模型窗口超出真实 observed horizon、runtime/code/environment/checkpoint closure 不一致。
