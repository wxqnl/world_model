# WM3D-V7 Native 5B 数据下载、转换与 cache 手册

这份文档从“空的数据盘”写到“可供正式训练只读挂载的 dataset seal”。所有命令都在
`wm3d_v7` 根目录执行，并设置 `PYTHONPATH`。**规划小时数只用于采购；最终时长、
episode 数和窗口数以冻结快照完成后的 receipt 为准。**

## 1. 要下载什么

| 逻辑源 | 上游仓库/内部来源 | 规划体量 | 规划小时 | 进入训练前的形态 |
|---|---|---:|---:|---|
| V7 formal residual | 现有约495h剔除旧MG 40k | 站点实测 | 约397h | `normalized_manifest` |
| RoboCasa full MG | `ember-lab-berkeley/robocasa365-pretrain-mg` | 约315GB | 约1,615h | 单个 LeRobot root，12D action |
| AgiBotWorld2026 Imitation | `agibot-world/AgiBotWorld2026` 的 `ImitationLearning/` | 12.8TB 总仓的一部分 | 约295h | tar 展开后的 LeRobot collection |
| AgiBotWorld2026 Rich | 同仓 `RichInteraction/` | 同上 | 约155h | LeRobot collection |
| AgiBotWorld2026 RL | 同仓 `ReinforcementLearning/` | 同上 | 约211h | LeRobot collection |
| AgiBot Beta | gated `agibot-world/AgiBotWorld-Beta` | 48.1TB | 2,976.4h | 官方 converter 逐 task 生成的 LeRobot collection |
| Beta 官方转换器 | gated `agibot-world/AgiBotWorld-Alpha` 的 `scripts/convert_to_lerobot.py` | 可忽略 | 0h | 冻结脚本 + 独立 LeRobot-v2 转换镜像 |

当前无重叠规划合计约 **5,649.4h**。现有 V7 formal 约495h中已有约98h的
RoboCasa MG 40k；它必须先从 legacy manifest 中按 `provenance_dataset` 精确剔除，
再用完整1,615h MG替换。不能把495和1,615直接相加。AgiBotWorld2026 的
`simulation/` 默认不下载、不计真实机器人小时；除非完成重复版本审计，否则不能混入。

上游入口：

- <https://huggingface.co/datasets/ember-lab-berkeley/robocasa365-pretrain-mg>
- <https://huggingface.co/datasets/agibot-world/AgiBotWorld2026>
- <https://huggingface.co/datasets/agibot-world/AgiBotWorld-Beta>
- <https://huggingface.co/datasets/agibot-world/AgiBotWorld-Alpha/tree/main/scripts>

Beta 当前文件树没有其 README 引用的 `scripts/`，而 Alpha 官方仓库保留了相同
`src_path/task_id/tgt_path` 合同的转换脚本。因此这里把 **Beta 数据**和 **Alpha
官方转换器**作为两个独立快照冻结；任务转换 receipt 会同时绑定 Beta materialization
receipt、Alpha converter download receipt、converter 文件 SHA 和独立转换环境
receipt。官方 Alpha 文档要求 LeRobot dataset v2.0；训练镜像使用的新运行时不能冒充
这个旧 API。交付包因此把转换镜像固定到 LeRobot `0.1.0` 的提交
`8e7d6970eaf5a64b8af6ec45586d201b8ca9ef16`，源码归档、`pyproject.toml` 和
`poetry.lock` 均有固定 SHA。Beta、Alpha 均为 gated 数据，必须先由数据负责人接受
各自的上游许可。`HF_TOKEN` 只能由集群 secret manager 注入环境，禁止写进 Git、
YAML、命令行历史或日志。

## 2. 推荐目录

```text
/raw/wm3d_v7_native5b/
├── snapshots/                         # 上游冻结快照，只读
│   ├── robocasa_full/
│   ├── agibot_world_2026_snapshot/
│   ├── agibot_beta_snapshot/
│   └── agibot_alpha_converter_snapshot/ # 只含 README + 官方 converter
└── materialized/                      # 解包/转换后只读
    ├── agibot2026_imitation/          # 多个独立 LeRobot roots
    ├── agibot2026_rich/
    ├── agibot2026_reinforcement/
    ├── agibot_beta_raw/               # Beta tar 安全物化后的官方目录树
    └── agibot_beta/                   # converter 生成的 task_XXXXXX roots

/datasets/wm3d_v7_native5b_5650h_v1/  # VGGT/action/task cache + seal
/releases/wm3d_v7_native5b_v1/        # locks、schema 报告、receipt、配置
```

原始快照、materialized 数据和正式 cache 应做独立存储快照。训练节点只读挂载
`/datasets/...`，不直接读取 60TB 级供应商目录。

## 3. 冻结 raw-source lock

```bash
cd /workspace/wm3d_v7
export PYTHONPATH=/workspace/wm3d_v7
export RELEASE=/releases/wm3d_v7_native5b_v1
export RAW=/raw/wm3d_v7_native5b
mkdir -p "${RELEASE}" "${RAW}/snapshots" "${RAW}/materialized"

cp configs/scale5b/raw_sources.lock.template.yaml \
  "${RELEASE}/raw_sources.lock.yaml"
```

在私有 lock 中把四个 `revision` 都替换成 Hugging Face 仓库的 **40 位小写 commit
SHA**。branch、tag、短 SHA 和 `REPLACE_...` 都会被下载器拒绝。数据负责人同时填写
许可审批编号；规划 TB/h 不作为完整性证据。

先做不落盘检查：

```bash
/opt/wm3d/bin/python scripts/scale5b/download_raw_snapshots.py \
  --lock "${RELEASE}/raw_sources.lock.yaml" \
  --raw-root "${RAW}/snapshots" \
  --dry-run
```

## 4. 下载四个不可变快照

在有外网、支持 Xet 的数据节点运行；不要在训练节点临时下载。

```bash
# HF_TOKEN 已由 secret manager 注入当前 job，不要 echo。
/opt/wm3d/bin/python scripts/scale5b/download_raw_snapshots.py \
  --lock "${RELEASE}/raw_sources.lock.yaml" \
  --raw-root "${RAW}/snapshots" \
  --source robocasa_full

/opt/wm3d/bin/python scripts/scale5b/download_raw_snapshots.py \
  --lock "${RELEASE}/raw_sources.lock.yaml" \
  --raw-root "${RAW}/snapshots" \
  --source agibot_world_2026_snapshot

/opt/wm3d/bin/python scripts/scale5b/download_raw_snapshots.py \
  --lock "${RELEASE}/raw_sources.lock.yaml" \
  --raw-root "${RAW}/snapshots" \
  --source agibot_beta_snapshot

/opt/wm3d/bin/python scripts/scale5b/download_raw_snapshots.py \
  --lock "${RELEASE}/raw_sources.lock.yaml" \
  --raw-root "${RAW}/snapshots" \
  --source agibot_alpha_converter_snapshot
```

下载中断后，确认仍是同一 lock，再显式加 `--resume`。无匹配完成 receipt 的已有目录
不会被当成成功；已完成且 revision 一致时命令可重入返回 `already_complete`。每个目标
最终含 `.wm3d_v7_download_receipt.json`，记录 repo、revision、文件数和总字节；
转换器工具快照还逐文件记录 SHA-256，wrapper 会在执行前复核脚本内容。

## 5. 展开 AgiBotWorld2026 三个 collection

上游每个 tar 是独立 LeRobot v2.1 根，episode index 会在包间重新从 0 开始。必须用
collection adapter，不能把所有 parquet 粗暴拼成一个 LeRobot root。

```bash
mkdir -p \
  "${RAW}/materialized/agibot2026_imitation" \
  "${RAW}/materialized/agibot2026_rich" \
  "${RAW}/materialized/agibot2026_reinforcement"

/opt/wm3d/bin/python scripts/scale5b/safe_extract_lerobot_collection.py \
  --archive-root "${RAW}/snapshots/agibot_world_2026_snapshot/ImitationLearning" \
  --output-root "${RAW}/materialized/agibot2026_imitation"

/opt/wm3d/bin/python scripts/scale5b/safe_extract_lerobot_collection.py \
  --archive-root "${RAW}/snapshots/agibot_world_2026_snapshot/RichInteraction" \
  --output-root "${RAW}/materialized/agibot2026_rich"

/opt/wm3d/bin/python scripts/scale5b/safe_extract_lerobot_collection.py \
  --archive-root "${RAW}/snapshots/agibot_world_2026_snapshot/ReinforcementLearning" \
  --output-root "${RAW}/materialized/agibot2026_reinforcement"
```

大规模站点可把命令做成 Slurm array，传入相同 `--num-shards N` 和不同
`--shard-id 0..N-1`。空 shard 会显式返回 `empty_shard`，不会误报失败。分片由
archive 相对路径哈希决定，稳定且无交叉写入。工具拒绝
绝对路径、`..`、符号/硬链接、设备文件；先写同盘临时目录，验证 `meta/info.json`
后原子发布。失败临时目录保留为证据，不自动覆盖或删除。

解包命令成功还不等于 collection 完整。三个类别都必须各自运行 closure；closure
把冻结快照的 download receipt、精确 archive 集合和每个 LeRobot root 绑定起来：

```bash
export AGIBOT2026_RECEIPT="${RAW}/snapshots/agibot_world_2026_snapshot/.wm3d_v7_download_receipt.json"

/opt/wm3d/bin/python scripts/scale5b/safe_extract_lerobot_collection.py \
  --archive-root "${RAW}/snapshots/agibot_world_2026_snapshot/ImitationLearning" \
  --output-root "${RAW}/materialized/agibot2026_imitation" \
  --finalize --download-receipt "${AGIBOT2026_RECEIPT}"

/opt/wm3d/bin/python scripts/scale5b/safe_extract_lerobot_collection.py \
  --archive-root "${RAW}/snapshots/agibot_world_2026_snapshot/RichInteraction" \
  --output-root "${RAW}/materialized/agibot2026_rich" \
  --finalize --download-receipt "${AGIBOT2026_RECEIPT}"

/opt/wm3d/bin/python scripts/scale5b/safe_extract_lerobot_collection.py \
  --archive-root "${RAW}/snapshots/agibot_world_2026_snapshot/ReinforcementLearning" \
  --output-root "${RAW}/materialized/agibot2026_reinforcement" \
  --finalize --download-receipt "${AGIBOT2026_RECEIPT}"
```

三个输出根都必须出现
`.wm3d_v7_collection_materialization_receipt.json`。`scan_sources.py` 会再次验证其
schema 和 SHA；每个 archive receipt 还绑定原始归档的 SHA-256。缺任一个 closure
都不能开始 cache。

## 6. 安全物化并转换 AgiBot Beta

Beta 仓库展示的是 TAR，不是 converter 可直接读取的 episode 目录。先生成真实 task
列表，再安全物化 `observations/parameters/proprio_stats`。工具会绑定下载 receipt、拒绝
路径穿越/链接/设备文件、按 archive 哈希稳定分片，并在所有 task 的三类 episode 集合
精确相等后才发布最终 receipt。

```bash
export BETA_SNAPSHOT="${RAW}/snapshots/agibot_beta_snapshot"
export BETA_CONVERTER_SNAPSHOT="${RAW}/snapshots/agibot_alpha_converter_snapshot"
export BETA_CONVERTER="${BETA_CONVERTER_SNAPSHOT}/scripts/convert_to_lerobot.py"
export BETA_CONVERTER_RECEIPT="${BETA_CONVERTER_SNAPSHOT}/.wm3d_v7_download_receipt.json"
export BETA_RAW="${RAW}/materialized/agibot_beta_raw"
export BETA_CONVERTED="${RAW}/materialized/agibot_beta"

test -f "${BETA_CONVERTER}"
test -f "${BETA_CONVERTER_RECEIPT}"

/opt/wm3d/bin/python scripts/scale5b/list_agibot_beta_tasks.py \
  --raw-root "${BETA_SNAPSHOT}" \
  --output "${RELEASE}/agibot_beta_task_ids.txt"

/opt/wm3d/bin/python scripts/scale5b/safe_materialize_agibot_beta.py prepare \
  --snapshot-root "${BETA_SNAPSHOT}" \
  --output-root "${BETA_RAW}"
```

正式解包建议 256 个 array shard；每个 archive 只属于一个 shard：

```bash
export BETA_EXTRACT_SHARDS=256
sbatch --array=0-255%64 --export=ALL --wrap='\
  cd /workspace/wm3d_v7 && \
  export PYTHONPATH=/workspace/wm3d_v7 && \
  /opt/wm3d/bin/python scripts/scale5b/safe_materialize_agibot_beta.py extract \
    --snapshot-root "${BETA_SNAPSHOT}" \
    --output-root "${BETA_RAW}" \
    --num-shards "${BETA_EXTRACT_SHARDS}" \
    --shard-id "${SLURM_ARRAY_TASK_ID}"'
```

等所有 array job 成功后，执行全量 closure：

```bash
/opt/wm3d/bin/python scripts/scale5b/safe_materialize_agibot_beta.py finalize \
  --snapshot-root "${BETA_SNAPSHOT}" \
  --output-root "${BETA_RAW}"

test -f "${BETA_RAW}/.wm3d_v7_beta_materialization_receipt.json"
```

任一 shard 失败时保留临时文件和缺失 receipt 作为证据，不要直接覆盖。先按报错 archive
定位并按站点变更流程处理；没有最终 receipt，后续 converter 和 formal seal 都不得通过。

接着切换到 `CLUSTER_RUNBOOK.md` 第2节构建并发布的 **AgiBot converter 镜像**。以下
命令必须在该镜像里执行；不能改回 `/opt/wm3d/bin/python`。先复核镜像内 receipt：

```bash
export BETA_CONVERTER_PYTHON=/opt/agibot-converter/bin/python
export BETA_CONVERTER_ENV_RECEIPT=/opt/agibot-converter/environment_receipt.json
test -x "${BETA_CONVERTER_PYTHON}"
test -f "${BETA_CONVERTER_ENV_RECEIPT}"

"${BETA_CONVERTER_PYTHON}" \
  /opt/agibot-converter-tools/verify_agibot_converter_environment.py \
  --contract /opt/agibot-converter/environment_contract.json \
  --revision-file /opt/agibot-converter/LEROBOT_REVISION \
  --receipt "${BETA_CONVERTER_ENV_RECEIPT}"
```

然后用**独立冻结的 Alpha 官方 converter 快照**把 Beta 转成 LeRobot collection：

```bash
mkdir -p "${BETA_CONVERTED}"

# 先验证一个真实 task；352 仅是官方文档示例，若列表中没有则换成第一行。
"${BETA_CONVERTER_PYTHON}" scripts/scale5b/convert_agibot_beta_task.py \
  --raw-root "${BETA_RAW}" \
  --output-root "${BETA_CONVERTED}" \
  --vendor-converter "${BETA_CONVERTER}" \
  --converter-download-receipt "${BETA_CONVERTER_RECEIPT}" \
  --converter-environment-receipt "${BETA_CONVERTER_ENV_RECEIPT}" \
  --task-id 352
```

单 task 验证通过后，用数组转换剩余 task。已完成 task 会凭 converter SHA 和 receipt
安全返回 `already_complete`：

```bash
export N_TASKS="$(wc -l < "${RELEASE}/agibot_beta_task_ids.txt")"
# 下面的 array job 必须由站点 Slurm container 插件绑定到已审 converter 镜像。
sbatch --array="0-$((N_TASKS-1))%64" --export=ALL --wrap='\
  cd /workspace/wm3d_v7 && \
  export PYTHONPATH=/workspace/wm3d_v7 && \
  "${BETA_CONVERTER_PYTHON}" scripts/scale5b/convert_agibot_beta_task.py \
    --raw-root "${BETA_RAW}" \
    --output-root "${BETA_CONVERTED}" \
    --vendor-converter "${BETA_CONVERTER}" \
    --converter-download-receipt "${BETA_CONVERTER_RECEIPT}" \
    --converter-environment-receipt "${BETA_CONVERTER_ENV_RECEIPT}" \
    --task-list "${RELEASE}/agibot_beta_task_ids.txt" \
    --array-index "${SLURM_ARRAY_TASK_ID}"'
```

数组完成后必须做 task 集合 closure；它会拒绝缺 task、多 task、临时目录、
converter 文件、converter download receipt 或 materialization receipt 漂移：

```bash
"${BETA_CONVERTER_PYTHON}" scripts/scale5b/convert_agibot_beta_task.py \
  --raw-root "${BETA_RAW}" \
  --output-root "${BETA_CONVERTED}" \
  --vendor-converter "${BETA_CONVERTER}" \
  --converter-download-receipt "${BETA_CONVERTER_RECEIPT}" \
  --converter-environment-receipt "${BETA_CONVERTER_ENV_RECEIPT}" \
  --task-list "${RELEASE}/agibot_beta_task_ids.txt" \
  --finalize

test -f "${BETA_CONVERTED}/.wm3d_v7_beta_conversion_collection_receipt.json"
```

每个 task receipt 与最终 collection receipt 都必须同时绑定 converter 环境 receipt
SHA 和 LeRobot revision；换镜像或改包版本后，旧 task 不会被当成可重入成功。如果冻结
的 Alpha revision 中 converter CLI 与
`--src_path/--task_id/--tgt_path` 不同，必须审查后形成新 wrapper 版本；禁止拿浮动
`main`、第三方 converter 或另一个数据版本猜格式。转换输出还要经过下一节的全 root
schema 审计。

## 7. Schema 审计：这里不允许“差不多”

RoboCasa：

```bash
/opt/wm3d/bin/python scripts/scale5b/inspect_lerobot_schema.py \
  --root "${RAW}/snapshots/robocasa_full" \
  --output "${RELEASE}/schema_robocasa.json"
```

AgiBot 三类与 Beta（`--max-roots` 必须覆盖全部 root）：

```bash
/opt/wm3d/bin/python scripts/scale5b/inspect_lerobot_schema.py \
  --root "${RAW}/materialized/agibot2026_imitation" \
  --collection --max-roots 1000000 --require-homogeneous \
  --output "${RELEASE}/schema_agibot2026_imitation.json"

/opt/wm3d/bin/python scripts/scale5b/inspect_lerobot_schema.py \
  --root "${RAW}/materialized/agibot2026_rich" \
  --collection --max-roots 1000000 --require-homogeneous \
  --output "${RELEASE}/schema_agibot2026_rich.json"

/opt/wm3d/bin/python scripts/scale5b/inspect_lerobot_schema.py \
  --root "${RAW}/materialized/agibot2026_reinforcement" \
  --collection --max-roots 1000000 --require-homogeneous \
  --output "${RELEASE}/schema_agibot2026_reinforcement.json"

/opt/wm3d/bin/python scripts/scale5b/inspect_lerobot_schema.py \
  --root "${RAW}/materialized/agibot_beta" \
  --collection --max-roots 1000000 --require-homogeneous \
  --output "${RELEASE}/schema_agibot_beta.json"
```

模板的 common layout 是：

- RoboCasa：三相机，action `4 base + 1 mode + 3 translation + 3 axis-angle + 1 gripper`；
- AgiBot G2：`14 arm + 2 gripper + 2 base + 2 waist + 2 head = 22D`，三相机键
  `top_head/hand_left/hand_right`，常用 `observation.state` 22D。

若报告出现多个 `schema_sha256`、不同 robot type、相机键或 action/state 宽度，必须按
真实 embodiment 拆成独立 source/layout，再修改：

- `configs/scale5b/dataset_inventory_5650h.template.yaml`
- `configs/scale5b/source_layouts_5650h.template.json`

`scan_sources.py` 会逐列、逐宽度 fail-closed。不要在 encoder 里加供应商字段 alias。
force/tactile/LiDAR 只有在冻结 schema 真有对应列时才加入 auxiliary contract；缺失值要
有 per-dimension validity mask，不能用 NaN 或全零假装事实。

## 8. 生成无 RoboCasa-MG 重叠的 V7 residual

正式输入不能直接使用原来的约495h manifest，因为其中约98h `robocasa365_mg`
会被完整 MG 重复覆盖。先让内部 V7 exporter 给每个 `EpisodeDescriptor` 写入精确
`provenance_dataset`；旧 MG 40k 统一标成 `robocasa365_mg`，禁止根据路径模糊猜测。
然后执行：

```bash
export WM3D_V7_LEGACY_ROOT=/raw/internal/v7_formal

/opt/wm3d/bin/python scripts/scale5b/prepare_legacy_residual_manifest.py \
  --input "${WM3D_V7_LEGACY_ROOT}/native5b_episode_manifest_full.jsonl" \
  --output "${WM3D_V7_LEGACY_ROOT}/native5b_episode_manifest.jsonl" \
  --exclude-provenance robocasa365_mg

test -f "${WM3D_V7_LEGACY_ROOT}/native5b_episode_manifest.jsonl.receipt.json"
```

审阅 receipt：被剔除和保留的 episode 必须都非空；约98h/397h只是规划参考，实测值
由 receipt 和后续 `source_scan.json` 决定。source layout 还会再次拒绝任何
`provenance_dataset=robocasa365_mg` 的残留行，形成双闸门。

residual manifest 每行必须是完整 `EpisodeDescriptor`：安全相对 parquet/video 路径、
行区间、FPS、三视角记录、动作/aux 列映射和非空 `provenance_dataset`。文件内 split
会被全局 seed 重算。raw_root、列名、action mapping 和相机 feature key 必须与 source
layout 完全一致。

## 9. 准备离线 encoder 资产

cache/训练节点不联网。先在 staging 下载并固定：VGGT 源码 commit、
`facebook/VGGT-1B` model revision、`google/flan-t5-xl` revision，然后发布无符号链接的
portable bundle：

```bash
export ASSET_ROOT=/datasets/wm3d_v7_native5b_encoder_assets_v1
/opt/wm3d/bin/python scripts/scale5b/prepare_encoder_assets.py \
  --vggt-source-root /staging/vggt \
  --vggt-source-commit "${VGGT_SOURCE_COMMIT}" \
  --vggt-model facebook/VGGT-1B \
  --vggt-snapshot "/staging/hf/vggt/${VGGT_REVISION}" \
  --vggt-revision "${VGGT_REVISION}" \
  --task-model google/flan-t5-xl \
  --task-snapshot "/staging/hf/flan-t5-xl/${TASK_REVISION}" \
  --task-revision "${TASK_REVISION}" \
  --output-root "${ASSET_ROOT}"

/opt/wm3d/bin/python scripts/scale5b/verify_encoder_assets.py \
  --asset-root "${ASSET_ROOT}" \
  --deep
```

T5 只离线建立 task bank，不进入训练图。

## 10. 编译合同并扫描所有 episode

```bash
export DATASET_ROOT=/datasets/wm3d_v7_native5b_5650h_v1
export BOOTSTRAP_ROOT="${RELEASE}/dataset_bootstrap"
export WM3D_V7_LEGACY_ROOT=/raw/internal/v7_formal
export ROBOCASA_FULL_ROOT="${RAW}/snapshots/robocasa_full"
export AGIBOT_2026_IMITATION_ROOT="${RAW}/materialized/agibot2026_imitation"
export AGIBOT_2026_RICH_ROOT="${RAW}/materialized/agibot2026_rich"
export AGIBOT_2026_REINFORCEMENT_ROOT="${RAW}/materialized/agibot2026_reinforcement"
export AGIBOT_BETA_ROOT="${RAW}/materialized/agibot_beta"

mkdir -p "${BOOTSTRAP_ROOT}"
test ! -e "${BOOTSTRAP_ROOT}/dataset_contract.json"
test ! -e "${DATASET_ROOT}"

/opt/wm3d/bin/python scripts/scale5b/compile_dataset_contract.py \
  --inventory configs/scale5b/dataset_inventory_5650h.template.yaml \
  --output "${BOOTSTRAP_ROOT}/dataset_contract.json"

/opt/wm3d/bin/python scripts/scale5b/scan_sources.py \
  --dataset-contract "${BOOTSTRAP_ROOT}/dataset_contract.json" \
  --source-layouts configs/scale5b/source_layouts_5650h.template.json \
  --output-root "${DATASET_ROOT}"
```

审阅 `${DATASET_ROOT}/receipts/source_scan.json`：每源必须有 train/val、正的实测小时；
所有 parquet 行区间、required column/vector width、视频普通文件和路径边界必须通过。
此 receipt 的实测值才替代规划表。

## 11. 分布式 action/aux 统计

```bash
export REPO_ROOT=/workspace/wm3d_v7
export NUM_SHARDS=256
export GLOBAL_SAMPLE_BUDGET=8000000
sbatch --array=0-255%64 \
  "${REPO_ROOT}/scripts/scale5b/sbatch_action_stats_array.sh"

/opt/wm3d/bin/python scripts/scale5b/build_action_stats.py merge \
  --partials "${DATASET_ROOT}"/control/action_stats_partials/partial_*.npz \
  --output "${DATASET_ROOT}/control/action_stats.json" \
  --clip 5.0
```

默认全语料总采样预算 8,000,000 行，不是每 shard 8M。merge 拒绝缺 shard、重复 shard、
lineage 混合、非有限 continuous action 和超过全局预算。

## 12. Task bank 与 VGGT cache

```bash
export ENCODER_ASSET_ROOT=/datasets/wm3d_v7_native5b_encoder_assets_v1
export TASK_MODEL_REVISION=<40位不可变commit>
sbatch "${REPO_ROOT}/scripts/scale5b/sbatch_task_bank.sh"

export VGGT_REVISION=<40位不可变commit>
export NUM_SHARDS=1024
sbatch --array=0-1023%128   "${REPO_ROOT}/scripts/scale5b/sbatch_encode_array.sh"
```

每个事务 part 包含：

- int8 per-vector 三视角 VGGT token + scale + view mask；
- FP16 depth、3D point、confidence、camera evidence；
- 可随机访问的 JPEG RGB pack；
- 30 Hz grouped action、contact/gripper 与维度有效 mask；
- typed context-only auxiliary token；
- window parquet、manifest 和 commit receipt。

缺相机只发布 invalid mask，不生成黑图。collection 内不同包的 episode index 即使都从 0
开始，也会因相对 root 哈希获得唯一 episode ID。

## 13. Merge、seal、深验

确认 `[0, NUM_SHARDS)` 每个 worker receipt 都存在后：

```bash
/opt/wm3d/bin/python scripts/scale5b/merge_and_seal.py \
  --dataset-root "${DATASET_ROOT}" \
  --num-encoder-shards 1024 \
  --index-rows-per-file 1000000

/opt/wm3d/bin/python scripts/scale5b/verify_dataset.py \
  --dataset-root "${DATASET_ROOT}" \
  --mode control \
  --sample-windows-per-source 4

/opt/wm3d/bin/python scripts/scale5b/verify_dataset.py \
  --dataset-root "${DATASET_ROOT}" \
  --mode deep \
  --sample-windows-per-source 16
```

最终 seal 绑定合同、layout、episode plan、action stats、task bank、encoder asset receipt、
所有 worker/part/index/payload manifest。训练启动只重验 control plane 和 manifest，不会
每次重新读完约 100TB payload。

## 14. 容量预算与 GO/NO-GO

约 5,649.4h 在 5 Hz 下约 1.017 亿视觉帧。仅 `3×144×2048` int8 token 就接近
90TB；加 scale、显式几何、JPEG、动作、索引、临时 part 和存储快照，cache 热层建议
100–130TB。再加约 60TB 原始供应商数据、转换副本和冗余，正式配置应准备 **200TB
usable**（物理 raw 250–300TB）。

以下任一项为 NO-GO：revision 非 40 位、许可未签、schema 异构未拆、列宽不符、重复
数据未审、任一源无 val、小时数仍只用规划值、cache receipt 不全、deep verify 失败、
磁盘不足或训练需要联网下载。
