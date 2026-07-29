# WM3D-V7 Native 5B 环境、集群启动与恢复手册

## 1. 正式拓扑

| 模式 | 节点/GPU | HSDP mesh | 预期用途 |
|---|---:|---|---|
| 推荐 | 16×8 H200 141GB = 128 GPU | replicate 16 × shard 8 | 3–5 周目标 |
| 最低可行 | 8×8 H200 = 64 GPU | replicate 8 × shard 8 | 约6–9周 |

每个 8-GPU NVLink/NVSwitch 节点构成一个 FSDP shard group，节点间经 IB 做 replicate。
推荐 400Gb/s IB；200Gb/s 只能在 canary 实测吞吐达标后接受。H200 的主要优势是
141GB HBM3e 和带宽，不应假设它会让 H100 训练时间自动减半。

正式作业要求：H200 每卡至少135,000MiB、uncorrected ECC=0、节点内完整 NVLink、
IB active、`/dev/shm>=64GB`、memlock unlimited、文件描述符>=1,048,576、dataset 与
output 文件系统启动时各至少10TB余量。

## 2. 构建锁定环境

本交付有两个镜像，不能混用：

1. **AgiBot dataset-v2 转换镜像**：CPU 数据节点使用，服务于 Beta 原始格式转换；
2. **WM3D-V7 Native5B 训练镜像**：H200 cache、canary 和正式训练使用。

### 2.1 AgiBot dataset-v2 转换镜像

官方 Alpha converter 使用 LeRobot dataset v2.0 API。该 API 与正式训练镜像里的新依赖
生命周期不同，所以转换镜像独立锁定 Python 3.10.15、LeRobot 0.1.0 提交
`8e7d6970eaf5a64b8af6ec45586d201b8ca9ef16`。Dockerfile 会复核源码 tar、
`pyproject.toml`、`poetry.lock` 三个 SHA，按上游 lock 安装依赖，再生成环境 receipt。

```bash
cd /workspace/wm3d_v7
export PYTHONPATH=/workspace/wm3d_v7

export AGIBOT_CONVERTER_BASE_IMAGE='python:3.10.15-slim-bookworm@sha256:<审定digest>'
export AGIBOT_CONVERTER_IMAGE_TAG='registry.internal/wm3d/v7-agibot-v2-converter:<release-sha>'
environments/scale5b/build_agibot_converter_image.sh
docker push "${AGIBOT_CONVERTER_IMAGE_TAG}"
```

记录 `docker image inspect` 的 `sha256:...`，并按站点转换成 SIF/Enroot 时再次记录最终
artifact SHA。转换 job 必须使用该 artifact，并能看到：

- `/opt/agibot-converter/bin/python`
- `/opt/agibot-converter/environment_contract.json`
- `/opt/agibot-converter/environment_receipt.json`
- `/opt/agibot-converter/LEROBOT_REVISION`
- `/opt/agibot-converter-tools/verify_agibot_converter_environment.py`

每个 job 开始时都要复核 receipt；`convert_agibot_beta_task.py` 还会在当前 Python 里
重新核对关键 package/import，且把环境 receipt SHA 写入每个 task receipt。该镜像不
用于 WM3D 训练，也不需要 GPU。

发布镜像后，把以下三个文件原样导出为一个可搬运的 runtime bundle。receipt 只记录
同目录相对文件名，因此整个目录可以复制到 release 存储；改动任一字节都会在最终
handoff manifest 阶段失败。

```bash
export CONVERTER_BUNDLE=/releases/wm3d_v7_native5b_<release-sha>/agibot_converter_runtime
mkdir -p "${CONVERTER_BUNDLE}"
test ! -e "${CONVERTER_BUNDLE}/environment_contract.json"
test ! -e "${CONVERTER_BUNDLE}/environment_receipt.json"
test ! -e "${CONVERTER_BUNDLE}/LEROBOT_REVISION"

CID="$(docker create "${AGIBOT_CONVERTER_IMAGE_TAG}")"
trap 'docker rm -f "${CID}" >/dev/null 2>&1 || true' EXIT
docker cp "${CID}:/opt/agibot-converter/environment_contract.json" \
  "${CONVERTER_BUNDLE}/environment_contract.json"
docker cp "${CID}:/opt/agibot-converter/environment_receipt.json" \
  "${CONVERTER_BUNDLE}/environment_receipt.json"
docker cp "${CID}:/opt/agibot-converter/LEROBOT_REVISION" \
  "${CONVERTER_BUNDLE}/LEROBOT_REVISION"
docker rm "${CID}"
trap - EXIT

test -f "${CONVERTER_BUNDLE}/environment_contract.json"
test -f "${CONVERTER_BUNDLE}/environment_receipt.json"
test -f "${CONVERTER_BUNDLE}/LEROBOT_REVISION"
```

### 2.2 WM3D-V7 Native5B 正式训练镜像

只能在 x86_64 Linux 构建。Python 3.10、PyTorch/CUDA/NCCL 和全部 wheel 已锁在
`environments/scale5b/`。

```bash
cd /workspace/wm3d_v7
export PYTHONPATH=/workspace/wm3d_v7

PYTHON_BIN=python3.10 \
  environments/scale5b/build_wheelhouse.sh \
  /workspace/wm3d_v7/environments/scale5b/wheelhouse

export BASE_IMAGE='nvidia/cuda@sha256:<审定的cuda-12.8.1-cudnn镜像digest>'
export IMAGE_TAG='registry.internal/wm3d/v7-native5b:<release-sha>'
environments/scale5b/build_image.sh
docker push "${IMAGE_TAG}"
```

`BASE_IMAGE` 必须是 digest，tag 会被拒绝。Dockerfile 离线安装 wheelhouse 并生成：

- `/opt/wm3d/environment_contract.json`
- `/opt/wm3d/environment_receipt.json`
- `/opt/wm3d/bin/python`
- `/opt/wm3d/bin/torchrun`

按站点需要转换为 SIF/Enroot/OCI 后，再记录最终 artifact SHA。训练节点不能 `pip
install`、不能下载模型。容器必须把 repo/dataset/encoder assets 只读挂载，把
logs/output/checkpoints 读写挂载，并暴露8张卡、NVLink和IB设备。

## 3. Release qualification

在最终容器内先跑 CPU/static/unit 门禁：

```bash
cd /workspace/wm3d_v7
scripts/scale5b/qualify_release.sh
```

再在空闲的同型号节点跑 2-GPU FSDP2+DCP 精确保存/恢复：

```bash
RUN_GPU_SMOKE=1 \
GPU_SMOKE_ROOT=/scratch/qualification/v7_native5b_2gpu_<release-sha> \
CUDA_VISIBLE_DEVICES=0,1 \
  scripts/scale5b/qualify_release.sh
```

然后跑单节点8卡：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  /opt/wm3d/bin/torchrun --standalone --nproc-per-node=8 \
  tests/scale5b_fsdp2_smoke.py \
  --root /scratch/qualification/v7_native5b_8gpu_<release-sha>
```

qualification 会检查全部 runtime 源码、单测、精确 `4,956,589,929` 参数、
V7-only dependency boundary、FSDP2/HSDP API 和 DCP bit-exact forward restore。
smoke checkpoint 是一次性证据，不能给 formal resume 使用。

## 4. Commit 后 seal code

选择审过的 `v7` commit/tag；V7 Native5B scope 必须干净：

```bash
export REPO=/workspace/wm3d_v7
export RELEASE=/releases/wm3d_v7_native5b_<release-sha>
mkdir -p "${RELEASE}"
cd "${REPO}"

/opt/wm3d/bin/python scripts/scale5b/seal_code.py \
  --repo-root "${REPO}" \
  --output "${RELEASE}/code_receipt.json"
```

receipt 绑定 git commit、scope 状态以及所有 Native5B runtime/config/test/doc/env 文件的
SHA/size。没有 `allow-dirty` 逃生开关；变更代码就必须新 commit、新 receipt、新 lineage。

## 5. Materialize 1k canary 和 600k formal

模板故意不可直接启动，所有 `__MATERIALIZE_REQUIRED__` 必须由 receipt 填入。

```bash
export DATASET=/datasets/wm3d_v7_native5b_5650h_v1
export ASSET_ROOT=/datasets/wm3d_v7_native5b_encoder_assets_v1
export RUN_ROOT=/checkpoints/wm3d_v7_native5b_5b_formal_v1
export RUN_LINEAGE=<64位小写hex，正式run唯一>

/opt/wm3d/bin/python scripts/scale5b/materialize_config.py \
  --template configs/scale5b/wm3d_v7_native5b_h200.template.yaml \
  --dataset-root "${DATASET}" \
  --code-receipt "${RELEASE}/code_receipt.json" \
  --code-root "${REPO}" \
  --environment-contract /opt/wm3d/environment_contract.json \
  --environment-receipt /opt/wm3d/environment_receipt.json \
  --output-root "${RUN_ROOT}" \
  --output-config "${RELEASE}/formal_128h200.yaml" \
  --run-name wm3d_v7_native5b_5b_formal_v1 \
  --run-lineage "${RUN_LINEAGE}" \
  --world-size 128 --shard-degree 8 \
  --global-batch-size 128 --micro-batch-size 1
```

64 GPU 仍建议 global batch128；materializer 会导出 gradient accumulation=2。128 GPU
为 accumulation=1。不要手改 materialized YAML；要改就修改模板、重跑 qualification、
重新 seal 和 materialize。

1k canary 使用完全相同的模型、数据、loss 和拓扑，但独立 output/lineage：

```bash
export CANARY_ROOT=/checkpoints/wm3d_v7_native5b_canary1k_v1
export CANARY_LINEAGE=<另一个64位小写hex>

/opt/wm3d/bin/python scripts/scale5b/materialize_config.py \
  --template configs/scale5b/wm3d_v7_native5b_h200_canary1k.template.yaml \
  --dataset-root "${DATASET}" \
  --code-receipt "${RELEASE}/code_receipt.json" \
  --code-root "${REPO}" \
  --environment-contract /opt/wm3d/environment_contract.json \
  --environment-receipt /opt/wm3d/environment_receipt.json \
  --output-root "${CANARY_ROOT}" \
  --output-config "${RELEASE}/canary1k_128h200.yaml" \
  --run-name wm3d_v7_native5b_canary1k_v1 \
  --run-lineage "${CANARY_LINEAGE}" \
  --world-size 128 --shard-degree 8 \
  --global-batch-size 128 --micro-batch-size 1
```

canary 只用于测 HBM、吞吐、loss、checkpoint 和数据质量。formal 必须用独立初始化与
lineage，**不能从 canary checkpoint 续训**。

## 6. 生成最终 handoff manifest

```bash
export CONVERTER_BUNDLE="${RELEASE}/agibot_converter_runtime"

/opt/wm3d/bin/python scripts/scale5b/create_handoff_manifest.py \
  --config "${RELEASE}/formal_128h200.yaml" \
  --repo-root "${REPO}" \
  --dataset-root "${DATASET}" \
  --asset-root "${ASSET_ROOT}" \
  --container-artifact /releases/containers/wm3d_v7_native5b_<sha>.sif \
  --converter-container-artifact /releases/containers/wm3d_v7_agibot_converter_<sha>.sif \
  --converter-environment-receipt "${CONVERTER_BUNDLE}/environment_receipt.json" \
  --output "${RELEASE}/handoff_manifest.json"
```

该 manifest 把 config、code、训练环境、dataset、encoder assets、训练容器，以及
AgiBot converter 容器和三文件 runtime bundle 绑成一个原子证据。缺任何一项都不允许
发起正式 allocation。

## 7. 全集群 preflight 与 canary 启动

```bash
export CONFIG="${RELEASE}/canary1k_128h200.yaml"
export REPO_ROOT="${REPO}"
export LOG_ROOT="/logs/wm3d_v7_native5b_canary1k_$(date +%Y%m%dT%H%M%S)"
export RDZV_ID="wm3d-v7-native5b-canary1k-<release-sha>"
export MASTER_PORT=29400

sbatch --nodes=16 scripts/scale5b/sbatch_native5b_h200.sh
```

Slurm 脚本每节点只启动1个 torchrun launcher，由它生成8个 worker；elastic restart
固定为0。训练前先在全 world size 跑 preflight，检查：

- host/GPU UUID 唯一，恰好8或16节点×8卡；
- 每节点8卡构成 NVLink/NVSwitch clique；
- H200 HBM、ECC、外部 compute process；
- IB link 速率和 full-world all-reduce 吞吐；
- `/dev/shm`、memlock、fd limit；
- dataset/output 每 rank 的独立磁盘门槛；
- code/environment/dataset seal 和 materialized training contract。

等 canary 自然停在 step1000，确认
`checkpoints/step_00001000/COMMITTED.json`，再审核：

- 峰值 HBM 至少15%余量；
- steady-state seconds/step 与3–5周预算；
- source mix 每100步严格10/15/10/8/12/45；
- token/RGB/depth/point/camera/action/contact loss 与 gradient finite；
- RGB 边缘/频谱、运动区域、三视角一致性；
- 每个已启用 action group 有非零 finite 梯度；
- DCP save/verify/load 与 sampler/RNG 连续性。

## 8. Formal 启动

只有 canary 签字后：

```bash
export CONFIG="${RELEASE}/formal_128h200.yaml"
export LOG_ROOT="/logs/wm3d_v7_native5b_5b_formal_v1_$(date +%Y%m%dT%H%M%S)"
export RDZV_ID="wm3d-v7-native5b-formal-v1-<release-sha>"
export MASTER_PORT=29400
unset RESUME_CHECKPOINT

sbatch --nodes=16 scripts/scale5b/sbatch_native5b_h200.sh
```

默认600,000 optimizer steps、global batch128，约7,680万 global samples 和约4,420亿
state-token presentations。这是训练预算，不是运行时间承诺；最终是否保留600k由 canary
吞吐和收敛曲线签字决定。

## 9. 监控

每10–30分钟记录：

- optimizer step、秒/步、样本/秒、data wait；
- 100步 source cycle；
- total/token/RGB/depth/point/camera/action/contact loss 和 gradient norm；
- 128卡 util/HBM/温度/uncorrected ECC；
- IB/NCCL error、网络吞吐；
- dataset/output/log 文件系统余量；
- 正在发布的 DCP incomplete tree 与编号 checkpoint 验证时间。

出现 nonfinite、OOM、CUDA/NCCL、数据/schema、I/O、ECC、磁盘门槛或拓扑/进程变化时，
先保留证据并停在当前控制面，不自动删除数据或 checkpoint，不静默换拓扑/配置。

## 10. 编号 checkpoint 与精确恢复

只有包含 `COMMITTED.json`、`MANIFEST.json`、`metadata.json`、DCP shards 和每 rank RNG
文件的 `checkpoints/step_XXXXXXXX/` 才是 checkpoint。没有 `latest`。

恢复示例：

```bash
export RESUME_CHECKPOINT="${RUN_ROOT}/checkpoints/step_00020000"
export CONFIG="${RELEASE}/formal_128h200.yaml"
export LOG_ROOT="/logs/wm3d_v7_native5b_resume_00020000_$(date +%Y%m%dT%H%M%S)"
export RDZV_ID="wm3d-v7-native5b-resume-00020000-<release-sha>"

sbatch --nodes=16 scripts/scale5b/sbatch_native5b_h200.sh
```

恢复严格要求同一 run lineage、training contract、dataset seal、world size、shard degree、
文件集合/SHA。会恢复 model、optimizer、step-addressed sampler、schedule 以及每 rank 的
Python/NumPy/CPU/CUDA RNG。formal 不支持随意 reshard 到另一拓扑。

## 11. 最终交给训练同事的文件

1. 审过的 V7 commit/tag；
2. 本中文文档；
3. raw-source lock 与三类 schema 报告；
4. container/SIF artifact + SHA；
5. environment contract/receipt；
6. encoder asset bundle/receipt；
7. dataset contract/layout/source scan/final seal；
8. code receipt；
9. materialized canary/formal YAML；
10. handoff manifest；
11. full-cluster preflight 与 canary 报告；
12. formal launch、编号 resume 命令和站点故障升级联系人。
