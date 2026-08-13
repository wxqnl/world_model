# WM3D 真实公开小样本一键验收

这个入口用于在新服务器交付前，真实走完一次统一 Stage0 链路。它使用公开的
`lerobot/aloha_sim_insertion_human` 固定 commit，只选 episode 0 和 30：前者进入
train split，后者进入 val split。它不是性能 benchmark，也不能替代正式数据训练或
LIBERO 闭环分数。

小样本不会沿用正式数据的 98/1/1 比例：两个 episode 在该比例下可能同时落进 train。
入口显式封存 `seed=3407, train=0.5, val=0.49`，并在最终验收逐行确认 episode 0
只能是 train、episode 30 只能是 val；这组比例只属于 smoke，不会写入正式 data profile。

## 1. 运行前确认

需要两张空闲 NVIDIA GPU、Python 3.10、可访问 Hugging Face/GitHub 的网络，以及一个
clean Git commit。请先阅读：

- ALOHA 小样本的上游许可证；
- `configs/adapters/aloha_sim_insertion_human.yaml`；
- 该 adapter 的双臂 group、50 Hz 原始时间戳以及 opaque continuous gripper 通道说明。

上游只声明第 7/14 维是 motor channel，没有声明开合极性，所以本 smoke 不把它伪装成
`absolute_gripper_open01`。

## 2. 一条命令

默认使用 GPU 0、1，并在 work root 内创建独立 Python 环境：

```bash
./run_wm3d.sh smoke-real \
  --work-root /data/wm3d_smoke_real \
  --operator "姓名或工号" \
  --gpus 0,1 \
  --accept-dataset-license \
  --confirm-adapter-semantics
```

如果服务器不能直连 `huggingface.co`，但有兼容且受信任的 HTTPS endpoint，可显式传
`--hf-endpoint https://hf-mirror.com`。endpoint 会写进不可变 smoke plan；source
revision 和上游 file-list 仍由原锁定脚本在线核对，不能用镜像名绕过 commit。

新服务器会下载固定 revision 的 ALOHA 小数据、Qwen3-VL embedding 模型、VGGT-1B
权重和固定 VGGT source commit。若节点已有经过审计的精确 snapshot，可显式传入：

```bash
./run_wm3d.sh smoke-real \
  --work-root /data/wm3d_smoke_real \
  --operator "姓名或工号" --gpus 0,1 \
  --accept-dataset-license --confirm-adapter-semantics \
  --qwen-model-snapshot /abs/path/9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda \
  --vggt-model-snapshot /abs/path/860abec7937da0a4c03c41d3c269c366e82abdf9 \
  --vggt-source-root /abs/path/vggt-source
```

显式资产不是“信任路径”：入口会封存 snapshot 的精确 file-set/type/size closure，
随后由 Qwen/VGGT 的本地只读加载完整消费权重；VGGT source 则做全文件内容 SHA，且
额外验证固定 commit 的关键实现 SHA。这样不会在每次重入时重复哈希十余 GB 权重，
路径名、文件集合、大小、source 内容或 encoder 实际加载任一不符都会停止。

## 3. 实际执行内容

```text
独立环境 receipt
→ 固定 source lock/file list
→ 可重入下载
→ schema audit 与人工语义确认 receipt
→ 双臂 inventory/profile/task bank
→ 两 GPU episode cache worker 与 seal
→ 1B window 与 grouped normalization
→ sealed runtime
→ 两 rank FSDP2 full preflight
→ 进程 A：0→1，提交 step_00000001
→ 进程 B：exact resume 1→2，提交 step_00000002
→ 进程 C：固定 val window offline eval
```

`preflight` 本身也是 torchrun 入口；world size 大于 1 时必须使用：

```bash
./run_wm3d.sh preflight \
  --nnodes=1 --nproc_per_node=2 --node_rank=0 \
  --master_addr=127.0.0.1 --master_port=29631 -- \
  --runtime /abs/path/runtime.yaml
```

统一入口自动追加 `--preflight-only`。不要在应用参数里重复传入。

## 4. 中断与重入

原命令可以原样再次执行。每个步骤 receipt 同时绑定：

- clean code commit 与 smoke plan；
- 实际命令；
- 输入 SHA；
- 输出文件或目录内容 closure。

只有这些内容完全一致才会 `verified-skip`。若训练在 committed DCP 已发布、步骤 receipt
尚未写出时中断，入口会逐文件验证 `MANIFEST.json` 的 size/SHA、`COMMITTED.json`、
runtime、lineage、sampler progress 和 gradient ownership，全部通过后才补步骤 receipt。
不完整 checkpoint、错误 work root 复用或产物篡改都会 fail closed。

## 5. 验收结果

成功后只认：

```text
<work-root>/SMOKE_REAL_ACCEPTANCE.json
```

总 receipt 必须满足 `passed=true`，并绑定 environment、模型资产、source lock、下载、
schema/adapter、data profile、task bank、episode/window seal、normalization、runtime、两个
完整编号 DCP、独立进程 exact resume 和 finite val eval。不要把 `latest`、控制台最后一行
或只有目录名的 checkpoint 当作验收证据。
