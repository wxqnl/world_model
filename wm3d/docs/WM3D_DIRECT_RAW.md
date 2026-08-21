# WM3D direct_raw：无视觉 latent cache 的正式训练路径

`direct_raw` 是 1B 和 5B 共用的正式数据路径。它保留 WM3D 的原生 3D
目标和双通路 RGB 结构，但不再创建或读取 episode 级 VGGT latent cache、
streaming LRU 或 sidecar。

## 数据流

每个 rank 对当前 sealed sample 执行以下固定流程：

1. 从已经封存的 episode/window 元数据取出同一个 `T+K` observation ordinal；
2. 对每个真实相机只随机访问这些 RGB 帧；
3. 在 rank 内用 frozen VGGT 编码这一窗口；
4. P64 geometry token 进入 3D 主干，逐视角 P256 layer-4 appearance token
   进入 RGB 通路；
5. 正常运行 WM3D 的 RGB、depth、point、camera、action 和 state loss。

视频层先使用 Decord 随机访问；遇到 Decord 不支持的 AV1 等视频时，自动使用
PyAV 从前一个 keyframe seek 到最后一个目标帧。两种路径都按视频的精确 PTS
绑定 observation ordinal，不允许近邻猜帧，也不会完整解码 episode。

## 删除了什么，保留了什么

删除：

- episode 级 VGGT latent payload；
- 每 rank latent LRU；
- node42 一类外部 cache sidecar；
- cache 发布、收养、淘汰和 prefetch 竞态。

保留：

- task bank、episode index、window index 和 grouped normalization；
- frozen VGGT encoder；
- 训练期 VGGT depth/point/camera teacher heads，用于产生不变的监督目标；
- WM3D 自己的 RGB/depth/point/camera/action/state 输出 heads；
- P64 geometry 主干和逐视角 P256 appearance 通路。

因此，direct 并不会让模型失去 depth、point 或 camera 输出。推理时不需要运行
VGGT 的 teacher decoder heads；最终输出仍由 WM3D 自己的 heads 产生。

## 为什么仍有 streaming-prepare

命令名沿用现有 handoff，但 direct 模式的 `streaming-prepare` 只生成小型、
可校验的采样元数据和 normalization，不生成任何 RGB/3D latent。它可以在 CPU
上一次完成，训练期间没有 cache worker 或 sidecar。

## 1B

```bash
./run_wm3d.sh 1b init canary1k /data/wm3d_1b/control/direct_canary.env
# 编辑路径、模型 snapshot 和许可
./run_wm3d.sh 1b doctor /data/wm3d_1b/control/direct_canary.env
./run_wm3d.sh 1b task-bank /data/wm3d_1b/control/direct_canary.env
./run_wm3d.sh 1b cache-plan /data/wm3d_1b/control/direct_canary.env
./run_wm3d.sh 1b streaming-prepare /data/wm3d_1b/control/direct_canary.env
./run_wm3d.sh 1b runtime /data/wm3d_1b/control/direct_canary.env
./run_wm3d.sh 1b preflight /data/wm3d_1b/control/direct_canary.env
./run_wm3d.sh 1b train /data/wm3d_1b/control/direct_canary.env
```

Canary 通过后，用新的 site 文件执行 `formal100k`。不要复用 canary 输出目录。

## 5B

```bash
./run_wm3d.sh 5b init canary1k /data/wm3d/control/direct_5b_canary.env
# 编辑 8 节点共享路径和 rendezvous
./run_wm3d.sh 5b doctor /data/wm3d/control/direct_5b_canary.env
./run_wm3d.sh 5b task-bank /data/wm3d/control/direct_5b_canary.env
./run_wm3d.sh 5b cache-plan /data/wm3d/control/direct_5b_canary.env
./run_wm3d.sh 5b streaming-prepare /data/wm3d/control/direct_5b_canary.env
./run_wm3d.sh 5b runtime /data/wm3d/control/direct_5b_canary.env
./run_wm3d.sh 5b preflight /data/wm3d/control/direct_5b_canary.env
./run_wm3d.sh 5b train /data/wm3d/control/direct_5b_canary.env
```

5B 使用同一实现，只把 geometry grid 改为模型 profile 的 P144；appearance 仍是
逐视角 P256 layer 4。不存在单独的 5B cache 实现。

## 关键参数

- `DIRECT_ENCODE_CHUNK_ROWS=32`：VGGT 每次编码的时空行数；encoder OOM 时自动
  减半，最低到 `DIRECT_MINIMUM_CHUNK_ROWS`。
- `DIRECT_PREFETCH_WINDOWS`：CPU 上有界的 uint8 RGB window 队列；官方 1B 默认 16、
  5B 默认 8，分别覆盖两个 micro-batch。它不是 latent cache，内存上限不随训练步数增长。
- `DIRECT_VIDEO_INDEX_CACHE_ASSETS=128`：只缓存小型 PTS 数组，不缓存 RGB 或
  latent。
- `DIRECT_DECODE_WORKERS=4`：同一个 episode 内相机解码并行度。
- `DIRECT_APPEARANCE_FEATURE_LAYER=4`：清晰 RGB 路径使用浅层逐视角特征。

## 稳定性边界

Direct 路径的内存随 `batch × (T+K) × views` 有界，不随训练步数或 episode
长度增长。encoder OOM 只降低 frozen VGGT chunk，不改变 micro batch、global
batch、模型、数据或 loss。Checkpoint 仍由既有分布式 checkpoint 合同管理；
direct adapter 没有可训练参数，不进入 optimizer 或 checkpoint。VGGT 的各 chunk
直接写入一份预分配输出，避免先保留全部 chunk 再 `cat` 所产生的双份峰值显存。

## 已完成验证

- RoboCasa 整文件路径与 sealed full decoder 的 RGB 逐像素一致；
- OXE 大 PTS offset 的 recorded segment 采用与 sealed decoder 完全相同的半开区间，
  预处理输入仅剩 resize 后 uint8 存储舍入误差（MAE 约 0.000304，最大 0.5/255）；
- 2480 帧 AV1 episode 只请求 24 行时，PyAV fallback 只从前一 keyframe 解码到目标帧，
  不会 materialize 整个 episode；
- 真实 VGGT 输出 geometry、appearance、RGB、depth、point、camera 全部有限；
- 两卡 1.327B FSDP2 已完成完整 objective、backward、optimizer、COMMITTED checkpoint
  与独立 exact-resume 读取；
- 全量单元/合同回归覆盖 direct、旧 cache 兼容路径和 rank-invariant RGB decoder。

Direct 的目标是去掉会随 episode/LRU/sidecar 状态变化的故障面。它仍要在线支付 frozen
VGGT 计算，因此不能承诺比完整热 cache 更快；收益是墙钟更可预测、内存有界、没有缓存
发布/收养/淘汰竞态，并且输入 latent 不再经过 int8 量化。
