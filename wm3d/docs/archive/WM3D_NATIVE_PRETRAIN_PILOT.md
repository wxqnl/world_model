# WM3D Native 预训练

## 2026-09-24：DROID 官方坏视频修复

正式训练在 step2000 保存后遇到 DROID AV1 视频解码错误，随后其他 rank 等待超时。
原始下载文件与上游旧版相同。上游已发布 [16 个视频的修复](https://huggingface.co/datasets/lerobot/droid_1.0.1/discussions/8)。
用户已批准采用修复文件，并授权这类明确故障修复后直接续训。监控周期改为两小时。

原始文件和封存索引保留。修复视频发布在同一 source root 的
`.wm3d_rgb_repairs/droid_official_pr8/` 下；读取器只有在
`WM3D_RAW_ASSET_REPAIRS` 指向完整修复清单时才选择这 16 个文件。
原文件仍通过原封存校验，修复文件另行校验。其余文件沿用原路径。
16 个修复视频共 2,261,384 帧，已完整解码并逐项核对原记录中的时间戳，
帧数、顺序和分辨率均一致。不采用跳帧、补帧或忽略解码错误。
上游重新编码改变了 RGB 字节，因此这是明确记录的数据修订。

正式根的 `data_repairs/droid_official_pr8.json` 绑定这次修订，
只允许从本正式 step2000 首次接入；之后 checkpoint 保存完整
`raw_asset_repairs` 清单，恢复时须匹配。优化器、采样器和 RNG 继续读回，
episode、动作标签、归一化和 source 比例不变。执行代码使用独立分支
`codex/native-droid-repair-20260924`，没有修改模型、loss、batch 或 50K 预算。
实际恢复状态和当前步数，以新续训日志为准。

## 2026-09-24：冻结特征复用与性能维护

用户要求立即停止优化，不再等第500步保存。13:27两机停止，日志虽到500，
但validation/save尚未完成；完整恢复点为本50K正式根的step100。
不得把日志步数当作已保存checkpoint。旧日志保留，101之后的未保存更新需要重算。

恢复代码为两机的 /data/Minko/wm3d_native_vggt_reuse_20260924，
入口为正式根的 resume_cache_step100.sh，服务名称为
wm3d-native-formal50k-cache-r100-n43/n44-20260924。
恢复仍用原runtime和step100完整optimizer、sampler、RNG；
50K、micro6/global96、FP32主参数、BF16计算、IB、模型、loss和原数据比例均不变。
代码执行版本变化记录在正式根 cache_optimization_transition.json；
launcher核对实际干净代码及验收记录后才允许执行兼容恢复，不伪称执行版本没变。

新增启用两个执行选项：冻结VGGT的8GiB/rank CPU特征缓存，以及官方foreach梯度裁剪。
VGGT只复用同一物理帧和同一有效相机组合的冻结输出，完整保留current/K8时间顺序；
未来特征仍只进入loss。无teacher forcing、量化、分辨率变化或新loss。
CPU索引解析缓存也随此代码部署；原完整索引的字段、顺序、比例和源文件校验不变。

真实VGGT验收使用原正式采样器rank0/8、step481–496的32个batch，
共192窗口、7个source。所有缓存开关输出逐元素相同，最大差0，CPU/CUDA RNG不变；
输出是可被可导模块使用的普通张量。1728个时间行只需编码709行。
两节点平均VGGT耗时2.71215→1.14356秒，减少57.84%；这是冻结编码环节，
不能写成整步加速58%。记录在正式根cache_optimization/receipt_rank00/08.json。

真实16卡DTensor裁剪对照覆盖全部2144个可训练tensor形状，用step100保存的
first moment作为回放梯度，包括有/无裁剪和无梯度参数。六轮对照的范数及裁剪后
数值逐元素一致。排除两轮预热、每轮取最慢rank，默认/foreach平均为0.12897/0.08769秒；
单轮波动较大，约0.04秒收益只是辅助，不能当主要瓶颈。全局非有限检查保持开启。
记录在grad_clip_optimization/，脚本为scripts/benchmark_native_grad_clip.py。
首次裁剪测试仅启动路径错误，未启动GPU worker；修正为python -m torch.distributed.run后通过。

GPU/功率验收看恢复后真实训练samples/s和完整时间窗口。400W恒定不是通过条件。
本轮未改MoT、FSDP边界或attention mask；MoT直接调用block子函数，不能机械逐层wrap。
CPU检查669 passed、4 skipped，不代替以上真实GPU对照或RGB/action质量验收。

## 当前正式任务：50K、fused AdamW、100Gbps IB（2026-09-24）

用户已授权43/44两台16卡训练50,000步。本节取代下文历史的2000步预算和TCP启动方式。
旧2000步attempt2通过预检后，在训练索引加载期间按用户要求干净停止；
未发生优化器更新、没有checkpoint，也不是运行故障。新任务保持fresh Wan初始化。

正式根：`/data/Minko/wm3d_native_1000h_formal_2node16_micro6_step50000_20260924`。
两机代码仍是`/data/Minko/wm3d_native_pretrain_perf_20260924`。
micro6、累积1、global96、验证micro1；模型、原1000h索引/比例、loss和FP32主参数不变。
总步数50,000、warmup1000、stable_fraction0.55，原峰值1e-4、最小1e-5和video LR比例0.1不变；
因此从第27,950步起才进入余弦下降，而不是沿用2000步的短调度。
每10步记录，每500步验证20批；保存100/500/1000/2000以及每5000步。
配置原件在checks目录的`h100_16_native_formal50000_micro6_fused_ib_20260924.yaml`。
约480万窗口次、0.975个窗口等效epoch，不保证每个episode或每个窗口都访问一次，也不证明收敛。

本次另外启用已连通的100Gbps InfiniBand：43固定`mlx5_4:1`、44固定`mlx5_6:1`，
两端均Active、SM LID25。44的400Gbps端口仍Initializing，不改网络设备或共享fabric。
`NCCL_IB_DISABLE=0`、`NCCL_NET=IB`和精确`NCCL_IB_HCA`选择防止静默回退；
TCP网卡`bond0.1411`仅用于bootstrap。NCCL NVLS仍关闭，FSDP2仍为节点内shard8/节点间复制。

实际链路测试98.4Gbps。NCCL 2.26.2通信A/B使用正式checkpoint的2144个可训练tensor形状，
每rank的FP32 padded gradient为3,278,446,328字节，在真实2×8 mesh上同时执行8组跨节点all-reduce。
TCP与IB各16份receipt通过，全buffer归约值正确。排除2次预热，4次测量均取最慢rank：
TCP平均8.96636秒，IB平均2.13845秒，通信环节减少6.82791秒、快4.19倍。
这是生产梯度规模的通信测量，不是完整训练step；不能直接声称整步快4.19倍。
证据保存在checks的`replica_tcp/`、`replica_ib/`和对应日志，脚本为
`benchmark_native_replica_collective.py`。容量checkpoint只用于读取形状，绝非正式初始化。
整步速度仍须排除加载/验证/保存、保留实际I/O慢步，从正式日志测量。

单checkpoint约76GiB，50K全保存在43本地会超出可用空间。
每小时监控要在保存后验证COMMITTED、16distcp/16rank_state/owner和双机可读，
始终保留最新两个完整正式checkpoint在43；更早的已被取代checkpoint可低优先级完整迁往44
独立归档目录，核对全文件数量/字节/内容后保留原路径符号链接，不直接删证据。
43至少留出两份checkpoint加50GiB空间，不在checkpoint写入期间迁移，不让归档抢占训练I/O。
50K完成后先验收，不自动延长或启动32卡。

### 真实更新已开始

本正式任务已通过双机预检并进入优化器更新。2026-09-24约11:58已记录step50，
梯度所有权检查通过，训练loss和梯度均有限。step31–50平均11.1794秒/步、
8.5872 global samples/s；保留真实I/O等待，未含验证或保存。
相对此前TCP/single的18.5723秒/步，短窗口耗时下降约39.8%。这不是完整长训均速。

### 索引启动缓存修复

启动慢的一项主因是每个rank在train/val、preflight/train之间重复解析同一份约12GB
JSONL。完整索引有12,765,292行；1000h清单仍只选择原4,922,451个train窗口，
validation仍为127,991。此次缓存只保存原解析器输出的不可变元数据，不缓存视频，
不改变source比例、窗口顺序、标签、归一化或训练采样。

修复位于双方的`/data/Minko/wm3d_native_index_cache_20260924`，
分支`codex/native-index-cache-20260924`。默认缓存目录为输出根同级的
`.wm3d_parsed_index_cache`；可用`WM3D_INDEX_CACHE_DIR`指定私有节点本地目录，
显式设为空则关闭磁盘缓存。train/val复用进程内解析结果，preflight/train复用磁盘结果。
每次调用仍验证原始文件内容；缓存损坏或过期时回到原解析器。构建加排他锁，
写完后原子发布，已有缓存并行读取；数据集建好后释放未选中的全量行引用。

真实全索引首次校验、解析和缓存写入571.77秒，随后校验加磁盘读入89.95秒。
逐条比较全部记录的字段与顺序，结果一致。该计时不包含训练所需的其他数据集组装、
模型加载或16rank并发争用，不能声称整个启动只需90秒。
聚焦检查38项通过，全仓CPU检查668 passed、4 skipped；不等同GPU生成质量验收。
生产数据集组装另耗train262.79秒、val55.23秒，含原有manifest和元数据验证；
全量CPU验收完成，4,922,451 train / 127,991 val及18源整数窗口计数均通过。
报告为checks目录的`index_cache_full_receipt_20260924.json`。
已用低优先级限速复制完成的8.28GiB缓存到44本地私有目录；44独立进程实际
校验原文件并读回12,765,292行，用时87.58秒。双机均已具备可复用缓存。
这些是单进程CPU实测，没有为测整个新启动耗时中断当前正式训练。

当前50K仍运行上方perf工作树。它已完成索引加载，不为应用此补丁再中断训练，
也不热改运行代码或放宽checkpoint身份校验。缓存修复供后续新启动使用。

以下为资格、数据恢复与旧预算的历史记录。


## 2026-09-24 正式训练恢复与优化

用户已确认43/44两台共16卡，正式首段为2000更新步。此前根
`/data/Minko/wm3d_native_1000h_formal_2node16_micro6_step2000_20260924`
于08:30启动，完成日志step30后中断。两机BMC在09:28记录电源冗余丢失，
09:29:54–56记录多路AC lost，约09:32恢复供电并自动开机；没有正常关机记录，
训练末尾未见OOM或Traceback。故障属于供电中断；具体上游原因仍需运维核实。
首个step100未保存，不能恢复step30，旧根仅保留证据。

新正式根为
`/data/Minko/wm3d_native_1000h_formal_2node16_micro6_step2000_20260924_attempt2`，
代码为双方同路径`/data/Minko/wm3d_native_pretrain_perf_20260924`。
恢复原数据/文本/输出NFS挂载后，重新由通用Wan与Wan派生权重初始化。
不使用资格checkpoint作为训练初始化。micro6、累积1、global96、原1000h索引、
LR、loss、3D/action、分辨率和K8均不变。下方“待拓扑确认”的内容是历史准备记录。

本次只增加封存的`optimizer.implementation: fused`选项，默认仍为`single`。
沿用PyTorch AdamW的相同公式，主参数和两个moment仍为FP32；fused的step标量
按PyTorch要求放到参数所在设备。不是低精度优化器，也没有改FSDP划分或注意力mask。
重启后的完整训练吞吐尚待实测，不能把算子速度作为整步加速倍数。

`scripts/benchmark_native_adamw.py`使用既有真实生产checkpoint的全部2144个
可训练tensor（6,542,457,322参数），按原2×8 DTensor拓扑比较single与fused。
只作优化器回放：相同保存moment作为回放梯度，包含不更新参数；不代表模型前后向或质量。
16份receipt均通过，8次更新后参数/moment/step符合rtol2e-5、atol2e-7；
全局最大绝对差3.8147e-6（大幅值元素使用相对容差），不声称逐位等价。
三个真实专家tensor的fused GPU-step DCP保存读回逐元素一致。
排除2次预热、对每轮取最慢rank后，single均值0.76487秒，fused0.05150秒，
该环节快14.85倍，节约0.71338秒/更新。重启前整步约18.57秒，
不能据此声称整个训练快14.85倍。记录在
`/data/Minko/wm3d_native_adamw_replay_2node16_20260924`。

## 目标与初始化边界

训练自己的 native 3D world model，改善 RGB 预测并支持 action 下游任务。
WSA 是代码和训练流程的参考，不是成品权重来源。保留通用 Wan2.2 视频/VAE 先验，
action、3D 专家的通用 Transformer 层沿用 WSA 官方的 Wan 插值与 alpha 缩放初始化；
机器人输入输出头、native 3D 状态和 WM3D 接口仍是新参数。已获用户确认。
这不是继承 WSA 的机器人能力，也不等于从随机参数训练整个视频生成器；
不能期待一百步就获得 WSA 成品模型的能力。

| 组件 | 采用的实现 | 权重与监督 |
| --- | --- | --- |
| 视频 | 官方 Wan VideoDiT、VAE、flow scheduler 与 sampler | Wan 视频先验；骨干低学习率，VAE 冻结 |
| Action | 官方 ActionDiT 与 flow objective | Wan 派生骨干、新输入输出头；真实 grouped action、timestamp、mask |
| 3D | MoT 内部的逐视角 K8×P64 trajectory queries | Wan 派生骨干、新状态接口；冻结 VGGT 的所有未来时刻 target |
| Robot/task | WM3D 已有数据桥与 conditioning | 当前状态、历史动作、任务和同一归一化 profile |

Wan 派生资产由官方 `preprocess_expert_backbones.py` 在 CPU 上生成，WM3D 只提供薄入口，
不改插值方法。当前双机资产已放在 `/data/Minko/models/WM3D-Wan-Backbones`。
新环境首次准备时运行一次；命令拒绝覆盖既有文件：

```bash
FOUNDATION_PYTHON_BIN=/data/Minko/.venvs/wm3d-foundation-v10-final/bin/python \
  ./run_wm3d.sh native-backbones --model configs/model/native_wan.yaml
```

没有新增 RGB decoder、第二套 action head 或 posthoc dynamics loss。输出是视频、
生成 action 与显式未来 native 3D tokens；不能将 token 预测直接宣称为高精度度量点云。
候选 action 反事实排序暂不作为本阶段的主要验收条件。

## 3D 如何参与 RGB 与 action

当前多视角图像经冻结 VGGT 得到每视角 P64 状态。它被投影到同视角 K8 个未来 query，
与可学习的位置/query 参数一起进入 Future3D。每个未来 query 保留视角、时间和空间槽位，
在官方 MoT 的每层与 RGB/action 专家交互，最后一对一读出 2048D native 3D token。
不把最终单帧 token 复制成 K 帧，也不在 RGB/action 算完后另加一个预测头冒充核心状态。

沿用 `WSALarge` 的注意力边界，不使用 `WSALargeJoint`：当前视频只读当前视频，
未来视频读取 3D；3D/action 读取观测和彼此，但不能读取加噪的未来 RGB。
训练中的未来 RGB/action 是 flow 目标对应的加噪变量；推理从噪声生成，不能把干净未来
标签塞进 context。未来 VGGT target 只在 loss 回调读取。VGGT 各时间行独立编码，
避免把未来信息混入当前观测特征。

3D 监督沿用 upstream 的 cosine + layer-normalized MSE，覆盖所有有效 K8 时刻，
权重为 0.1。视频/action 各为 1.0。逐 horizon 的 delta 方向、幅度和 copy-last error
只用于诊断，不再叠加一组运动 loss。缺失视角、动作和维度保留原 mask。

## 数据

复用 `/data/Minko/wm3d_v7_world_1000h_data_20260909/train_episode_allowlist.json`，
对应约 999.999 小时、253,207 个唯一训练 episode、4,922,451 个训练窗口、18 个 source。
小时数按 episode 时长，不重复累计重叠窗口。保留所有 OXE，以及原清单选定的 RoboCasa
episode；按既有选中窗口数比例采样，不重新做 source 均衡或运动筛选。
127,991 个 validation 窗口不受训练 allowlist 过滤。

数据 profile 和 metadata 位于 `/data/Minko/wm3d_full_robot_data_20260904`。
新增的 `episode_selection.py` 只应用既定清单，核对 source、episode 和窗口覆盖；
source adapter、物理语义、归一化和原始文件都不改。
WSA 使用的其他高质量数据尚未加入；先核实下载位置、许可证和 adapter，不能把未读过的
数据写成已参与训练。

## 双机试训配置

使用 `configs/model/native_wan.yaml`、`configs/objective/native_joint.yaml` 和
`configs/runtime/h100_16_native_pilot.yaml`。仅在 43、44 现有任务全部结束后启动；不使用 41。

- 16 GPU，micro-batch 1、累积 1、global batch 16，100 个更新步。
- FP32 主参数及优化器状态，BF16 计算；Wan 视频骨干学习率为新模块的 0.1 倍。
- FSDP2 节点内 8 卡分片、节点间复制；保留官方 activation checkpoint。
- 官方 MoT 会直接调用 block 子函数，因此本路线先用 root-only FSDP，不能机械套逐层 wrap。
- 每 20 步验证；只在第 100 步保存完整 checkpoint。预计单 checkpoint 约 76 GiB，开训前
  同时预留 checkpoint 和临时写入空间。试训数据流不变，后续按实测显存提高 micro-batch。
- `h100_32_native.yaml` 仅为 4×8 卡拓扑配置起点；预算、batch 和质量尚未验证，不能自动长训。

保持现有 `run_wm3d.sh` 入口、trainer 和 DCP checkpoint，不新增平行训练器。
下面命令在 43 上 `ssh New-H100-3` 后、`cd /data/Minko` 执行；44 也经 43 进入。
代码根为 `/data/Minko/wm3d_native_pretrain_1000h_20260923`，44 使用同路径同步副本。

```bash
cd /data/Minko/wm3d_native_pretrain_1000h_20260923/wm3d
export PYTHON_BIN=/data/Minko/.venvs/wm3d-foundation-v10-final/bin/python
export FOUNDATION_PYTHON_BIN="$PYTHON_BIN"
./run_wm3d.sh check

RUN=/data/Minko/wm3d_native_1000h_pilot_2node16_20260923
./run_wm3d.sh runtime \
  --model configs/model/native_wan.yaml \
  --data /data/Minko/wm3d_full_robot_data_20260904/data_profile.yaml \
  --runtime configs/runtime/h100_16_native_pilot.yaml \
  --objective configs/objective/native_joint.yaml \
  --data-mode direct_raw \
  --streaming-metadata-seal /data/Minko/wm3d_full_robot_data_20260904/streaming_metadata_seal.json \
  --environment-lock /data/Minko/.venvs/wm3d-foundation-v10-final/environment_receipt.json \
  --run-name native_1000h_pilot --run-lineage native_1000h \
  --output-root "$RUN" --output "$RUN/runtime.yaml"
```

先确认双方模型、metadata、原始视频、text context 和共享输出目录可访问。双机以相同
runtime 执行以下命令，`NODE_RANK` 在 43 为 0、44 为 1；端口须先确认未被其他任务使用：

```bash
export NCCL_IB_DISABLE=1 NCCL_NVLS_ENABLE=0
export NCCL_SOCKET_IFNAME=bond0.1411 NCCL_SOCKET_FAMILY=AF_INET
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MALLOC_ARENA_MAX=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WM3D_VGGT_SOURCE_ROOT=/data/Minko/wm3d_vggt_source_20260822
export WM3D_VGGT_MODEL_SNAPSHOT=/data/Minko/models/VGGT-1B
export WM3D_DIRECT_PREPARED_ROW_CACHE_BYTES_PER_RANK=8589934592
export WM3D_DIRECT_PREFETCH_WORKERS=2 WM3D_DIRECT_ASYNC_PIPELINE=0
./run_wm3d.sh foundation-preflight \
  --nnodes=2 --nproc_per_node=8 --node_rank="$NODE_RANK" \
  --master_addr=172.27.0.6 --master_port=29983 -- --runtime "$RUN/runtime.yaml"

# preflight 成功且两个节点仍空闲后，以同一组参数调用 foundation-train。
./run_wm3d.sh foundation-train \
  --nnodes=2 --nproc_per_node=8 --node_rank="$NODE_RANK" \
  --master_addr=172.27.0.6 --master_port=29983 -- --runtime "$RUN/runtime.yaml"
```

## 下载期间先用可读数据测试

用户于 2026-09-23 要求不等待全部下载，先测已有数据。44 已空闲，43 仍有其他任务，
因此先用 `h100_8_native_qualification.yaml` 在 44 做完整模型 8 卡、20 步资格测试。
这不是缩小模型的模拟实验：模型、RGB 分辨率、K8、三视角、action 与 3D loss 均不变；
仅 batch 为 8，资格调度为 20 步，第 20 步保存，以验证真实前后向与 checkpoint。

资格数据位于 `/data/Minko/wm3d_native_available_qualification_data_20260923`。
使用三个已存在的 RoboCasa source 及已恢复的 OXE mutex/hydra：每源取既定 1000h
清单中前 8 个训练 episode、原 split 的前 2 个验证 episode，不按动作或预测质量挑选。
继承原数据的物理标签、camera 绑定、数值归一化和文本向量，不改正式数据文件。
小集只判断工程是否跑通，不证明 1000h 收敛或跨 source 能力。

首次资格根 `/data/Minko/wm3d_native_available_q20_8gpu_20260923` 已保留：8 卡预检通过，
首个真实前后向完成，但指标累积时发现 upstream 部分日志为 Python float，训练器要求
Tensor，尚未完成优化器更新。现仅把日志统一转成 Tensor，total 的反传图保持原样。
修复后 fresh 重测使用 44 的
`/data/Minko/wm3d_native_available_q20_8gpu_20260923_attempt2`，不复用失败运行。

attempt2 已完成真实 20 次优化器更新并正常退出。step20 的 COMMITTED、
8 个 distcp 分片和 8 个 rank state 已齐全，checkpoint 约 76 GiB。
MoT、native 3D、action time、proprio 和 robot conditioning 均有有限非零梯度。
step20 total 1.711914、video 0.676270、action 0.803711、3D 2.324219、grad norm 8.433914。
各步使用不同样本及扩散噪声，不能据此推断质量趋势。
排除冷启动与验证后的 step12–20 平均 5.39 秒/步，范围 4.71–6.20 秒；
单机 global batch 8、短程小集吞吐约 1.48 samples/s，不能外推为全量双机吞吐。

Demo 导出也已补齐 native FSDP 路径：按保存时的 8/16 卡拓扑严格读回 checkpoint，
各 rank 显式 unshard 后独立采样并分目录输出，不伪装单卡同拓扑；生成输入去除未来
3D 标签，逐 horizon 3D 指标仅在预测完成后计算。独立读回/生成服务为
`wm3d-native-available-q20-eval-20260923`，日志在 attempt2 根的 `logs/eval.log`；
读回和生成已正常退出，8 份 receipt 均生成，future action 标签对独立 policy 的差异严格为 0。
`foundation-demo` 接受 `--` 分隔的 torchrun 拓扑与应用参数。

生成质量尚不达标：8 个样本未来帧间变化保持率为 13.1%–37.3%，方向余弦约 -0.004–0.048；
RGB L1 均差于 copy-last，action MAE 均高于归一化零值基线。不能把画面有变化当作正确运动。
对 Hydra 和 RoboCasa 各一组完整 9 帧做时序检查，保留场景结构，但生成机械臂运动仍弱，
后期细节模糊；GIF 时间 0–2.8 秒仅为播放时间，非机器人控制时钟。没有单色或逐帧完全相同。
这轮仅训练 20 步，说明工程链可运行，不能确认质量，也不能单凭它归因为模型结构错误。

接续双机资格根为 `/data/Minko/wm3d_native_available_q100_2node16_20260923`。
43/44 均已空闲，使用上述相同资格数据与模型，fresh Wan 初始化；16 卡 micro1/acc1/global16，
100 步，不 resume 8 卡 checkpoint。配置从 `h100_16_native_pilot.yaml` 仅改资格身份、
移除全量 allowlist 字段（资格 profile 已只含原清单选中的 episode），未改正式清单或 source 比例。
运行配置 `/data/Minko/wm3d_native_pretrain_checks_20260923/h100_16_available_qualification.yaml`。
服务 `wm3d-native-available-q100-n43-20260923` 与 `wm3d-native-available-q100-n44-20260923`；
日志在运行根 `logs/preflight_n43.log`、`preflight_n44.log`、`train_n43.log`、`train_n44.log`。
双机资格于 2026-09-24 00:25 正常完成 100 步，step100 COMMITTED、16 distcp、
16 rank_state、梯度所有权和双机读取均通过。step100 total 1.103271、video 0.246033、
action 0.775513、3D 0.816650、grad norm 3.969389。末尾 step82–100 为 14.67 秒/步、
1.09 samples/s。同 16 卡拓扑独立读回与 joint 生成已正常完成，16 份 receipt 的
future-label policy 差异均严格为 0，没有单色或逐帧完全相同。

固定 seed7340 与 q20 对齐了 8 组 source/sample，配对均值如下：

| 生成诊断 | q20 | q100 | 参考基线 |
| --- | ---: | ---: | ---: |
| Action 归一化 MAE | 1.09212 | 0.82993 | 归一化零值 0.69487 |
| 3D 第八时刻 delta error | 1.22391 | 0.71420 | copy-last 0.71763 |
| 3D 第八时刻方向余弦 | 0.19513 | 0.30589 | — |
| RGB 帧间变化保持率 | 23.91% | 65.58% | — |
| RGB 时序方向余弦 | 0.02557 | 0.01842 | — |
| RGB L1 | 0.09392 | 0.09606 | copy-last 0.06447 |

Action/3D 有学习趋势，但 RGB 变化增加不等于运动正确。完整 9 帧检查显示姿态偏差、
重影和尾部模糊；16 个 q100 样本中 RGB 没有优于 copy-last，policy 仅 1 个优于归一化零值。
归一化零值对应 source offset，不是物理 no-op。样本仅覆盖 Hydra/RoboCasa composite；
两轮训练曝光量和调度不同，不是只改变步数的严格消融，更不能据此证明成熟质量。
报告位于 q100 根的 `eval/qualification_report_20260924.md` 和 `eval/q20_q100_comparison.json`。

正式启动须等待完整数据恢复、必要验收与拓扑确认。“双卡”指两台 16 卡还是两张卡
尚待用户确认；当前仅继续已批准的 43/44 资格和准备，不擅自启动 32 卡。
首轮正式按 2000 更新步的保守分段预算准备，不采用规划文件的 100000 步。

## Micro-batch 容量测量

独立工作树使用同一生产模型、真实数据、实际 AdamW 更新和累积 1，依次测试
micro1、micro2，后者稳定才测试更大值。保持分辨率、K8、3D/action 条件、
loss、学习率及原数据采样语义不变。每次新建资格根，不复用资格 checkpoint 初始化。

设置 `WM3D_PROFILE_GPU_MEMORY=1` 后，现有 trainer 会写各 rank 的
`gpu_memory_rank_00000.jsonl` 等记录，包含初始化、前后向、optimizer、每个验证 batch
的 allocated/reserved 峰值、设备剩余显存与实际 tensor shapes。默认关闭时不执行 CUDA
同步或采样。开启后的少量同步/I/O 开销应在吞吐报告中说明。

以所有 rank 的最坏峰值为准，至少保留 4–6 GiB，选择稳定且 samples/s 更好的最大 batch；
不能只看加载期或 nvidia-smi 瞬时值。OOM 只允许发生在独立短资格进程，立即停止该组，
不影响其他任务。完成后仍需完整数据预检和真实读取，五源小集不代表全量就绪。

2026-09-24 的同模型短程容量结果（step12–20，排除冷启动及验证计时窗口）：

| train micro / global | 秒/步 | samples/s | 前后向峰值 allocated / reserved GiB | 阶段末最少空闲 GiB |
| --- | ---: | ---: | ---: | ---: |
| 1 / 16 | 14.82 | 1.079 | 51.27 / 76.83 | 3.22 |
| 2 / 32 | 14.96 | 2.138 | 51.88 / 77.39 | 3.32 |
| 4 / 64 | 15.76 | 4.061 | 62.10 / 70.47 | 8.78 |

三档均为 fresh Wan、20 次真实更新和两次验证，无 OOM 或非有限值，owner 梯度有限非零。
验证 batch 保持 micro1/global16：小集 mutex 验证只有 21 个窗口，不能无放回组成 global32。
第一次 micro2 在此容量检查退出，未训练、未 OOM；未通过复制窗口或删除 source 绕过检查。
micro4 验证峰值 allocated 42.56 GiB，吞吐约为 micro1 的 3.76 倍。
reserved 包含可回收缓存，不能等同活跃张量；不同 batch 的缓存布局也不同。
上述结果只覆盖五源小集。小集最小 train 容量 70，不支持更大的 global batch，
因此更大档位使用恢复后的完整原索引，未复制窗口或删除 source。
小集的配置、逐 rank 显存和日志分别保留在
`/data/Minko/wm3d_native_capacity_micro1_2node16_20260924`、
`/data/Minko/wm3d_native_capacity_micro2_2node16_20260924_attempt2`、
`/data/Minko/wm3d_native_capacity_micro4_2node16_20260924`。

### 全量容量验收与选档

2026-09-24 的全量 micro4 attempt2 与 micro6 均正常完成：生产模型 fresh Wan 初始化，
原 1000h 索引及比例，累积1；各20次实际更新、step10/20各4批验证。
16rank的optimizer均到step20，COMMITTED、16 distcp、16 rank_state及梯度所有权通过，
双机保存记录和分片可读性一致。没有OOM或非有限值，ECC均为0。
本轮没有重复完整模型生成读回，不能把容量验收当作RGB/VLA质量验收。

| 全量 train micro / global | 秒/步 | samples/s | 峰值 allocated / reserved GiB | 阶段末最少空闲 GiB |
| --- | ---: | ---: | ---: | ---: |
| 4 / 64 | 16.7786 | 3.81439 | 62.3296 / 70.4473 | 8.8032 |
| 6 / 96 | 20.3846 | 4.70945 | 72.7140 / 76.5293 | 4.9067 |

速度均取step12–20，排除加载、验证及保存窗口；保留真实的数据等待。
micro6的DROID慢步为40.62秒，rank0新增prefetch等待21.01秒，已计入平均值。
micro6吞吐提高23.47%，但余量已接近4–6GiB安全边界，故选micro6，不再试更大档。
最大实际action为[6,32,42]、history为[6,128,42]，video/native3D形状保持原合同。
20步实际抽到6源，不冒充18源GPU全覆盖；此前CPU真实读取已覆盖18源104个样本。
记录在 `/data/Minko/wm3d_native_pretrain_checks_20260923/full_capacity_micro6_report_20260924.md`。

正式首段配置已准备在上述checks目录的 `h100_16_native_formal2000_micro6_20260924.yaml`，
仅用于用户确认43/44两台16卡后启动。2000更新步，micro6/acc1/global96，验证micro1；
optimizer峰值学习率不变，视频骨干保持0.1倍。warmup100步，log每10步，
每100步验证20批，checkpoint为100/500/1000/2000。不得resume任何资格checkpoint。
2000步并不代表完整遍历1000h；按短程实测纯更新约11.3小时，另加加载、验证、保存和I/O波动。
截至本次记录，正式拓扑尚待确认，未启动正式训练，未使用32卡。

## 验收与当前证据

截至 2026-09-24，含显存测量和全量采样接口修复，全仓检查为 659 passed、1 skipped。真实 Wan 权重的 CPU 建模已完成，
未加载 WSA checkpoint；总参数约 7.247B、可训练参数约 6.542B，包含约 5.000B 视频骨干、
1.021B action、0.433B 3D。参数注册已去重，避免重复保存同一 MoT 权重。
真实 8/16 卡训练、保存、同拓扑独立读回和无未来标签生成均已运行；生成质量尚未达标。
不能表述为 WM3D 已收敛或优于 WSA。

试训先确认真实数据能读、各训练 owner 梯度有限非零、优化器确实更新参数、训练/推理边界
一致和 checkpoint 可恢复，再记录稳态秒/步、samples/s、峰值显存、解码与 VGGT 耗时。
用固定真实样本比较无目标输入的 joint RGB、独立 action 和每个未来时刻的 3D：

- RGB：target/prediction/copy-last 按同 K8 时序并排，区分运动方向、幅度和静态清晰度。
- 3D：逐 horizon 方向、幅度、error 相对 copy-last；测试 observation/3D 消融是否影响生成。
- Action：物理反归一化误差、方向和幅值、gripper、task/vision/state/history 消融；不能用
  world token gain 代替 policy 能力，也不能把训练标签参与的 flow loss 当作生成质量。
- 保存完整 step100 后做独立进程读回。有限的 100 步只资格验证和测速，不证明最终 VLA
  或约 1000h 的充分收敛；闭环任务留给后续阶段，不能只用 LIBERO 下结论。

原始数据已于 2026-09-24 02:18 恢复完成：此前 43/44/42 未找到完整 OXE/DROID 文件，RoboCasa
原文件仍在。已按原清单逐文件核对公开资产，14 个 OXE source 对应 main，DROID 对应
v3.0；恢复约 412 GiB 的原文件，不更换索引、adapter 或采样比例，不使用 41。
下载服务 `wm3d-native-raw-recovery-20260923` 在 43 运行，数据写入 44 的
`/data/Minko/raw_recovery_20260923`，43 通过 NFS 访问同一路径。
进度、日志与完成记录在 `/data/Minko/wm3d_native_pretrain_checks_20260923`，分别为
`raw_recovery_progress.json`、`raw_recovery.log`、`RAW_RECOVERY_COMPLETE.json`。
双方 `/shared/Minko/datasets` 已只读映射恢复目录，以支持已恢复 source 的独立资格小集。
15 个恢复源均有完成记录，下载服务正常退出。原始完整索引的 train/val 共 104 个真实
样本已通过 RGB/action 输入转换、finite 与文本 context 检查；18 个训练源均覆盖，
原 buds 验证集为零窗口，未添加或删除数据。总量仍为 4,922,451 train / 127,991 val。
这项 CPU 读取检查不替代全量 GPU 容量和生成质量验收。开训前仍核对 NFS 和磁盘空间。

43 的其他任务和 44 单机资格读回均已结束。每次启动前仍需查占用，
不得停止或共享其他项目 GPU。全量恢复后使用完整索引执行预检与真实读取；q100 已完成，不机械重复相同资格。
正式启动仍需拓扑确认，且不得绕过缺失 source 开训。

### 全量采样接口修复

首次全量 micro4 预检在构造采样器时退出，没有训练更新或 checkpoint。
1000h episode 清单适配器返回小数比例，整数采样器将其截为零。小集资格直接使用
整数权重的独立 profile，未触发这条接口。修复后传入各源原始选中窗口数，
不更改选中的 episode、窗口、比例或归一化；采样器也显式拒绝小数权重。
新增清单与生产 sampler 联动测试，防止仅在函数内检查比例而漏掉调用方类型。

计数权重对应 4,922,451 个调度位置。现只缓存原算法的每轮洗牌及 source occurrence
位置，不改 seed、顺序或断点恢复寻址。真实清单的首次构建约 4.55 秒，随后 2000 次
寻址共约 0.10 秒；跨轮、乱序访问与恢复地址测试通过。这是 CPU 调度开销，不是 GPU 吞吐。
原失败根 `/data/Minko/wm3d_native_full_capacity_micro4_2node16_20260924` 保留。
后续使用独立 attempt2 根 fresh Wan 运行同一 micro4/acc1/global64、验证 micro1/global16
的全量20步容量资格，不复用旧资格checkpoint；随后micro6也已完成，结果见上方全量容量节。

## 来源与实验材料说明

本计划使用 academic-research-suite 的 experiment-agent / code experiment plan 工作流，
形成于 2026-09-23。材料来源：本机实际代码/资产、既定 1000h receipt、官方仓库；
验证状态为代码、8/16 卡训练/保存/独立读回/推理工程检查已验；已有部分学习趋势，但本文不证明生成质量。

- [WSA 官方实现](https://github.com/zaleni/WSA)：MoT、WSALarge mask、flow training 和 sampler。
- [WSA 预训练入口](https://github.com/zaleni/WSA/blob/main/launch/wsa_large_pretrain.sh)：
  pipeline 参考。已采用其窄专家 Wan 派生初始化，不使用 WSA 成品 checkpoint。
- [Wan2.2 官方实现](https://github.com/Wan-Video/Wan2.2)：通用视频/VAE 先验。
