# WM3D V8 5B 训练交接

5B 只保留一条交付路径。请直接按
[`WM3D_5B_SCALING.md`](WM3D_5B_SCALING.md) 从“只填写模型和数据目录”开始执行，
不要复用旧 runtime、旧 site 或任何 canary/checkpoint。

当前 `v8` 的 1B 与 5B 共用同一套实现语义：

- source-normalized action 先按封存 profile 还原为共同物理单位；
- group-preserving factual action 在 block 0 前进入 future state；
- factual P144 是 5B 唯一运动所有者，RGB 使用其 backward flow；
- 高频 refiner 只能补有界高通细节；
- future candidate 严格不进入 policy/action-free trunk；
- absolute future P256、teacher forcing 和 renderer-only action 通路均关闭。

先跑 64 卡 1K canary 并完成 checkpoint/resume/eval 门槛；通过后再从全新的 step 0
启动正式 600K。数据可来自魔搭、Hugging Face 或内部存储，入口只接收本地
`MODEL_ROOT` 和 `DATA_ROOT`，不会把下载来源写死。
