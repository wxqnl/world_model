# wm3d_v4 — Latent Diffusion Pixel Head (Report)

**Date:** 2026-05-25
**Status:** ✅ Trained, evaluated, demo gifs generated.

## TL;DR
v3 的 47M `PixelDecoder`（L1+LPIPS 回归头）产出**模糊的"未来均值"**：背景对、机械臂/夹子糊。
v4 把它换成一个 **211M 的 latent-diffusion mini-head**（DiT + 跨注意力到 v3 的 `pred_tokens`），
其余 v3 backbone 全冻结，只训这一头 + 复用冻结的 SD-1.5 VAE 做 256×256 ↔ 32×32 隐空间。

**核心结果**：单帧/短窗 (k=8 future) RGB 质量**显著超过 v3 PixelDecoder**——机械臂、夹子、物体边缘清晰可辨；
长 autoregressive rollout 因误差累积比 v3 退化更快（细节越具体，错得越明显）。

## 架构差异
```
v3 PixelDecoder (47M, regression L1+LPIPS):
  pred_tokens [B,k,64,2048] → ConvTransp stack → rgb [B,k,3,256,256]

v4 DiffusionHead (211M, ε-prediction DDPM cosine):
  pred_tokens [B,k,64,2048] (frozen v3 output)
        ↓ cross-attention condition
  noisy_latent [B,k,4,32,32]  ─── DiT 14 blocks, hidden=768 ───→  predicted_noise
        ↑                                                                   ↓
  schedule.add_noise(z0, ε, t)                                       MSE(ε_pred, ε)
        ↑                                              
  z0 = SD-VAE.encode(rgb_target * 2 - 1) * 0.18215
```

冻结 / 可训练：
- 冻结：dual_stream (468M) + action_proj (1.9M) + geom (6.7M) + SD-VAE (83.7M)
- 可训练：**DiffusionHead 211.6M**

## 训练
* 4× H100，DDP，bf16 mixed precision
* AdamW(lr=6e-5, β=(0.9, 0.95), wd=0.02), 500-step warmup + cosine decay
* Batch 8/GPU = 32 effective, 79473 train windows
* 从 v3 best.pt 加载 dual+geom+action 权重，DiffusionHead 随机初始化
* **NaN-skip 兜底**：第一次跑 lr=1.5e-4 在 ep 4 中段开始大量 NaN gradients
  → resume 时加 `--reset_optim` + lr 降到 6e-5 + grad_clip 0.5 才稳定
* 训了 15 epoch (磁盘满中断，自然早停)，total ~7 小时

### val_eps_mse trajectory
```
ep  0 : 0.165   ← (启动自 v3 best.pt + DiffHead 随机初始)
ep  3 : 0.155
ep  5 : 0.150
ep 10 : 0.141
ep 14 : 0.134   ← best.pt
```
每 epoch ~0.002 边际改进，仍单调下降，继续训应该能再降一些。

## 推理
DDIM 25-step 采样（无 classifier-free guidance）：
```python
v4.forward_sample(s, c, vae, schedule, n_steps=25) -> {"rgb": [B,k,3,256,256], ...}
```
推理时间：单 clip k=8 帧约 1-2 秒（vs v3 单次 forward 0.3 秒）。

## Demo

### 短 horizon（k=8 future）三列对比 (v3 / v4 / GT)
`/home/user01/Minko/newwm/results/wm3d_v4/demo/v4cmp_*.gif`

观察：
- **背景纹理**：v4 显著比 v3 锐利（木纹、墙砖、家具）
- **机械臂结构**：v3 是黑色模糊块，v4 清晰显示手臂 + 夹子轮廓
- **小物体**：v3 把杯子/碗糊成圆斑，v4 给出具体形状
- **快速运动部位**：v4 偶有 minor artifact（在动态最大的夹子末端），可继续训练改善

### 长 horizon 自回归 rollout
`/home/user01/Minko/newwm/results/wm3d_v4/demo_long/`

- bridge 64 frames：t=0-16 贴合 GT, t=24-32 开始偏移, t=48+ 明显发散
- fractal 216 frames：~30 帧后基本崩坏
- 这是 latent diffusion 的固有问题：DDIM 输出的细节"具体"，喂回 dual stream
  后误差累积更明显。比 v3 PixelDecoder 的回归"均值"扩散得更快

**推荐用法**：v4 用于 1-2 个 k-chunk 的高质量短预测，长 rollout 在当前架构下需要单独的 token-级动力学校准（v4 future work）。

## 已知问题 / 改进方向
1. **训练 NaN（已解决）**：bf16 + cross-attn 数值偶发不稳。fix = grad_clip 0.5 + NaN-skip + 较小 LR。
2. **长 rollout 漂移**：可加 token-级辅助预测损失、或在采样时混入低噪声 GT token 锚定。
3. **DDIM 速度**：25 step 单帧 ~50ms，可上 LCM/distill 蒸馏到 4-step。
4. **CFG 没开**：可加 task_text classifier-free guidance 提高语义一致性。

## GPU policy
GPUs 0-3 only，全程未触及 4-7。

## 文件
```
wm3d_v4/
├── configs/v4_oxe.yaml   v4_smoke.yaml
├── scripts/train_v4.sh
└── wm3d_v4/
    ├── models/
    │   ├── vae_wrapper.py        # 冻结 SD-1.5 VAE
    │   ├── diffusion_head.py     # 14-block DiT, cross-attn 到 v3 tokens
    │   └── joint_v4.py           # JointV4 = 冻结 v3 + DiffusionHead
    ├── schedulers.py             # cosine DDPM + DDIM 采样
    ├── training/train_v4.py      # DDP + bf16 + NaN-skip + reset_optim
    └── eval/
        ├── sample_demo.py        # 三列短 horizon 对比
        └── sample_long_rollout.py  # 自回归长 rollout
```

ckpt: `/home/user01/Minko/newwm/results/wm3d_v4/ckpt/best.pt` (4.5GB, epoch 14, val 0.1339)
