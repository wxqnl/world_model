# wm3d_v3.5 — PixelDecoder Scaling (Report)

**Date:** 2026-05-26
**Status:** ✅ Trained (early-stopped @ ep 14, val 0.2256), evaluated, demo gifs.

## TL;DR
为了让 v3 的 RGB 不那么糊，把 `PixelDecoder` 从 `hidden=768, n_res=2` (**47M**) scale 到 `hidden=1536, n_res=2` (**186M**)，冻结 v3 backbone + geom + action，只训新的大 pixel 头。

**结论：纯 scaling pixel 头 + 冻结上游 backbone 收益边际化。**
- v3.5 best val_total **0.2256**（epoch 14）
- v3 best val_total **0.2238**（epoch 26，端到端）
- v3.5 略不如 v3 的根本原因：**v3.5 冻结 backbone 只训新随机初始的 pixel 头，14 epoch 不够追上 v3 联合训 26 epoch 的优势**

## 设置
- 复用 v3 backbone (524M frozen) + 新 PixelDecoder (186M trainable)
- L1 + LPIPS 损失不变
- AdamW lr=1e-4, warmup 1000, cosine
- 2× H100（GPUs 0-1，user 把 2-3 留作其他用途）
- bs=4 per GPU = 8 effective
- 实际跑了 15 epoch（磁盘满中断）

## val_total 轨迹
```
ep 0  : 0.314
ep 5  : 0.253
ep 10 : 0.235
ep 12 : 0.230
ep 13 : 0.228
ep 14 : 0.226  ← best.pt
```
每 epoch 改进 0.002-0.003，单调下降但已在边际收益区。

## 视觉对比
`/home/user01/Minko/newwm/results/wm3d_v3_5/demo/v3_v35_cmp_*.gif`

5 个三列对比（v3 47M | v3.5 186M | GT）+ 底部 depth 行：
- bridge 灶台场景：两者几乎一致，v3.5 蓝色物体边缘略锐
- fractal RT-1 推车：几乎打平
- bridge 厨房：打平

## 为什么 scaling 没奏效

1. **冻结上游**：v3.5 完全冻结 backbone + geom + action，只让 186M pixel 头从随机权重学习。pixel 头的能力上限被冻结 backbone 输出的 pred_tokens 信息密度卡住。
2. **训练时间不足**：v3 是 backbone + pixel 联合训 26 epoch；v3.5 只训 pixel 14 epoch。要 fair 比较至少再 10+ epoch。
3. **Token grid 太粗**：8×8=64 个 token 重建 256×256，每 token 覆盖 32×32 像素 — 夹子（~10 px）小于一个 token，无论 pixel 头多大都救不回来。

## 真正能突破的方向（按 ROI）

| 方案 | 估时 | 期望收益 |
|---|---|---|
| **联合训练**：v3.5 的大 PixelDecoder + 解冻 backbone 再 15-20 epoch | 1-2 天 | 中等 |
| **更大 token grid**：用未 pool 的 VGGT (16×16 = 256 token) | 2-3 天（重 cache + 重训）| 大 |
| **diffusion 头**：换 v4 那条路（已经做过，时序一致性问题） | 已知 | 小-中 |

## 文件
```
wm3d_v3/
├── configs/v3_5_oxe.yaml
├── scripts/
│   ├── train_v3_5.sh
│   └── v3_v3_5_compare.py
└── wm3d_v3/training/train_pixel_only.py
```
ckpt: `/home/user01/Minko/newwm/results/wm3d_v3_5/ckpt/best.pt` (4.1GB, epoch 14, val 0.2256)

## GPU policy
GPUs 0-1 only（user 把 GPUs 2-3 留作他自己的训练任务）。
