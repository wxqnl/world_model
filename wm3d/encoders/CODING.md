# 编码缓存实现导读

本目录负责把公开 RGB 视频转换成 WM3D 训练需要的冻结视觉与几何表示。正式预训练不加载 VGGT；这里的
输出会先写入 sealed cache，再由 `wm3d.data` 读取。

## 1. 组件边界

| 文件 | 真实职责 |
|---|---|
| `vggt_encoder.py` | 加载固定源码与固定 revision 的 VGGT，提取 patch token 和原始几何 head 输出 |
| `vggt_features.py` | 组织三视角与时间、pool 到 12×12、处理缺失相机、形成 WM3D cache 字段 |

VGGT 提供观测表示和 pseudo-label，不负责未来预测，也不在 world-model optimizer 中。

## 2. 为什么使用 VGGT

[VGGT](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_VGGT_Visual_Geometry_Grounded_Transformer_CVPR_2025_paper.html)
在一个模型中产生 camera、depth、point map 和跨视角几何 token。WM3D 使用它解决两件事：

1. 把来源不同的普通 RGB 数据映射到统一的 2048D 几何特征接口；
2. 为没有完整传感器 GT 的数据生成 depth、point、camera 和 confidence 监督。

代价是几何监督上限受 VGGT 误差影响。代码不会把 pseudo-label 描述成传感器真值。

## 3. 资产必须固定

`VGGTEncoder.__init__()` 不允许省略 revision：

```python
if model_revision is None:
    raise ValueError("model_revision is required for VGGTEncoder")

snapshot_path = Path(snapshot_download(
    repo_id=model_name,
    revision=model_revision,
    local_files_only=local_files_only,
)).resolve(strict=True)

if snapshot_path.name != str(model_revision):
    raise RuntimeError("VGGT snapshot revision mismatch")
```

代码还检查导入的 `VGGT` class 是否来自登记的本地源码树：

```python
source_file = Path(inspect.getsourcefile(VGGT) or "").resolve(strict=True)
source_file.relative_to(source_root)
```

这样可以防止环境中另一个同名 Python 包被意外导入。源码、权重文件集和 revision 最终都会写入 asset
receipt；cache seal 与 receipt 绑定。

## 4. 时间维为什么折入 batch

`VGGTFeatureEncoder.forward()` 的输入是：

```text
[B,T,V,3,H,W]
```

经过：

```python
active_images = images.index_select(2, active_indices)
flat = active_images.reshape(
    batch * times,
    active_views,
    channels,
    height,
    width,
)
encoded = self.encoder(flat)
```

VGGT 实际看到的是：

```text
[B*T,V,3,H,W]
```

因此同一时间的 head/left/right 相机可以联合建立几何，不同时间被视为不同 batch item，不能在 VGGT 内
互相注意。这个 reshape 是 cache 层的未来泄漏保护；如果把 T 和 V 一起放进 VGGT sequence，context token
可能已经编码了未来图像，后续 causal world model 再严格也无法修复。

## 5. 缺失相机不是黑图

代码要求同一个 encoder batch 使用稳定的相机可用布局：

```python
availability = view_mask[0, 0]
if not bool((view_mask == availability[None, None]).all()):
    raise ValueError(
        "one encoder batch must use a stable camera-availability layout"
    )
active_indices = torch.nonzero(availability).flatten()
active_images = images.index_select(2, active_indices)
```

缺失相机在送入 VGGT 前被移除，不用全黑图占位。黑图仍会产生 feature 和几何响应，容易被模型误当作真实
证据。VGGT 返回结果后，代码再按固定的三个 view slot 填回零值并写 `view_mask`。

约束的另一面是：一个 cache batch 不能混合不同相机布局。构建器需要先按 embodiment/view layout 分组。

## 6. Patch token 的提取与 pooling

VGGT wrapper 调用 aggregator：

```python
aggregated_tokens, patch_start_idx = self.model.aggregator(images)
tokens = aggregated_tokens[-1]
patch_tokens = tokens[:, :, int(patch_start_idx):, :]
pooled = self._pool_patch_tokens(patch_tokens).to(torch.float16)
```

使用最后一层聚合 token，并去掉 patch 起点之前的特殊 token。原始方形 patch grid 经自适应平均池化到 12×12：

```python
x = patch_tokens.reshape(b * t, grid, grid, d)
x = x.permute(0, 3, 1, 2)
x = F.adaptive_avg_pool2d(x.float(), (12, 12))
x = x.permute(0, 2, 3, 1).reshape(b, t, 144, d)
```

Pooling 在 FP32 中执行，结果存 FP16/BF16。正式接口要求 `d == 2048`，维度漂移会直接报错。

12×12 是空间分辨率与长序列成本之间的训练配置。它保留二维 lattice，但不是显式 voxelization；更高 P 是否
改善小物体和 RGB 清晰度需要用吞吐、显存和下游效果共同评估。

## 7. 显式几何 head

正式 cache 要求 VGGT 的三个 head 都存在：

```python
depth, depth_conf = depth_head(...)
pose_enc = camera_head(aggregated_tokens)[-1]
world_points, world_points_conf = point_head(...)
```

缺少任一 head 时：

```python
if encoded.get("geom_extra_missing"):
    raise RuntimeError("formal wm3d requires all VGGT geometry heads")
```

Depth、point 与 confidence 都 pool 到相同的 12×12 patch grid。Camera 使用 VGGT pose encoding 前 9 维：

```python
if active_pose.shape[-1] < 9:
    raise RuntimeError(...)
active_pose = active_pose[..., :9]
```

模型后续预测的是这 9D 表示，并非直接约束后的 SE(3) 矩阵。

## 8. Geometry confidence

Depth 与 point confidence 合成为：

```python
active_confidence = torch.sqrt(
    depth_confidence.float().clamp_min(0.0)
    * point_confidence.float().clamp_min(0.0)
)
active_confidence = active_confidence / active_confidence.amax(
    dim=(-1, -2), keepdim=True
).clamp_min(1.0e-6)
```

几何平均要求 depth 和 point 两种证据都可靠；随后按每个样本/时间的视角与 patch 最大值归一化到 `[0,1]`。
这是本项目的 confidence 合成规则，不是 VGGT 论文规定的概率校准。

## 9. `view_tokens` 与 `world_tokens`

编码器同时输出两种 token：

```python
weights = confidence[..., None]
world_tokens = (view_tokens.float() * weights).sum(dim=2) / (
    weights.sum(dim=2).clamp_min(1.0e-6)
)
```

| 字段 | 形状 | 用途 |
|---|---|---|
| `view_tokens` | `[B,T,3,144,2048]` | 模型 context 输入，保留逐视角信息 |
| `world_tokens` | `[B,T,144,2048]` | future token target 与低频 memory summary |

Context 输入由模型自己的 `MultiViewTokenFuser` 学习融合；future target 使用冻结 confidence 加权融合。这两个
方向不完全对称：context fuser 当前不直接读取 geometry confidence。

## 10. RGB 与最终输出

原始 RGB 双线性缩放到 384×384，并存为 uint8：

```python
rgb = F.interpolate(
    images.reshape(batch * times * views, 3, height, width),
    size=(384, 384),
    mode="bilinear",
    align_corners=False,
    antialias=True,
)
rgb = rgb.mul(255.0).round().clamp(0, 255).to(torch.uint8)
```

完整输出为：

```python
{
    "view_tokens": ...,          # BF16
    "view_mask": ...,            # bool
    "world_tokens": ...,         # BF16
    "frame_summary": ...,        # BF16
    "rgb": ...,                  # uint8
    "depth": ...,                # FP16
    "point": ...,                # FP16
    "geometry_confidence": ...,  # FP16
    "camera_pose": ...,           # FP32
}
```

Cache 层只负责当前观测编码，不产生未来预测。

## 11. 已知边界

- VGGT pseudo-label 不是传感器 GT，正式数据如果有可靠 RGB-D/标定，可考虑保留独立 GT 字段用于评测或混合监督；
- adaptive average pooling 会损失小物体边界，P144 是算力折中；
- confidence 是相对归一化值，不是跨数据集可比较的绝对概率；
- camera pose 截取前 9 维，依赖固定 VGGT revision 的 pose contract；
- 每个 encoder batch 要有固定 view layout，cache builder 必须按布局分桶。

这些边界都位于离线观测接口，不改变 state trunk 对未来世界的所有权。
