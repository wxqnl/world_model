# 模型与正式预训练

## 模型

WM3D 使用 Wan2.2-TI2V-5B 视频先验，复用 WSA 的 Mixture-of-Transformers 训练／采样实现，联合学习动作和 native 3D 未来轨迹。正式预训练从 Wan 视频骨干和声明的分支初始权重开始，未加载训练完成的 WSA checkpoint。

| 项目 | 正式设置 |
| --- | --- |
| 模型入口 | `FoundationWorldModel` |
| 历史／未来容量 | T16 / K8 |
| 几何空间查询 | P64，最多三个视角 |
| 视频画布 | 384×320 |
| 3D query / readout | slot-noise / WSA perceiver |
| 监督 | 视频、动作、逐时刻 native 3D 轨迹 |
| 几何教师 | 冻结 VGGT-Omega 512；时间折入 batch，未来特征只作监督 |
| 机器人接口 | 最多三组；每组动作14维、状态10维；缺失维度与相机使用有效掩码 |
| 训练精度 | BF16 计算，FP32 主参数、Adam 状态与梯度归约 |

关键实现位于 `wm3d/models/foundation_world_model.py`、`foundation_native.py`、
`wm3d/data/foundation_bridge.py`、`foundation_input.py` 和 `wm3d/encoders/native_omega.py`。
生产模型配置另存为 `configs/model/native_trajectory.yaml`，保留原运行声明的站点资产路径，部署时需显式绑定本地资产。

## 已完成的正式预训练

正式运行在 43／44／41 的 24×H100 上完成 167,446 次 optimizer 更新。数据与来源采样倍数按最终正式 runtime 执行；这里的总步数不等同于所有来源各完整遍历两遍。

最终 native checkpoint `step_00167446` 已完整提交、登记，并完成自身 CPU 全状态读回：
模型 2,354 个状态 tensor、2,157 个 Adam 参数状态及全部一／二阶 moment、24 个 rank 的 sampler 与 Python／NumPy／Torch CPU／CUDA RNG 均已反序列化，数值检查通过。正常保留的预训练完整 checkpoint 为 150k、160k 和 167446。

训练状态和质量评测分别记录。旧 milestone 评测已停止；已有九次结果及 100k 失败现场保留，不补记为质量通过。

## 接续

新的下游 SFT 严格继承最终预训练模型的全部参数与 buffers，创建自己的 Adam、采样器与 LR 日程。下游中断恢复加载该下游运行自己的完整 DCP，包含 Adam、采样进度及同 rank RNG。

站点上的最终源 checkpoint、CPU 完整性证明和 model-only 交接文件保存在运行目录；Git 仅保存实现和数值配置。当前三项下游的协议见 [BENCHMARKS.md](BENCHMARKS.md)。
