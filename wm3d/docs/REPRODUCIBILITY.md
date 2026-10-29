# 代码来源与复现

## 本次发布的来源

| 运行版本 | 源提交 | 发布形态 |
| --- | --- | --- |
| LIBERO／公共实现 | `a90d4f1` | `wm3d/` 公共代码树 |
| WorldArena | `d0835b5` | 相对公共树的 `variants/worldarena.patch` |
| RoboNet 当前恢复版本 | `d427667` | 相对公共树的 `variants/robonet.patch` |

三个来源均从干净的 Git checkout 导出。差异补丁包含该运行的数据适配、模型时间坐标接口和训练／恢复改动，省去两份重复公共代码。公共实现中的 LIBERO 生命周期内存处理与各运行版本按源文件保留。

## 使用运行差异

从仓库根目录，在新的独立 checkout 中选择一份补丁：

```bash
git apply --check wm3d/variants/worldarena.patch
git apply wm3d/variants/worldarena.patch
```

RoboNet 对应 `wm3d/variants/robonet.patch`。两份补丁分别相对于本分支公共实现生成，不叠加使用。应用后提交本地代码版本，再按新的提交身份物化 runtime。清理后的文档、配置和源码同时保留在 checkout 中。

活跃服务器副本未替换，运行服务、GPU分配、代码绑定与日志路径未改变。当前服务器的完整 runtime 含站点路径、数据 stat 绑定、来源权威和 checkpoint 身份，继续保存在原运行目录。

## 环境与数据

依赖锁位于 `environments/`。上游 WSA、Wan、VGGT-Omega 和各数据集分别按其许可证获取，权重与数据不进入本仓库。冻结教师和原生 grouped ABI 的来源仍由物化 runtime 绑定。

`configs/benchmarks/` 保存六份数值 profile；`configs/model/native_trajectory.yaml` 保存最终预训练的模型声明。运行前必须为本机环境、数据、文本特征、模型资产和输出目录生成真实 runtime，并完成资源 preflight。

## 保存与恢复

预训练最终完整 checkpoint 保留原 Adam、sampler 和24 rank RNG。新 SFT 只继承全部模型参数与 buffers；SFT 恢复读取本次自己的完整状态。RoboNet 的恢复代码保留显式源 runtime、完整 CPU 证明与 decoder-only continuation 约束。

本次发布检查运行现有完整 CPU 回归和补丁应用检查。GPU验证证据来自原运行目录，不在代码整理阶段占用训练卡或重跑训练。
