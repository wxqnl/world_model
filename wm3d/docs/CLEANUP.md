# 2026-10-29 整理记录

## 保留

- 当前预训练／SFT公共实现、完整 checkpoint／恢复代码、数据 ABI 和现有回归测试。
- WorldArena、RoboNet 运行版本的全部源码差异，以补丁保存。
- 六份下游数值配方与最终预训练模型声明。
- 服务器上所有模型、数据、正式训练日志、结果、失败首因和恢复证明。

## 删除与归档

- 删除无入口或回归引用的历史 `launch_attempt21_eval.sh`、`benchmark_native_vggt_cache.py`。
- 删除四份已被当前说明替代、无代码引用的实验笔记：`factual_motion_microprobe.md`、`v10_conservative_conditioning.md`、`v10_fp32_adapter_updates.md`、`v10_pretrained_action_prior.md`。
- 旧 native RGB、1B／H200、Stage1、WSA adapter及 pilot 文档移入 `docs/archive/`；旧链接同步调整。
- 两份单用途 optimizer／grad-clip 性能脚本及历史门禁工具保留为开发工具，供既有回归和历史复现；当前入口不主动调度它们。
- `__pycache__`、pytest／ruff缓存、临时补丁备份与日志、权重、环境、凭证均不进入发布树。忽略规则覆盖后续生成的同类产物。

本次使用独立发布 checkout。运行中的代码、日志、readonly绑定和共享 checkpoint 操作锁均未修改。文档测试同步调整归档路径；一项历史 step783 集成测试与同文件其他测试采用相同的真实 checkpoint 存在条件。该历史资产已按原轮转退出，整理时不恢复或模拟它；其余配置、采样与篡改断言保留。

## 检查与临时产物

完整 CPU 回归结果为812 passed、6 skipped。缺失历史 checkpoint 或可选运行条件的集成检查保留明确跳过原因；本次未占用训练GPU。两份代码差异补丁通过应用检查，文档相对链接检查通过。

两份已成功完成的 CPU 数据准备 stdout 进度日志已删除；对应 COMPLETE.json、归一化、原生目录索引和 dispatch 凭证保留。删除前检查了本机 FD、mmap 和直接命令引用，未发现占用或未知读取错误。发布时导出的三份临时代码树在生成差异补丁后清除。
