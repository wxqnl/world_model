# 运行差异

- `worldarena.patch`：WorldArena，源提交 `d0835b5`。
- `robonet.patch`：RoboNet 当前恢复版本，源提交 `d427667`。

公共树来自 `a90d4f1`。在仓库根目录使用 `git apply --check wm3d/variants/<name>.patch`，再在新的 checkout 中应用单份补丁。两者不叠加。

差异涵盖实际数据 reader、时间坐标、采样、验证／选择或恢复行为；现有训练服务不切换到本发布目录。
