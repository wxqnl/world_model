# 集群站点配置

当前默认模型为 native direct RGB。5B 只使用 `h200_5b_direct.env.example`，
通过下面的入口生成站点文件，不再维护重复的 h200_5b.env.example：

```bash
./run_wm3d.sh 5b configure /共享目录/模型 /共享目录/已下载数据
```

完整顺序见 ../../docs/WM3D_5B_SCALING.md。先资格，再 fresh 正式；
已有 checkpoint 只能按同一 run 的恢复合同读取。

1B 兼容入口是 `./run_wm3d.sh 1b init`，模板为 h100_1b_direct.env.example。
当前本地 16×H100 全量正式训练使用独立封存 runtime，不能用单节点模板覆盖。

站点文件只保存环境、路径和拓扑；数据范围由已审计 profile 决定。
下载来源不限于 Hugging Face，已有模型/数据不要求重新下载。
旧缓存模式保留给已有工作流；新交付默认 direct_raw，不缓存全部视频特征。

填写后的 site、token、运行日志、数据和 checkpoint 不提交到 Git。
