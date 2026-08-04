# 集群配置

`h200.env.example` 是新站点唯一需要复制和编辑的环境变量样例：

```bash
./wm3d.sh init site.env
```

必须填写共享 `WORK_ROOT`、Hugging Face token 文件、Slurm partition/account，以及数据许可
确认。路径必须在所有计算节点上保持一致；不要把 token 内容直接写进 `site.env`。

`./wm3d.sh doctor site.env` 会检查普通文件权限、Python 环境、Slurm、共享目录和必要命令；
`./wm3d.sh plan site.env` 只打印完整计划，不提交任务。正式 preflight 还会在计算节点核验 H200
型号、8 卡 NVLink clique、InfiniBand、磁盘、共享内存和 GPU 空闲状态。

更换集群时只修改站点变量，不要把绝对路径写进训练 YAML 或 Python 源码。
