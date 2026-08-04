# Python 环境

WM3D 使用普通 Python 3.10 `venv`，不依赖 Docker、Conda 或系统服务修改。

| 文件 | 作用 |
|---|---|
| `requirements.lock` | 主训练/数据环境的固定依赖与哈希 |
| `environment_contract.json` | Python、PyTorch、CUDA、NCCL 等运行时契约 |
| `bootstrap_environment.sh` | 创建主 venv、安装 lock、生成环境 receipt |
| `verify_environment.py` | 对照契约核验当前解释器 |
| `agibot_converter_environment_contract.json` | AgiBot 官方转换器的隔离环境契约 |
| `bootstrap_agibot_converter_environment.sh` | 仅在需要转换 Beta 时延迟创建隔离 venv |
| `prepare_lerobot_converter_build.py` | 对固定上游源码做哈希约束的兼容修补 |
| `verify_agibot_converter_environment.py` | 核验转换环境与官方工具 |

新服务器执行：

```bash
./wm3d.sh init site.env
./wm3d.sh setup site.env
./wm3d.sh doctor site.env
```

主环境安装成功后会产生可哈希 receipt。AgiBot 转换环境由 `prepare` 阶段按需创建，避免其
依赖污染训练环境。若要升级任何包，必须同时更新 lock、环境契约、测试和 smoke 证据；不要
在计算节点临时 `pip install` 后继续正式训练。
