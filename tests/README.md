# 测试

| 文件 | 覆盖范围 |
|---|---|
| `test_model.py` | 5B 参数冻结、native forward/backward、未来 action 无泄漏、sampler 恢复、V7 血统审计 |
| `test_data_pipeline.py` | 数据契约、action/aux 对齐、codec、seal 与 dataset 读取 |
| `test_download.py` | revision lock、断点下载、路径和内容校验 |
| `test_handoff.py` | 环境/资产/code receipt、物化配置、checkpoint、NVLink/IB preflight |
| `test_public_smoke.py` | 公开 ALOHA smoke 的入口和报告契约 |

运行：

```bash
source site.env
"${PYTHON_BIN}" -m ruff check .
"${PYTHON_BIN}" -m pytest -q
find . -type f -name '*.sh' -print0 | xargs -0 -n1 bash -n
```

单测通过只证明代码契约；新服务器仍须运行真实数据、真实 VGGT 和 GPU0–1 的 smoke。任何模型
形状、数据字段、恢复语义或依赖 lock 改动都要补相应回归测试。
