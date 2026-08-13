# Repository guide

本分支只发布 `wm3d/` 中的当前 WM3D 实现。上一代训练代码不得重新加入仓库；旧数据只能
通过 `wm3d/run_wm3d.sh legacy-residual-import` 进入当前 ABI。

所有项目命令从 `wm3d/` 执行。修改提交前必须运行：

```bash
PYTHON_BIN=.venv/bin/python ./run_wm3d.sh check
```

对外名称统一使用 `WM3D` / `wm3d`。已经封存的 schema 字符串属于兼容 ABI，不作为项目
名称，也不能在没有迁移方案和真实 replay 的情况下改写。
