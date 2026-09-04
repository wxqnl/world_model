# Model profiles

1B 与 5B 只能通过本目录中的容量 profile 区分；不得复制模型实现或增加独立训练器。
所有 shape 上限必须与统一数据 ABI 和 runtime validation 保持一致。

Native direct RGB 使用 `native_1b_v8_native_direct_rgb.yaml` / `native_5b_v8_native_direct_rgb.yaml`，与 transport 共用现代物理 factual pass，不能通过 renderer 开关改变 policy / action 编码。
输出路径、资格边界与当前证据见 `docs/NATIVE_DIRECT_RGB.md`。旧 profile 保留兼容已有 checkpoint，不表示已通过当前正式训练验收。
