# Objective profiles

本目录声明 Stage0 各监督项及权重。新增或关闭 lane 时，必须同步 coverage 门禁、离线评测
receipt 和对应测试，不能让零 coverage 的 masked loss 被当作有效结果。

verified V7 60K 的生产合同不启用额外 wrong-action RGB objective；
`context_pixel_action_rank_weight` 与 `context_pixel_action_separation_weight` 必须为 0。
wrong-action forward 只保留为显式诊断实验，不能进入正式 1B/5B profile，也不能用
RGB 排序结果代替 factual P64 方向与误差验收。
