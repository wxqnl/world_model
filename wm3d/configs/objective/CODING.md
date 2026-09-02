# Objective profiles

本目录声明 Stage0 各监督项及权重。新增或关闭 lane 时，必须同步 coverage 门禁、离线评测
receipt 和对应测试，不能让零 coverage 的 masked loss 被当作有效结果。

verified V7 60K 的生产合同在后半程启用稀疏 wrong-action RGB curriculum：
step 30000 开始、10000 step 线性增权、每 8 step 取 1 个样本，rank/separation
权重分别为 2.0/0.5。正式 1B/5B profile 必须保持这一调度；短资格训练可以按总步数
等比例缩放起点与 ramp，但不能把它常驻到每一步。RGB 排序结果仍不能替代 factual P64
方向、误差以及 policy/action-free 隔离验收。
