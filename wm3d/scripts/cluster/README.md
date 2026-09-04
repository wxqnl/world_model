# 集群入口

wm3d_1b.sh 和 wm3d_5b.sh 组合现有生产命令，不维护第二套模型或训练器。
5B 操作只看 ../../docs/WM3D_5B_SCALING.md。

configure_5b_inputs.py 负责本地输入识别、路径检查和 site 生成。
check_5b_contract.py 检查当前 5B 模型/目标及实际 sealed runtime 是否一致，
doctor、runtime 和启动前均使用同一个检查。Meta 参数构建不是 GPU 资格。

输入检查只报告 ready_for_preflight。真实集群 preflight、模型前后向、
checkpoint 和离线评测决定后续资格；不能用文件存在或空 receipt 宣布可以训练。
