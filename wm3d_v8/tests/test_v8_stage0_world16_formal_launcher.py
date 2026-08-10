from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs"
    / (
        "wm3d_v8_stage0_causal_dual_view_actionpolicy_"
        "formal100k_world16_node43_node44_v1.yaml"
    )
)
NODE_LAUNCHER = (
    ROOT
    / "scripts"
    / ("launch_wm3d_v8_stage0_causal_dual_view_formal100k_world16_node_v1.sh")
)
ORCHESTRATOR = (
    ROOT
    / "scripts"
    / ("start_wm3d_v8_stage0_causal_dual_view_formal100k_world16_v1.sh")
)


def test_world16_formal_config_is_fresh_global64_and_gate_bound() -> None:
    config = yaml.safe_load(CONFIG.read_text())
    contract = config["contract"]
    train = config["train"]

    assert config["_base_"] == (
        "wm3d_v8_stage0_causal_dual_view_actionpolicy_formal.yaml"
    )
    assert contract["canary_gate_receipt"].endswith(
        "/audits/training_canary100_v10/formal_gate.json"
    )
    assert contract["canary_gate_receipt_sha256"] == (
        "c935b45e3a7900c1a2b1c1cc40ae47dc4dd3e7e10821b149e9a5942ffcda0fd5"
    )
    assert train["num_nodes"] == 2
    assert train["gpus_per_node"] == 8
    assert train["batch_size_per_gpu"] == 2
    assert train["gradient_accumulation_steps"] == 2
    assert train["effective_global_batch"] == 64
    assert train["max_steps"] == 100000
    assert train["resume_checkpoint"] is None
    assert train["pretrained_world_checkpoint"] is None
    assert train["fresh_initialization_required"] is True
    assert train["checkpoint_milestone_steps"][0] == 1000


def test_world16_launchers_bind_only_node43_node44_and_hard_stop_1k() -> None:
    node = NODE_LAUNCHER.read_text()
    start = ORCHESTRATOR.read_text()

    assert "--nnodes=2" in node
    assert "--nproc_per_node=8" in node
    assert "--stop_after_step 1000" in node
    assert "MASTER_ADDR=${MASTER_ADDR:-172.27.0.6}" in node
    assert "mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_5,mlx5_6,mlx5_7,mlx5_8" in node
    assert "mlx5_0,mlx5_1,mlx5_4,mlx5_5,mlx5_7,mlx5_8,mlx5_9,mlx5_10" in node
    assert "172.27.0.6" in start
    assert "172.27.0.7" in start
    assert "172.27.0.4" not in start
    assert "172.27.0.5" not in start
    assert "concurrent_two_host_absolute_time_gate" in start
    assert "partial launch cleanup" in start
    assert "invocation_hard_stop_step" in start
    assert "WM3D_V8_STAGE0_FORMAL" in node
    assert "WM3D_V8_STAGE0_FORMAL" in start
    assert "PYTHONDONTWRITEBYTECODE=1" in node
    assert "PYTHONDONTWRITEBYTECODE=1" in start
