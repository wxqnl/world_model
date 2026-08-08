import json
import subprocess
import sys
from pathlib import Path

from scripts.cache_v7_stage1_planner_root_contexts import context_state_steps
from wm3d_v3.stage1_planner.train import (
    DYNAMICS_PREFIXES,
    PLANNER_PREFIXES,
    SERVING_GUARD_PREFIXES,
)


ROOT = Path(__file__).resolve().parents[1]


def test_root_context_schedule_is_real_t16_and_never_reads_future() -> None:
    steps = context_state_steps(83, frames=16, stride=4)
    assert len(steps) == 16
    assert steps == tuple(range(23, 84, 4))
    assert max(steps) == 83


def test_stage1_trainable_allowlists_exclude_serving_action_heads() -> None:
    for prefix in (*DYNAMICS_PREFIXES, *PLANNER_PREFIXES):
        assert not prefix.startswith(SERVING_GUARD_PREFIXES)
    assert "world.action_policy." not in DYNAMICS_PREFIXES
    assert "world.action_proj." not in DYNAMICS_PREFIXES


def test_stage1_configs_never_authorize_automatic_phase_promotion() -> None:
    common = (ROOT / "configs/wm3d_v7_stage1_planner_native3d_common.yaml").read_text()
    launcher = (ROOT / "scripts/launch_wm3d_v7_stage1_planner.sh").read_text()
    assert "automatic_phase_promotion: false" in common
    assert "repeated_root_context_forbidden: true" in common
    assert "current_pinned_robocasa_runtime_causal_replay" in common
    assert "REPLACE_AFTER_STAGE0_STEP100000_HARD_STOP" in common
    assert "EXECUTE_WM3D_V7_STAGE1_PLANNER_PHASE" in launcher
    assert "wm3d_v3.stage1_planner.train" in launcher


def test_planner_head_source_does_not_accept_candidate_actions() -> None:
    source = (ROOT / "wm3d_v3/stage1_planner/planner_head.py").read_text()
    forward = source[source.index("    def forward(") : source.index("\n\ndef planning_score")]
    assert "candidate_actions" not in forward
    assert "future_tokens" in forward
    assert "depth" in forward and "point" in forward and "pose" in forward


def test_branch_cache_cannot_replace_history_with_repeated_root() -> None:
    source = (ROOT / "scripts/cache_v7_stage1_planner_branches.py").read_text()
    assert "encode_repeated_context" not in source
    assert 'context_archive["anchor_codes"]' in source
    assert "runtime/context root RGB mismatch" in source


def test_candidate_harvester_uses_the_real_pose_only_flow_width() -> None:
    source = (ROOT / "scripts/harvest_wm3d_v7_stage1_planner_candidates.py").read_text()
    assert "flow_action_dim != 6" in source
    assert "(1, core_horizon, flow_action_dim)" in source
    assert "(1, core_horizon, 7)" not in source


def test_candidate_index_sealer_merges_real_payloads_deterministically(tmp_path: Path) -> None:
    roles = [
        "direct",
        "flow_0",
        "flow_1",
        "flow_2",
        "flow_3",
        "grip_open",
        "grip_close",
        "arm_hold",
        "pose_reverse",
        "pose_half",
    ]
    checkpoint_sha = "a" * 64
    shards = []
    for index, (split, root_id) in enumerate((("val", "root-b"), ("train", "root-a"))):
        payload = tmp_path / f"candidate-{index}.pt"
        payload.write_bytes(f"payload-{root_id}".encode())
        row = {
            "schema": "wm3d_v7_stage1_planner_candidates_v1",
            "root_id": root_id,
            "task": "pick_place",
            "split": split,
            "split_group": f"episode-{index}",
            "branch_roles": roles,
            "candidate_path": str(payload),
            "root_context_sha256": "b" * 64,
            "stage0_checkpoint_sha256": checkpoint_sha,
            "future_observation_leakage": False,
        }
        shard = tmp_path / f"index.shard-{index}.jsonl"
        shard.write_text(json.dumps(row) + "\n")
        shards.append(shard)

    output = tmp_path / "index.jsonl"
    command = [
        sys.executable,
        str(ROOT / "scripts/seal_wm3d_v7_stage1_planner_indices.py"),
        "--kind",
        "candidates",
        "--expected-roots",
        "2",
        "--output",
        str(output),
    ]
    for shard in shards:
        command.extend(("--input", str(shard)))
    subprocess.run(command, check=True, cwd=ROOT, capture_output=True, text=True)

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    report = json.loads(output.with_suffix(".seal.json").read_text())
    assert [row["root_id"] for row in rows] == ["root-a", "root-b"]
    assert report["passed"] is True
    assert report["roots"] == 2
    assert report["splits"] == {"test": 0, "train": 1, "val": 1}
    assert report["stage0_checkpoint_sha256"] == checkpoint_sha
