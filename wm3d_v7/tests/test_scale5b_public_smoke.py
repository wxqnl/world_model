from __future__ import annotations

import json
from pathlib import Path
import subprocess

import torch
import yaml

from scripts.scale5b.build_task_bank_smoke import deterministic_embedding
from wm3d_v3.data.scale5b_contracts import DatasetContract
from wm3d_v3.data.scale5b_sources import (
    SourceLayout,
    deterministic_split,
    plan_shard,
)
from wm3d_v3.models.native5b import (
    NativeWM3D5B,
    config_from_mapping,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "scale5b"
UPSTREAM_REVISION = "cc571a3c661df81b566dbfde3d5c1e85fcdf7884"


def test_public_smoke_contract_is_production_shape_and_bimanual() -> None:
    inventory = yaml.safe_load(
        (CONFIG_ROOT / "dataset_inventory_smoke_aloha.yaml").read_text(
            encoding="utf-8"
        )
    )
    contract = DatasetContract.from_mapping(inventory)
    assert (contract.T, contract.P, contract.K, contract.token_dim) == (
        24,
        144,
        16,
        2048,
    )
    assert contract.source_order == ("aloha_smoke",)
    assert contract.source_weights == {"aloha_smoke": 100}
    groups = contract.embodiments[0].action_groups
    assert [len(group.dimensions) for group in groups] == [6, 1, 6, 1]
    assert sum(len(group.dimensions) for group in groups) == 14

    layouts = json.loads(
        (CONFIG_ROOT / "source_layouts_smoke_aloha.json").read_text(
            encoding="utf-8"
        )
    )
    layout = SourceLayout.from_mapping(layouts["layouts"][0])
    assert layout.view_keys == {
        "head": "observation.images.top",
        "left_hand": None,
        "right_hand": None,
    }
    assert [item.indices for item in layout.action_columns] == [
        (0, 1, 2, 3, 4, 5),
        (6,),
        (7, 8, 9, 10, 11, 12),
        (13,),
    ]

    episode_ids = [f"aloha_smoke:{index:09d}" for index in range(50)]
    splits = {
        deterministic_split("aloha_smoke", item, seed=700, train_fraction=0.8)
        for item in episode_ids
    }
    assert {"train", "val"}.issubset(splits)
    assert {plan_shard(item, 2) for item in episode_ids} == {0, 1}


def test_public_smoke_keeps_exact_5b_core_and_pinned_data() -> None:
    template = yaml.safe_load(
        (CONFIG_ROOT / "wm3d_v7_native5b_smoke2.template.yaml").read_text(
            encoding="utf-8"
        )
    )
    with torch.device("meta"):
        model = NativeWM3D5B(config_from_mapping(template["model"]))
    assert model.parameter_counts()["total"] == 4_956_589_929
    assert template["distributed"]["expected_world_size"] == 2
    assert template["distributed"]["shard_degree"] == 2
    assert template["train"]["total_steps"] == 1

    lock = yaml.safe_load(
        (CONFIG_ROOT / "raw_sources_smoke_aloha.lock.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert lock["sources"]["aloha_smoke"]["revision"] == UPSTREAM_REVISION


def test_public_smoke_hash_embedding_and_shell_are_deterministic() -> None:
    first = deterministic_embedding("insert the peg")
    second = deterministic_embedding("insert the peg")
    other = deterministic_embedding("move the cube")
    assert first.shape == (2048,)
    assert torch.isfinite(first).all()
    assert torch.equal(first, second)
    assert not torch.equal(first, other)
    result = subprocess.run(
        ["bash", "-n", str(ROOT / "scripts" / "scale5b" / "run_public_smoke.sh")],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
