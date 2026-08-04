from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import torch
import yaml

from scripts.internal.build_task_bank import deterministic_embedding
from wm3d.data.contracts import DatasetContract
from wm3d.data.sources import (
    SourceLayout,
    deterministic_split,
    plan_shard,
)
from wm3d.models.wm3d import (
    WM3D,
    config_from_mapping,
)


ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG_ROOT = ROOT / "configs" / "smoke"
TRAIN_CONFIG_ROOT = ROOT / "configs" / "train"
UPSTREAM_REVISION = "cc571a3c661df81b566dbfde3d5c1e85fcdf7884"


def test_public_smoke_contract_is_production_shape_and_bimanual() -> None:
    inventory = yaml.safe_load(
        (SMOKE_CONFIG_ROOT / "aloha_dataset.yaml").read_text(
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
        (SMOKE_CONFIG_ROOT / "aloha_layouts.json").read_text(
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
        (TRAIN_CONFIG_ROOT / "5b_smoke.yaml").read_text(
            encoding="utf-8"
        )
    )
    with torch.device("meta"):
        model = WM3D(config_from_mapping(template["model"]))
    assert model.parameter_counts()["total"] == 4_956_589_929
    assert template["distributed"]["expected_world_size"] == 2
    assert template["distributed"]["shard_degree"] == 2
    assert template["train"]["total_steps"] == 1

    lock = yaml.safe_load(
        (SMOKE_CONFIG_ROOT / "aloha_sources.lock.yaml").read_text(
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
    shell = (ROOT / "scripts" / "run_public_smoke.sh").read_text()
    assert "export PYTHONDONTWRITEBYTECODE=1" in shell
    assert (
        'ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"'
        in shell
    )
    assert not torch.equal(first, other)
    result = subprocess.run(
        ["bash", "-n", str(ROOT / "scripts" / "run_public_smoke.sh")],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_sealed_vggt_source_import_does_not_write_bytecode(tmp_path: Path) -> None:
    source = tmp_path / "source"
    package = source / "vggt"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "probe.py").write_text("VALUE = 7\n", encoding="utf-8")
    program = "\n".join(
        (
            "import os",
            "from pathlib import Path",
            f"root = Path({str(source)!r})",
            "os.environ['WM3D_VGGT_SOURCE_ROOT'] = str(root)",
            (
                "from wm3d.encoders.vggt_encoder import "
                "_ensure_local_vggt_on_path"
            ),
            "assert _ensure_local_vggt_on_path() == root",
            "import vggt.probe",
            "assert vggt.probe.VALUE == 7",
            "assert not list(root.rglob('*.pyc'))",
        )
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT)
    environment.pop("PYTHONDONTWRITEBYTECODE", None)
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert not list(source.rglob("*.pyc"))
