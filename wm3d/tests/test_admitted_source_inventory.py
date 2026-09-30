from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import yaml

from tests.test_source_inventory import _raw_fixture
from wm3d.data.file_binding import bind_stat_asset
from wm3d.data.grouped_robot import bimanual_arm_spec
from wm3d.data.manifest_contract import ManifestContractError, validate_source_manifest
from wm3d.data.source_adapters import load_adapter_contract
from wm3d.data.source_inventory import SourceInventoryError, scan_lerobot_source

ADMISSION = "test-admission"
CATALOG = {0: {"episode_id": "original-episode-27", "split": "val"}}


def _setup(root, monkeypatch, *, duplicate_time=False):
    adapter_path = _raw_fixture(root, duplicate_time=duplicate_time)
    binding = bind_stat_asset(root, "adapter.yaml", admission_id=ADMISSION)
    adapter = load_adapter_contract(adapter_path, expected_binding=binding)

    def no_sha(*args, **kwargs):
        raise AssertionError("admitted inventory must not call SHA")

    for module in ("source_inventory", "manifest_contract", "source_adapters"):
        monkeypatch.setattr(f"wm3d.data.{module}.sha256_file", no_sha)
    monkeypatch.setattr("wm3d.data.source_inventory.canonical_sha256", no_sha)
    monkeypatch.setattr("wm3d.data.source_inventory.deterministic_split", no_sha)
    monkeypatch.setattr("wm3d.data.manifest_contract.canonical_timestamp_sha256", no_sha)
    kwargs = dict(
        root=root, source="aloha", embodiment=bimanual_arm_spec(), adapter=adapter,
        split_seed=7, train_fraction=0.8, validation_fraction=0.1, default_task="",
        admission_id=ADMISSION, episode_catalog=deepcopy(CATALOG),
    )
    return kwargs, binding


def _write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def test_inventory_admitted_manifest_without_hashing_preserves_real_clocks_and_split(
    tmp_path, monkeypatch
):
    kwargs, binding = _setup(tmp_path, monkeypatch)
    rows, receipt = scan_lerobot_source(**kwargs)
    row = rows[0]
    assert row["schema"] == "wm3d_admitted_source_manifest_v1"
    assert row["episode_id"] == "original-episode-27" and row["split"] == "val"
    assert row["observation_clock"]["timestamp_binding"] == "recorded_in_admitted_payload"
    assert row["observation_clock"]["min_dt_s"] == pytest.approx(0.031)
    assert row["observation_clock"]["max_dt_s"] == pytest.approx(0.063)
    assert "sha256" not in json.dumps(rows)
    assert receipt["adapter_contract_binding"] == binding
    assert receipt["SHA_scans"] is False
    assert not receipt["global_deduplication_performed_by_inventory"]
    assert set(receipt["input_metadata_bindings"]) == {
        "meta/info.json", "meta/episodes.jsonl", "meta/tasks.jsonl"
    }
    manifest = _write_rows(tmp_path / "manifest.jsonl", rows)
    result = validate_source_manifest(
        manifest, expected_source="aloha",
        expected_embodiment=bimanual_arm_spec(), admission_id=ADMISSION,
    )
    assert result["episodes"] == 1
    with pytest.raises(ManifestContractError, match="schema"):
        validate_source_manifest(manifest, expected_source="aloha")


@pytest.mark.parametrize("catalog", [
    None, {}, {0: {"episode_id": "", "split": "train"}},
    {0: {"episode_id": "episode", "split": "unknown"}},
    {0: CATALOG[0], 1: CATALOG[0]},
    {1: {"episode_id": "different", "split": "train"}},
])
def test_inventory_requires_explicit_nonduplicated_matching_episode_catalog(
    tmp_path, monkeypatch, catalog
):
    kwargs, _ = _setup(tmp_path, monkeypatch)
    kwargs["episode_catalog"] = catalog
    with pytest.raises(SourceInventoryError, match="catalog"):
        scan_lerobot_source(**kwargs)


def test_inventory_rejects_admission_mismatch_and_implicit_downgrade(tmp_path, monkeypatch):
    kwargs, _ = _setup(tmp_path, monkeypatch)
    with pytest.raises(SourceInventoryError, match="admission"):
        scan_lerobot_source(**{**kwargs, "admission_id": "different"})
    with pytest.raises(SourceInventoryError, match="explicit admission"):
        scan_lerobot_source(**{**kwargs, "admission_id": None})


def test_stat_manifest_rejects_changed_member_admission_and_clock_fields(tmp_path, monkeypatch):
    kwargs, _ = _setup(tmp_path, monkeypatch)
    rows, _ = scan_lerobot_source(**kwargs)
    for changed in ("member", "clock"):
        value = deepcopy(rows[0])
        if changed == "member":
            value["assets"][1]["binding"]["admission_id"] = "different"
        else:
            value["observation_clock"]["timestamp_binding"] = "nominal_fps"
        manifest = _write_rows(tmp_path / "manifest.jsonl", [value])
        with pytest.raises(ManifestContractError, match="admission|clock binding"):
            validate_source_manifest(
                manifest, expected_source="aloha", admission_id=ADMISSION
            )


def test_admitted_inventory_still_rejects_duplicate_recorded_timestamps(tmp_path, monkeypatch):
    kwargs, _ = _setup(tmp_path, monkeypatch, duplicate_time=True)
    with pytest.raises(SourceInventoryError, match="strictly increasing"):
        scan_lerobot_source(**kwargs)


def test_admitted_inventory_rejects_metadata_mutation_during_scan(tmp_path, monkeypatch):
    import wm3d.data.source_inventory as inventory
    kwargs, _ = _setup(tmp_path, monkeypatch)
    original = inventory._task_text

    def mutate(*args, **values):
        path = tmp_path / "meta/tasks.jsonl"
        path.write_text(path.read_text() + " ")
        return original(*args, **values)

    monkeypatch.setattr(inventory, "_task_text", mutate)
    with pytest.raises(ValueError, match="identity changed"):
        scan_lerobot_source(**kwargs)


def test_inventory_cli_admitted_receipt_roundtrip_and_wrong_audit_rejected(
    tmp_path, monkeypatch
):
    kwargs, adapter_binding = _setup(tmp_path, monkeypatch)
    script = Path(__file__).resolve().parents[1] / "scripts/data/materialize_source_inventory.py"
    spec = importlib.util.spec_from_file_location("inventory_admitted_cli", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "sha256_file", lambda *a, **k: pytest.fail("SHA in CLI"))
    (tmp_path / "binding.json").write_text(json.dumps(adapter_binding))
    template = {
        "schema": "wm3d_v8_data_profile_v4",
        "sources": [{"name": "aloha", "embodiment": bimanual_arm_spec().name}],
        "embodiments": [asdict(bimanual_arm_spec())],
    }
    (tmp_path / "template.yaml").write_text(yaml.safe_dump(template))
    catalog = {
        "schema": "wm3d_admitted_episode_catalog_v1", "admission_id": ADMISSION,
        "source": "aloha", "episodes": {"0": CATALOG[0]},
    }
    (tmp_path / "catalog.json").write_text(json.dumps(catalog))
    audit = {
        "schema": module.ADMITTED_ADAPTER_AUDIT_SCHEMA, "source": "aloha",
        "admission_id": ADMISSION, "adapter_contract_binding": adapter_binding,
        "data_template_binding": bind_stat_asset(
            tmp_path, "template.yaml", admission_id=ADMISSION
        ),
        "episode_catalog_binding": bind_stat_asset(
            tmp_path, "catalog.json", admission_id=ADMISSION
        ),
        "structural_checks": "pass", "semantic_review": "operator_confirmed_fail_closed",
    }
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps(audit))
    monkeypatch.setattr(sys, "argv", [
        "inventory", "--source", "aloha", "--raw-root", str(tmp_path),
        "--data-template", str(tmp_path / "template.yaml"),
        "--adapter-contract", str(tmp_path / "adapter.yaml"),
        "--adapter-contract-binding", str(tmp_path / "binding.json"),
        "--admission-id", ADMISSION, "--episode-catalog", str(tmp_path / "catalog.json"),
        "--adapter-audit-receipt", str(audit_path),
        "--output-manifest", str(tmp_path / "manifest.jsonl"),
        "--output-receipt", str(tmp_path / "receipt.json"),
    ])
    module.main()
    receipt = json.loads((tmp_path / "receipt.json").read_text())
    assert receipt["manifest_validation"]["episodes"] == 1
    assert receipt["manifest_binding"]["kind"] == "admitted_file_stat_v1"
    assert "sha256" not in json.dumps(receipt)
    before = (tmp_path / "manifest.jsonl").read_bytes()
    audit["semantic_review"] = "not_reviewed"
    audit_path.write_text(json.dumps(audit))
    with pytest.raises(RuntimeError, match="does not authorize"):
        module.main()
    assert (tmp_path / "manifest.jsonl").read_bytes() == before
