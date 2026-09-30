from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import yaml

from tests.test_admitted_source_inventory import _setup, _write_rows, ADMISSION
from tests.test_unified_manifest_contract import _profile, _source_row
from wm3d.data.file_binding import bind_stat_asset
from wm3d.data.grouped_robot import bimanual_arm_spec
from wm3d.data.manifest_contract import (
    ADMITTED_DATA_PROFILE_SCHEMA, ADMITTED_CACHE_INDEX_SCHEMA,
    ManifestContractError, load_data_profile,
)
from wm3d.data.source_inventory import scan_lerobot_source


def bind(path):
    return bind_stat_asset(Path("/"), path.absolute().as_posix().lstrip("/"),
                           admission_id=ADMISSION)


def write(path, value):
    path.write_text(yaml.safe_dump(value, sort_keys=False))
    return path


def setup_profile(tmp_path, monkeypatch):
    old_root = tmp_path / "legacy"
    old_root.mkdir()
    manifest = _write_rows(old_root / "source.jsonl", [_source_row()])
    old = _profile(manifest)
    old["cache_representation"]["view_slots"] = ["overhead", "head"]
    old_path = write(old_root / "profile.yaml", old)
    seal_path = old_root / "seal.json"
    seal_path.write_text(json.dumps({
        "schema": "wm3d_streaming_metadata_v1", "data_profile_path": str(old_path),
        "data_profile_sha256": "9" * 64,
        "source_manifest_sha256_by_name": {"toy": old["sources"][0]["manifest_sha256"]},
        "adapter_contract_sha256_by_name": {"toy": old["sources"][0]["adapter_contract_sha256"]},
    }))
    selection = old_root / "selection.json"
    selection.write_text('{"toy":["episode-0001"]}')
    kwargs, adapter_binding = _setup(tmp_path / "new", monkeypatch)
    rows, inventory = scan_lerobot_source(**kwargs)
    new_manifest = _write_rows(tmp_path / "new" / "manifest.jsonl", rows)
    legacy_source = {**old["sources"][0], "origin": "inherited_legacy",
                     "manifest_binding": bind(manifest),
                     "adapter_contract_binding": bind(Path(old["sources"][0]["adapter_config"]))}
    new_source = {
        "name": "aloha", "adapter": "lerobot", "raw_root": str(tmp_path/"new"),
        "adapter_config": str(tmp_path/"new"/"adapter.yaml"), "adapter_contract_binding": adapter_binding,
        "manifest": str(new_manifest), "manifest_binding": bind(new_manifest),
        "embodiment": bimanual_arm_spec().name, "weight": 1, "license_id": "test",
        "origin": "admitted",
    }
    value = {
        **old, "schema": ADMITTED_DATA_PROFILE_SCHEMA, "admission_id": ADMISSION,
        "cache": {**old["cache"], "schema": ADMITTED_CACHE_INDEX_SCHEMA, "resume": "receipt_and_stat"},
        "sources": [legacy_source, new_source],
        "embodiments": old["embodiments"] + [asdict(bimanual_arm_spec())],
        "inherited": {
            "profile_path": str(old_path), "profile_binding": bind(old_path),
            "metadata_seal_path": str(seal_path), "metadata_seal_binding": bind(seal_path),
            "selection_path": str(selection), "selection_binding": bind(selection),
        },
    }
    path = write(tmp_path / "combined.yaml", value)
    return path, value, inventory


def test_combined_profile_preserves_old_and_new_sources_without_SHA(tmp_path, monkeypatch):
    path, value, _ = setup_profile(tmp_path, monkeypatch)
    profile = load_data_profile(path, expected_binding=bind(path))
    assert profile.profile_sha256 == "" and profile.file_binding == bind(path)
    assert profile.admission_id == ADMISSION
    assert profile.source_order == ("toy", "aloha")
    old, new = profile.sources
    assert old.origin == "inherited_legacy"
    assert old.manifest_sha256 == value["sources"][0]["manifest_sha256"]
    assert new.manifest_sha256 == "" and new.origin == "admitted"
    assert new.manifest_binding == value["sources"][1]["manifest_binding"]
    assert profile.inherited == value["inherited"]
    # The legacy entry point must not silently downgrade its contract.
    with pytest.raises(ManifestContractError, match="schema"):
        load_data_profile(path)


@pytest.mark.parametrize("mutation", [
    "omit_legacy", "legacy_weight", "legacy_embodiment", "legacy_digest",
    "representation", "new_fake_sha", "new_admission", "relabel_legacy",
])
def test_profile_rejects_legacy_drift_and_implicit_binding_changes(tmp_path, monkeypatch, mutation):
    path, value, _ = setup_profile(tmp_path, monkeypatch)
    if mutation == "omit_legacy":
        value["sources"].pop(0)
    elif mutation == "legacy_weight":
        value["sources"][0]["weight"] += 1
    elif mutation == "legacy_embodiment":
        value["embodiments"][0]["groups"][0]["action_frame"] = "world"
    elif mutation == "legacy_digest":
        value["sources"][0]["manifest_sha256"] = "0"*64
    elif mutation == "representation":
        value["cache_representation"]["rgb_size"] *= 2
    elif mutation == "new_fake_sha":
        value["sources"][1]["manifest_sha256"] = "0"*64
    elif mutation == "new_admission":
        value["sources"][1]["manifest_binding"]["admission_id"] = "wrong"
    else:
        value["sources"][0]["origin"] = "admitted"
        value["sources"][0].pop("manifest_sha256")
        value["sources"][0].pop("adapter_contract_sha256")
    write(path, value)
    with pytest.raises((ManifestContractError, ValueError)):
        load_data_profile(path, expected_binding=bind(path))


@pytest.mark.parametrize("target", ["profile", "manifest", "selection"])
def test_profile_checks_identity_even_without_revalidating_manifest_rows(tmp_path, monkeypatch, target):
    path, value, _ = setup_profile(tmp_path, monkeypatch)
    expected = bind(path)
    target_path = {
        "profile": path, "manifest": Path(value["sources"][1]["manifest"]),
        "selection": Path(value["inherited"]["selection_path"]),
    }[target]
    target_path.write_text(target_path.read_text()+" ")
    with pytest.raises(ValueError, match="identity changed"):
        load_data_profile(path, expected_binding=expected, verify_source_manifests=False)


def test_admitted_profile_cli_from_bound_inventory_and_semantic_audit(tmp_path, monkeypatch):
    path, value, inventory = setup_profile(tmp_path, monkeypatch)
    template_value = deepcopy(value)
    template_value["schema"] = "wm3d_v8_data_profile_v4"
    template_value.pop("admission_id")
    template = write(tmp_path/"template.yaml", template_value)
    tb = bind(template)
    new = value["sources"][1]
    catalog = tmp_path/"catalog.json"
    catalog.write_text(json.dumps({"source": "aloha", "episodes": {"0": {"episode_id":"original-episode-27", "split":"val"}}}))
    audit_path = tmp_path/"audit.json"
    audit = {
        "schema": "wm3d_admitted_source_adapter_audit_receipt_v1",
        "source": "aloha", "admission_id": ADMISSION,
        "structural_checks": "pass", "semantic_review": "operator_confirmed_fail_closed",
        "adapter_contract_binding": new["adapter_contract_binding"],
        "data_template_binding": tb, "episode_catalog_binding": bind(catalog),
    }
    audit_path.write_text(json.dumps(audit))
    rp = tmp_path/"inventory.json"
    receipt = {
        **inventory, "data_template_path": str(template), "data_template_binding": tb,
        "adapter_contract_path": new["adapter_config"],
        "adapter_contract_binding": new["adapter_contract_binding"],
        "manifest_path": new["manifest"], "manifest_binding": new["manifest_binding"],
        "adapter_audit_receipt_path": str(audit_path), "adapter_audit_receipt_binding": bind(audit_path),
        "episode_catalog_path": str(catalog), "episode_catalog_binding": bind(catalog),
        "manifest_validation": {"episodes": 1},
    }
    rp.write_text(json.dumps(receipt))
    script = Path(__file__).resolve().parents[1]/"scripts/data/materialize_data_profile.py"
    spec = importlib.util.spec_from_file_location("admitted_profile_cli", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "sha256_file", lambda *a, **k: pytest.fail("SHA in admitted profile CLI"))
    output, receipt_path = tmp_path/"published.yaml", tmp_path/"published.receipt.json"
    monkeypatch.setattr(sys, "argv", [
        "data-profile", "--template", str(template), "--admission-id", ADMISSION,
        "--inventory", f"aloha={rp}", "--output", str(output), "--receipt", str(receipt_path),
    ])
    module.main()
    actual = json.loads(receipt_path.read_text())
    assert actual["source_count"] == 2 and actual["new_source_count"] == 1
    assert actual["SHA_scans"] is False and actual["training_ready"] is False
    assert actual["data_profile_binding"] == bind(output)
    assert load_data_profile(output, expected_binding=actual["data_profile_binding"]).source_order == ("toy","aloha")
    before = output.read_bytes()
    audit["semantic_review"] = "not_reviewed"
    audit_path.write_text(json.dumps(audit))
    receipt["adapter_audit_receipt_binding"] = bind(audit_path)
    rp.write_text(json.dumps(receipt))
    with pytest.raises(RuntimeError, match="physical adapter audit"):
        module.main()
    assert output.read_bytes() == before
