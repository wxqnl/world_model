from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

from packaging.utils import canonicalize_name
import pytest
import torch
import yaml

from scripts.cluster.preflight_cluster import (
    _parse_ib_rate_gbps,
    _parse_nvlink_topology,
)
from scripts.cluster.seal_code import DEFAULT_PATTERNS
from wm3d.data.assets import (
    ASSET_RECEIPT_SCHEMA,
    verify_asset_bundle,
)
from wm3d.data.contracts import (
    ContractError,
    DatasetContract,
    DatasetSeal,
    FileEvidence,
    atomic_write_json,
    canonical_sha256,
    sha256_file,
)
from wm3d.training.config import (
    CODE_RECEIPT_SCHEMA,
    training_contract_sha256,
    verify_code_receipt,
)
from wm3d.training.checkpoint import (
    CHECKPOINT_SCHEMA,
    COMMIT_SCHEMA,
    CheckpointIntegrityError,
    CheckpointManager,
)
from wm3d.training.environment import (
    ENVIRONMENT_CONTRACT_SCHEMA,
    EnvironmentContractError,
    create_environment_receipt,
    verify_environment_receipt,
)


def test_code_seal_covers_native_encoder_dependency_closure() -> None:
    required = {
        "wm3d/__init__.py",
        "wm3d/data/__init__.py",
        "wm3d/encoders/__init__.py",
        "wm3d/encoders/vggt_features.py",
        "wm3d/encoders/vggt_encoder.py",
        "wm3d/models/__init__.py",
        "wm3d/training/__init__.py",
    }
    root = Path(__file__).resolve().parents[1]
    covered = {
        path.relative_to(root).as_posix()
        for pattern in DEFAULT_PATTERNS
        for path in root.glob(pattern)
        if path.is_file()
    }
    assert required.issubset(covered)


def test_environment_receipt_binds_current_runtime(tmp_path: Path) -> None:
    contract = {
        "schema": ENVIRONMENT_CONTRACT_SCHEMA,
        "python_major_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
        "torch": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda),
        "minimum_nccl": [0, 0, 0],
        "packages": {"numpy": importlib.metadata.version("numpy")},
    }
    contract_path = tmp_path / "environment_contract.json"
    contract_path.write_text(
        json.dumps(contract, sort_keys=True),
        encoding="utf-8",
    )
    receipt_path = tmp_path / "environment_receipt.json"
    receipt = create_environment_receipt(
        contract_path=contract_path,
        output_path=receipt_path,
    )
    verified = verify_environment_receipt(
        receipt_path,
        expected_sha256=canonical_sha256(receipt),
        contract_path=contract_path,
        check_current=True,
    )
    assert verified["environment"]["pass"]
    receipt_alias = tmp_path / "environment_receipt_alias.json"
    receipt_alias.symlink_to(receipt_path)
    with pytest.raises(EnvironmentContractError, match="regular file"):
        verify_environment_receipt(
            receipt_alias,
            expected_sha256=canonical_sha256(receipt),
            contract_path=contract_path,
            check_current=False,
        )
    with pytest.raises(EnvironmentContractError, match="receipt SHA"):
        verify_environment_receipt(
            receipt_path,
            expected_sha256="0" * 64,
            contract_path=contract_path,
            check_current=False,
        )


def test_encoder_asset_receipt_detects_tamper(tmp_path: Path) -> None:
    root = tmp_path / "assets"
    asset_paths = {
        "vggt_source": root / "vggt/source",
        "vggt_model": root / "vggt/model/aaaaaaaa",
        "task_model": root / "task/model/bbbbbbbb",
    }
    for index, path in enumerate(asset_paths.values()):
        path.mkdir(parents=True)
        (path / "payload.bin").write_bytes(bytes([index + 1]) * 11)
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            files[relative] = {
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    receipt = {
        "schema": ASSET_RECEIPT_SCHEMA,
        "assets": {
            name: {"path": path.relative_to(root).as_posix()}
            for name, path in asset_paths.items()
        },
        "files": files,
    }
    receipt["content_sha256"] = canonical_sha256(receipt)
    (root / "receipt.json").write_text(
        json.dumps(receipt, sort_keys=True),
        encoding="utf-8",
    )
    assert verify_asset_bundle(root, deep=True)["pass"]
    alias = tmp_path / "asset_alias"
    alias.symlink_to(root, target_is_directory=True)
    with pytest.raises(ContractError, match="real directory"):
        verify_asset_bundle(alias, deep=False)
    target = asset_paths["task_model"] / "payload.bin"
    target.write_bytes(b"tampered")
    with pytest.raises(ContractError, match="verification failed"):
        verify_asset_bundle(root, deep=True)


def test_code_receipt_rejects_dirty_scope(tmp_path: Path) -> None:
    root = tmp_path / "wm3d_v7"
    root.mkdir()
    source = root / "native.py"
    source.write_text("VALUE = 7\n", encoding="utf-8")
    receipt = {
        "schema": CODE_RECEIPT_SCHEMA,
        "root_layout": "wm3d",
        "created_at_utc": "2026-07-29T00:00:00Z",
        "include_patterns": ["native.py"],
        "git_commit": "a" * 40,
        "scoped_git_status": "?? native.py",
        "files": {
            "native.py": {
                "size": source.stat().st_size,
                "sha256": sha256_file(source),
            }
        },
    }
    path = tmp_path / "code_receipt.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="dirty"):
        verify_code_receipt(
            path,
            expected_sha256=canonical_sha256(receipt),
            repo_root=root,
        )
    receipt["scoped_git_status"] = ""
    path.write_text(json.dumps(receipt), encoding="utf-8")
    assert (
        verify_code_receipt(
            path,
            expected_sha256=canonical_sha256(receipt),
            repo_root=root,
        )["git_commit"]
        == "a" * 40
    )


def _materialization_contract() -> DatasetContract:
    return DatasetContract.from_mapping(
        {
            "name": "synthetic_wm3d_handoff",
            "feature_fps": 5.0,
            "action_fps": 30.0,
            "T": 24,
            "P": 144,
            "K": 16,
            "token_dim": 2048,
            "task_dim": 2048,
            "num_views": 3,
            "max_action_groups": 8,
            "max_action_dim": 16,
            "action_substeps": 6,
            "max_group_id": 64,
            "max_embodiments": 256,
            "max_aux_tokens": 8,
            "aux_dim": 256,
            "max_aux_type_id": 64,
            "source_order": ["droid"],
            "sources": [
                {
                    "name": "droid",
                    "adapter": "lerobot",
                    "raw_root": "/synthetic",
                    "license_id": "synthetic-test",
                    "nominal_hours": 1.0,
                    "weight": 100,
                    "embodiment_names": ["single_arm"],
                    "split_seed": 17,
                    "train_fraction": 0.8,
                }
            ],
            "embodiments": [
                {
                    "name": "single_arm",
                    "embodiment_id": 0,
                    "views": ["head", "left_hand", "right_hand"],
                    "action_groups": [
                        {
                            "name": "arm",
                            "group_id": 0,
                            "dimensions": ["x", "y", "z", "rx", "ry", "rz"],
                            "rate_hz": 30.0,
                            "control_mode": "delta_pose",
                        },
                        {
                            "name": "gripper",
                            "group_id": 1,
                            "dimensions": ["close"],
                            "rate_hz": 30.0,
                            "control_mode": "discrete_gripper",
                        },
                    ],
                }
            ],
        }
    )


def test_materialize_config_verifies_full_seal_and_has_no_placeholders(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    dataset_root = tmp_path / "dataset"
    (dataset_root / "control").mkdir(parents=True)
    payload_manifest = (
        dataset_root / "payload" / "parts" / "part-00000000" / "manifest.json"
    )
    payload_manifest.parent.mkdir(parents=True)
    contract = _materialization_contract()
    contract_path = dataset_root / "control" / "dataset_contract.json"
    atomic_write_json(contract_path, contract.as_dict())
    atomic_write_json(
        payload_manifest,
        {"schema": "synthetic_payload_manifest_v1", "windows": 2},
    )
    seal = DatasetSeal(
        dataset_schema=contract.schema,
        dataset_contract_sha256=contract.sha256,
        control_files={
            "control/dataset_contract.json": FileEvidence(
                size=contract_path.stat().st_size,
                sha256=sha256_file(contract_path),
            )
        },
        payload_manifest_files={
            "payload/parts/part-00000000/manifest.json": FileEvidence(
                size=payload_manifest.stat().st_size,
                sha256=sha256_file(payload_manifest),
            )
        },
        source_window_counts={"droid": {"train": 1, "val": 1}},
        source_hours={"droid": 1.0},
        created_at_utc="2026-07-29T00:00:00Z",
    )
    seal_path = dataset_root / "receipts" / "dataset_seal.json"
    atomic_write_json(seal_path, seal.as_dict())

    code_root = tmp_path / "code"
    code_root.mkdir()
    source = code_root / "native.py"
    source.write_text("VALUE = 7\n", encoding="utf-8")
    code_receipt = {
        "schema": CODE_RECEIPT_SCHEMA,
        "root_layout": "wm3d",
        "created_at_utc": "2026-07-29T00:00:00Z",
        "include_patterns": ["native.py"],
        "git_commit": "a" * 40,
        "scoped_git_status": "",
        "files": {
            "native.py": {
                "size": source.stat().st_size,
                "sha256": sha256_file(source),
            }
        },
    }
    code_receipt_path = tmp_path / "code_receipt.json"
    atomic_write_json(code_receipt_path, code_receipt)

    environment_contract = {
        "schema": ENVIRONMENT_CONTRACT_SCHEMA,
        "python_major_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
        "torch": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda),
        "minimum_nccl": [0, 0, 0],
        "packages": {"numpy": importlib.metadata.version("numpy")},
    }
    environment_contract_path = tmp_path / "environment_contract.json"
    atomic_write_json(environment_contract_path, environment_contract)
    environment_receipt_path = tmp_path / "environment_receipt.json"
    create_environment_receipt(
        contract_path=environment_contract_path,
        output_path=environment_receipt_path,
    )

    output_config = tmp_path / "materialized.yaml"
    command = [
        sys.executable,
        str(repo_root / "scripts" / "cluster" / "materialize_config.py"),
        "--template",
        str(repo_root / "configs" / "train" / "5b_h200.yaml"),
        "--dataset-root",
        str(dataset_root),
        "--code-receipt",
        str(code_receipt_path),
        "--code-root",
        str(code_root),
        "--environment-contract",
        str(environment_contract_path),
        "--environment-receipt",
        str(environment_receipt_path),
        "--output-root",
        str(tmp_path / "formal_run"),
        "--output-config",
        str(output_config),
        "--run-name",
        "synthetic_wm3d_formal",
        "--run-lineage",
        "b" * 64,
        "--world-size",
        "64",
        "--shard-degree",
        "8",
        "--global-batch-size",
        "128",
        "--micro-batch-size",
        "1",
    ]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(repo_root)
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert json.loads(result.stdout)["pass"]
    materialized_text = output_config.read_text(encoding="utf-8")
    assert "__MATERIALIZE_REQUIRED__" not in materialized_text
    materialized = yaml.safe_load(materialized_text)
    assert materialized["distributed"]["expected_world_size"] == 64
    assert materialized["train"]["gradient_accumulation"] == 2
    assert materialized["run"]["training_contract_sha256"] == (
        training_contract_sha256(materialized)
    )
    assert materialized["data"]["seal_receipt_sha256"] == seal.sha256

    payload_manifest.write_text('{"tampered":true}\n', encoding="utf-8")
    failed = subprocess.run(
        [
            *command[:],
            "--output-config",
            str(tmp_path / "must_not_exist.yaml"),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert failed.returncode != 0
    assert "dataset seal verification failed" in failed.stderr
    assert not (tmp_path / "must_not_exist.yaml").exists()


def test_checkpoint_manifest_rejects_path_escape(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint = checkpoint_root / "step_00000001"
    checkpoint.mkdir(parents=True)
    outside = checkpoint_root / "outside.bin"
    outside.write_bytes(b"outside")
    metadata = {
        "schema": CHECKPOINT_SCHEMA,
        "step": 1,
        "run_lineage": "lineage",
    }
    metadata_path = checkpoint / "metadata.json"
    atomic_write_json(metadata_path, metadata)
    manifest = {
        "schema": CHECKPOINT_SCHEMA,
        "step": 1,
        "files": {
            "../outside.bin": {
                "size": outside.stat().st_size,
                "sha256": sha256_file(outside),
            },
            "metadata.json": {
                "size": metadata_path.stat().st_size,
                "sha256": sha256_file(metadata_path),
            },
        },
    }
    manifest_path = checkpoint / "MANIFEST.json"
    atomic_write_json(manifest_path, manifest)
    commit = {
        "schema": COMMIT_SCHEMA,
        "step": 1,
        "metadata_sha256": sha256_file(metadata_path),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_content_sha256": canonical_sha256(manifest),
        "run_lineage": "lineage",
    }
    atomic_write_json(checkpoint / "COMMITTED.json", commit)
    with pytest.raises(CheckpointIntegrityError, match="unsafe relative path"):
        CheckpointManager(checkpoint_root).verify(checkpoint)


def test_preflight_requires_full_eight_gpu_nvlink_clique() -> None:
    header = "        " + " ".join(f"GPU{index}" for index in range(8))
    rows = []
    for source in range(8):
        links = ["X" if source == target else "NV18" for target in range(8)]
        rows.append(f"GPU{source} " + " ".join(links) + " 0-31")
    topology = "\n".join([header, *rows])
    assert _parse_nvlink_topology(topology)["pass"]
    with pytest.raises(RuntimeError, match="NVLink clique is incomplete"):
        _parse_nvlink_topology(topology.replace("NV18", "SYS", 1))


def test_preflight_parses_infiniband_rate_fail_closed() -> None:
    assert _parse_ib_rate_gbps("400 Gb/sec (4X NDR)") == 400.0
    assert _parse_ib_rate_gbps("200 Gb/sec (4X HDR)") == 200.0
    with pytest.raises(RuntimeError, match="unrecognized"):
        _parse_ib_rate_gbps("Unknown")
    with pytest.raises(RuntimeError, match="non-positive"):
        _parse_ib_rate_gbps("0 Gb/sec")


def test_environment_setup_is_plain_and_beta_converter_is_lazy() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    setup_script = (repo_root / "environments" / "bootstrap_environment.sh").read_text(
        encoding="utf-8"
    )
    entry = (repo_root / "wm3d.sh").read_text(encoding="utf-8")
    pipeline = (repo_root / "scripts" / "pipeline.py").read_text(encoding="utf-8")
    assert '-m venv "${ENV_PREFIX}"' in setup_script
    assert "docker" not in setup_script.lower()
    assert "micromamba" not in setup_script.lower()
    assert "bootstrap_agibot_converter_environment.sh" not in entry
    assert "bootstrap_agibot_converter_environment.sh" in pipeline


def test_lerobot_converter_repair_is_exact_and_fail_closed() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module = __import__(
        "environments.prepare_lerobot_converter_build",
        fromlist=["normalize_requirements"],
    )
    newline = bytes((10,))
    continuation = bytes((92, 10))
    block = (
        b'pyav==13.1.0 ; python_version >= "3.10" '
        + continuation
        + b"    --hash=sha256:deadbeef"
        + newline
    )
    raw = b"first==1" + newline + block + b"last==1" + newline
    normalized = module.normalize_requirements(
        raw,
        expected_block_sha256=hashlib.sha256(block).hexdigest(),
    )
    assert b"pyav==" not in normalized
    assert b"av==13.1.0" in normalized
    assert module.OFFICIAL_AV_CP310_LINUX_X86_64_SHA256.encode() in normalized
    with pytest.raises(ValueError, match="拒绝静默修补"):
        module.normalize_requirements(raw, expected_block_sha256="0" * 64)

    original = b"[tool.poetry.dependencies]" + newline + b'pyav = ">=12.0.5"' + newline
    expected = original.replace(b"pyav", b"av")
    patched = module.patch_pyproject(
        original,
        original_sha256=hashlib.sha256(original).hexdigest(),
        patched_sha256=hashlib.sha256(expected).hexdigest(),
    )
    assert patched == expected

    contract = json.loads(
        (
            repo_root / "environments" / "agibot_converter_environment_contract.json"
        ).read_text(encoding="utf-8")
    )
    assert contract["packages"]["av"] == "13.1.0"
    assert "av" in contract["required_imports"]


def test_requirements_lock_exactly_matches_environment_contract() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    environment_root = repo_root / "environments"
    contract = json.loads(
        (environment_root / "environment_contract.json").read_text(encoding="utf-8")
    )
    expected = {
        canonicalize_name(name): str(version)
        for name, version in contract["packages"].items()
    }
    locked = {}
    for raw_line in (
        (environment_root / "requirements.lock")
        .read_text(encoding="utf-8")
        .splitlines()
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        name, version = line.split("==", 1)
        locked[canonicalize_name(name)] = version
    assert locked == expected


def test_planning_inventory_and_source_layouts_remain_aligned() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config_root = repo_root / "configs" / "data"
    inventory = yaml.safe_load(
        (config_root / "public_6106h.yaml").read_text(encoding="utf-8")
    )
    source_names = [source["name"] for source in inventory["sources"]]
    assert inventory["source_order"] == source_names
    assert sum(source["weight"] for source in inventory["sources"]) == 100
    nominal_hours = sum(
        float(source["nominal_hours"]) for source in inventory["sources"]
    )
    assert nominal_hours == pytest.approx(6_106.4)
    assert nominal_hours == pytest.approx(
        float(inventory["notes"]["nominal_total_hours"])
    )
    layouts = json.loads(
        (config_root / "public_6106h_layouts.json").read_text(encoding="utf-8")
    )
    assert [layout["source"] for layout in layouts["layouts"]] == source_names


def test_data_pipeline_keeps_bootstrap_outside_formal_dataset_root() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    pipeline = (repo_root / "scripts" / "pipeline.py").read_text(encoding="utf-8")
    assert 'self.bootstrap = self.release / "dataset_bootstrap"' in pipeline
    assert 'contract = self.bootstrap / "dataset_contract.json"' in pipeline
    assert 'self.dataset / "bootstrap"' not in pipeline
    assert "LEGACY_ROOT" not in pipeline
    assert "LEGACY_MANIFEST_FULL" not in pipeline
    assert "prepare_legacy_residual_manifest.py" not in pipeline
    assert "GLOBAL_SAMPLE_BUDGET" in pipeline


def test_canary_template_preserves_formal_model_data_and_loss_contract() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config_root = repo_root / "configs" / "train"
    formal = yaml.safe_load((config_root / "5b_h200.yaml").read_text(encoding="utf-8"))
    canary = yaml.safe_load(
        (config_root / "5b_h200_canary.yaml").read_text(encoding="utf-8")
    )
    for section in (
        "model",
        "model_budget",
        "hardware",
        "data",
        "distributed",
        "optimizer",
        "loss",
    ):
        assert canary[section] == formal[section]
    assert canary["train"]["total_steps"] == 1000
    assert max(canary["train"]["checkpoint_steps"]) == 1000
    assert canary["train"]["checkpoint_interval"] == 0
    assert canary["schedule"]["warmup_steps"] < canary["train"]["total_steps"]
    assert formal["train"]["total_steps"] == 600_000


def test_every_operational_script_is_documented_and_referenced() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    scripts = sorted(
        path
        for root in (repo_root / "scripts", repo_root / "environments")
        for path in root.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix in {".py", ".sh"}
    )
    text_paths = [
        path
        for path in repo_root.rglob("*")
        if path.is_file()
        and ".git" not in path.parts
        and "__pycache__" not in path.parts
        and path.suffix in {".md", ".py", ".sh", ".yaml", ".json", ".example"}
    ]
    assert scripts
    for script in scripts:
        readme = script.parent / "README.md"
        assert readme.is_file(), f"missing directory README for {script}"
        assert script.name in readme.read_text(encoding="utf-8"), (
            f"{script.name} is not documented in {readme}"
        )
        references = [
            path
            for path in text_paths
            if path != script and script.name in path.read_text(encoding="utf-8")
        ]
        assert len(references) >= 2, f"orphan operational script: {script}"
