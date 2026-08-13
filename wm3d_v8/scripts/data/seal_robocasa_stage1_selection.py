#!/usr/bin/env python3
"""Seal the reviewed RoboCasa Stage1 root selection against real source data."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import yaml

from wm3d_v3.data.manifest_contract import SHA256_RE, canonical_sha256
from wm3d_v3.stage1_planner.replay_authority import (
    REPLAY_SELECTION_FIELDS,
    REPLAY_SELECTION_SCHEMA,
)
from wm3d_v3.stage1_planner.rollout_audit import (
    TrustedOutputRoot,
    read_regular_bytes,
)
from wm3d_v3.training.launch_qualification import verify_clean_runtime_checkout


POLICY_SCHEMA = "wm3d_v8_robocasa_stage1_selection_policy_v1"
CANDIDATE_SCHEMA = "wm3d_v7_stage1_planner_candidates_v2"
CANDIDATE_SEAL_SCHEMA = "wm3d_v7_stage1_planner_index_seal_v1"
SOURCE_MANIFEST_SCHEMA = "wm3d_v8_source_manifest_v4"
_SPLITS = ("train", "val", "test")
_POLICY_ROW_FIELDS = {"split", "source", "root_id", "episode_id", "t0"}
_CANDIDATE_SEAL_FIELDS = {
    "schema", "kind", "output", "output_sha256", "roots", "splits",
    "payload_schema", "stage0_checkpoint_sha256", "input_indices", "passed",
}


def _sha(value: object, label: str) -> str:
    if type(value) is not str or SHA256_RE.fullmatch(value) is None:
        raise RuntimeError(f"{label} must be a lowercase SHA256 string")
    return value


def _structured(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = yaml.safe_load(payload)
    except yaml.YAMLError as error:
        raise RuntimeError(f"{label} is invalid YAML") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a mapping")
    return value


def _json(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{label} is invalid JSON") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be an object")
    return value


def _jsonl(payload: bytes, label: str) -> list[dict[str, Any]]:
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise RuntimeError(f"{label} is not UTF-8 JSONL") from error
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"{label} row {number} is invalid JSON") from error
        if not isinstance(row, dict):
            raise RuntimeError(f"{label} row {number} must be an object")
        rows.append(row)
    if not rows:
        raise RuntimeError(f"{label} is empty")
    return rows


def _regular(path: Path, label: str) -> tuple[Path, bytes, str]:
    try:
        return read_regular_bytes(path, label)
    except (OSError, ValueError) as error:
        raise RuntimeError(str(error)) from error


def _model_xml_sha256(payload: bytes) -> str:
    try:
        canonical = gzip.decompress(payload).decode("utf-8").encode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise RuntimeError("source model.xml.gz is invalid") from error
    return hashlib.sha256(canonical).hexdigest()


def _canonical_json_sha256(payload: bytes, label: str) -> str:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{label} is invalid JSON") from error
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def _tracked_policy(
    repo: Path, commit: str, path: Path
) -> tuple[Path, bytes, str]:
    resolved, payload, digest = _regular(path, "selection policy")
    try:
        package_root = repo.resolve(strict=True)
        repository = Path(
            subprocess.check_output(
                ["git", "-C", str(repo), "rev-parse", "--show-toplevel"],
                text=True,
            ).strip()
        ).resolve(strict=True)
        relative = resolved.relative_to(repository).as_posix()
        package_relative = package_root.relative_to(repository)
    except (ValueError, subprocess.CalledProcessError) as error:
        raise RuntimeError("selection policy must be inside the clean checkout") from error
    expected_relative = (
        package_relative / "configs/data/stage1_robocasa_real_4roots.template.yaml"
    ).as_posix()
    if relative != expected_relative:
        raise RuntimeError("selection policy is not the reviewed formal four-root profile")
    tracked = subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=repository,
        check=True,
        capture_output=True,
    ).stdout
    if tracked != payload:
        raise RuntimeError("selection policy bytes differ from the clean commit")
    return resolved, payload, digest


def _policy(value: dict[str, Any]) -> list[dict[str, Any]]:
    notes = value.get("notes")
    policy = notes.get("stage1_selection_policy") if isinstance(notes, dict) else None
    if not isinstance(policy, dict) or set(policy) != {
        "schema", "expected_split_counts", "expected_data_profile_sha256",
        "expected_candidate_index_sha256", "expected_candidate_seal_sha256",
        "expected_stage0_checkpoint_sha256", "rows",
    }:
        raise RuntimeError("selection policy fields mismatch")
    if policy["schema"] != POLICY_SCHEMA:
        raise RuntimeError("selection policy schema mismatch")
    for name in (
        "expected_data_profile_sha256", "expected_candidate_index_sha256",
        "expected_candidate_seal_sha256", "expected_stage0_checkpoint_sha256",
    ):
        _sha(policy[name], f"selection policy {name}")
    counts = policy["expected_split_counts"]
    rows = policy["rows"]
    if (
        not isinstance(counts, dict)
        or set(counts) != set(_SPLITS)
        or counts != {"train": 2, "val": 1, "test": 1}
        or not isinstance(rows, list)
        or len(rows) != 4
    ):
        raise RuntimeError("formal selection must be train2/val1/test1 over four roots")
    observed = {split: 0 for split in _SPLITS}
    roots: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != _POLICY_ROW_FIELDS:
            raise RuntimeError("selection policy row fields mismatch")
        split = row["split"]
        root_id = _sha(row["root_id"], "selection policy root id")
        if (
            split not in observed
            or type(row["source"]) is not str
            or not row["source"]
            or type(row["episode_id"]) is not int
            or row["episode_id"] < 0
            or type(row["t0"]) is not int
            or row["t0"] < 0
            or root_id in roots
        ):
            raise RuntimeError("selection policy row identity is invalid")
        roots.add(root_id)
        observed[split] += 1
    if observed != counts:
        raise RuntimeError("selection policy split closure mismatch")
    return rows


def _build_receipt(
    *,
    code_commit: str,
    code_repo: Path,
    policy_path: Path,
    policy_sha: str,
    profile_path: Path,
    profile_sha: str,
    candidate_index: Path,
    candidate_sha: str,
    candidate_seal: Path,
    candidate_seal_sha: str,
    policy_rows: list[dict[str, Any]],
    policy_contract: dict[str, Any],
    profile: dict[str, Any],
    candidate_payload: bytes,
    candidate_seal_payload: bytes,
) -> dict[str, Any]:
    if (
        profile_sha != policy_contract["expected_data_profile_sha256"]
        or candidate_sha != policy_contract["expected_candidate_index_sha256"]
        or candidate_seal_sha != policy_contract["expected_candidate_seal_sha256"]
    ):
        raise RuntimeError("selection inputs differ from tracked expected SHA authority")
    if profile.get("schema") != "wm3d_v8_data_profile_v4":
        raise RuntimeError("materialized data profile schema mismatch")
    sources = profile.get("sources")
    if not isinstance(sources, list) or not sources:
        raise RuntimeError("materialized data profile has no sources")
    source_rows: dict[str, dict[str, Any]] = {}
    manifest_rows: dict[tuple[str, int], dict[str, Any]] = {}
    for source in sources:
        if not isinstance(source, dict) or type(source.get("name")) is not str:
            raise RuntimeError("materialized data profile source is invalid")
        name = source["name"]
        if name in source_rows:
            raise RuntimeError("materialized data profile has duplicate source")
        raw_root = Path(str(source.get("raw_root", "")))
        if raw_root.is_symlink() or not raw_root.resolve(strict=True).is_dir():
            raise RuntimeError(f"source {name} raw root is invalid")
        manifest_path, manifest_payload, manifest_sha = _regular(
            Path(str(source.get("manifest", ""))), f"source {name} manifest"
        )
        if manifest_sha != _sha(
            source.get("manifest_sha256"), f"source {name} manifest SHA"
        ):
            raise RuntimeError(f"source {name} manifest SHA drift")
        source_rows[name] = dict(source, raw_root=str(raw_root.resolve(strict=True)))
        for row in _jsonl(manifest_payload, f"source {name} manifest"):
            episode_text = row.get("episode_id")
            prefix = f"{name}:"
            if (
                row.get("schema") != SOURCE_MANIFEST_SCHEMA
                or row.get("source") != name
                or type(episode_text) is not str
                or not episode_text.startswith(prefix)
            ):
                raise RuntimeError(f"source {name} manifest row identity is invalid")
            try:
                episode_id = int(episode_text[len(prefix):])
            except ValueError as error:
                raise RuntimeError("source manifest episode id is invalid") from error
            key = (name, episode_id)
            if key in manifest_rows:
                raise RuntimeError("source manifest has duplicate episode")
            manifest_rows[key] = dict(
                row,
                _manifest_path=str(manifest_path),
                _manifest_sha256=manifest_sha,
            )

    seal = _json(candidate_seal_payload, "candidate index seal")
    output = seal.get("output")
    if (
        set(seal) != _CANDIDATE_SEAL_FIELDS
        or seal.get("schema") != CANDIDATE_SEAL_SCHEMA
        or seal.get("passed") is not True
        or seal.get("kind") != "candidates"
        or seal.get("payload_schema") != CANDIDATE_SCHEMA
        or type(output) is not str
        or Path(output).is_symlink()
        or Path(output).resolve(strict=True) != candidate_index
        or seal.get("output_sha256") != candidate_sha
        or type(seal.get("roots")) is not int
        or seal["roots"] < 4
    ):
        raise RuntimeError("candidate index seal does not authorize the index")
    _sha(seal.get("stage0_checkpoint_sha256"), "candidate seal Stage0 SHA")
    if (
        seal["stage0_checkpoint_sha256"]
        != policy_contract["expected_stage0_checkpoint_sha256"]
    ):
        raise RuntimeError("candidate seal Stage0 checkpoint differs from tracked policy")
    splits = seal.get("splits")
    if (
        not isinstance(splits, dict)
        or set(splits) != set(_SPLITS)
        or any(type(value) is not int or value < 0 for value in splits.values())
        or sum(splits.values()) != seal["roots"]
    ):
        raise RuntimeError("candidate index seal split closure is invalid")
    inputs = seal.get("input_indices")
    if not isinstance(inputs, list) or not inputs:
        raise RuntimeError("candidate index seal input closure is empty")
    for item in inputs:
        if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
            raise RuntimeError("candidate index seal input fields mismatch")
        if type(item["path"]) is not str:
            raise RuntimeError("candidate index seal input path must be a string")
        _path, _payload, observed = _regular(
            Path(item["path"]), "candidate index seal input"
        )
        if observed != _sha(item["sha256"], "candidate index seal input SHA"):
            raise RuntimeError("candidate index seal input SHA drift")
    candidates: dict[str, dict[str, Any]] = {}
    for row in _jsonl(candidate_payload, "candidate index"):
        root_id = row.get("root_id")
        if row.get("schema") != CANDIDATE_SCHEMA or type(root_id) is not str:
            raise RuntimeError("candidate row schema/identity is invalid")
        _sha(root_id, "candidate root id")
        if root_id in candidates:
            raise RuntimeError("candidate index has duplicate root")
        candidates[root_id] = row
    observed_splits = {
        split: sum(row.get("split") == split for row in candidates.values())
        for split in _SPLITS
    }
    if len(candidates) != seal["roots"] or observed_splits != splits:
        raise RuntimeError("candidate index rows differ from its seal")

    sealed_rows = []
    for policy_row in policy_rows:
        root_id = policy_row["root_id"]
        candidate = candidates.get(root_id)
        source = source_rows.get(policy_row["source"])
        manifest = manifest_rows.get((policy_row["source"], policy_row["episode_id"]))
        if candidate is None or source is None or manifest is None:
            raise RuntimeError("selection policy row is absent from sealed source/candidates")
        if (
            candidate.get("split") != policy_row["split"]
            or candidate.get("episode_id") != policy_row["episode_id"]
            or candidate.get("t0") != policy_row["t0"]
            or candidate.get("episode_root_index") != policy_row["t0"]
            or Path(str(candidate.get("source_dataset", ""))).resolve(strict=True)
            != Path(source["raw_root"])
            or manifest.get("split") != policy_row["split"]
        ):
            raise RuntimeError("selection policy differs from source/candidate identity")
        candidate_path, _candidate_bytes, candidate_payload_sha = _regular(
            Path(str(candidate.get("candidate_path", ""))), "selected candidate payload"
        )
        root_context_path, _context_bytes, root_context_sha = _regular(
            Path(str(candidate.get("root_context_path", ""))), "selected root context"
        )
        if (
            candidate_payload_sha
            != _sha(candidate.get("payload_sha256"), "candidate payload SHA")
            or root_context_sha
            != _sha(candidate.get("root_context_sha256"), "root context SHA")
        ):
            raise RuntimeError("selected candidate payload/root context SHA drift")
        assets = manifest.get("assets")
        primary = (
            [item for item in assets if item.get("role") == "primary_payload"]
            if isinstance(assets, list)
            else []
        )
        if len(primary) != 1:
            raise RuntimeError("selected source episode lacks primary payload authority")
        source_episode_sha = _sha(primary[0].get("sha256"), "source episode SHA")
        source_episode, _episode_bytes, observed_episode_sha = _regular(
            Path(source["raw_root"]) / str(primary[0].get("path", "")),
            "selected source episode",
        )
        if observed_episode_sha != source_episode_sha:
            raise RuntimeError("selected source episode SHA drift")
        source_dataset = Path(source["raw_root"])
        extras_root = source_dataset / f"extras/episode_{policy_row['episode_id']:06d}"
        states_path, _states_payload, states_sha = _regular(
            extras_root / "states.npz", "selected source states"
        )
        model_xml_path, model_xml_payload, model_xml_file_sha = _regular(
            extras_root / "model.xml.gz", "selected source model XML"
        )
        ep_meta_path, ep_meta_payload, ep_meta_file_sha = _regular(
            extras_root / "ep_meta.json", "selected source episode metadata"
        )
        dataset_meta_path, _dataset_meta_payload, dataset_meta_sha = _regular(
            source_dataset / "extras/dataset_meta.json",
            "selected source dataset metadata",
        )
        modality_path, _modality_payload, modality_sha = _regular(
            source_dataset / "meta/modality.json", "selected source modality"
        )
        sealed_rows.append({
            "split": policy_row["split"],
            "source": policy_row["source"],
            "root_id": root_id,
            "episode_id": policy_row["episode_id"],
            "episode_root_index": policy_row["t0"],
            "t0": policy_row["t0"],
            "source_dataset_path": str(source_dataset),
            "source_manifest_path": manifest["_manifest_path"],
            "source_manifest_sha256": manifest["_manifest_sha256"],
            "source_manifest_row_sha256": canonical_sha256({
                key: value for key, value in manifest.items() if not key.startswith("_")
            }),
            "source_episode_path": str(source_episode),
            "source_episode_sha256": source_episode_sha,
            "states_path": str(states_path),
            "states_sha256": states_sha,
            "model_xml_gz_path": str(model_xml_path),
            "model_xml_gz_sha256": model_xml_file_sha,
            "model_xml_sha256": _model_xml_sha256(model_xml_payload),
            "ep_meta_path": str(ep_meta_path),
            "ep_meta_file_sha256": ep_meta_file_sha,
            "ep_meta_sha256": _canonical_json_sha256(
                ep_meta_payload, "selected source episode metadata"
            ),
            "dataset_meta_path": str(dataset_meta_path),
            "dataset_meta_sha256": dataset_meta_sha,
            "modality_path": str(modality_path),
            "modality_sha256": modality_sha,
            "candidate_index_row_sha256": canonical_sha256(candidate),
            "candidate_payload_path": str(candidate_path),
            "candidate_payload_sha256": candidate_payload_sha,
            "root_context_path": str(root_context_path),
            "root_context_sha256": root_context_sha,
        })
    sealed_rows.sort(key=lambda row: (_SPLITS.index(row["split"]), row["root_id"]))
    counts = {
        split: sum(row["split"] == split for row in sealed_rows)
        for split in _SPLITS
    }
    receipt = {
        "schema": REPLAY_SELECTION_SCHEMA,
        "code_commit": code_commit,
        "code_repo_path": str(code_repo),
        "selection_policy_path": str(policy_path),
        "selection_policy_sha256": policy_sha,
        "data_profile_path": str(profile_path),
        "data_profile_sha256": profile_sha,
        "candidate_index_path": str(candidate_index),
        "candidate_index_sha256": candidate_sha,
        "candidate_index_seal_path": str(candidate_seal),
        "candidate_index_seal_sha256": candidate_seal_sha,
        "selection_count": counts,
        "rows": sealed_rows,
        "rows_sha256": canonical_sha256(sealed_rows),
        "passed": True,
    }
    if set(receipt) != REPLAY_SELECTION_FIELDS:
        raise AssertionError("internal selection receipt fields drifted")
    return receipt


def rebuild_selection_receipt(
    receipt: dict[str, Any], *, expected_code_commit: str
) -> dict[str, Any]:
    """Re-derive a selection seal from its committed policy and all referents."""
    if set(receipt) != REPLAY_SELECTION_FIELDS:
        raise RuntimeError("selection receipt exact fields mismatch")
    if (
        receipt.get("schema") != REPLAY_SELECTION_SCHEMA
        or receipt.get("passed") is not True
        or receipt.get("code_commit") != expected_code_commit
        or type(receipt.get("code_repo_path")) is not str
    ):
        raise RuntimeError("selection receipt schema/commit mismatch")
    repo = Path(receipt["code_repo_path"])
    code_commit = verify_clean_runtime_checkout(repo, expected_code_commit)
    repo = repo.resolve(strict=True)
    policy_path, policy_payload, policy_sha = _tracked_policy(
        repo, code_commit, Path(receipt["selection_policy_path"])
    )
    profile_path, profile_payload, profile_sha = _regular(
        Path(receipt["data_profile_path"]), "materialized data profile"
    )
    candidate_index, candidate_payload, candidate_sha = _regular(
        Path(receipt["candidate_index_path"]), "candidate index"
    )
    candidate_seal, candidate_seal_payload, candidate_seal_sha = _regular(
        Path(receipt["candidate_index_seal_path"]), "candidate index seal"
    )
    rebuilt = _build_receipt(
        code_commit=code_commit,
        code_repo=repo,
        policy_path=policy_path,
        policy_sha=policy_sha,
        profile_path=profile_path,
        profile_sha=profile_sha,
        candidate_index=candidate_index,
        candidate_sha=candidate_sha,
        candidate_seal=candidate_seal,
        candidate_seal_sha=candidate_seal_sha,
        policy_rows=_policy(_structured(policy_payload, "selection policy")),
        policy_contract=_structured(policy_payload, "selection policy")["notes"][
            "stage1_selection_policy"
        ],
        profile=_structured(profile_payload, "materialized data profile"),
        candidate_payload=candidate_payload,
        candidate_seal_payload=candidate_seal_payload,
    )
    if rebuilt != receipt:
        raise RuntimeError("selection receipt differs from committed derivation")
    return rebuilt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-policy", type=Path, required=True)
    parser.add_argument("--data-profile", type=Path, required=True)
    parser.add_argument("--candidate-index", type=Path, required=True)
    parser.add_argument("--candidate-index-seal", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[2]
    code_commit = verify_clean_runtime_checkout(repo, args.code_commit)
    policy_path, policy_payload, policy_sha = _tracked_policy(
        repo, code_commit, args.selection_policy
    )
    profile_path, profile_payload, profile_sha = _regular(
        args.data_profile, "materialized data profile"
    )
    candidate_index, candidate_payload, candidate_sha = _regular(
        args.candidate_index, "candidate index"
    )
    candidate_seal, candidate_seal_payload, candidate_seal_sha = _regular(
        args.candidate_index_seal, "candidate index seal"
    )
    receipt = _build_receipt(
        code_commit=code_commit,
        code_repo=repo.resolve(strict=True),
        policy_path=policy_path,
        policy_sha=policy_sha,
        profile_path=profile_path,
        profile_sha=profile_sha,
        candidate_index=candidate_index,
        candidate_sha=candidate_sha,
        candidate_seal=candidate_seal,
        candidate_seal_sha=candidate_seal_sha,
        policy_rows=_policy(_structured(policy_payload, "selection policy")),
        policy_contract=_structured(policy_payload, "selection policy")["notes"][
            "stage1_selection_policy"
        ],
        profile=_structured(profile_payload, "materialized data profile"),
        candidate_payload=candidate_payload,
        candidate_seal_payload=candidate_seal_payload,
    )
    payload = (json.dumps(receipt, sort_keys=True, indent=2) + "\n").encode()
    output_path = args.output.absolute()
    with TrustedOutputRoot(output_path.parent, label="selection seal parent") as output_scope:
        output_scope.publish(output_path, payload, label="Stage1 selection seal")
    print(json.dumps({
        "schema": REPLAY_SELECTION_SCHEMA,
        "output": str(output_path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "selection_count": receipt["selection_count"],
        "passed": True,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
