from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from wm3d_v3.stage1.action_contract import canonical_dataset_name
from wm3d_v3.stage1.droid_interval_action import DROID_INTERVAL_ACTION_KIND
from wm3d_v3.stage1.immutable_artifact import publish_immutable_bytes


class Stage1ManifestError(ValueError):
    pass


@dataclass(frozen=True)
class DeduplicatedRecords:
    records: tuple[dict[str, Any], ...]
    source_count: int
    duplicate_count: int


@dataclass(frozen=True)
class ManifestBuildResult:
    path: Path
    sha256: str
    source_sha256: str
    source_count: int
    unique_count: int
    output_count: int
    duplicate_count: int
    excluded_count: int
    domain_counts: dict[str, int]


def _canonical_record(raw: Mapping[str, Any]) -> dict[str, Any]:
    record = dict(raw)
    record.pop("repeat_weight", None)
    clip_id = str(record.get("clip_id", "")).strip()
    if not clip_id:
        raise Stage1ManifestError("manifest record has no clip_id")
    record["clip_id"] = clip_id
    dataset = canonical_dataset_name(str(record.get("dataset", "")))
    record["dataset"] = dataset
    if dataset == "droid" and record.get("action_kind") != DROID_INTERVAL_ACTION_KIND:
        raise Stage1ManifestError(
            f"legacy DROID action kind is forbidden: {clip_id}: "
            f"{record.get('action_kind')!r}"
        )
    return record


def deduplicate_records(
    records: Iterable[Mapping[str, Any]],
) -> DeduplicatedRecords:
    by_clip: dict[str, dict[str, Any]] = {}
    source_count = 0
    duplicate_count = 0
    for raw in records:
        source_count += 1
        record = _canonical_record(raw)
        clip_id = record["clip_id"]
        previous = by_clip.get(clip_id)
        if previous is None:
            by_clip[clip_id] = record
            continue
        duplicate_count += 1
        if previous != record:
            raise Stage1ManifestError(
                f"conflicting duplicate manifest record: {clip_id}"
            )
    return DeduplicatedRecords(
        records=tuple(by_clip[key] for key in sorted(by_clip)),
        source_count=source_count,
        duplicate_count=duplicate_count,
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open() as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise Stage1ManifestError(
                        f"manifest row {line_number} is not a mapping"
                    )
                rows.append(payload)
    except (OSError, json.JSONDecodeError) as exc:
        raise Stage1ManifestError(f"cannot read source manifest {path}: {exc}") from exc
    return rows


def build_training_manifest(
    source_path: str | Path,
    output_path: str | Path,
    *,
    excluded_clip_ids: Iterable[str] = (),
) -> ManifestBuildResult:
    source = Path(source_path).resolve(strict=True)
    output = Path(output_path)
    deduplicated = deduplicate_records(_read_jsonl(source))
    excluded = {str(value) for value in excluded_clip_ids}
    selected = tuple(
        record
        for record in deduplicated.records
        if record["clip_id"] not in excluded
    )
    encoded = b"".join(
        (
            json.dumps(
                record,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        for record in selected
    )
    published = publish_immutable_bytes(output, encoded)
    counts = Counter(str(record["dataset"]) for record in selected)
    return ManifestBuildResult(
        path=output,
        sha256=published.sha256,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        source_count=deduplicated.source_count,
        unique_count=len(deduplicated.records),
        output_count=len(selected),
        duplicate_count=deduplicated.duplicate_count,
        excluded_count=len(deduplicated.records) - len(selected),
        domain_counts=dict(sorted(counts.items())),
    )
