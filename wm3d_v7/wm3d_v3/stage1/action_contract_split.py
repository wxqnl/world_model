from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import posixpath
import re
from typing import Mapping, Sequence

from wm3d_v3.data.manifest import OXEClipRecord
from wm3d_v3.stage1.action_contract import (
    action_contract_key,
    canonical_dataset_name,
)
from wm3d_v3.stage1.droid_interval_action import DROID_INTERVAL_ACTION_KIND


class ActionContractSplitError(ValueError):
    pass


@dataclass(frozen=True)
class FrozenContractSplit:
    contract_key: str
    calibration_clip_ids: tuple[str, ...]
    qualification_clip_ids: tuple[str, ...]
    confirmation_clip_ids: tuple[str, ...]
    calibration_group_ids: tuple[str, ...]
    qualification_group_ids: tuple[str, ...]
    confirmation_group_ids: tuple[str, ...]
    clip_to_group_id: dict[str, str]
    clip_to_group_id_sha256: str
    source_unique_count: int
    source_duplicate_count: int
    source_ineligible_count: int

    @property
    def heldout_clip_ids(self) -> tuple[str, ...]:
        return self.qualification_clip_ids + self.confirmation_clip_ids


_DROID_IDENTITY_RE = re.compile(
    r"^(?P<namespace>droid(?:/[^/]+)*)/"
    r"(?P<label>episode|session|trajectory|traj)[_=-]"
    r"(?P<identifier>[^_/]+)(?:[_/]|$)",
    re.IGNORECASE,
)


def _canonical_identity_path(value: str) -> str:
    normalized = posixpath.normpath(str(value).replace("\\", "/"))
    return "" if normalized == "." else normalized


def _identity_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def independence_group_id(record: OXEClipRecord) -> str:
    """Return the physical trajectory identity used to isolate formal splits."""
    domain = canonical_dataset_name(record.dataset)
    if domain != "droid":
        tar_path = _canonical_identity_path(record.tar_path)
        pickle_member = _canonical_identity_path(record.pickle_member)
        if not tar_path or not pickle_member:
            raise ActionContractSplitError(
                f"missing tar/member group identity for {record.clip_id}"
            )
        return f"tar_member:{_identity_digest(f'{tar_path}|{pickle_member}')}"

    clip_id = _canonical_identity_path(record.clip_id)
    match = _DROID_IDENTITY_RE.match(clip_id)
    if match is None:
        return f"droid_clip:{_identity_digest(clip_id)}"
    namespace = _canonical_identity_path(match.group("namespace"))
    label = match.group("label").lower()
    identifier = match.group("identifier")
    return f"droid:{namespace}|{label}={identifier}"


def _record_order(
    record: OXEClipRecord,
    *,
    contract_key: str,
    seed: int,
) -> tuple[str, str]:
    digest = hashlib.sha256(
        f"{int(seed)}|{contract_key}|{record.clip_id}".encode("utf-8")
    ).hexdigest()
    return digest, record.clip_id


def _group_order(
    group_id: str,
    *,
    contract_key: str,
    seed: int,
) -> tuple[str, str]:
    digest = hashlib.sha256(
        f"{int(seed)}|{contract_key}|{group_id}".encode("utf-8")
    ).hexdigest()
    return digest, group_id


def _clip_to_group_id_sha256(clip_to_group_id: Mapping[str, str]) -> str:
    encoded = json.dumps(
        dict(sorted(clip_to_group_id.items())),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_frozen_contract_split(split: FrozenContractSplit) -> None:
    partitions = (
        (
            "calibration",
            split.calibration_clip_ids,
            split.calibration_group_ids,
        ),
        (
            "qualification",
            split.qualification_clip_ids,
            split.qualification_group_ids,
        ),
        (
            "confirmation",
            split.confirmation_clip_ids,
            split.confirmation_group_ids,
        ),
    )
    clip_ids = tuple(
        clip_id
        for _, partition_clip_ids, _ in partitions
        for clip_id in partition_clip_ids
    )
    if len(set(clip_ids)) != len(clip_ids):
        raise ActionContractSplitError("split clip IDs are not unique")
    if set(split.clip_to_group_id) != set(clip_ids):
        raise ActionContractSplitError("clip-to-group mapping is incomplete")
    if split.clip_to_group_id_sha256 != _clip_to_group_id_sha256(
        split.clip_to_group_id
    ):
        raise ActionContractSplitError("clip-to-group mapping hash mismatch")

    partition_group_sets: list[set[str]] = []
    for name, partition_clip_ids, partition_group_ids in partitions:
        group_ids = set(partition_group_ids)
        if not group_ids:
            raise ActionContractSplitError(f"{name}_group_ids are missing")
        mapped_group_ids = {
            split.clip_to_group_id[clip_id] for clip_id in partition_clip_ids
        }
        if group_ids != mapped_group_ids:
            raise ActionContractSplitError(
                f"{name} group IDs do not match clip-to-group mapping"
            )
        partition_group_sets.append(group_ids)

    for index, first in enumerate(partition_group_sets):
        for second in partition_group_sets[index + 1 :]:
            if first & second:
                raise ActionContractSplitError(
                    "split partitions share independence groups"
                )


def frozen_contract_split_from_mapping(
    payload: Mapping[str, object],
) -> FrozenContractSplit:
    required_fields = {
        "contract_key",
        "calibration_clip_ids",
        "qualification_clip_ids",
        "confirmation_clip_ids",
        "calibration_group_ids",
        "qualification_group_ids",
        "confirmation_group_ids",
        "clip_to_group_id",
        "clip_to_group_id_sha256",
        "source_unique_count",
        "source_duplicate_count",
        "source_ineligible_count",
    }
    missing = sorted(required_fields.difference(payload))
    if missing:
        raise ActionContractSplitError(
            f"frozen split is missing required fields: {missing}"
        )

    def sequence(name: str, *, expected: int | None = None) -> tuple[str, ...]:
        raw = payload[name]
        if not isinstance(raw, (list, tuple)):
            raise ActionContractSplitError(f"{name} must be a sequence")
        values = tuple(str(value) for value in raw)
        if expected is not None and len(values) != expected:
            raise ActionContractSplitError(
                f"{name} must contain exactly {expected} clips"
            )
        if len(values) != len(set(values)):
            raise ActionContractSplitError(f"{name} contains duplicates")
        return values

    raw_mapping = payload["clip_to_group_id"]
    if not isinstance(raw_mapping, dict):
        raise ActionContractSplitError("clip_to_group_id must be a mapping")
    split = FrozenContractSplit(
        contract_key=str(payload["contract_key"]),
        calibration_clip_ids=sequence(
            "calibration_clip_ids",
            expected=32,
        ),
        qualification_clip_ids=sequence(
            "qualification_clip_ids",
            expected=32,
        ),
        confirmation_clip_ids=sequence(
            "confirmation_clip_ids",
            expected=32,
        ),
        calibration_group_ids=sequence("calibration_group_ids"),
        qualification_group_ids=sequence("qualification_group_ids"),
        confirmation_group_ids=sequence("confirmation_group_ids"),
        clip_to_group_id={
            str(clip_id): str(group_id)
            for clip_id, group_id in raw_mapping.items()
        },
        clip_to_group_id_sha256=str(payload["clip_to_group_id_sha256"]),
        source_unique_count=int(payload["source_unique_count"]),
        source_duplicate_count=int(payload["source_duplicate_count"]),
        source_ineligible_count=int(payload["source_ineligible_count"]),
    )
    validate_frozen_contract_split(split)
    return split


def freeze_contract_splits(
    records: Sequence[OXEClipRecord],
    *,
    seed: int,
    calibration_count: int = 32,
    qualification_count: int = 32,
    confirmation_count: int = 32,
    minimum_frames: int = 20,
) -> dict[str, FrozenContractSplit]:
    calibration_count = int(calibration_count)
    qualification_count = int(qualification_count)
    confirmation_count = int(confirmation_count)
    minimum_frames = int(minimum_frames)
    counts = (calibration_count, qualification_count, confirmation_count)
    if any(value < 1 for value in counts):
        raise ActionContractSplitError("split counts must all be positive")
    if minimum_frames < 20:
        raise ActionContractSplitError("minimum_frames cannot be below 20")

    grouped: dict[str, dict[str, OXEClipRecord]] = {}
    duplicates: Counter[str] = Counter()
    ineligible: Counter[str] = Counter()
    for record in records:
        domain = canonical_dataset_name(record.dataset)
        if domain == "droid" and record.action_kind != DROID_INTERVAL_ACTION_KIND:
            raise ActionContractSplitError(
                f"legacy DROID action kind is forbidden: {record.action_kind!r}"
            )
        key = action_contract_key(record)
        clip_records = grouped.setdefault(key, {})
        existing = clip_records.get(record.clip_id)
        if existing is not None:
            if existing != record:
                raise ActionContractSplitError(
                    f"conflicting duplicate record: {record.clip_id}"
                )
            duplicates[key] += 1
            continue
        if int(record.n_frames) < minimum_frames:
            ineligible[key] += 1
            continue
        clip_records[record.clip_id] = record

    if not grouped:
        raise ActionContractSplitError("manifest has no contract records")

    required = sum(counts)
    result: dict[str, FrozenContractSplit] = {}
    for key, clip_map in sorted(grouped.items()):
        ordered = sorted(
            clip_map.values(),
            key=lambda record: _record_order(
                record,
                contract_key=key,
                seed=int(seed),
            ),
        )
        if len(ordered) < required:
            raise ActionContractSplitError(
                f"contract {key} requires {required} unique eligible clips, "
                f"got {len(ordered)}"
            )

        records_by_group: dict[str, list[OXEClipRecord]] = {}
        for record in ordered:
            records_by_group.setdefault(independence_group_id(record), []).append(
                record
            )
        if len(records_by_group) < len(counts):
            raise ActionContractSplitError(
                f"contract {key} requires at least {len(counts)} independent "
                f"groups, got {len(records_by_group)}"
            )

        ordered_group_ids = sorted(
            records_by_group,
            key=lambda group_id: _group_order(
                group_id,
                contract_key=key,
                seed=int(seed),
            ),
        )
        partition_clip_ids: list[tuple[str, ...]] = []
        partition_group_ids: list[tuple[str, ...]] = []
        clip_to_group_id: dict[str, str] = {}
        next_group_index = 0
        for count in counts:
            selected_clip_ids: list[str] = []
            selected_group_ids: list[str] = []
            while len(selected_clip_ids) < count:
                if next_group_index >= len(ordered_group_ids):
                    raise ActionContractSplitError(
                        f"contract {key} lacks independent groups for split "
                        "allocation"
                    )
                group_id = ordered_group_ids[next_group_index]
                next_group_index += 1
                selected_group_ids.append(group_id)
                remaining = count - len(selected_clip_ids)
                for record in records_by_group[group_id][:remaining]:
                    selected_clip_ids.append(record.clip_id)
                    clip_to_group_id[record.clip_id] = group_id
            partition_clip_ids.append(tuple(selected_clip_ids))
            partition_group_ids.append(tuple(selected_group_ids))

        frozen = FrozenContractSplit(
            contract_key=key,
            calibration_clip_ids=partition_clip_ids[0],
            qualification_clip_ids=partition_clip_ids[1],
            confirmation_clip_ids=partition_clip_ids[2],
            calibration_group_ids=partition_group_ids[0],
            qualification_group_ids=partition_group_ids[1],
            confirmation_group_ids=partition_group_ids[2],
            clip_to_group_id=dict(sorted(clip_to_group_id.items())),
            clip_to_group_id_sha256=_clip_to_group_id_sha256(
                clip_to_group_id
            ),
            source_unique_count=len(ordered),
            source_duplicate_count=int(duplicates[key]),
            source_ineligible_count=int(ineligible[key]),
        )
        validate_frozen_contract_split(frozen)
        result[key] = frozen
    return result

