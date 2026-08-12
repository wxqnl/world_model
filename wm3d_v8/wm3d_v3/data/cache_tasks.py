"""Content-addressed task planning and atomic claims for parallel cache jobs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

from .manifest_contract import safe_relative_path, sha256_file


CACHE_TASK_SCHEMA = "wm3d_v8_episode_cache_task_v4"
CACHE_TASK_RECEIPT_SCHEMA = "wm3d_v8_cache_task_receipt_v1"
CACHE_EPISODE_SEAL_SCHEMA = "wm3d_v8_episode_cache_seal_v4"
CACHE_WINDOW_SEAL_SCHEMA = "wm3d_v8_window_index_seal_v3"
HEX64 = re.compile(r"[0-9a-f]{64}")


class CacheTaskError(RuntimeError):
    pass


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def cache_task_from_mapping(value: Mapping[str, Any]) -> "CacheTask":
    """Parse a planned task and re-derive its content address fail-closed."""

    required = {
        "schema",
        "task_id",
        "source",
        "episode_id",
        "payload",
        "payload_sha256",
        "payload_row_start",
        "payload_row_stop",
        "source_record_sha256",
        "assets",
        "views",
        "task_text",
        "embodiment",
        "split",
        "observation_samples",
        "observation_clock",
        "robot_groups",
        "source_manifest_sha256",
        "adapter_contract_sha256",
        "encoder_contract_sha256",
        "task_encoder_contract_sha256",
        "task_bank_index_sha256",
        "representation_contract_sha256",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise CacheTaskError("cache task fields mismatch")
    if value["schema"] != CACHE_TASK_SCHEMA:
        raise CacheTaskError(f"cache task schema must be {CACHE_TASK_SCHEMA}")
    # Source manifest schema/duration participate in the original source-row
    # digest but are intentionally not duplicated into the task.  The planner
    # therefore stores and the parser validates the digest format, while the
    # task identity below binds that digest transitively.
    source_record_sha = str(value["source_record_sha256"])
    if HEX64.fullmatch(source_record_sha) is None:
        raise CacheTaskError("source_record_sha256 is invalid")
    assets = tuple(
        (str(item["role"]), str(item["path"]), str(item["sha256"]))
        for item in value["assets"]
    )
    if (
        not assets
        or any(not role or HEX64.fullmatch(digest) is None for role, _path, digest in assets)
        or len({role for role, _path, _digest in assets}) != len(assets)
    ):
        raise CacheTaskError("cache task assets are empty, duplicated, or invalid")
    views = tuple(
        (
            str(item["name"]),
            str(item["asset_role"]),
            str(item["segment_kind"]),
            None if item["start_s"] is None else float(item["start_s"]),
            None if item["stop_s"] is None else float(item["stop_s"]),
        )
        for item in value["views"]
    )
    if not views or len({name for name, *_rest in views}) != len(views):
        raise CacheTaskError("cache task views are empty or duplicated")
    asset_roles = {role for role, _path, _digest in assets}
    if any(role not in asset_roles for _name, role, *_rest in views):
        raise CacheTaskError("cache task view references a missing asset role")
    identity = task_identity(
        source=str(value["source"]),
        episode_id=str(value["episode_id"]),
        payload_sha256=str(value["payload_sha256"]),
        source_record_sha256=source_record_sha,
        source_manifest_sha256=str(value["source_manifest_sha256"]),
        adapter_contract_sha256=str(value["adapter_contract_sha256"]),
        encoder_contract_sha256=str(value["encoder_contract_sha256"]),
        task_encoder_contract_sha256=str(value["task_encoder_contract_sha256"]),
        task_bank_index_sha256=str(value["task_bank_index_sha256"]),
        representation_contract_sha256=str(value["representation_contract_sha256"]),
    )
    if identity != value["task_id"]:
        raise CacheTaskError("cache task identity does not match its bound contracts")
    task = CacheTask(
        task_id=identity,
        source=str(value["source"]),
        episode_id=str(value["episode_id"]),
        payload=safe_relative_path(str(value["payload"])),
        payload_sha256=str(value["payload_sha256"]),
        payload_row_start=int(value["payload_row_start"]),
        payload_row_stop=int(value["payload_row_stop"]),
        source_record_sha256=source_record_sha,
        assets=tuple(
            (role, safe_relative_path(path), digest) for role, path, digest in assets
        ),
        views=views,
        task_text=str(value["task_text"]),
        embodiment=str(value["embodiment"]),
        split=str(value["split"]),
        observation_samples=int(value["observation_samples"]),
        observation_clock=dict(value["observation_clock"]),
        robot_groups=dict(value["robot_groups"]),
        source_manifest_sha256=str(value["source_manifest_sha256"]),
        adapter_contract_sha256=str(value["adapter_contract_sha256"]),
        encoder_contract_sha256=str(value["encoder_contract_sha256"]),
        task_encoder_contract_sha256=str(value["task_encoder_contract_sha256"]),
        task_bank_index_sha256=str(value["task_bank_index_sha256"]),
        representation_contract_sha256=str(value["representation_contract_sha256"]),
    )
    for name, digest in (
        ("payload", task.payload_sha256),
        ("source manifest", task.source_manifest_sha256),
        ("adapter", task.adapter_contract_sha256),
        ("encoder", task.encoder_contract_sha256),
        ("task encoder", task.task_encoder_contract_sha256),
        ("task bank index", task.task_bank_index_sha256),
        ("representation", task.representation_contract_sha256),
    ):
        if HEX64.fullmatch(digest) is None:
            raise CacheTaskError(f"cache task {name} SHA is invalid")
    if task.payload_row_start < 0 or task.payload_row_stop <= task.payload_row_start:
        raise CacheTaskError("cache task payload row slice is invalid")
    if task.observation_samples != task.payload_row_stop - task.payload_row_start:
        raise CacheTaskError("cache task observation cardinality differs from row slice")
    if task.as_dict() != value:
        raise CacheTaskError("cache task failed canonical round-trip")
    primary = [
        (path, digest)
        for role, path, digest in task.assets
        if role == "primary_payload"
    ]
    if primary != [(task.payload, task.payload_sha256)]:
        raise CacheTaskError("cache task primary payload binding mismatch")
    return task


@dataclass(frozen=True)
class CacheTask:
    task_id: str
    source: str
    episode_id: str
    payload: str
    payload_sha256: str
    payload_row_start: int
    payload_row_stop: int
    source_record_sha256: str
    assets: tuple[tuple[str, str, str], ...]
    views: tuple[tuple[str, str, str, float | None, float | None], ...]
    task_text: str
    embodiment: str
    split: str
    observation_samples: int
    observation_clock: Mapping[str, Any]
    robot_groups: Mapping[str, Any]
    source_manifest_sha256: str
    adapter_contract_sha256: str
    encoder_contract_sha256: str
    task_encoder_contract_sha256: str
    task_bank_index_sha256: str
    representation_contract_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": CACHE_TASK_SCHEMA,
            "task_id": self.task_id,
            "source": self.source,
            "episode_id": self.episode_id,
            "payload": self.payload,
            "payload_sha256": self.payload_sha256,
            "payload_row_start": self.payload_row_start,
            "payload_row_stop": self.payload_row_stop,
            "source_record_sha256": self.source_record_sha256,
            "assets": [
                {"role": role, "path": path, "sha256": digest}
                for role, path, digest in self.assets
            ],
            "views": [
                {
                    "name": name,
                    "asset_role": role,
                    "segment_kind": segment_kind,
                    "start_s": start_s,
                    "stop_s": stop_s,
                }
                for name, role, segment_kind, start_s, stop_s in self.views
            ],
            "task_text": self.task_text,
            "embodiment": self.embodiment,
            "split": self.split,
            "observation_samples": self.observation_samples,
            "observation_clock": self.observation_clock,
            "robot_groups": self.robot_groups,
            "source_manifest_sha256": self.source_manifest_sha256,
            "adapter_contract_sha256": self.adapter_contract_sha256,
            "encoder_contract_sha256": self.encoder_contract_sha256,
            "task_encoder_contract_sha256": self.task_encoder_contract_sha256,
            "task_bank_index_sha256": self.task_bank_index_sha256,
            "representation_contract_sha256": self.representation_contract_sha256,
        }


def task_identity(
    *,
    source: str,
    episode_id: str,
    payload_sha256: str,
    source_record_sha256: str,
    source_manifest_sha256: str,
    adapter_contract_sha256: str,
    encoder_contract_sha256: str,
    task_encoder_contract_sha256: str,
    task_bank_index_sha256: str,
    representation_contract_sha256: str,
) -> str:
    return canonical_sha256(
        {
            "source": source,
            "episode_id": episode_id,
            "payload_sha256": payload_sha256,
            "source_record_sha256": source_record_sha256,
            "source_manifest_sha256": source_manifest_sha256,
            "adapter_contract_sha256": adapter_contract_sha256,
            "encoder_contract_sha256": encoder_contract_sha256,
            "task_encoder_contract_sha256": task_encoder_contract_sha256,
            "task_bank_index_sha256": task_bank_index_sha256,
            "representation_contract_sha256": representation_contract_sha256,
        }
    )


def plan_tasks(
    rows: Iterable[Mapping[str, Any]],
    *,
    source_manifest_sha256: str,
    adapter_contract_sha256: str,
    encoder_contract_sha256: str,
    task_encoder_contract_sha256: str,
    task_bank_index_sha256: str,
    representation_contract_sha256: str,
    canonical_view_slots: Iterable[str] | None = None,
) -> tuple[CacheTask, ...]:
    for name, value in (
        ("source_manifest", source_manifest_sha256),
        ("adapter", adapter_contract_sha256),
        ("encoder", encoder_contract_sha256),
        ("task_encoder", task_encoder_contract_sha256),
        ("task_bank_index", task_bank_index_sha256),
        ("representation", representation_contract_sha256),
    ):
        if HEX64.fullmatch(value) is None:
            raise CacheTaskError(f"{name} contract digest must be SHA256")
    slots = None if canonical_view_slots is None else tuple(canonical_view_slots)
    if slots is not None and (
        not slots or any(not item for item in slots) or len(set(slots)) != len(slots)
    ):
        raise CacheTaskError("canonical_view_slots must be unique/non-empty")
    tasks: list[CacheTask] = []
    seen: set[str] = set()
    for row in rows:
        source = str(row["source"])
        episode_id = str(row["episode_id"])
        payload_sha = str(row["payload_sha256"])
        if HEX64.fullmatch(payload_sha) is None:
            raise CacheTaskError(f"{episode_id}: payload digest is invalid")
        source_record_sha = canonical_sha256(row)
        row_start = int(row["payload_row_start"])
        row_stop = int(row["payload_row_stop"])
        if row_start < 0 or row_stop <= row_start:
            raise CacheTaskError(f"{episode_id}: invalid payload row slice")
        raw_assets = row["assets"]
        if not isinstance(raw_assets, list) or not raw_assets:
            raise CacheTaskError(f"{episode_id}: source assets are missing")
        assets: list[tuple[str, str, str]] = []
        for raw_asset in raw_assets:
            if not isinstance(raw_asset, dict) or set(raw_asset) != {
                "role",
                "path",
                "sha256",
            }:
                raise CacheTaskError(f"{episode_id}: invalid source asset entry")
            digest = str(raw_asset["sha256"])
            if HEX64.fullmatch(digest) is None:
                raise CacheTaskError(f"{episode_id}: invalid source asset digest")
            assets.append((str(raw_asset["role"]), str(raw_asset["path"]), digest))
        raw_views = row.get("views")
        if not isinstance(raw_views, list) or not raw_views:
            raise CacheTaskError(f"{episode_id}: source views are missing")
        views: list[tuple[str, str, str, float | None, float | None]] = []
        for raw_view in raw_views:
            if not isinstance(raw_view, dict) or set(raw_view) != {
                "name",
                "asset_role",
                "segment_kind",
                "start_s",
                "stop_s",
            }:
                raise CacheTaskError(f"{episode_id}: invalid source view entry")
            views.append(
                (
                    str(raw_view["name"]),
                    str(raw_view["asset_role"]),
                    str(raw_view["segment_kind"]),
                    None if raw_view["start_s"] is None else float(raw_view["start_s"]),
                    None if raw_view["stop_s"] is None else float(raw_view["stop_s"]),
                )
            )
        source_view_names = tuple(item[0] for item in views)
        if len(set(source_view_names)) != len(source_view_names):
            raise CacheTaskError(f"{episode_id}: duplicate source view names")
        if slots is not None:
            unknown_views = sorted(set(source_view_names) - set(slots))
            if unknown_views:
                raise CacheTaskError(
                    f"{episode_id}: source views not in canonical slots: {unknown_views}"
                )
        task_text = str(row.get("task_text", ""))
        embodiment = str(row.get("embodiment", ""))
        split = str(row.get("split", ""))
        observation_samples = int(row.get("observation_samples", 0))
        observation_clock = row.get("observation_clock")
        robot_groups = row.get("robot_groups")
        if not task_text.strip() or not embodiment or split not in {"train", "val", "test"}:
            raise CacheTaskError(f"{episode_id}: task/embodiment/split is invalid")
        if (
            observation_samples != row_stop - row_start
            or not isinstance(observation_clock, dict)
            or int(observation_clock.get("sample_count", -1)) != observation_samples
            or not isinstance(robot_groups, dict)
            or not robot_groups
        ):
            raise CacheTaskError(
                f"{episode_id}: observation clock/robot-group evidence is invalid"
            )
        identity = task_identity(
            source=source,
            episode_id=episode_id,
            payload_sha256=payload_sha,
            source_record_sha256=source_record_sha,
            source_manifest_sha256=source_manifest_sha256,
            adapter_contract_sha256=adapter_contract_sha256,
            encoder_contract_sha256=encoder_contract_sha256,
            task_encoder_contract_sha256=task_encoder_contract_sha256,
            task_bank_index_sha256=task_bank_index_sha256,
            representation_contract_sha256=representation_contract_sha256,
        )
        if identity in seen:
            raise CacheTaskError(f"duplicate cache task identity {identity}")
        seen.add(identity)
        tasks.append(
            CacheTask(
                identity,
                source,
                episode_id,
                str(row["payload"]),
                payload_sha,
                row_start,
                row_stop,
                source_record_sha,
                tuple(assets),
                tuple(views),
                task_text,
                embodiment,
                split,
                observation_samples,
                dict(observation_clock),
                dict(robot_groups),
                source_manifest_sha256,
                adapter_contract_sha256,
                encoder_contract_sha256,
                task_encoder_contract_sha256,
                task_bank_index_sha256,
                representation_contract_sha256,
            )
        )
    return tuple(tasks)


class AtomicTaskClaim:
    """One worker owns one task; completed matching receipts are skipped."""

    def __init__(self, root: Path, task: CacheTask):
        self.root = Path(root)
        self.task = task
        self.claim = self.root / "claims" / f"{task.task_id}.claim"
        self.receipt = self.root / "receipts" / f"{task.task_id}.json"
        self._descriptor: int | None = None

    def completed(self) -> bool:
        if not self.receipt.is_file() or self.receipt.is_symlink():
            return False
        value = json.loads(self.receipt.read_text(encoding="utf-8"))
        if value.get("schema") != CACHE_TASK_RECEIPT_SCHEMA:
            raise CacheTaskError(f"receipt schema mismatch: {self.receipt}")
        if value.get("task") != self.task.as_dict():
            raise CacheTaskError(f"receipt task mismatch: {self.receipt}")
        outputs = value.get("outputs")
        if not isinstance(outputs, dict) or not outputs:
            raise CacheTaskError(f"receipt has no outputs: {self.receipt}")
        for path_value, evidence in outputs.items():
            path = Path(path_value)
            if not path.is_absolute() or path.is_symlink() or not path.is_file():
                raise CacheTaskError(f"receipt output is invalid: {path}")
            if (
                not isinstance(evidence, dict)
                or set(evidence) != {"sha256", "size_bytes"}
                or HEX64.fullmatch(str(evidence["sha256"])) is None
                or int(evidence["size_bytes"]) <= 0
                or path.stat().st_size != int(evidence["size_bytes"])
            ):
                raise CacheTaskError(f"receipt output evidence mismatch: {path}")
        return True

    def __enter__(self) -> "AtomicTaskClaim":
        if self.completed():
            return self
        self.claim.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._descriptor = os.open(
                self.claim, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640
            )
            os.write(self._descriptor, f"pid={os.getpid()}\n".encode())
            os.fsync(self._descriptor)
        except FileExistsError as exc:
            raise CacheTaskError(f"task already claimed: {self.task.task_id}") from exc
        return self


    def publish_receipt(self, outputs: Mapping[Path, str]) -> None:
        if self._descriptor is None:
            raise CacheTaskError("task was not claimed")
        normalized = {
            str(Path(path).resolve(strict=True)): {
                "sha256": digest,
                "size_bytes": Path(path).stat().st_size,
            }
            for path, digest in outputs.items()
        }
        value = {
            "schema": CACHE_TASK_RECEIPT_SCHEMA,
            "task": self.task.as_dict(),
            "outputs": normalized,
        }
        payload = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
        self.receipt.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.receipt.with_name(
            f".{self.receipt.name}.tmp.{os.getpid()}"
        )
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, self.receipt)
        except FileExistsError:
            if self.receipt.read_bytes() != payload:
                raise CacheTaskError("non-identical task receipt already exists")
        finally:
            temporary.unlink(missing_ok=True)

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._descriptor is not None:
            os.close(self._descriptor)
            self._descriptor = None
        # A claim is removed only after success; failed claims remain as
        # explicit evidence and require operator inspection before retry.
        if exc_type is None and self.receipt.is_file():
            self.claim.unlink()
