"""Strict data and lineage contracts for native WM3D-V7 5B pretraining.

The production dataset is intentionally split into:

* a small, hashable control plane (contracts, indexes and shard manifests);
* immutable payload shards referenced by content digest; and
* a seal receipt binding the complete control plane.

Normal launch preflight verifies the complete control plane and payload
metadata.  A separate deep verifier can re-hash every payload shard before a
formal data release without forcing every training restart to scan 100 TB.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any, Iterable, Mapping


DATASET_SCHEMA = "wm3d_v7_native5b_dataset_v1"
SEAL_SCHEMA = "wm3d_v7_native5b_dataset_seal_v1"
MANIFEST_SCHEMA = "wm3d_v7_native5b_shard_manifest_v1"
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,127}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ContractError(ValueError):
    """Raised when a production data or resume contract is malformed."""


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _validate_name(value: str, field_name: str) -> str:
    value = str(value)
    if not _NAME_RE.fullmatch(value):
        raise ContractError(f"{field_name} is not a canonical name: {value!r}")
    return value


def _validate_sha(value: str, field_name: str) -> str:
    value = str(value)
    if not _SHA256_RE.fullmatch(value):
        raise ContractError(f"{field_name} is not a lowercase SHA-256: {value!r}")
    return value


def safe_relative_path(value: str) -> PurePosixPath:
    raw = str(value)
    path = PurePosixPath(raw)
    if not raw or path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ContractError(f"unsafe relative path: {raw!r}")
    if "\\" in raw or "\x00" in raw:
        raise ContractError(f"non-portable relative path: {raw!r}")
    return path


def resolve_real_directory(path: Path, field_name: str = "directory") -> Path:
    """Resolve a directory while rejecting a symlink at the supplied root."""

    input_path = Path(path)
    try:
        info = os.lstat(input_path)
    except OSError as exc:
        raise ContractError(
            f"{field_name} is unavailable: {input_path}: {exc}"
        ) from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ContractError(f"{field_name} is not a real directory: {input_path}")
    return input_path.resolve(strict=True)


def resolve_regular_file(root: Path, relative: str) -> Path:
    """Resolve a regular file under root while rejecting symlink traversal."""

    rel = safe_relative_path(relative)
    root = resolve_real_directory(root, "sealed root")
    current = root
    for part in rel.parts:
        current = current / part
        info = os.lstat(current)
        if stat.S_ISLNK(info.st_mode):
            raise ContractError(f"symlink is forbidden in sealed path: {relative}")
    if not current.is_file():
        raise ContractError(f"sealed path is not a regular file: {relative}")
    if root not in current.resolve(strict=True).parents:
        raise ContractError(f"sealed path escaped dataset root: {relative}")
    return current


def atomic_write_json(path: Path, value: Any, *, exclusive: bool = True) -> None:
    """Durably publish canonical JSON using same-directory atomic rename."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive and path.exists():
        raise FileExistsError(path)
    suffix = f".tmp.{os.getpid()}.{os.urandom(6).hex()}"
    temporary = path.with_name(path.name + suffix)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(temporary, flags, 0o640)
    try:
        payload = canonical_json_bytes(value)
        with os.fdopen(fd, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.close(fd)
        fd = -1
        if exclusive and path.exists():
            raise FileExistsError(path)
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if fd >= 0:
            os.close(fd)
        if temporary.exists():
            temporary.unlink()


@dataclass(frozen=True)
class ActionGroupSpec:
    name: str
    group_id: int
    dimensions: tuple[str, ...]
    rate_hz: float
    control_mode: str
    normalization: str = "robust_quantile"

    def validate(self, *, max_dim: int, max_group_id: int) -> None:
        _validate_name(self.name, "action group name")
        if not 0 <= int(self.group_id) < int(max_group_id):
            raise ContractError(f"group_id out of range for {self.name}")
        if not self.dimensions or len(self.dimensions) > int(max_dim):
            raise ContractError(
                f"{self.name} has {len(self.dimensions)} dimensions; max is {max_dim}"
            )
        if len(set(self.dimensions)) != len(self.dimensions):
            raise ContractError(f"duplicate dimensions in action group {self.name}")
        for dimension in self.dimensions:
            _validate_name(dimension, f"{self.name} dimension")
        if not 5.0 <= float(self.rate_hz) <= 100.0:
            raise ContractError(
                f"unsupported action rate for {self.name}: {self.rate_hz}"
            )
        _validate_name(self.control_mode, f"{self.name} control mode")
        _validate_name(self.normalization, f"{self.name} normalization")


@dataclass(frozen=True)
class AuxiliaryModalitySpec:
    """One context-only numeric sensor stream.

    ``type_id`` occupies one position in the first ``max_aux_type_id``
    dimensions of the packed auxiliary token.  Numeric values occupy the
    remaining dimensions.  This keeps heterogeneous sensors distinguishable
    without adding a learned producer to the data pipeline.
    """

    name: str
    type_id: int
    dimensions: tuple[str, ...]
    rate_hz: float
    discrete: bool = False
    normalization: str = "robust_quantile"

    def validate(
        self,
        *,
        aux_dim: int,
        max_aux_type_id: int,
    ) -> None:
        _validate_name(self.name, "auxiliary modality name")
        if not 0 <= int(self.type_id) < int(max_aux_type_id):
            raise ContractError(f"auxiliary type_id out of range for {self.name}")
        value_capacity = int(aux_dim) - int(max_aux_type_id)
        if not self.dimensions or 2 * len(self.dimensions) > value_capacity:
            raise ContractError(
                f"{self.name} has {len(self.dimensions)} dimensions; "
                f"packed auxiliary value+validity capacity is {value_capacity}"
            )
        if len(set(self.dimensions)) != len(self.dimensions):
            raise ContractError(
                f"duplicate dimensions in auxiliary modality {self.name}"
            )
        for dimension in self.dimensions:
            _validate_name(dimension, f"{self.name} auxiliary dimension")
        if not 1.0 <= float(self.rate_hz) <= 1000.0:
            raise ContractError(
                f"unsupported auxiliary rate for {self.name}: {self.rate_hz}"
            )
        _validate_name(self.normalization, f"{self.name} normalization")


@dataclass(frozen=True)
class EmbodimentSpec:
    name: str
    embodiment_id: int
    views: tuple[str, ...]
    action_groups: tuple[ActionGroupSpec, ...]
    auxiliary_modalities: tuple[AuxiliaryModalitySpec, ...] = ()

    def validate(
        self,
        *,
        max_views: int,
        max_groups: int,
        max_dim: int,
        max_group_id: int,
        max_embodiments: int,
        max_aux_tokens: int,
        aux_dim: int,
        max_aux_type_id: int,
    ) -> None:
        _validate_name(self.name, "embodiment name")
        if not 0 <= int(self.embodiment_id) < int(max_embodiments):
            raise ContractError(f"embodiment_id out of range for {self.name}")
        if not self.views or len(self.views) > int(max_views):
            raise ContractError(
                f"{self.name} has invalid view count: {len(self.views)}"
            )
        if len(set(self.views)) != len(self.views):
            raise ContractError(f"duplicate views for embodiment {self.name}")
        for view in self.views:
            _validate_name(view, f"{self.name} view")
        if not self.action_groups or len(self.action_groups) > int(max_groups):
            raise ContractError(
                f"{self.name} has invalid action-group count: {len(self.action_groups)}"
            )
        group_ids = [int(group.group_id) for group in self.action_groups]
        if len(group_ids) != len(set(group_ids)):
            raise ContractError(f"duplicate group ids for embodiment {self.name}")
        for group in self.action_groups:
            group.validate(max_dim=max_dim, max_group_id=max_group_id)
        if len(self.auxiliary_modalities) > int(max_aux_tokens):
            raise ContractError(
                f"{self.name} has {len(self.auxiliary_modalities)} auxiliary "
                f"modalities; max is {max_aux_tokens}"
            )
        auxiliary_names = [item.name for item in self.auxiliary_modalities]
        auxiliary_ids = [int(item.type_id) for item in self.auxiliary_modalities]
        if len(auxiliary_names) != len(set(auxiliary_names)):
            raise ContractError(f"duplicate auxiliary modality in {self.name}")
        if len(auxiliary_ids) != len(set(auxiliary_ids)):
            raise ContractError(f"duplicate auxiliary type_id in {self.name}")
        for modality in self.auxiliary_modalities:
            modality.validate(
                aux_dim=aux_dim,
                max_aux_type_id=max_aux_type_id,
            )


@dataclass(frozen=True)
class SourceSpec:
    name: str
    adapter: str
    raw_root: str
    license_id: str
    nominal_hours: float
    weight: int
    embodiment_names: tuple[str, ...]
    split_seed: int
    train_fraction: float = 0.98

    def validate(self) -> None:
        _validate_name(self.name, "source name")
        _validate_name(self.adapter, f"{self.name} adapter")
        if not str(self.raw_root):
            raise ContractError(f"source {self.name} has empty raw_root")
        if not str(self.license_id):
            raise ContractError(f"source {self.name} has empty license_id")
        if float(self.nominal_hours) <= 0:
            raise ContractError(f"source {self.name} has invalid nominal hours")
        if int(self.weight) <= 0:
            raise ContractError(f"source {self.name} has invalid sampling weight")
        if not self.embodiment_names:
            raise ContractError(f"source {self.name} has no embodiments")
        if not 0.5 <= float(self.train_fraction) < 1.0:
            raise ContractError(f"source {self.name} has invalid train fraction")


@dataclass(frozen=True)
class DatasetContract:
    name: str
    feature_fps: float
    action_fps: float
    T: int
    P: int
    K: int
    token_dim: int
    task_dim: int
    num_views: int
    max_action_groups: int
    max_action_dim: int
    action_substeps: int
    max_group_id: int
    max_embodiments: int
    max_aux_tokens: int
    aux_dim: int
    max_aux_type_id: int
    source_order: tuple[str, ...]
    sources: tuple[SourceSpec, ...]
    embodiments: tuple[EmbodimentSpec, ...]
    schema: str = DATASET_SCHEMA
    notes: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "DatasetContract":
        source_values = []
        for value in raw.get("sources", ()):
            item = dict(value)
            item["embodiment_names"] = tuple(item.get("embodiment_names", ()))
            source_values.append(SourceSpec(**item))
        embodiment_values = []
        for value in raw.get("embodiments", ()):
            item = dict(value)
            groups = []
            for group_value in item.get("action_groups", ()):
                group_item = dict(group_value)
                group_item["dimensions"] = tuple(group_item.get("dimensions", ()))
                groups.append(ActionGroupSpec(**group_item))
            item["views"] = tuple(item.get("views", ()))
            item["action_groups"] = tuple(groups)
            auxiliary = []
            for auxiliary_value in item.get("auxiliary_modalities", ()):
                auxiliary_item = dict(auxiliary_value)
                auxiliary_item["dimensions"] = tuple(
                    auxiliary_item.get("dimensions", ())
                )
                auxiliary.append(AuxiliaryModalitySpec(**auxiliary_item))
            item["auxiliary_modalities"] = tuple(auxiliary)
            embodiment_values.append(EmbodimentSpec(**item))
        values = dict(raw)
        values["source_order"] = tuple(values.get("source_order", ()))
        values["sources"] = tuple(source_values)
        values["embodiments"] = tuple(embodiment_values)
        contract = cls(**values)
        contract.validate()
        return contract

    def validate(self) -> None:
        if self.schema != DATASET_SCHEMA:
            raise ContractError(f"unsupported dataset schema: {self.schema}")
        _validate_name(self.name, "dataset name")
        if float(self.feature_fps) != 5.0:
            raise ContractError("native5b feature_fps must be exactly 5 Hz")
        expected_substeps = float(self.action_fps) / float(self.feature_fps)
        if expected_substeps != int(expected_substeps):
            raise ContractError("action_fps must be an integer multiple of feature_fps")
        if int(expected_substeps) != int(self.action_substeps):
            raise ContractError(
                "action_substeps does not match action_fps / feature_fps"
            )
        if (self.T, self.P, self.K, self.token_dim) != (24, 144, 16, 2048):
            raise ContractError(
                "production native5b representation must be T24/P144/K16/D2048"
            )
        if self.num_views != 3:
            raise ContractError("production native5b contract requires three views")
        if (
            int(self.max_aux_tokens),
            int(self.aux_dim),
            int(self.max_aux_type_id),
        ) != (8, 256, 64):
            raise ContractError(
                "production native5b auxiliary contract must be "
                "8 tokens / D256 / 64 type slots"
            )
        sources = {source.name: source for source in self.sources}
        embodiments = {item.name: item for item in self.embodiments}
        if len(sources) != len(self.sources) or len(embodiments) != len(
            self.embodiments
        ):
            raise ContractError("source and embodiment names must be unique")
        if tuple(self.source_order) != tuple(source.name for source in self.sources):
            raise ContractError("source_order must exactly match sources order")
        for source in self.sources:
            source.validate()
            unknown = set(source.embodiment_names).difference(embodiments)
            if unknown:
                raise ContractError(
                    f"source {source.name} references unknown embodiments {sorted(unknown)}"
                )
        for embodiment in self.embodiments:
            embodiment.validate(
                max_views=self.num_views,
                max_groups=self.max_action_groups,
                max_dim=self.max_action_dim,
                max_group_id=self.max_group_id,
                max_embodiments=self.max_embodiments,
                max_aux_tokens=self.max_aux_tokens,
                aux_dim=self.aux_dim,
                max_aux_type_id=self.max_aux_type_id,
            )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.as_dict())

    @property
    def source_weights(self) -> dict[str, int]:
        return {source.name: int(source.weight) for source in self.sources}


@dataclass(frozen=True)
class FileEvidence:
    size: int
    sha256: str

    def validate(self) -> None:
        if int(self.size) < 0:
            raise ContractError("negative file size in receipt")
        _validate_sha(self.sha256, "file sha256")


@dataclass(frozen=True)
class DatasetSeal:
    dataset_schema: str
    dataset_contract_sha256: str
    control_files: Mapping[str, FileEvidence]
    payload_manifest_files: Mapping[str, FileEvidence]
    source_window_counts: Mapping[str, Mapping[str, int]]
    source_hours: Mapping[str, float]
    created_at_utc: str
    schema: str = SEAL_SCHEMA

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "DatasetSeal":
        values = dict(raw)
        for field_name in ("control_files", "payload_manifest_files"):
            values[field_name] = {
                str(path): FileEvidence(**dict(evidence))
                for path, evidence in dict(values.get(field_name, {})).items()
            }
        receipt = cls(**values)
        receipt.validate()
        return receipt

    def validate(self) -> None:
        if self.schema != SEAL_SCHEMA:
            raise ContractError(f"unsupported seal schema: {self.schema}")
        if self.dataset_schema != DATASET_SCHEMA:
            raise ContractError("seal is not for native5b dataset schema")
        _validate_sha(self.dataset_contract_sha256, "dataset contract sha256")
        if not self.control_files or not self.payload_manifest_files:
            raise ContractError("seal must bind control and payload-manifest files")
        for mapping in (self.control_files, self.payload_manifest_files):
            for path, evidence in mapping.items():
                safe_relative_path(path)
                evidence.validate()
        if not self.source_window_counts:
            raise ContractError("seal has no window counts")
        for source, split_counts in self.source_window_counts.items():
            _validate_name(source, "receipt source")
            unknown_splits = set(split_counts).difference({"train", "val", "test"})
            if unknown_splits:
                raise ContractError(
                    f"source {source} has unknown splits {sorted(unknown_splits)}"
                )
            if int(split_counts.get("train", 0)) <= 0:
                raise ContractError(f"source {source} has no train windows")
            if int(split_counts.get("val", 0)) <= 0:
                raise ContractError(f"source {source} has no validation windows")
            if any(int(value) <= 0 for value in split_counts.values()):
                raise ContractError(f"source {source} has non-positive split count")
        if set(self.source_hours) != set(self.source_window_counts):
            raise ContractError("seal source-hours keys differ from window-count keys")
        for source, hours in self.source_hours.items():
            if float(hours) <= 0:
                raise ContractError(f"source {source} has non-positive hours")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.as_dict())


def evidence_for(root: Path, relatives: Iterable[str]) -> dict[str, FileEvidence]:
    result: dict[str, FileEvidence] = {}
    for relative in sorted(set(str(value) for value in relatives)):
        path = resolve_regular_file(root, relative)
        result[relative] = FileEvidence(
            size=path.stat().st_size, sha256=sha256_file(path)
        )
    return result


def verify_file_evidence(
    root: Path,
    evidence: Mapping[str, FileEvidence],
) -> list[str]:
    errors: list[str] = []
    for relative, expected in sorted(evidence.items()):
        try:
            path = resolve_regular_file(root, relative)
        except (ContractError, FileNotFoundError, OSError) as exc:
            errors.append(f"{relative}: {exc}")
            continue
        size = path.stat().st_size
        if size != int(expected.size):
            errors.append(f"{relative}: size {size} != {expected.size}")
            continue
        actual_sha = sha256_file(path)
        if actual_sha != expected.sha256:
            errors.append(f"{relative}: sha256 {actual_sha} != {expected.sha256}")
    return errors


def load_contract(path: Path) -> DatasetContract:
    input_path = Path(path)
    info = os.lstat(input_path)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ContractError(f"dataset contract is not a regular file: {input_path}")
    resolved = input_path.resolve(strict=True)
    return DatasetContract.from_mapping(
        json.loads(resolved.read_text(encoding="utf-8"))
    )


def load_seal(path: Path) -> DatasetSeal:
    input_path = Path(path)
    info = os.lstat(input_path)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ContractError(f"dataset seal is not a regular file: {input_path}")
    resolved = input_path.resolve(strict=True)
    return DatasetSeal.from_mapping(json.loads(resolved.read_text(encoding="utf-8")))


def verify_dataset_seal(
    root: Path,
    receipt_relative: str = "receipts/dataset_seal.json",
) -> dict[str, Any]:
    root = resolve_real_directory(root, "dataset root")
    receipt_path = resolve_regular_file(root, receipt_relative)
    receipt = load_seal(receipt_path)
    errors = verify_file_evidence(root, receipt.control_files)
    errors.extend(verify_file_evidence(root, receipt.payload_manifest_files))
    contract_candidates = [
        relative
        for relative in receipt.control_files
        if relative.endswith("dataset_contract.json")
    ]
    if len(contract_candidates) != 1:
        errors.append("seal must bind exactly one dataset_contract.json")
    else:
        contract = load_contract(resolve_regular_file(root, contract_candidates[0]))
        if contract.sha256 != receipt.dataset_contract_sha256:
            errors.append(
                "dataset contract digest mismatch: "
                f"{contract.sha256} != {receipt.dataset_contract_sha256}"
            )
        expected_sources = set(contract.source_order)
        if set(receipt.source_window_counts) != expected_sources:
            errors.append(
                "seal window-count sources differ from dataset contract: "
                f"{sorted(receipt.source_window_counts)} != "
                f"{sorted(expected_sources)}"
            )
        if set(receipt.source_hours) != expected_sources:
            errors.append(
                "seal source-hours keys differ from dataset contract: "
                f"{sorted(receipt.source_hours)} != "
                f"{sorted(expected_sources)}"
            )
    return {
        "schema": SEAL_SCHEMA,
        "pass": not errors,
        "errors": errors,
        "receipt_sha256": receipt.sha256,
        "dataset_contract_sha256": receipt.dataset_contract_sha256,
        "source_window_counts": receipt.source_window_counts,
        "source_hours": receipt.source_hours,
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
