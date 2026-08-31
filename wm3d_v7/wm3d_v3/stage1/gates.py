from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import random
import statistics
import tempfile
from typing import Any

from wm3d_v3.stage1 import artifacts as _artifacts
from wm3d_v3.stage1.artifacts import (
    ArtifactError,
    CandidateIdentity,
    atomic_write_json,
    canonical_json_bytes,
    fsync_directory,
    fsync_tree,
    load_candidate,
    read_bytes_no_follow,
    reject_symlink_components,
    rename_no_replace,
    require_directory_no_symlink,
    require_regular_file_no_symlink,
    sha256_file,
)
from wm3d_v3.stage1.closed_loop import (
    G4_SOURCE_ENUM_SCHEMA,
    G4_SOURCE_SCHEMA_VERSION,
    g4_expected_rgb_source,
    g4_expected_token_source,
)


REQUIRED_GATES = tuple(f"G{index}" for index in range(6))
REQUIRED_ACTION_MODES = frozenset({"zero", "reverse", "time_shift", "grip_toggle"})
REQUIRED_GRIP_CLASSES = frozenset({"up", "down", "non_transition"})
CANONICAL_BUNDLE_FILES = frozenset(
    {
        "config/stage1.yaml",
        "config/flow.yaml",
        "data/unique_manifest.jsonl",
        "data/action_frame_contract.json",
        "eval/short_multidomain.json",
        "eval/rolling_closed_loop.json",
    }
)


class GateError(RuntimeError):
    """Raised when gate evidence cannot produce a trustworthy verdict."""


@dataclass(frozen=True)
class GateVerdict:
    gate: str
    status: str
    failures: tuple[str, ...]
    details: Mapping[str, Any]

    @property
    def passed(self) -> bool:
        return self.status == "PASS"

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate": self.gate,
            "status": self.status,
            "passed": self.passed,
            "failures": list(self.failures),
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class G2Thresholds:
    psnr_delta_lower: float = -0.5
    ssim_delta_lower: float = -0.015
    lpips_improvement_lower: float = -0.02
    geometry_relative_lower: float = -0.05
    bridge_median_psnr_delta: float = 0.5
    catastrophic_psnr_delta: float = -5.0
    catastrophic_ssim_delta: float = -0.10
    catastrophic_lpips_improvement: float = -0.15
    catastrophic_geometry_relative: float = -0.25


@dataclass(frozen=True)
class G3Thresholds:
    separation_ci_lower: float = 0.0
    prediction_difference_floor: float = 0.0
    direction_agreement: float = 0.70


@dataclass(frozen=True)
class G4Thresholds:
    seam_ratio: float = 1.25
    bridge_drift_improvement: float = 0.15
    maximum_regression: float = 0.10
    minimum_chunks: int = 8
    minimum_future_frames: int = 64


@dataclass(frozen=True)
class G5Thresholds:
    world_size: int = 8
    node43_endpoint: str = "172.27.0.6"
    forbidden_node42_endpoint: str = "172.27.0.5"
    maximum_memory_fraction: float = 0.85
    average_step_ratio: float = 1.25
    rolling_step_ratio: float = 6.0
    stall_seconds: float = 120.0


def _verdict(
    gate: str,
    failures: Iterable[str],
    *,
    details: Mapping[str, Any] | None = None,
) -> GateVerdict:
    normalized = tuple(dict.fromkeys(str(failure) for failure in failures))
    return GateVerdict(
        gate=gate,
        status="PASS" if not normalized else "FAIL",
        failures=normalized,
        details={} if details is None else details,
    )


def _as_row(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if is_dataclass(value):
        return asdict(value)
    raise GateError(f"gate observation must be a mapping, got {type(value)!r}")


def _rows(values: Iterable[Any], gate: str) -> list[dict[str, Any]]:
    rows = [_as_row(value) for value in values]
    if not rows:
        raise GateError(f"{gate} requires nonempty evidence")
    return rows


def _text(row: Mapping[str, Any], key: str) -> str:
    value = str(row.get(key, "")).strip()
    if not value:
        raise GateError(f"gate observation has no {key}")
    return value


def _number(row: Mapping[str, Any], key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise GateError(f"gate observation has invalid {key}") from exc
    if not math.isfinite(value):
        raise GateError(f"gate observation has nonfinite {key}")
    return value


def _seed(row: Mapping[str, Any]) -> int:
    try:
        return int(row["seed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise GateError("gate observation has invalid seed") from exc


def _stable_random(label: str) -> random.Random:
    seed = int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:8], "big")
    return random.Random(seed)


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise GateError("cannot compute a percentile of no values")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(
        sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction
    )


def _bootstrap_mean_interval(
    values: Sequence[float],
    *,
    label: str,
    samples: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    if not values:
        raise GateError("bootstrap group has no clips")
    if samples < 100:
        raise GateError("bootstrap_samples must be at least 100")
    if len(values) == 1:
        return float(values[0]), float(values[0])
    generator = _stable_random(label)
    count = len(values)
    estimates = sorted(
        statistics.fmean(values[generator.randrange(count)] for _ in range(count))
        for _ in range(samples)
    )
    tail = (1.0 - confidence) / 2.0
    return _percentile(estimates, tail), _percentile(estimates, 1.0 - tail)


def _normalize_mode(value: object) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _group_label(kind: str, value: tuple[object, ...]) -> str:
    return f"{kind}=" + "/".join(str(part) for part in value)


def _evaluate_g0_core(
    candidate_dir: str | Path,
    *,
    expected_identity: CandidateIdentity | None = None,
    modules: Mapping[str, Any] | None = None,
    strict_loader: Any = None,
    bound_artifacts: Mapping[str, str | Path] | None = None,
    resume_dir: str | Path | None = None,
) -> GateVerdict:
    failures: list[str] = []
    details: dict[str, Any] = {}
    try:
        loaded = load_candidate(
            candidate_dir,
            expected_identity=expected_identity,
            modules=modules,
            strict_loader=strict_loader,
        )
        details["candidate_id"] = loaded.identity.candidate_id
        details["identity"] = loaded.identity.to_dict()
        if loaded.strict_load_report is not None:
            details["strict_load_report"] = {
                "loaded_files": list(loaded.strict_load_report.loaded_files),
                "missing_keys": list(loaded.strict_load_report.missing_keys),
                "unexpected_keys": list(loaded.strict_load_report.unexpected_keys),
                "shape_mismatches": list(loaded.strict_load_report.shape_mismatches),
            }
        expected_hashes = {
            "config": loaded.identity.config_sha256,
            "contract": loaded.identity.contract_sha256,
            "unique_manifest": loaded.identity.unique_manifest_sha256,
        }
        if bound_artifacts is not None:
            missing = set(expected_hashes) - set(bound_artifacts)
            extra = set(bound_artifacts) - set(expected_hashes)
            if missing or extra:
                failures.append(
                    f"bound artifact set mismatch: "
                    f"missing={sorted(missing)}, extra={sorted(extra)}"
                )
            for label in sorted(set(expected_hashes) & set(bound_artifacts)):
                actual = sha256_file(bound_artifacts[label])
                if actual != expected_hashes[label]:
                    failures.append(
                        f"{label} SHA256 mismatch: expected "
                        f"{expected_hashes[label]}, got {actual}"
                    )
        if resume_dir is not None:
            metadata = _artifacts._validate_resume_inventory(Path(resume_dir))
            details["resume_update_id"] = int(metadata["update_id"])
    except (ArtifactError, OSError, ValueError) as exc:
        failures.append(str(exc))
    return _verdict("G0", failures, details=details)


def evaluate_g1(
    contract_groups: Iterable[Any],
    *,
    expected_groups: Iterable[str] | None = None,
) -> GateVerdict:
    rows = _rows(contract_groups, "G1")
    failures: list[str] = []
    seen: set[str] = set()
    details: dict[str, Any] = {"groups": {}}

    for row in rows:
        group = str(row.get("group", row.get("contract_group", ""))).strip()
        if not group:
            failures.append("timing evidence contains an unnamed contract group")
            continue
        if group in seen:
            failures.append(f"duplicate timing contract group: {group}")
            continue
        seen.add(group)
        group_failures: list[str] = []
        if row.get("global_offset_passed") is not True:
            group_failures.append("global-offset test failed")
        if row.get("null_test_passed") is not True:
            group_failures.append("null test failed")

        consumers = row.get("consumer_indices")
        if not isinstance(consumers, Mapping) or len(consumers) < 2:
            group_failures.append("fewer than two timing consumers were recorded")
        else:
            normalized = {
                str(name): tuple(int(index) for index in indices)
                for name, indices in consumers.items()
            }
            if any(not indices for indices in normalized.values()):
                group_failures.append("a timing consumer resolved no indices")
            if len(set(normalized.values())) != 1:
                group_failures.append("timing consumers resolved different indices")
        failures.extend(f"{group}: {failure}" for failure in group_failures)
        details["groups"][group] = {
            "status": "PASS" if not group_failures else "FAIL",
            "failures": group_failures,
        }

    if expected_groups is not None:
        expected = {str(group) for group in expected_groups}
        missing = expected - seen
        extra = seen - expected
        if missing:
            failures.append(f"missing timing contract groups: {sorted(missing)}")
        if extra:
            failures.append(f"unexpected timing contract groups: {sorted(extra)}")
    return _verdict("G1", failures, details=details)


def _g2_deltas(row: Mapping[str, Any]) -> dict[str, float]:
    baseline_geometry = _number(row, "baseline_geometry")
    candidate_geometry = _number(row, "candidate_geometry")
    denominator = abs(baseline_geometry)
    if denominator <= 1e-12:
        geometry_delta = 0.0 if candidate_geometry == baseline_geometry else -math.inf
    else:
        geometry_delta = (candidate_geometry - baseline_geometry) / denominator
    if row.get("geometry_higher_is_better", True) is False:
        geometry_delta *= -1.0
    return {
        "PSNR": _number(row, "candidate_psnr") - _number(row, "baseline_psnr"),
        "SSIM": _number(row, "candidate_ssim") - _number(row, "baseline_ssim"),
        "LPIPS": _number(row, "baseline_lpips") - _number(row, "candidate_lpips"),
        "geometry": geometry_delta,
    }


def evaluate_g2(
    observations: Iterable[Any],
    *,
    expected_seeds: Iterable[int] | None = None,
    thresholds: G2Thresholds | None = None,
    bootstrap_samples: int = 2000,
) -> GateVerdict:
    rows = _rows(observations, "G2")
    thresholds = thresholds or G2Thresholds()
    failures: list[str] = []
    prepared: list[tuple[dict[str, Any], dict[str, float]]] = []
    seeds: set[int] = set()

    for row in rows:
        domain = _text(row, "domain")
        stratum = _text(row, "stratum")
        clip_id = _text(row, "clip_id")
        seed = _seed(row)
        seeds.add(seed)
        deltas = _g2_deltas(row)
        prepared.append((row, deltas))
        prefix = f"domain={domain}/stratum={stratum}/seed={seed}/clip={clip_id}"
        if row.get("motion_interval_passed") is not True:
            failures.append(f"{prefix}: frozen motion interval failed")
        if row.get("catastrophic_regression") is True:
            failures.append(f"{prefix}: preregistered catastrophic regression")
        catastrophic = {
            "PSNR": thresholds.catastrophic_psnr_delta,
            "SSIM": thresholds.catastrophic_ssim_delta,
            "LPIPS": thresholds.catastrophic_lpips_improvement,
            "geometry": thresholds.catastrophic_geometry_relative,
        }
        for metric, lower in catastrophic.items():
            if deltas[metric] < lower:
                failures.append(
                    f"{prefix}: catastrophic {metric} delta "
                    f"{deltas[metric]:.6g} < {lower:.6g}"
                )

    if expected_seeds is not None:
        expected = {int(seed) for seed in expected_seeds}
        if seeds != expected:
            failures.append(
                f"seed set mismatch: expected={sorted(expected)}, got={sorted(seeds)}"
            )

    groups: dict[tuple[str, str, int], list[dict[str, float]]] = defaultdict(list)
    for row, deltas in prepared:
        seed = _seed(row)
        groups[("domain", _text(row, "domain"), seed)].append(deltas)
        stratum = f"{_text(row, 'domain')}:{_text(row, 'stratum')}"
        groups[("stratum", stratum, seed)].append(deltas)

    limits = {
        "PSNR": thresholds.psnr_delta_lower,
        "SSIM": thresholds.ssim_delta_lower,
        "LPIPS": thresholds.lpips_improvement_lower,
        "geometry": thresholds.geometry_relative_lower,
    }
    group_details: dict[str, Any] = {}
    for (kind, name, seed), items in sorted(groups.items()):
        label = _group_label(kind, (name, f"seed={seed}"))
        metric_details: dict[str, Any] = {}
        for metric, lower_limit in limits.items():
            values = [item[metric] for item in items]
            lower, upper = _bootstrap_mean_interval(
                values,
                label=f"G2:{label}:{metric}",
                samples=bootstrap_samples,
            )
            sorted_values = sorted(values)
            tenth = _percentile(sorted_values, 0.10)
            worst = sorted_values[0]
            metric_details[metric] = {
                "mean": statistics.fmean(values),
                "ci95": [lower, upper],
                "p10": tenth,
                "worst": worst,
                "threshold": lower_limit,
            }
            if lower < lower_limit:
                failures.append(
                    f"{label}: {metric} 95% lower bound {lower:.6g} < {lower_limit:.6g}"
                )
            if tenth < lower_limit:
                failures.append(
                    f"{label}: {metric} 10th percentile {tenth:.6g} < {lower_limit:.6g}"
                )
        group_details[label] = metric_details

    bridge_rows: dict[int, list[float]] = defaultdict(list)
    for row, deltas in prepared:
        if _text(row, "domain").lower() == "bridge":
            bridge_rows[_seed(row)].append(deltas["PSNR"])
    if not bridge_rows:
        failures.append("Bridge evidence is missing")
    for seed, values in sorted(bridge_rows.items()):
        median = statistics.median(values)
        if median < thresholds.bridge_median_psnr_delta:
            failures.append(
                f"Bridge seed={seed}: median PSNR improvement {median:.6g} "
                f"< {thresholds.bridge_median_psnr_delta:.6g}"
            )

    return _verdict(
        "G2",
        failures,
        details={
            "seeds": sorted(seeds),
            "groups": group_details,
            "bridge_median_psnr_delta": {
                str(seed): statistics.median(values)
                for seed, values in sorted(bridge_rows.items())
            },
        },
    )


def evaluate_g3(
    observations: Iterable[Any],
    *,
    expected_seeds: Iterable[int] | None = None,
    thresholds: G3Thresholds | None = None,
    prediction_difference_floor: float | None = None,
    bootstrap_samples: int = 2000,
) -> GateVerdict:
    rows = _rows(observations, "G3")
    thresholds = thresholds or G3Thresholds()
    if prediction_difference_floor is not None:
        thresholds = G3Thresholds(
            separation_ci_lower=thresholds.separation_ci_lower,
            prediction_difference_floor=float(prediction_difference_floor),
            direction_agreement=thresholds.direction_agreement,
        )

    failures: list[str] = []
    seeds = {_seed(row) for row in rows}
    if expected_seeds is not None:
        expected = {int(seed) for seed in expected_seeds}
        if seeds != expected:
            failures.append(
                f"seed set mismatch: expected={sorted(expected)}, got={sorted(seeds)}"
            )

    groups: dict[tuple[str, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            _text(row, "domain"),
            _text(row, "stratum"),
            _seed(row),
            _normalize_mode(row.get("grip_class")),
        )
        groups[key].append(row)

    group_details: dict[str, Any] = {}
    for (domain, stratum, seed, grip_class), items in sorted(groups.items()):
        label = f"domain={domain}/stratum={stratum}/seed={seed}/grip={grip_class}"
        modes = {_normalize_mode(item.get("mode")) for item in items}
        missing_modes = REQUIRED_ACTION_MODES - modes
        if missing_modes:
            failures.append(f"{label}: missing action modes {sorted(missing_modes)}")
        if grip_class not in REQUIRED_GRIP_CLASSES:
            failures.append(f"{label}: unknown grip class")

        separations = [
            _number(item, "wrong_mse") - _number(item, "true_mse") for item in items
        ]
        differences = [_number(item, "prediction_difference") for item in items]
        directions = [_number(item, "direction_agreement") for item in items]
        separation_lower, separation_upper = _bootstrap_mean_interval(
            separations,
            label=f"G3:{label}:separation",
            samples=bootstrap_samples,
        )
        difference_lower, difference_upper = _bootstrap_mean_interval(
            differences,
            label=f"G3:{label}:prediction_difference",
            samples=bootstrap_samples,
        )
        direction = statistics.fmean(directions)
        if separation_lower <= thresholds.separation_ci_lower:
            failures.append(
                f"{label}: wrong_mse-true_mse 95% lower bound "
                f"{separation_lower:.6g} is not positive"
            )
        if difference_lower < thresholds.prediction_difference_floor:
            failures.append(
                f"{label}: prediction-difference 95% lower bound "
                f"{difference_lower:.6g} < "
                f"{thresholds.prediction_difference_floor:.6g}"
            )
        if direction < thresholds.direction_agreement:
            failures.append(
                f"{label}: direction agreement {direction:.6g} "
                f"< {thresholds.direction_agreement:.6g}"
            )
        group_details[label] = {
            "modes": sorted(modes),
            "separation_ci95": [separation_lower, separation_upper],
            "prediction_difference_ci95": [
                difference_lower,
                difference_upper,
            ],
            "direction_agreement": direction,
        }

    observed_grip_classes = {key[3] for key in groups}
    missing_grip_classes = REQUIRED_GRIP_CLASSES - observed_grip_classes
    if missing_grip_classes:
        failures.append(f"missing grip classes: {sorted(missing_grip_classes)}")
    return _verdict(
        "G3",
        failures,
        details={"seeds": sorted(seeds), "groups": group_details},
    )


def _contiguous(row: Mapping[str, Any]) -> bool:
    if "contiguous" in row:
        return row["contiguous"] is True
    starts = row.get("starts")
    if isinstance(starts, Sequence) and not isinstance(starts, (str, bytes)):
        normalized = [int(start) for start in starts]
        return len(normalized) >= 8 and all(
            right - left == 8 for left, right in zip(normalized, normalized[1:])
        )
    return False


def _future_frames(row: Mapping[str, Any]) -> int:
    if "future_frames" in row:
        return int(row["future_frames"])
    if "n_frames" in row:
        return max(0, int(row["n_frames"]) - 16)
    return 0


def _evaluate_g4_core(
    observations: Iterable[Any],
    *,
    required_domains: Iterable[str],
    expected_seeds: Iterable[int] | None = None,
    thresholds: G4Thresholds | None = None,
) -> GateVerdict:
    rows = [_as_row(value) for value in observations]
    thresholds = thresholds or G4Thresholds()
    domains = {str(domain) for domain in required_domains}
    if not domains:
        raise GateError("G4 requires configured domains")
    failures: list[str] = []
    unexpected = {_text(row, "domain") for row in rows} - domains
    if unexpected:
        failures.append(f"unexpected G4 domains: {sorted(unexpected)}")

    eligible: list[dict[str, Any]] = []
    domain_details: dict[str, Any] = {}
    for domain in sorted(domains):
        domain_rows = [row for row in rows if _text(row, "domain") == domain]
        eligible_rows = [
            row
            for row in domain_rows
            if int(row.get("chunks", 0)) >= thresholds.minimum_chunks
            and _future_frames(row) >= thresholds.minimum_future_frames
            and _contiguous(row)
        ]
        if not eligible_rows:
            domain_details[domain] = {
                "status": "NOT_APPLICABLE",
                "eligible_clips": 0,
            }
        else:
            domain_details[domain] = {
                "status": "APPLICABLE",
                "eligible_clips": len(eligible_rows),
            }
            eligible.extend(eligible_rows)

    if not eligible:
        failures.append("G4 has no eligible source-only closed-loop clips")
        return _verdict("G4", failures, details={"domains": domain_details})

    seeds = {_seed(row) for row in eligible}
    if expected_seeds is not None:
        expected = {int(seed) for seed in expected_seeds}
        if seeds != expected:
            failures.append(
                f"seed set mismatch: expected={sorted(expected)}, got={sorted(seeds)}"
            )

    hazards = (
        "black_frames",
        "dark_arm_collapse",
        "melting",
        "target_leakage",
        "action_reversal",
    )
    ratios: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    bridge_improvements: dict[int, list[float]] = defaultdict(list)
    clip_details: dict[str, Any] = {}
    for row in eligible:
        domain = _text(row, "domain")
        stratum = _text(row, "stratum")
        clip_id = _text(row, "clip_id")
        seed = _seed(row)
        prefix = f"domain={domain}/stratum={stratum}/seed={seed}/clip={clip_id}"
        seam = _number(row, "seam_error")
        within = _number(row, "within_chunk_error")
        seam_ratio = (
            0.0
            if seam == within == 0.0
            else (math.inf if within <= 0.0 else seam / within)
        )
        if seam_ratio > thresholds.seam_ratio:
            failures.append(
                f"{prefix}: seam ratio {seam_ratio:.6g} > {thresholds.seam_ratio:.6g}"
            )
        for hazard in hazards:
            if row.get(hazard) is True:
                failures.append(f"{prefix}: {hazard}")
        if row.get("source_ledger_passed", True) is not True:
            failures.append(f"{prefix}: source ledger failed")

        candidate_drift = _number(row, "candidate_drift_slope")
        baseline_drift = _number(row, "baseline_drift_slope")
        if baseline_drift <= 0.0:
            failures.append(f"{prefix}: baseline drift slope must be positive")
            drift_ratio = math.inf
        else:
            drift_ratio = candidate_drift / baseline_drift
        ratios[("domain", domain, seed)].append(drift_ratio)
        ratios[("stratum", f"{domain}:{stratum}", seed)].append(drift_ratio)
        if domain.lower() == "bridge":
            bridge_improvements[seed].append(1.0 - drift_ratio)
        clip_details[prefix] = {
            "seam_ratio": seam_ratio,
            "drift_ratio": drift_ratio,
            "chunks": int(row["chunks"]),
            "future_frames": _future_frames(row),
        }

    aggregate_details: dict[str, Any] = {}
    maximum_ratio = 1.0 + thresholds.maximum_regression
    for (kind, name, seed), values in sorted(ratios.items()):
        median_ratio = statistics.median(values)
        label = _group_label(kind, (name, f"seed={seed}"))
        aggregate_details[label] = {"median_drift_ratio": median_ratio}
        if median_ratio > maximum_ratio:
            failures.append(
                f"{label}: median drift regression "
                f"{median_ratio - 1.0:.2%} > "
                f"{thresholds.maximum_regression:.2%}"
            )

    if not bridge_improvements:
        failures.append("Bridge has no eligible G4 clips")
    for seed, values in sorted(bridge_improvements.items()):
        improvement = statistics.median(values)
        if improvement < thresholds.bridge_drift_improvement:
            failures.append(
                f"Bridge seed={seed}: median drift improvement "
                f"{improvement:.2%} < "
                f"{thresholds.bridge_drift_improvement:.2%}"
            )

    return _verdict(
        "G4",
        failures,
        details={
            "domains": domain_details,
            "seeds": sorted(seeds),
            "clips": clip_details,
            "aggregates": aggregate_details,
        },
    )


def _finite_values(report: Mapping[str, Any], key: str) -> list[float]:
    value = report.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise GateError(f"G5 {key} must be a sequence")
    result = [float(item) for item in value]
    if any(not math.isfinite(item) or item < 0.0 for item in result):
        raise GateError(f"G5 {key} contains an invalid duration")
    return result


def evaluate_g5(
    report: Mapping[str, Any],
    *,
    thresholds: G5Thresholds | None = None,
) -> GateVerdict:
    thresholds = thresholds or G5Thresholds()
    failures: list[str] = []
    details: dict[str, Any] = {}
    try:
        rank_ids = [int(rank) for rank in report.get("rank_ids", [])]
        expected_ranks = list(range(thresholds.world_size))
        if sorted(rank_ids) != expected_ranks or len(rank_ids) != len(set(rank_ids)):
            failures.append(
                f"rank set mismatch: expected={expected_ranks}, got={sorted(rank_ids)}"
            )

        endpoints = [str(endpoint) for endpoint in report.get("job_endpoints", [])]
        if not endpoints:
            failures.append("job topology has no endpoints")
        if any(thresholds.forbidden_node42_endpoint in item for item in endpoints):
            failures.append("node42 endpoint is present in the job topology")
        if any(thresholds.node43_endpoint not in item for item in endpoints):
            failures.append("job topology contains a non-node43 endpoint")

        memory = float(report["max_memory_fraction"])
        if memory > 1.0:
            memory /= 100.0
        if not math.isfinite(memory) or memory < 0.0:
            failures.append("max_memory_fraction is invalid")
        elif memory > thresholds.maximum_memory_fraction:
            failures.append(
                f"reserved memory {memory:.2%} > "
                f"{thresholds.maximum_memory_fraction:.2%}"
            )

        baseline = float(report["normal_smoke_step_time"])
        if not math.isfinite(baseline) or baseline <= 0.0:
            raise GateError("G5 normal_smoke_step_time must be positive")
        step_times = _finite_values(report, "step_times")
        rolling_times = _finite_values(report, "rolling_step_times")
        if not step_times:
            failures.append("no all-step timing samples")
        elif statistics.fmean(step_times) > thresholds.average_step_ratio * baseline:
            failures.append(
                f"average step ratio "
                f"{statistics.fmean(step_times) / baseline:.6g} > "
                f"{thresholds.average_step_ratio:.6g}"
            )
        if not rolling_times:
            failures.append("no rolling-step timing samples")
        elif max(rolling_times) > thresholds.rolling_step_ratio * baseline:
            failures.append(
                f"rolling step ratio {max(rolling_times) / baseline:.6g} > "
                f"{thresholds.rolling_step_ratio:.6g}"
            )

        normal_stalls = _finite_values(report, "normal_stalls")
        rolling_stalls = _finite_values(report, "rolling_stalls")
        if normal_stalls and max(normal_stalls) >= thresholds.stall_seconds:
            failures.append("normal step reached the 120-second stall limit")
        if rolling_stalls and max(rolling_stalls) >= thresholds.stall_seconds:
            failures.append("rolling step reached the 120-second stall limit")

        checkpoint_time = float(report["checkpoint_time"])
        checkpoint_sla = float(report["checkpoint_sla"])
        if (
            not math.isfinite(checkpoint_time)
            or not math.isfinite(checkpoint_sla)
            or checkpoint_time < 0.0
            or checkpoint_sla <= 0.0
        ):
            failures.append("checkpoint timing or SLA is invalid")
        elif checkpoint_time > checkpoint_sla:
            failures.append(
                f"checkpoint time {checkpoint_time:.6g}s > SLA {checkpoint_sla:.6g}s"
            )

        errors = report.get("errors")
        if not isinstance(errors, Sequence) or isinstance(errors, (str, bytes)):
            failures.append("compute/checkpoint errors must be a sequence")
        elif errors:
            failures.append(f"compute/checkpoint errors were recorded: {list(errors)}")

        details = {
            "rank_ids": rank_ids,
            "job_endpoints": endpoints,
            "max_memory_fraction": memory,
            "average_step_time": (statistics.fmean(step_times) if step_times else None),
            "maximum_rolling_step_time": max(rolling_times) if rolling_times else None,
            "checkpoint_time": checkpoint_time,
            "checkpoint_sla": checkpoint_sla,
        }
    except (KeyError, TypeError, ValueError, GateError) as exc:
        failures.append(str(exc))
    return _verdict("G5", failures, details=details)


def _normalize_verdict(value: GateVerdict | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, GateVerdict):
        return value.to_dict()
    if not isinstance(value, Mapping):
        raise GateError("gate verdict must be GateVerdict or a mapping")
    gate = str(value.get("gate", ""))
    status = str(value.get("status", ""))
    passed = value.get("passed")
    if status not in {"PASS", "FAIL"} or passed is not (status == "PASS"):
        raise GateError(f"inconsistent machine verdict for {gate or 'unknown gate'}")
    return {
        "gate": gate,
        "status": status,
        "passed": bool(passed),
        "failures": list(value.get("failures", [])),
        "details": dict(value.get("details", {})),
    }


def _legacy_write_gate_report(
    path: str | Path,
    *,
    candidate_identity: CandidateIdentity,
    evaluation_source_digest: str,
    clip_list_sha256: str,
    seeds: Iterable[int],
    raw_metrics: Mapping[str, Any],
    verdicts: Iterable[GateVerdict | Mapping[str, Any]],
) -> dict[str, Any]:
    normalized = [_normalize_verdict(verdict) for verdict in verdicts]
    names = [verdict["gate"] for verdict in normalized]
    if len(names) != len(set(names)):
        raise GateError("gate report contains duplicate gate verdicts")
    missing = set(REQUIRED_GATES) - set(names)
    extra = set(names) - set(REQUIRED_GATES)
    complete = not missing and not extra
    machine_pass = complete and all(verdict["passed"] for verdict in normalized)
    payload = {
        "schema_version": 1,
        "artifact_type": "wm3d_stage1_gate_report",
        "candidate_id": candidate_identity.candidate_id,
        "candidate_identity": candidate_identity.to_dict(),
        "evaluation_source_digest": _artifacts._require_sha256(
            evaluation_source_digest,
            "evaluation source digest",
        ),
        "clip_list_sha256": _artifacts._require_sha256(
            clip_list_sha256,
            "clip-list hash",
        ),
        "seeds": sorted({int(seed) for seed in seeds}),
        "raw_per_clip_metrics": dict(raw_metrics),
        "verdicts": sorted(normalized, key=lambda item: item["gate"]),
        "missing_gates": sorted(missing),
        "extra_gates": sorted(extra),
        "machine_verdict": "PASS" if machine_pass else "FAIL",
    }
    try:
        atomic_write_json(path, payload)
    except ArtifactError as exc:
        raise GateError(str(exc)) from exc
    return payload


def _read_json_file(path: str | Path, label: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(read_bytes_no_follow(path).decode("utf-8"))
    except (ArtifactError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GateError(f"{label} is not valid JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise GateError(f"{label} must be a JSON object")
    return payload


def _validate_passing_report(
    report: Mapping[str, Any],
    identity: CandidateIdentity,
) -> None:
    if report.get("artifact_type") != "wm3d_stage1_gate_report":
        raise GateError("promotion gate report has the wrong artifact type")
    if report.get("self_computed") is not True:
        raise GateError("promotion requires a self-computed gate report")
    if report.get("machine_verdict") != "PASS":
        raise GateError("promotion requires a PASS machine verdict")
    if report.get("candidate_id") != identity.candidate_id:
        raise GateError("gate report candidate_id does not match the candidate")
    identity_payload = report.get("candidate_identity")
    if not isinstance(identity_payload, Mapping):
        raise GateError("gate report has no candidate identity")
    try:
        report_identity = CandidateIdentity.from_dict(identity_payload)
    except ArtifactError as exc:
        raise GateError(str(exc)) from exc
    if report_identity != identity:
        raise GateError("gate report identity does not match the candidate")
    verdicts = report.get("verdicts")
    if not isinstance(verdicts, Sequence):
        raise GateError("gate report has no verdict list")
    normalized = [_normalize_verdict(verdict) for verdict in verdicts]
    if {verdict["gate"] for verdict in normalized} != set(REQUIRED_GATES):
        raise GateError("promotion requires exactly G0-G5 verdicts")
    if any(not verdict["passed"] for verdict in normalized):
        raise GateError("promotion gate report contains a failed verdict")
    if report.get("missing_gates") or report.get("extra_gates"):
        raise GateError("promotion requires a complete G0-G5 gate set")
    raw_metrics = report.get("raw_per_clip_metrics")
    if not isinstance(raw_metrics, Mapping) or not raw_metrics:
        raise GateError("promotion requires nonempty raw metrics")
    strict_load_report = report.get("strict_load_report")
    if not isinstance(strict_load_report, Mapping):
        raise GateError("promotion report is missing strict loader evidence")
    if set(strict_load_report.get("loaded_files", [])) != set(
        _artifacts.CANDIDATE_WEIGHT_FILES
    ):
        raise GateError("promotion strict loader evidence does not cover the exact three-file set")
    if (
        strict_load_report.get("missing_keys")
        or strict_load_report.get("unexpected_keys")
        or strict_load_report.get("shape_mismatches")
    ):
        raise GateError("promotion strict loader evidence is not fail-closed")
    evidence_hashes = report.get("evidence_sha256")
    if not isinstance(evidence_hashes, Mapping) or not evidence_hashes:
        raise GateError("promotion report is missing frozen evidence hashes")
    for required_key in (
        "frozen_manifest",
        "g1",
        "g2",
        "g3",
        "g4",
        "g5",
        "source_ledger",
    ):
        if required_key not in evidence_hashes:
            raise GateError(f"promotion report is missing {required_key} evidence binding")


def _copy_regular_file(source: Path, destination: Path) -> None:
    input_descriptor = _artifacts._open_regular_no_follow(source)
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        output_descriptor = os.open(destination, flags, 0o600)
    except Exception:
        os.close(input_descriptor)
        raise
    with os.fdopen(input_descriptor, "rb") as input_stream:
        with os.fdopen(output_descriptor, "wb") as output_stream:
            for block in iter(lambda: input_stream.read(1024 * 1024), b""):
                output_stream.write(block)
            output_stream.flush()
            os.fsync(output_stream.fileno())
    os.chmod(destination, 0o444)


def _validate_bundle_mapping(
    bundle_files: Mapping[str, str | Path],
    identity: CandidateIdentity,
) -> dict[str, Path]:
    if set(bundle_files) != set(CANONICAL_BUNDLE_FILES):
        missing = sorted(CANONICAL_BUNDLE_FILES - set(bundle_files))
        extra = sorted(set(bundle_files) - CANONICAL_BUNDLE_FILES)
        raise GateError(
            f"canonical bundle input mismatch: missing={missing}, extra={extra}"
        )
    normalized: dict[str, Path] = {}
    for relative, source in bundle_files.items():
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts:
            raise GateError(f"invalid canonical relative path: {relative}")
        try:
            normalized[relative] = require_regular_file_no_symlink(source)
        except ArtifactError as exc:
            raise GateError(str(exc)) from exc

    bindings = {
        "config/stage1.yaml": identity.config_sha256,
        "data/action_frame_contract.json": identity.contract_sha256,
        "data/unique_manifest.jsonl": identity.unique_manifest_sha256,
    }
    for relative, expected in bindings.items():
        actual = sha256_file(normalized[relative])
        if actual != expected:
            raise GateError(
                f"canonical input {relative} hash mismatch: "
                f"expected {expected}, got {actual}"
            )
    return normalized


def _strict_reload_promoted_weights(
    weights_dir: Path,
    identity: CandidateIdentity,
) -> None:
    actual_names = {entry.name for entry in os.scandir(weights_dir)}
    if actual_names != set(_artifacts.CANDIDATE_WEIGHT_FILES):
        raise GateError("promoted weights do not contain the strict three-file set")
    for filename in _artifacts.CANDIDATE_WEIGHT_FILES:
        path = require_regular_file_no_symlink(weights_dir / filename)
        if sha256_file(path) != identity.weight_sha256[filename]:
            raise GateError(f"promoted weight hash mismatch: {filename}")
        _artifacts._load_torch_state(path)


def _legacy_promote_candidate(
    candidate_dir: str | Path,
    destination: str | Path,
    *,
    gate_report: str | Path,
    bundle_files: Mapping[str, str | Path],
    expected_identity: CandidateIdentity | None = None,
) -> Path:
    candidate_path = Path(candidate_dir)
    try:
        reject_symlink_components(candidate_path)
        require_directory_no_symlink(candidate_path)
    except ArtifactError as exc:
        raise GateError(f"explicit candidate is invalid or a symlink: {exc}") from exc
    if _artifacts._STEP_PATTERN.fullmatch(candidate_path.name) is None:
        raise GateError(
            "promotion requires an explicit step_XXXXXXXX candidate directory"
        )
    if candidate_path.name in {"best.pt", "latest.pt"}:
        raise GateError("best.pt and latest.pt are invalid promotion inputs")

    try:
        loaded = load_candidate(
            candidate_path,
            expected_identity=expected_identity,
        )
    except ArtifactError as exc:
        raise GateError(str(exc)) from exc
    try:
        report_path = require_regular_file_no_symlink(gate_report)
    except ArtifactError as exc:
        raise GateError(str(exc)) from exc
    report = _read_json_file(report_path, "gate report")
    _validate_passing_report(report, loaded.identity)
    normalized_bundle = _validate_bundle_mapping(bundle_files, loaded.identity)

    target = Path(destination)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        reject_symlink_components(target.parent)
    except (ArtifactError, OSError) as exc:
        raise GateError(str(exc)) from exc
    if os.path.lexists(target):
        raise GateError(f"canonical destination already exists: {target}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp.", dir=target.parent))
    if temporary.stat().st_dev != target.parent.stat().st_dev:
        _artifacts._remove_tree(temporary)
        raise GateError("promotion temporary directory is on another filesystem")

    try:
        for filename in _artifacts.CANDIDATE_WEIGHT_FILES:
            _copy_regular_file(
                candidate_path / filename,
                temporary / "weights" / filename,
            )
        for relative, source in sorted(normalized_bundle.items()):
            _copy_regular_file(source, temporary / relative)
        _copy_regular_file(report_path, temporary / "eval/gate_report.json")
        _strict_reload_promoted_weights(temporary / "weights", loaded.identity)

        inventory: dict[str, dict[str, Any]] = {}
        for current, _, files in os.walk(temporary):
            for name in files:
                path = Path(current) / name
                relative = path.relative_to(temporary).as_posix()
                inventory[relative] = {
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
        manifest = {
            "schema_version": 1,
            "artifact_type": "wm3d_stage1_canonical_bundle",
            "candidate_id": loaded.identity.candidate_id,
            "candidate_identity": loaded.identity.to_dict(),
            "files": dict(sorted(inventory.items())),
            "gate_report_sha256": inventory["eval/gate_report.json"]["sha256"],
        }
        _artifacts._write_bytes_fsync(
            temporary / "manifest.json",
            canonical_json_bytes(manifest),
            mode=0o444,
        )
        fsync_tree(temporary)

        # This marker is deliberately the final file created in the temporary tree.
        _artifacts._write_bytes_fsync(
            temporary / "PROMOTED",
            f"{loaded.identity.candidate_id}\n".encode("ascii"),
            mode=0o444,
        )
        fsync_tree(temporary)
        for current, directories, _ in os.walk(temporary, topdown=False):
            for name in directories:
                os.chmod(Path(current) / name, 0o555)
        os.chmod(temporary, 0o555)
        rename_no_replace(temporary, target)
        fsync_directory(target.parent)
        return target
    except (ArtifactError, GateError, OSError, ValueError) as exc:
        _artifacts._remove_tree(temporary)
        if isinstance(exc, GateError):
            raise
        raise GateError(str(exc)) from exc


@dataclass(frozen=True)
class GateEvidencePaths:
    frozen_manifest: str | Path
    g1: str | Path
    g2: str | Path
    g3: str | Path
    g4: str | Path
    g5: str | Path
    source_ledger: str | Path | None = None


def _clip_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (_text(row, "domain"), _text(row, "stratum"), _text(row, "clip_id"))


def _report_digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(dict(payload))).hexdigest()


def _payload_candidate_identity(
    payload: Mapping[str, Any],
    *,
    label: str,
) -> CandidateIdentity:
    identity_payload = payload.get("candidate_identity")
    if not isinstance(identity_payload, Mapping):
        raise GateError(f"{label} has no candidate identity")
    try:
        identity = CandidateIdentity.from_dict(identity_payload)
    except ArtifactError as exc:
        raise GateError(f"{label} has an invalid candidate identity: {exc}") from exc
    if payload.get("candidate_id", identity.candidate_id) != identity.candidate_id:
        raise GateError(f"{label} candidate_id does not match candidate identity")
    return identity


def _require_evidence_candidate_identity(
    payload: Mapping[str, Any],
    *,
    label: str,
    expected_identity: CandidateIdentity | None,
) -> CandidateIdentity:
    identity = _payload_candidate_identity(payload, label=label)
    if expected_identity is not None and identity != expected_identity:
        raise GateError(f"{label} candidate identity does not match the candidate")
    return identity


def _candidate_identity_from_manifest(candidate_dir: str | Path) -> CandidateIdentity:
    directory = require_directory_no_symlink(candidate_dir)
    manifest_path = require_regular_file_no_symlink(directory / "candidate.json")
    payload = _read_json_file(manifest_path, "candidate manifest")
    if payload.get("artifact_type") != "wm3d_stage1_candidate":
        raise GateError("candidate manifest has the wrong artifact type")
    return _payload_candidate_identity(payload, label="candidate manifest")


def _load_frozen_manifest(paths: GateEvidencePaths) -> dict[str, Any]:
    frozen_path = require_regular_file_no_symlink(paths.frozen_manifest)
    payload = _read_json_file(frozen_path, "frozen manifest")
    if payload.get("artifact_type") != "wm3d_stage1_frozen_eval_manifest":
        raise GateError("frozen manifest has the wrong artifact type")
    seeds_raw = payload.get("seeds")
    if not isinstance(seeds_raw, Sequence) or isinstance(seeds_raw, (str, bytes)):
        raise GateError("frozen manifest has no seed list")
    seeds = tuple(sorted({int(seed) for seed in seeds_raw}))
    if not seeds:
        raise GateError("frozen manifest seed list is empty")
    groups_raw = payload.get("contract_groups")
    if not isinstance(groups_raw, Sequence) or isinstance(groups_raw, (str, bytes)):
        raise GateError("frozen manifest has no contract group list")
    contract_groups = tuple(sorted({str(group) for group in groups_raw if str(group).strip()}))
    if not contract_groups:
        raise GateError("frozen manifest contract group list is empty")
    clips_raw = payload.get("clips")
    if not isinstance(clips_raw, Sequence) or isinstance(clips_raw, (str, bytes)):
        raise GateError("frozen manifest has no clip list")
    clip_rows = [_as_row(row) for row in clips_raw]
    if not clip_rows:
        raise GateError("frozen manifest clip list is empty")
    clip_records = []
    for row in clip_rows:
        clip_records.append(
            {
                "domain": _text(row, "domain"),
                "stratum": _text(row, "stratum"),
                "clip_id": _text(row, "clip_id"),
                "g4_eligible": row.get("g4_eligible") is True,
            }
        )
    return {
        "path": frozen_path,
        "sha256": sha256_file(frozen_path),
        "seeds": seeds,
        "contract_groups": contract_groups,
        "clips": tuple(clip_records),
        "clip_keys": frozenset(
            (item["domain"], item["stratum"], item["clip_id"]) for item in clip_records
        ),
        "eligible_clip_keys": frozenset(
            (item["domain"], item["stratum"], item["clip_id"])
            for item in clip_records
            if item["g4_eligible"]
        ),
        "required_domains": frozenset(item["domain"] for item in clip_records),
    }


def _load_rows_evidence(
    path: str | Path,
    *,
    label: str,
    artifact_type: str,
    frozen_manifest_sha256: str,
    expected_identity: CandidateIdentity | None,
) -> tuple[list[dict[str, Any]], str]:
    evidence_path = require_regular_file_no_symlink(path)
    payload = _read_json_file(evidence_path, label)
    if payload.get("artifact_type") != artifact_type:
        raise GateError(f"{label} has the wrong artifact type")
    if payload.get("frozen_manifest_sha256") != frozen_manifest_sha256:
        raise GateError(f"{label} is not bound to the frozen manifest")
    _require_evidence_candidate_identity(
        payload,
        label=label,
        expected_identity=expected_identity,
    )
    rows = payload.get("rows")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise GateError(f"{label} has no row list")
    return [_as_row(row) for row in rows], sha256_file(evidence_path)


def _load_report_evidence(
    path: str | Path,
    *,
    label: str,
    artifact_type: str,
    frozen_manifest_sha256: str,
    expected_identity: CandidateIdentity | None,
) -> tuple[dict[str, Any], str]:
    evidence_path = require_regular_file_no_symlink(path)
    payload = _read_json_file(evidence_path, label)
    if payload.get("artifact_type") != artifact_type:
        raise GateError(f"{label} has the wrong artifact type")
    if payload.get("frozen_manifest_sha256") != frozen_manifest_sha256:
        raise GateError(f"{label} is not bound to the frozen manifest")
    _require_evidence_candidate_identity(
        payload,
        label=label,
        expected_identity=expected_identity,
    )
    report = payload.get("report")
    if not isinstance(report, Mapping):
        raise GateError(f"{label} has no report object")
    return dict(report), sha256_file(evidence_path)


def _require_exact_clip_set(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected: frozenset[tuple[str, str, str]],
    label: str,
) -> None:
    observed = frozenset(_clip_key(row) for row in rows)
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    if missing or unexpected:
        raise GateError(
            f"{label} must cover the complete frozen clip set: missing={missing}, unexpected={unexpected}"
        )


def _ledger_row_passes(row: Mapping[str, Any]) -> bool:
    if int(row.get("target_reads_after_chunk0", -1)) != 0:
        return False
    if row.get("source_enum_schema", G4_SOURCE_ENUM_SCHEMA) != G4_SOURCE_ENUM_SCHEMA:
        return False
    if int(row.get("source_schema_version", G4_SOURCE_SCHEMA_VERSION)) != G4_SOURCE_SCHEMA_VERSION:
        return False
    chunks = row.get("chunks")
    if not isinstance(chunks, Sequence) or isinstance(chunks, (str, bytes)) or not chunks:
        return False
    for expected_chunk, item in enumerate(chunks):
        chunk = _as_row(item)
        if int(chunk.get("chunk", -1)) != expected_chunk:
            return False
        token_source = str(chunk.get("token_source", "")).strip()
        rgb_source = str(chunk.get("rgb_source", "")).strip()
        expected_token = g4_expected_token_source(expected_chunk)
        expected_rgb = g4_expected_rgb_source(expected_chunk)
        if token_source != expected_token or rgb_source != expected_rgb:
            return False
        if str(chunk.get("token_context_source", "")).strip() != expected_token:
            return False
        if str(chunk.get("rgb_context_source", "")).strip() != expected_rgb:
            return False
        parent_chunk = None if expected_chunk == 0 else expected_chunk - 1
        if chunk.get("token_context_from_chunk") != parent_chunk:
            return False
        if chunk.get("rgb_context_from_chunk") != parent_chunk:
            return False
    return True


def _source_ledger_status(
    source_ledger: Sequence[Mapping[str, Any]],
    *,
    expected_eligible_clips: frozenset[tuple[str, str, str]],
) -> dict[tuple[str, str, str], bool]:
    rows = [_as_row(row) for row in source_ledger]
    _require_exact_clip_set(
        rows,
        expected=expected_eligible_clips,
        label="source ledger",
    )
    return {_clip_key(row): _ledger_row_passes(row) for row in rows}


def evaluate_g0(
    candidate_dir: str | Path,
    *,
    expected_identity: CandidateIdentity | None = None,
    modules: Mapping[str, Any] | None = None,
    strict_loader: Any = None,
    bound_artifacts: Mapping[str, str | Path] | None = None,
    resume_dir: str | Path | None = None,
) -> GateVerdict:
    failures: list[str] = []
    if modules is None and strict_loader is None:
        failures.append("strict loader or target modules are required")
    if bound_artifacts is None:
        failures.append("bound artifacts are required")
    verdict = _evaluate_g0_core(
        candidate_dir,
        expected_identity=expected_identity,
        modules=modules,
        strict_loader=strict_loader,
        bound_artifacts=bound_artifacts,
        resume_dir=resume_dir,
    )
    return _verdict("G0", [*failures, *verdict.failures], details=verdict.details)


def evaluate_g4(
    observations: Iterable[Any],
    *,
    required_domains: Iterable[str],
    expected_seeds: Iterable[int] | None = None,
    thresholds: G4Thresholds | None = None,
    source_ledger: Sequence[Mapping[str, Any]] | None = None,
    expected_eligible_clips: frozenset[tuple[str, str, str]] | None = None,
) -> GateVerdict:
    rows = [_as_row(value) for value in observations]
    failures: list[str] = []
    if expected_eligible_clips is not None:
        observed = frozenset(_clip_key(row) for row in rows)
        missing = sorted(expected_eligible_clips - observed)
        unexpected = sorted(observed - expected_eligible_clips)
        if missing or unexpected:
            failures.append(
                f"G4 rows must cover the complete eligible frozen set: missing={missing}, unexpected={unexpected}"
            )
    if source_ledger is None:
        failures.append("source ledger evidence is required")
        ledger_status: dict[tuple[str, str, str], bool] = {}
    else:
        expected = expected_eligible_clips or frozenset(_clip_key(row) for row in rows)
        ledger_status = _source_ledger_status(
            source_ledger,
            expected_eligible_clips=expected,
        )
    annotated = []
    for row in rows:
        bound = dict(row)
        bound["source_ledger_passed"] = ledger_status.get(_clip_key(row), False)
        annotated.append(bound)
    verdict = _evaluate_g4_core(
        annotated,
        required_domains=required_domains,
        expected_seeds=expected_seeds,
        thresholds=thresholds,
    )
    return _verdict("G4", [*failures, *verdict.failures], details=verdict.details)


def _compute_self_computed_report(
    *,
    candidate_dir: str | Path,
    expected_identity: CandidateIdentity | None,
    strict_loader: Any,
    bound_artifacts: Mapping[str, str | Path],
    evidence: GateEvidencePaths,
    modules: Mapping[str, Any] | None = None,
    resume_dir: str | Path | None = None,
) -> dict[str, Any]:
    candidate_identity = expected_identity or _candidate_identity_from_manifest(
        candidate_dir
    )
    frozen = _load_frozen_manifest(evidence)
    g1_rows, g1_sha = _load_rows_evidence(
        evidence.g1,
        label="G1 evidence",
        artifact_type="wm3d_stage1_g1_rows",
        frozen_manifest_sha256=frozen["sha256"],
        expected_identity=candidate_identity,
    )
    g2_rows, g2_sha = _load_rows_evidence(
        evidence.g2,
        label="G2 evidence",
        artifact_type="wm3d_stage1_g2_rows",
        frozen_manifest_sha256=frozen["sha256"],
        expected_identity=candidate_identity,
    )
    g3_rows, g3_sha = _load_rows_evidence(
        evidence.g3,
        label="G3 evidence",
        artifact_type="wm3d_stage1_g3_rows",
        frozen_manifest_sha256=frozen["sha256"],
        expected_identity=candidate_identity,
    )
    g4_rows, g4_sha = _load_rows_evidence(
        evidence.g4,
        label="G4 evidence",
        artifact_type="wm3d_stage1_g4_rows",
        frozen_manifest_sha256=frozen["sha256"],
        expected_identity=candidate_identity,
    )
    g5_report, g5_sha = _load_report_evidence(
        evidence.g5,
        label="G5 evidence",
        artifact_type="wm3d_stage1_g5_report",
        frozen_manifest_sha256=frozen["sha256"],
        expected_identity=candidate_identity,
    )
    if evidence.source_ledger is None:
        raise GateError("source ledger evidence is required")
    source_ledger_rows, source_ledger_sha = _load_rows_evidence(
        evidence.source_ledger,
        label="source ledger",
        artifact_type="wm3d_stage1_source_ledger",
        frozen_manifest_sha256=frozen["sha256"],
        expected_identity=candidate_identity,
    )
    g4_payload = _read_json_file(evidence.g4, "G4 evidence")
    if g4_payload.get("source_ledger_sha256") != source_ledger_sha:
        raise GateError("G4 evidence source ledger binding does not match")
    _require_exact_clip_set(g2_rows, expected=frozen["clip_keys"], label="G2 rows")
    _require_exact_clip_set(g3_rows, expected=frozen["clip_keys"], label="G3 rows")
    _require_exact_clip_set(
        g4_rows,
        expected=frozen["eligible_clip_keys"],
        label="G4 rows",
    )
    verdicts = [
        evaluate_g0(
            candidate_dir,
            expected_identity=expected_identity,
            modules=modules,
            strict_loader=strict_loader,
            bound_artifacts=bound_artifacts,
            resume_dir=resume_dir,
        ),
        evaluate_g1(g1_rows, expected_groups=frozen["contract_groups"]),
        evaluate_g2(g2_rows, expected_seeds=frozen["seeds"]),
        evaluate_g3(g3_rows, expected_seeds=frozen["seeds"]),
        evaluate_g4(
            g4_rows,
            required_domains=frozen["required_domains"],
            expected_seeds=frozen["seeds"],
            source_ledger=source_ledger_rows,
            expected_eligible_clips=frozen["eligible_clip_keys"],
        ),
        evaluate_g5(g5_report),
    ]
    normalized = [_normalize_verdict(verdict) for verdict in verdicts]
    names = [verdict["gate"] for verdict in normalized]
    missing = sorted(set(REQUIRED_GATES) - set(names))
    extra = sorted(set(names) - set(REQUIRED_GATES))
    machine_pass = not missing and not extra and all(verdict["passed"] for verdict in normalized)
    bound_hashes = {
        key: sha256_file(path)
        for key, path in sorted(bound_artifacts.items())
    }
    evidence_hashes = {
        "frozen_manifest": frozen["sha256"],
        "g1": g1_sha,
        "g2": g2_sha,
        "g3": g3_sha,
        "g4": g4_sha,
        "g5": g5_sha,
        "source_ledger": source_ledger_sha,
    }
    clip_list_sha256 = _report_digest({"clips": list(frozen["clips"])})
    evaluation_source_digest = _report_digest(
        {
            "bound_artifacts": bound_hashes,
            "evidence_sha256": evidence_hashes,
        }
    )
    strict_load_report = verdicts[0].details.get("strict_load_report")
    payload = {
        "schema_version": 1,
        "artifact_type": "wm3d_stage1_gate_report",
        "self_computed": True,
        "candidate_id": verdicts[0].details.get("candidate_id"),
        "candidate_identity": verdicts[0].details.get("identity"),
        "evaluation_source_digest": evaluation_source_digest,
        "clip_list_sha256": clip_list_sha256,
        "seeds": list(frozen["seeds"]),
        "raw_per_clip_metrics": {
            "clip_count": len(frozen["clip_keys"]),
            "g2_row_count": len(g2_rows),
            "g3_row_count": len(g3_rows),
            "g4_row_count": len(g4_rows),
            "eligible_g4_clip_count": len(frozen["eligible_clip_keys"]),
        },
        "verdicts": sorted(normalized, key=lambda item: item["gate"]),
        "missing_gates": missing,
        "extra_gates": extra,
        "machine_verdict": "PASS" if machine_pass else "FAIL",
        "strict_load_report": strict_load_report,
        "bound_artifact_sha256": bound_hashes,
        "evidence_sha256": evidence_hashes,
        "frozen_manifest_sha256": frozen["sha256"],
        "source_ledger_sha256": source_ledger_sha,
    }
    return payload


def write_gate_report(
    path: str | Path,
    *,
    candidate_identity: CandidateIdentity | None = None,
    evaluation_source_digest: str | None = None,
    clip_list_sha256: str | None = None,
    seeds: Iterable[int] | None = None,
    raw_metrics: Mapping[str, Any] | None = None,
    verdicts: Iterable[GateVerdict | Mapping[str, Any]] | None = None,
    candidate_dir: str | Path | None = None,
    expected_identity: CandidateIdentity | None = None,
    strict_loader: Any = None,
    bound_artifacts: Mapping[str, str | Path] | None = None,
    evidence: GateEvidencePaths | None = None,
    modules: Mapping[str, Any] | None = None,
    resume_dir: str | Path | None = None,
) -> dict[str, Any]:
    if candidate_dir is not None or evidence is not None or strict_loader is not None:
        if candidate_dir is None or evidence is None or strict_loader is None or bound_artifacts is None:
            raise GateError(
                "self-computed gate report requires candidate_dir, strict_loader, bound_artifacts, and evidence"
            )
        payload = _compute_self_computed_report(
            candidate_dir=candidate_dir,
            expected_identity=expected_identity,
            strict_loader=strict_loader,
            bound_artifacts=bound_artifacts,
            evidence=evidence,
            modules=modules,
            resume_dir=resume_dir,
        )
        try:
            atomic_write_json(path, payload)
        except ArtifactError as exc:
            raise GateError(str(exc)) from exc
        return payload
    if (
        candidate_identity is None
        or evaluation_source_digest is None
        or clip_list_sha256 is None
        or seeds is None
        or raw_metrics is None
        or verdicts is None
    ):
        raise GateError("manual gate report inputs are incomplete")
    return _legacy_write_gate_report(
        path,
        candidate_identity=candidate_identity,
        evaluation_source_digest=evaluation_source_digest,
        clip_list_sha256=clip_list_sha256,
        seeds=seeds,
        raw_metrics=raw_metrics,
        verdicts=verdicts,
    )


def promote_candidate(
    candidate_dir: str | Path,
    destination: str | Path,
    *,
    gate_report: str | Path,
    bundle_files: Mapping[str, str | Path],
    expected_identity: CandidateIdentity | None = None,
    strict_loader: Any = None,
    bound_artifacts: Mapping[str, str | Path] | None = None,
    evidence: GateEvidencePaths | None = None,
    modules: Mapping[str, Any] | None = None,
    resume_dir: str | Path | None = None,
) -> Path:
    candidate_path = Path(candidate_dir)
    try:
        reject_symlink_components(candidate_path)
        require_directory_no_symlink(candidate_path)
    except ArtifactError as exc:
        raise GateError(f"explicit candidate is invalid or a symlink: {exc}") from exc
    if _artifacts._STEP_PATTERN.fullmatch(candidate_path.name) is None:
        raise GateError(
            "promotion requires an explicit step_XXXXXXXX candidate directory"
        )
    if candidate_path.name in {"best.pt", "latest.pt"}:
        raise GateError("best.pt and latest.pt are invalid promotion inputs")
    report_path = require_regular_file_no_symlink(gate_report)
    report = _read_json_file(report_path, "gate report")
    loaded = load_candidate(
        candidate_path,
        expected_identity=expected_identity,
        modules=modules,
        strict_loader=strict_loader,
    )
    _validate_passing_report(report, loaded.identity)
    if evidence is None or strict_loader is None or bound_artifacts is None:
        raise GateError(
            "promotion requires strict_loader, bound_artifacts, and evidence"
        )
    recomputed = _compute_self_computed_report(
        candidate_dir=candidate_path,
        expected_identity=expected_identity,
        strict_loader=strict_loader,
        bound_artifacts=bound_artifacts,
        evidence=evidence,
        modules=modules,
        resume_dir=resume_dir,
    )
    if canonical_json_bytes(report) != canonical_json_bytes(recomputed):
        raise GateError("promotion gate report does not match recomputed evidence")
    return _legacy_promote_candidate(
        candidate_path,
        destination,
        gate_report=report_path,
        bundle_files=bundle_files,
        expected_identity=expected_identity,
    )
