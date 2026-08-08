from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np


ACTION_GATE_SCHEMA = "wm3d_v6_stage1_action_gate_v2"
CANDIDATE_OFFSETS = (-2, -1, 0, 1, 2)
NORMALIZATION_SCALE_FLOOR = 0.05
NORMALIZATION_WINSOR_LIMITS = (-8.0, 8.0)
FORMAL_MIN_RESAMPLES = 10_000
_TEST_MIN_RESAMPLES = 100
_FAMILY_SOURCE = {
    "state": "proprioceptive",
    "gripper": "proprioceptive",
    "flow": "exteroceptive",
    "geometry": "exteroceptive",
}
_ALLOWED_FAMILIES = frozenset(
    {"state", "flow", "geometry", "gripper", "first_motion"}
)
FORMAL_OXE_COHORT_ID = "legacy_oxe5_fps5_delta_xyz_rpy_gripper_v1"
FORMAL_OXE_COHORT_METHOD = "pooled_oxe_strict_state_flow_v1"
FORMAL_OXE_METHOD = "independent_qualification_confirmation_v2"
FORMAL_OXE_CONTRACT_KEYS = (
    "bridge|5|delta_xyz+rpy+gripper",
    "fractal20220817_data|5|delta_xyz+rpy+gripper",
    "jaco_play|5|delta_xyz+rpy+gripper",
    "kuka|5|delta_xyz+rpy+gripper",
    "taco_play|5|delta_xyz+rpy+gripper",
)
FORMAL_DROID_METHOD = "droid_exact_interval_n_minus_one_v1"
FORMAL_DROID_CONTRACT_KEY = (
    "droid|5|cartesian_target_interval_delta+rpy+gripper"
)
_DIAGNOSTIC_FAMILIES = frozenset({"gripper", "first_motion"})


class ActionContractEvidenceError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.context = dict(context or {})


@dataclass(frozen=True)
class ModalityEvidence:
    family: str
    observed: float
    null_samples: tuple[float, ...]
    direction: int
    scale_epsilon: float = 1e-6
    source_class: str = ""
    informative: bool = True
    robot_mask_motion_coverage: float | None = None
    valid_depth_coverage: float | None = None
    support_count: int | None = None
    support_fraction: float | None = None
    minimum_support_count: int | None = None
    fallback_used: bool = False


@dataclass(frozen=True)
class ClipOffsetEvidence:
    clip_id: str
    offset: int
    modalities: Mapping[str, ModalityEvidence]


@dataclass(frozen=True)
class OffsetSelectionReport:
    selected_offset: int
    clip_count: int
    family_names: tuple[str, ...]
    mean_score_by_offset: Mapping[int, float]
    effect_by_challenger: Mapping[int, float]
    raw_p_by_challenger: Mapping[int, float]
    holm_p_by_challenger: Mapping[int, float]


def normalize_modality_score(
    *,
    observed: float,
    null_samples: Sequence[float],
    direction: int,
    scale_epsilon: float = 1e-6,
) -> float:
    null = np.asarray(tuple(null_samples), dtype=np.float64)
    if null.size < 2 or not np.isfinite(null).all():
        raise ActionContractEvidenceError(
            "each modality requires at least two finite null samples"
        )
    if direction not in (-1, 1):
        raise ActionContractEvidenceError("modality direction must be -1 or 1")
    observed = float(observed)
    scale_epsilon = float(scale_epsilon)
    if not np.isfinite(observed) or not (scale_epsilon > 0.0):
        raise ActionContractEvidenceError(
            "observed score and scale_epsilon must be finite and positive"
        )
    scale = max(
        float(null.std(ddof=0)), scale_epsilon, NORMALIZATION_SCALE_FLOOR
    )
    normalized = float(direction) * (observed - float(null.mean())) / scale
    return float(np.clip(normalized, *NORMALIZATION_WINSOR_LIMITS))


def _family_scores(row: ClipOffsetEvidence) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for modality_name, modality in sorted(row.modalities.items()):
        family = str(modality.family).strip()
        if not family:
            raise ActionContractEvidenceError(
                f"empty independence family for modality {modality_name}"
            )
        grouped.setdefault(family, []).append(
            normalize_modality_score(
                observed=modality.observed,
                null_samples=modality.null_samples,
                direction=modality.direction,
                scale_epsilon=modality.scale_epsilon,
            )
        )
    return {
        family: float(np.mean(values))
        for family, values in sorted(grouped.items())
    }


def _unique_best(scores: Mapping[int, float], label: str) -> int:
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    if len(ranked) != 5:
        raise ActionContractEvidenceError(
            f"{label} must contain exactly five candidate offsets"
        )
    if np.isclose(ranked[0][1], ranked[1][1], rtol=0.0, atol=1e-12):
        raise ActionContractEvidenceError(f"{label} has no unique best offset")
    return int(ranked[0][0])


def _holm_adjust(raw_p: Mapping[int, float]) -> dict[int, float]:
    ordered = sorted(raw_p.items(), key=lambda item: (item[1], item[0]))
    count = len(ordered)
    adjusted: dict[int, float] = {}
    running = 0.0
    for rank, (key, value) in enumerate(ordered):
        candidate = min(1.0, float(value) * (count - rank))
        running = max(running, candidate)
        adjusted[int(key)] = running
    return adjusted


def select_offset_contract(
    rows: Sequence[ClipOffsetEvidence],
    *,
    bootstrap_resamples: int = 1000,
    seed: int = 0,
    min_clips: int = 32,
    min_effect: float = 0.05,
    alpha: float = 0.01,
) -> OffsetSelectionReport:
    if int(bootstrap_resamples) < 1000:
        raise ActionContractEvidenceError(
            "bootstrap_resamples must be at least 1000"
        )
    by_clip: dict[str, dict[int, dict[str, float]]] = {}
    for row in rows:
        offset = int(row.offset)
        if offset not in range(-2, 3):
            raise ActionContractEvidenceError(f"invalid candidate offset {offset}")
        clip_rows = by_clip.setdefault(str(row.clip_id), {})
        if offset in clip_rows:
            raise ActionContractEvidenceError(
                f"duplicate evidence for clip={row.clip_id} offset={offset}"
            )
        clip_rows[offset] = _family_scores(row)

    if len(by_clip) < int(min_clips):
        raise ActionContractEvidenceError(
            f"need at least {min_clips} clips, got {len(by_clip)}"
        )

    expected_offsets = set(range(-2, 3))
    family_names: tuple[str, ...] | None = None
    for clip_id, offset_rows in sorted(by_clip.items()):
        if set(offset_rows) != expected_offsets:
            raise ActionContractEvidenceError(
                f"clip {clip_id} does not contain all five offsets"
            )
        for family_scores in offset_rows.values():
            current = tuple(sorted(family_scores))
            if family_names is None:
                family_names = current
            elif current != family_names:
                raise ActionContractEvidenceError(
                    f"inconsistent independence families for clip {clip_id}"
                )
    if family_names is None or len(family_names) < 2:
        raise ActionContractEvidenceError(
            "at least two independence families are required"
        )

    clip_ids = tuple(sorted(by_clip))
    family_best: dict[str, int] = {}
    for family in family_names:
        family_means = {
            offset: float(
                np.mean(
                    [by_clip[clip_id][offset][family] for clip_id in clip_ids]
                )
            )
            for offset in range(-2, 3)
        }
        family_best[family] = _unique_best(
            family_means, f"family {family}"
        )
    if len(set(family_best.values())) != 1:
        raise ActionContractEvidenceError(
            f"independence families disagree on best offset: {family_best}"
        )

    clip_scores = {
        offset: np.asarray(
            [
                np.mean(
                    [
                        by_clip[clip_id][offset][family]
                        for family in family_names
                    ]
                )
                for clip_id in clip_ids
            ],
            dtype=np.float64,
        )
        for offset in range(-2, 3)
    }
    means = {
        offset: float(values.mean())
        for offset, values in clip_scores.items()
    }
    selected = _unique_best(means, "aggregate evidence")
    if selected != next(iter(family_best.values())):
        raise ActionContractEvidenceError(
            "aggregate and family best offsets disagree"
        )

    rng = np.random.default_rng(int(seed))
    effect_by_challenger: dict[int, float] = {}
    raw_p_by_challenger: dict[int, float] = {}
    for challenger in range(-2, 3):
        if challenger == selected:
            continue
        difference = clip_scores[selected] - clip_scores[challenger]
        effect = float(difference.mean())
        effect_by_challenger[challenger] = effect
        if effect < float(min_effect):
            raise ActionContractEvidenceError(
                f"normalized effect below {min_effect}: "
                f"selected={selected} challenger={challenger} effect={effect}"
            )
        sample_indices = rng.integers(
            0,
            difference.size,
            size=(int(bootstrap_resamples), difference.size),
        )
        bootstrap_means = difference[sample_indices].mean(axis=1)
        raw_p_by_challenger[challenger] = float(
            (np.count_nonzero(bootstrap_means <= 0.0) + 1)
            / (int(bootstrap_resamples) + 1)
        )

    holm = _holm_adjust(raw_p_by_challenger)
    if max(holm.values()) >= float(alpha):
        raise ActionContractEvidenceError(
            f"Holm-corrected bootstrap p-value is not below {alpha}: {holm}"
        )
    return OffsetSelectionReport(
        selected_offset=selected,
        clip_count=len(clip_ids),
        family_names=family_names,
        mean_score_by_offset=means,
        effect_by_challenger=effect_by_challenger,
        raw_p_by_challenger=raw_p_by_challenger,
        holm_p_by_challenger=holm,
    )


@dataclass(frozen=True)
class ActionGateConfig:
    permutation_resamples: int = FORMAL_MIN_RESAMPLES
    bootstrap_resamples: int = FORMAL_MIN_RESAMPLES
    sign_flip_resamples: int = FORMAL_MIN_RESAMPLES
    qualification_alpha: float = 0.01
    confirmation_alpha: float = 0.01
    min_dz: float = 0.30
    min_bootstrap_win_frequency: float = 0.80
    target_length: int = 16
    null_repeats: int = 256
    block_size: int = 2
    seed: int = 0
    test_mode: bool = False

    def __post_init__(self) -> None:
        minimum = _TEST_MIN_RESAMPLES if self.test_mode else FORMAL_MIN_RESAMPLES
        for name in (
            "permutation_resamples",
            "bootstrap_resamples",
            "sign_flip_resamples",
        ):
            if int(getattr(self, name)) < minimum:
                mode = "test" if self.test_mode else "formal"
                raise ActionContractEvidenceError(
                    f"{mode} {name} must be at least {minimum}"
                )
        if not (0.0 < float(self.qualification_alpha) <= 0.01):
            raise ActionContractEvidenceError(
                "qualification alpha must be in (0, 0.01]"
            )
        if not (0.0 < float(self.confirmation_alpha) <= 0.01):
            raise ActionContractEvidenceError(
                "confirmation alpha must be in (0, 0.01]"
            )
        if float(self.min_dz) < 0.30:
            raise ActionContractEvidenceError(
                "minimum paired dz must be at least 0.30"
            )
        if not (0.80 <= float(self.min_bootstrap_win_frequency) <= 1.0):
            raise ActionContractEvidenceError(
                "bootstrap win frequency threshold must be in [0.80, 1]"
            )

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        test_mode: bool = False,
    ) -> ActionGateConfig:
        if payload.get("schema_version") != ACTION_GATE_SCHEMA:
            raise ActionContractEvidenceError(
                f"unexpected action gate schema: {payload.get('schema_version')!r}"
            )
        if payload.get("immutable") is not True:
            raise ActionContractEvidenceError(
                "action gate config must declare immutable=true"
            )
        if tuple(payload.get("candidate_offsets", ())) != CANDIDATE_OFFSETS:
            raise ActionContractEvidenceError(
                f"candidate offsets must be {list(CANDIDATE_OFFSETS)}"
            )
        split = _gate_mapping(payload, "clip_split")
        expected_split = {
            "method": "frozen_artifact",
            "derivation": "sha256(seed|contract_key|independence_group_id)",
            "calibration": 32,
            "qualification": 32,
            "confirmation": 32,
            "evidence_total": 64,
        }
        if dict(split) != expected_split:
            raise ActionContractEvidenceError(
                "clip split must bind the frozen SHA256 seed/contract/clip "
                "artifact with calibration32/qualification32/confirmation32"
            )
        normalization = _gate_mapping(payload, "normalization")
        expected_normalization = {
            "scale_floor": 0.05,
            "winsorize": [-8.0, 8.0],
            "flat_null_and_observed": "uninformative",
        }
        if dict(normalization) != expected_normalization:
            raise ActionContractEvidenceError(
                "normalization must use floor 0.05, winsor [-8, 8], and "
                "flat-null/flat-observed exclusion"
            )
        families = _gate_mapping(payload, "families")
        expected_families = {
            "required": ["state"],
            "visual_preference": ["flow", "geometry"],
            "diagnostic_only": ["gripper", "first_motion"],
            "required_source_classes": [
                "proprioceptive",
                "exteroceptive",
            ],
            "flow_geometry_agreement": True,
            "qualification_min_informative_clips": 24,
        }
        if dict(families) != expected_families:
            raise ActionContractEvidenceError(
                "family policy must require state, prefer flow with geometry "
                "fallback, and keep gripper/first_motion diagnostic-only"
            )
        evidence = _gate_mapping(payload, "evidence")
        expected_evidence = {
            "target_length": 16,
            "null_repeats": 256,
            "block_size": 2,
        }
        if dict(evidence) != expected_evidence:
            raise ActionContractEvidenceError(
                "evidence settings must be target_length=16, "
                "null_repeats=256, block_size=2"
            )
        qualification = _gate_mapping(payload, "qualification")
        confirmation = _gate_mapping(payload, "confirmation")
        if set(qualification) != {
            "method",
            "permutation_resamples",
            "alpha",
            "min_dz",
            "bootstrap_resamples",
            "min_bootstrap_win_frequency",
        }:
            raise ActionContractEvidenceError(
                "qualification config has unregistered fields"
            )
        if set(confirmation) != {
            "method",
            "sign_flip_resamples",
            "holm_alpha",
            "min_dz",
            "reselect",
        }:
            raise ActionContractEvidenceError(
                "confirmation config has unregistered fields"
            )
        if qualification["method"] != "within_clip_offset_label_max_t":
            raise ActionContractEvidenceError(
                "qualification must use offset-label maxT permutation"
            )
        if confirmation["method"] != "paired_sign_flip_holm":
            raise ActionContractEvidenceError(
                "confirmation must use paired sign-flip Holm tests"
            )
        if confirmation["reselect"] is not False:
            raise ActionContractEvidenceError(
                "confirmation must preregister reselect=false"
            )
        if float(qualification["min_dz"]) != float(confirmation["min_dz"]):
            raise ActionContractEvidenceError(
                "qualification and confirmation dz thresholds must match"
            )
        return cls(
            permutation_resamples=int(qualification["permutation_resamples"]),
            bootstrap_resamples=int(qualification["bootstrap_resamples"]),
            sign_flip_resamples=int(confirmation["sign_flip_resamples"]),
            qualification_alpha=float(qualification["alpha"]),
            confirmation_alpha=float(confirmation["holm_alpha"]),
            min_dz=float(qualification["min_dz"]),
            min_bootstrap_win_frequency=float(
                qualification["min_bootstrap_win_frequency"]
            ),
            target_length=int(evidence["target_length"]),
            null_repeats=int(evidence["null_repeats"]),
            block_size=int(evidence["block_size"]),
            seed=int(payload.get("seed", 0)),
            test_mode=bool(test_mode),
        )


@dataclass(frozen=True)
class QualificationReport:
    selected_offset: int
    clip_ids: tuple[str, ...]
    aggregate_clip_count: int
    informative_clip_count_by_family: Mapping[str, int]
    mean_score_by_offset: Mapping[int, float]
    dz_by_challenger: Mapping[int, float]
    max_t_p_by_challenger: Mapping[int, float]
    bootstrap_win_frequency: float
    family_best_by_family: Mapping[str, int]
    family_dz_by_challenger: Mapping[str, Mapping[int, float]]
    family_max_t_p_by_challenger: Mapping[str, Mapping[int, float]]
    family_bootstrap_win_frequency: Mapping[str, float]
    visual_best_by_family: Mapping[str, int]


@dataclass(frozen=True)
class ConfirmationFamilyReport:
    clip_ids: tuple[str, ...]
    clip_count: int
    dz_by_challenger: Mapping[int, float]
    raw_p_by_challenger: Mapping[int, float]
    holm_p_by_challenger: Mapping[int, float]


@dataclass(frozen=True)
class ConfirmationReport:
    tested_offset: int
    clip_ids: tuple[str, ...]
    by_family: Mapping[str, ConfirmationFamilyReport]


@dataclass(frozen=True)
class ActionContractV2Report:
    selected_offset: int
    clip_count: int
    required_families: tuple[str, str]
    required_source_classes: tuple[str, str]
    eligible_families: tuple[str, ...]
    binding_families: tuple[str, ...]
    diagnostic_families: tuple[str, ...]
    split_artifact_sha256: str | None
    split_partition_sha256: str
    qualification: QualificationReport
    confirmation: ConfirmationReport
    flow_geometry_agree: bool | None


@dataclass(frozen=True)
class ExpectedOffsetFamilyFalsificationReport:
    best_offset: int
    clip_ids: tuple[str, ...]
    clip_count: int
    mean_score_by_offset: Mapping[int, float]
    challenger_over_expected_dz: Mapping[int, float]
    raw_p_by_challenger: Mapping[int, float]
    holm_p_by_challenger: Mapping[int, float]
    conflicting_challengers: tuple[int, ...]


@dataclass(frozen=True)
class ExpectedOffsetPartitionFalsificationReport:
    expected_offset: int
    clip_ids: tuple[str, ...]
    by_family: Mapping[str, ExpectedOffsetFamilyFalsificationReport]


@dataclass(frozen=True)
class CohortSubsetFalsificationReport:
    selected_offset: int
    member_contract_keys: tuple[str, ...]
    frozen_qualification_clip_count: int
    frozen_confirmation_clip_count: int
    eligible_families: tuple[str, ...]
    binding_families: tuple[str, ...]
    qualification: ExpectedOffsetPartitionFalsificationReport
    confirmation: ExpectedOffsetPartitionFalsificationReport


@dataclass(frozen=True)
class PooledOXEActionContractReport:
    selected_offset: int
    clip_count: int
    frozen_qualification_clip_count: int
    frozen_confirmation_clip_count: int
    required_families: tuple[str, str]
    required_source_classes: tuple[str, str]
    binding_families: tuple[str, ...]
    eligible_families: tuple[str, ...]
    diagnostic_families: tuple[str, ...]
    split_artifact_sha256: str
    split_partition_sha256: str
    member_split_partition_sha256: Mapping[str, str]
    member_falsification: Mapping[str, CohortSubsetFalsificationReport]
    qualification: ExpectedOffsetPartitionFalsificationReport
    confirmation: ExpectedOffsetPartitionFalsificationReport
    geometry_policy: str


@dataclass(frozen=True)
class ExactDerivedDroidReport:
    selected_offset: int
    clip_count: int
    required_families: tuple[str, str]
    required_source_classes: tuple[str, str]
    split_artifact_sha256: str
    split_partition_sha256: str
    qualification: ExpectedOffsetPartitionFalsificationReport
    confirmation: ExpectedOffsetPartitionFalsificationReport
    separation_basis: str
    separation_kind: str = "non_statistical_exact_construction"
    statistical_separation_claimed: bool = False


def _gate_mapping(
    payload: Mapping[str, Any],
    key: str,
) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ActionContractEvidenceError(f"action gate {key} must be a mapping")
    return value


def deterministic_clip_split(
    clip_ids: Sequence[str],
    *,
    seed: int = 0,
    contract_key: str = "test",
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    normalized = tuple(str(clip_id) for clip_id in clip_ids)
    if len(normalized) != 64 or len(set(normalized)) != 64:
        raise ActionContractEvidenceError(
            "action contract requires exactly 64 unique clips"
        )
    if any(not clip_id for clip_id in normalized):
        raise ActionContractEvidenceError("clip IDs must be nonempty")
    ranked = sorted(
        normalized,
        key=lambda clip_id: (
            hashlib.sha256(
                f"{int(seed)}|{contract_key}|{clip_id}".encode("utf-8")
            ).digest(),
            clip_id,
        ),
    )
    return tuple(ranked[:32]), tuple(ranked[32:])


def _resolve_split(
    clip_ids: Sequence[str],
    gate: ActionGateConfig,
    *,
    contract_key: str,
    qualification_clip_ids: Sequence[str] | None,
    confirmation_clip_ids: Sequence[str] | None,
    split_artifact_sha256: str | None,
    expected_partition_count: int = 32,
) -> tuple[tuple[str, ...], tuple[str, ...], str]:
    if qualification_clip_ids is None and confirmation_clip_ids is None:
        if not gate.test_mode:
            raise ActionContractEvidenceError(
                "formal v2 evaluation requires explicit qualification and "
                "confirmation clip IDs from the frozen split artifact"
            )
        qualification, confirmation = deterministic_clip_split(
            clip_ids,
            seed=gate.seed,
            contract_key=contract_key or "test",
        )
    elif qualification_clip_ids is None or confirmation_clip_ids is None:
        raise ActionContractEvidenceError(
            "qualification and confirmation clip IDs must be supplied together"
        )
    else:
        qualification = tuple(str(value) for value in qualification_clip_ids)
        confirmation = tuple(str(value) for value in confirmation_clip_ids)
    if (
        len(qualification) != expected_partition_count
        or len(confirmation) != expected_partition_count
        or len(set(qualification)) != expected_partition_count
        or len(set(confirmation)) != expected_partition_count
        or set(qualification).intersection(confirmation)
    ):
        raise ActionContractEvidenceError(
            "frozen split must contain disjoint "
            f"qualification{expected_partition_count}/"
            f"confirmation{expected_partition_count}"
        )
    if set(qualification).union(confirmation) != set(clip_ids):
        raise ActionContractEvidenceError(
            "frozen split IDs do not exactly match the "
            f"{expected_partition_count * 2} evidence clips"
        )
    if not gate.test_mode:
        digest = str(split_artifact_sha256 or "")
        if (
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ActionContractEvidenceError(
                "formal v2 evaluation requires the frozen split artifact SHA256"
            )
    identity = hashlib.sha256()
    identity.update(str(contract_key).encode("utf-8"))
    for label, values in (
        ("qualification", qualification),
        ("confirmation", confirmation),
    ):
        identity.update(b"\0")
        identity.update(label.encode("ascii"))
        for value in values:
            identity.update(b"\0")
            identity.update(value.encode("utf-8"))
    return qualification, confirmation, identity.hexdigest()


def _registered_source(modality: ModalityEvidence) -> str:
    family = str(modality.family).strip()
    source = str(modality.source_class).strip()
    expected = _FAMILY_SOURCE.get(family)
    if not source and expected:
        source = expected
    if source not in {"proprioceptive", "exteroceptive"}:
        raise ActionContractEvidenceError(
            f"modality family {family!r} requires a registered source class"
        )
    if expected and source != expected:
        raise ActionContractEvidenceError(
            f"family {family} must use source class {expected}, got {source}"
        )
    return source


def _finite_unit_interval(value: Any, label: str) -> float:
    numeric = float(value)
    if not np.isfinite(numeric) or not (0.0 <= numeric <= 1.0):
        raise ActionContractEvidenceError(
            f"{label} must be finite in [0, 1], got {value!r}"
        )
    return numeric


def _visual_support_audit(
    modality: ModalityEvidence,
) -> dict[str, float | int | bool] | None:
    required = (
        'robot_mask_motion_coverage',
        'valid_depth_coverage',
        'support_count',
        'support_fraction',
        'minimum_support_count',
    )
    present = any(getattr(modality, field) is not None for field in required)
    if not present and modality.fallback_used is False:
        return None
    missing = [field for field in required if getattr(modality, field) is None]
    if missing:
        raise ActionContractEvidenceError(
            'visual modality support audit is incomplete: '
            f'family={modality.family} missing={missing}'
        )
    support_count = int(modality.support_count)
    minimum_support_count = int(modality.minimum_support_count)
    if support_count < 0 or minimum_support_count < 0:
        raise ActionContractEvidenceError(
            'visual modality support counts must be nonnegative: '
            f'family={modality.family} support_count={support_count} '
            f'minimum_support_count={minimum_support_count}'
        )
    if not isinstance(modality.fallback_used, bool):
        raise ActionContractEvidenceError(
            'visual modality fallback_used flag must be boolean'
        )
    return {
        'robot_mask_motion_coverage': _finite_unit_interval(
            modality.robot_mask_motion_coverage,
            f'{modality.family} robot_mask_motion_coverage',
        ),
        'valid_depth_coverage': _finite_unit_interval(
            modality.valid_depth_coverage,
            f'{modality.family} valid_depth_coverage',
        ),
        'support_count': support_count,
        'support_fraction': _finite_unit_interval(
            modality.support_fraction,
            f'{modality.family} support_fraction',
        ),
        'minimum_support_count': minimum_support_count,
        'fallback_used': modality.fallback_used,
    }


def _locally_informative(modality: ModalityEvidence) -> bool:
    if not isinstance(modality.informative, bool):
        raise ActionContractEvidenceError(
            'modality informative flag must be boolean'
        )
    if not modality.informative:
        return False
    null = np.asarray(modality.null_samples, dtype=np.float64)
    if null.size < 2 or not np.isfinite(null).all():
        normalize_modality_score(
            observed=modality.observed,
            null_samples=modality.null_samples,
            direction=modality.direction,
            scale_epsilon=modality.scale_epsilon,
        )
    if modality.family in {'flow', 'geometry'}:
        audit = _visual_support_audit(modality)
        if audit is not None:
            if audit['fallback_used']:
                return False
            if audit['support_count'] < audit['minimum_support_count']:
                return False
            if (
                audit['support_fraction'] <= 0.0
                or audit['robot_mask_motion_coverage'] <= 0.0
            ):
                return False
            if (
                modality.family == 'geometry'
                and audit['valid_depth_coverage'] <= 0.0
            ):
                return False
    return not (
        float(np.ptp(null)) <= 1e-12
        and math.isclose(
            float(modality.observed),
            float(null.mean()),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    )


def _flat_modalities(
    rows_by_offset: Mapping[int, ClipOffsetEvidence],
) -> frozenset[str]:
    names = set.intersection(
        *(set(row.modalities) for row in rows_by_offset.values())
    )
    flat: set[str] = set()
    for name in names:
        modalities = [
            rows_by_offset[offset].modalities[name]
            for offset in CANDIDATE_OFFSETS
        ]
        observed = np.asarray(
            [float(modality.observed) for modality in modalities]
        )
        if (
            all(
                float(
                    np.ptp(
                        np.asarray(modality.null_samples, dtype=np.float64)
                    )
                )
                <= 1e-12
                for modality in modalities
            )
            and float(np.ptp(observed)) <= 1e-12
        ):
            flat.add(name)
    return frozenset(flat)


def _prepare_v2_scores(
    rows: Sequence[ClipOffsetEvidence],
    *,
    expected_clip_count: int = 64,
) -> tuple[
    dict[str, dict[int, dict[str, float]]],
    dict[str, str],
    tuple[str, ...],
    tuple[str, ...],
]:
    raw: dict[str, dict[int, ClipOffsetEvidence]] = {}
    for row in rows:
        clip_id = str(row.clip_id)
        offset = int(row.offset)
        if offset not in CANDIDATE_OFFSETS:
            raise ActionContractEvidenceError(f"invalid candidate offset {offset}")
        clip_rows = raw.setdefault(clip_id, {})
        if offset in clip_rows:
            raise ActionContractEvidenceError(
                f"duplicate evidence for clip={clip_id} offset={offset}"
            )
        clip_rows[offset] = row
    if len(raw) != expected_clip_count:
        raise ActionContractEvidenceError(
            "action contract requires exactly "
            f"{expected_clip_count} unique clips"
        )

    expected_offsets = set(CANDIDATE_OFFSETS)
    modality_names: frozenset[str] | None = None
    for clip_id, clip_rows in sorted(raw.items()):
        if set(clip_rows) != expected_offsets:
            raise ActionContractEvidenceError(
                f"clip {clip_id} does not contain all five offsets"
            )
        for offset, row in clip_rows.items():
            names = frozenset(row.modalities)
            if not names:
                raise ActionContractEvidenceError(
                    f"clip={clip_id} offset={offset} has no modalities"
                )
            if modality_names is None:
                modality_names = names
            elif names != modality_names:
                raise ActionContractEvidenceError(
                    "all evidence rows must contain the same registered "
                    f"modalities; mismatch at clip={clip_id} offset={offset}"
                )

    source_sets: dict[str, set[str]] = {}
    seen: set[str] = set()
    scored: dict[str, dict[int, dict[str, float]]] = {}
    for clip_id, clip_rows in sorted(raw.items()):
        flat = _flat_modalities(clip_rows)
        scored[clip_id] = {}
        for offset in CANDIDATE_OFFSETS:
            grouped: dict[str, list[float]] = {}
            for name, modality in sorted(clip_rows[offset].modalities.items()):
                family = str(modality.family).strip()
                if family not in _ALLOWED_FAMILIES:
                    raise ActionContractEvidenceError(
                        f"unregistered modality family {family!r} for {name}"
                    )
                seen.add(family)
                source_sets.setdefault(family, set()).add(
                    _registered_source(modality)
                )
                if name in flat or not _locally_informative(modality):
                    continue
                grouped.setdefault(family, []).append(
                    normalize_modality_score(
                        observed=modality.observed,
                        null_samples=modality.null_samples,
                        direction=modality.direction,
                        scale_epsilon=modality.scale_epsilon,
                    )
                )
            scored[clip_id][offset] = {
                family: float(np.mean(values))
                for family, values in sorted(grouped.items())
            }
    source_by_family: dict[str, str] = {}
    for family, sources in sorted(source_sets.items()):
        if len(sources) != 1:
            raise ActionContractEvidenceError(
                f"family {family} spans multiple source classes: {sorted(sources)}"
            )
        source_by_family[family] = next(iter(sources))
    families = tuple(sorted(seen))
    diagnostic = tuple(
        family for family in families if family in _DIAGNOSTIC_FAMILIES
    )
    return scored, source_by_family, families, diagnostic


def _usable_clip_ids(
    scores: Mapping[str, Mapping[int, Mapping[str, float]]],
    clip_ids: Sequence[str],
    family: str,
) -> tuple[str, ...]:
    return tuple(
        clip_id
        for clip_id in clip_ids
        if all(
            family in scores[clip_id][offset]
            for offset in CANDIDATE_OFFSETS
        )
    )


def _qualification_eligible_families(
    scores: Mapping[str, Mapping[int, Mapping[str, float]]],
    qualification_clip_ids: Sequence[str],
    families: Sequence[str],
    *,
    min_informative: int = 24,
) -> tuple[str, ...]:
    return tuple(
        family
        for family in families
        if len(_usable_clip_ids(scores, qualification_clip_ids, family))
        >= min_informative
    )


def _v2_matrix(
    scores: Mapping[str, Mapping[int, Mapping[str, float]]],
    clip_ids: Sequence[str],
    family: str,
) -> np.ndarray:
    return np.asarray(
        [
            [scores[clip_id][offset][family] for offset in CANDIDATE_OFFSETS]
            for clip_id in clip_ids
        ],
        dtype=np.float64,
    )


def _v2_unique_best(values: np.ndarray, label: str) -> int:
    ranked = np.argsort(-values, kind="stable")
    if math.isclose(
        float(values[int(ranked[0])]),
        float(values[int(ranked[1])]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ActionContractEvidenceError(f"{label} has no unique best offset")
    return CANDIDATE_OFFSETS[int(ranked[0])]


def _paired_dz(difference: np.ndarray) -> float:
    return float(difference.mean()) / max(
        float(difference.std(ddof=1)),
        1e-12,
    )


def _max_t_p_values(
    matrix: np.ndarray,
    selected_index: int,
    *,
    resamples: int,
    rng: np.random.Generator,
) -> dict[int, float]:
    clip_count, candidate_count = matrix.shape
    observed = {
        challenger: math.sqrt(clip_count)
        * _paired_dz(matrix[:, selected_index] - matrix[:, index])
        for index, challenger in enumerate(CANDIDATE_OFFSETS)
        if index != selected_index
    }
    counts = {challenger: 0 for challenger in observed}
    remaining = int(resamples)
    while remaining:
        batch = min(remaining, 512)
        permutations = np.argsort(
            rng.random((batch, clip_count, candidate_count)),
            axis=2,
        )
        source = np.broadcast_to(
            matrix,
            (batch, clip_count, candidate_count),
        )
        permuted = np.take_along_axis(source, permutations, axis=2)
        max_t = np.zeros(batch)
        for left in range(candidate_count):
            for right in range(left + 1, candidate_count):
                difference = permuted[:, :, left] - permuted[:, :, right]
                statistic = (
                    np.abs(difference.mean(axis=1))
                    * math.sqrt(clip_count)
                    / np.maximum(difference.std(axis=1, ddof=1), 1e-12)
                )
                max_t = np.maximum(max_t, statistic)
        for challenger, statistic in observed.items():
            counts[challenger] += int(np.count_nonzero(max_t >= statistic))
        remaining -= batch
    return {
        challenger: float((count + 1) / (int(resamples) + 1))
        for challenger, count in counts.items()
    }


def _bootstrap_win_frequency_v2(
    matrix: np.ndarray,
    selected_index: int,
    *,
    resamples: int,
    rng: np.random.Generator,
) -> float:
    wins = 0
    remaining = int(resamples)
    challengers = [
        index for index in range(matrix.shape[1]) if index != selected_index
    ]
    while remaining:
        batch = min(remaining, 1024)
        sampled = rng.integers(
            0,
            matrix.shape[0],
            size=(batch, matrix.shape[0]),
        )
        means = matrix[sampled].mean(axis=1)
        wins += int(
            np.count_nonzero(
                means[:, selected_index]
                > means[:, challengers].max(axis=1) + 1e-12
            )
        )
        remaining -= batch
    return float(wins / int(resamples))


def _qualification_family_gate(
    scores: Mapping[str, Mapping[int, Mapping[str, float]]],
    clip_ids: tuple[str, ...],
    family: str,
    selected: int,
    gate: ActionGateConfig,
    *,
    seed_offset: int,
) -> tuple[int, dict[int, float], dict[int, float], float]:
    matrix = _v2_matrix(scores, clip_ids, family)
    best = _v2_unique_best(
        matrix.mean(axis=0),
        f"qualification required family {family}",
    )
    if best != selected:
        raise ActionContractEvidenceError(
            "qualification required families disagree: "
            f"aggregate={selected} family={family} best={best}"
        )
    selected_index = CANDIDATE_OFFSETS.index(selected)
    effects: dict[int, float] = {}
    for index, challenger in enumerate(CANDIDATE_OFFSETS):
        if index == selected_index:
            continue
        effect = _paired_dz(matrix[:, selected_index] - matrix[:, index])
        effects[challenger] = effect
        if effect < gate.min_dz:
            raise ActionContractEvidenceError(
                "qualification required family paired dz below threshold: "
                f"family={family} selected={selected} "
                f"challenger={challenger} dz={effect:.6g}"
            )
    max_t_p = _max_t_p_values(
        matrix,
        selected_index,
        resamples=gate.permutation_resamples,
        rng=np.random.default_rng(gate.seed + seed_offset),
    )
    if max(max_t_p.values()) >= gate.qualification_alpha:
        raise ActionContractEvidenceError(
            "qualification required family maxT p-value is not below "
            f"{gate.qualification_alpha}: family={family} p={max_t_p}"
        )
    win_frequency = _bootstrap_win_frequency_v2(
        matrix,
        selected_index,
        resamples=gate.bootstrap_resamples,
        rng=np.random.default_rng(gate.seed + seed_offset + 1),
    )
    if win_frequency < gate.min_bootstrap_win_frequency:
        raise ActionContractEvidenceError(
            "qualification required family bootstrap win frequency below "
            f"threshold: family={family} {win_frequency:.6g} < "
            f"{gate.min_bootstrap_win_frequency}"
        )
    return best, effects, max_t_p, win_frequency

def _qualification_v2(
    scores: Mapping[str, Mapping[int, Mapping[str, float]]],
    clip_ids: tuple[str, ...],
    required: tuple[str, str],
    families: tuple[str, ...],
    eligible: tuple[str, ...],
    gate: ActionGateConfig,
    *,
    binding_families: Sequence[str] | None = None,
) -> QualificationReport:
    binding = tuple(required if binding_families is None else binding_families)
    usable_by_family = {
        family: _usable_clip_ids(scores, clip_ids, family)
        for family in families
    }
    aggregate_ids = tuple(
        clip_id
        for clip_id in clip_ids
        if any(
            clip_id in set(usable_by_family[family])
            for family in required
        )
    )
    aggregate = np.asarray(
        [
            [
                float(
                    np.mean(
                        [
                            scores[clip_id][offset][family]
                            for family in required
                            if clip_id in set(usable_by_family[family])
                        ]
                    )
                )
                for offset in CANDIDATE_OFFSETS
            ]
            for clip_id in aggregate_ids
        ],
        dtype=np.float64,
    )
    means = aggregate.mean(axis=0)

    visual_best: dict[str, int] = {}
    for family in ("flow", "geometry"):
        if family in eligible and family in binding:
            family_ids = usable_by_family[family]
            visual_best[family] = _v2_unique_best(
                _v2_matrix(scores, family_ids, family).mean(axis=0),
                f"qualification {family}",
            )
    if {"flow", "geometry"}.issubset(visual_best) and (
        visual_best["flow"] != visual_best["geometry"]
    ):
        raise ActionContractEvidenceError(
            "eligible flow and geometry disagree in qualification: "
            f"{visual_best}",
            code="OXE_GEOMETRY_CONFLICT",
            context={"visual_best_by_family": visual_best},
        )
    selected = _v2_unique_best(means, "qualification aggregate")
    selected_index = CANDIDATE_OFFSETS.index(selected)

    if any(value != selected for value in visual_best.values()):
        raise ActionContractEvidenceError(
            "qualification aggregate and eligible visual family disagree: "
            f"selected={selected} visual={visual_best}",
            code="OXE_DIRECTION_REVERSAL",
            context={
                "selected_offset": selected,
                "visual_best_by_family": visual_best,
            },
        )

    family_best: dict[str, int] = {}
    family_effects: dict[str, dict[int, float]] = {}
    family_max_t: dict[str, dict[int, float]] = {}
    family_bootstrap: dict[str, float] = {}
    for family_index, family in enumerate(binding):
        (
            family_best[family],
            family_effects[family],
            family_max_t[family],
            family_bootstrap[family],
        ) = _qualification_family_gate(
            scores,
            usable_by_family[family],
            family,
            selected,
            gate,
            seed_offset=100 + family_index * 10,
        )

    effects: dict[int, float] = {}
    for index, challenger in enumerate(CANDIDATE_OFFSETS):
        if index == selected_index:
            continue
        effect = _paired_dz(aggregate[:, selected_index] - aggregate[:, index])
        effects[challenger] = effect
        if effect < gate.min_dz:
            raise ActionContractEvidenceError(
                "qualification paired dz below threshold: "
                f"selected={selected} challenger={challenger} dz={effect:.6g}"
            )
    max_t_p = _max_t_p_values(
        aggregate,
        selected_index,
        resamples=gate.permutation_resamples,
        rng=np.random.default_rng(gate.seed),
    )
    if max(max_t_p.values()) >= gate.qualification_alpha:
        raise ActionContractEvidenceError(
            "qualification maxT p-value is not below "
            f"{gate.qualification_alpha}: {max_t_p}"
        )
    win_frequency = _bootstrap_win_frequency_v2(
        aggregate,
        selected_index,
        resamples=gate.bootstrap_resamples,
        rng=np.random.default_rng(gate.seed + 1),
    )
    if win_frequency < gate.min_bootstrap_win_frequency:
        raise ActionContractEvidenceError(
            "qualification bootstrap win frequency below threshold: "
            f"{win_frequency:.6g} < {gate.min_bootstrap_win_frequency}"
        )
    return QualificationReport(
        selected_offset=selected,
        clip_ids=clip_ids,
        aggregate_clip_count=len(aggregate_ids),
        informative_clip_count_by_family={
            family: len(values)
            for family, values in sorted(usable_by_family.items())
        },
        mean_score_by_offset={
            offset: float(means[index])
            for index, offset in enumerate(CANDIDATE_OFFSETS)
        },
        dz_by_challenger=effects,
        max_t_p_by_challenger=max_t_p,
        bootstrap_win_frequency=win_frequency,
        family_best_by_family=family_best,
        family_dz_by_challenger=family_effects,
        family_max_t_p_by_challenger=family_max_t,
        family_bootstrap_win_frequency=family_bootstrap,
        visual_best_by_family=visual_best,
    )


def _sign_flip_p_value(
    difference: np.ndarray,
    *,
    resamples: int,
    rng: np.random.Generator,
) -> float:
    observed = float(difference.mean())
    count = 0
    remaining = int(resamples)
    while remaining:
        batch = min(remaining, 2048)
        signs = rng.integers(
            0,
            2,
            size=(batch, difference.size),
            dtype=np.int8,
        )
        signs = signs * 2 - 1
        count += int(
            np.count_nonzero((signs * difference[None, :]).mean(axis=1) >= observed)
        )
        remaining -= batch
    return float((count + 1) / (int(resamples) + 1))


def _confirmation_v2(
    scores: Mapping[str, Mapping[int, Mapping[str, float]]],
    clip_ids: tuple[str, ...],
    selected: int,
    required: Sequence[str],
    gate: ActionGateConfig,
    *,
    min_informative: int = 24,
    partition_clip_count: int = 32,
) -> ConfirmationReport:
    selected_index = CANDIDATE_OFFSETS.index(selected)
    results: dict[str, ConfirmationFamilyReport] = {}
    for family_index, family in enumerate(required):
        family_clip_ids = _usable_clip_ids(scores, clip_ids, family)
        if len(family_clip_ids) < min_informative:
            raise ActionContractEvidenceError(
                "confirmation fixed family gate requires at least "
                f"{min_informative} informative clips: family={family} "
                f"({min_informative}/{partition_clip_count} gate), "
                f"got {len(family_clip_ids)}"
            )
        matrix = _v2_matrix(scores, family_clip_ids, family)
        effects: dict[int, float] = {}
        raw_p: dict[int, float] = {}
        rng = np.random.default_rng(gate.seed + 10 + family_index)
        for index, challenger in enumerate(CANDIDATE_OFFSETS):
            if index == selected_index:
                continue
            difference = matrix[:, selected_index] - matrix[:, index]
            effect = _paired_dz(difference)
            effects[challenger] = effect
            if effect < gate.min_dz:
                raise ActionContractEvidenceError(
                    "confirmation fixed offset failed paired dz: "
                    f"family={family} selected={selected} "
                    f"challenger={challenger} dz={effect:.6g}"
                )
            raw_p[challenger] = _sign_flip_p_value(
                difference,
                resamples=gate.sign_flip_resamples,
                rng=rng,
            )
        holm = _holm_adjust(raw_p)
        if max(holm.values()) >= gate.confirmation_alpha:
            raise ActionContractEvidenceError(
                "confirmation fixed offset failed Holm sign-flip threshold: "
                f"family={family} selected={selected} p={holm}"
            )
        results[family] = ConfirmationFamilyReport(
            clip_ids=family_clip_ids,
            clip_count=len(family_clip_ids),
            dz_by_challenger=effects,
            raw_p_by_challenger=raw_p,
            holm_p_by_challenger=holm,
        )
    return ConfirmationReport(
        tested_offset=selected,
        clip_ids=clip_ids,
        by_family=results,
    )


def _expected_offset_falsification(
    scores: Mapping[str, Mapping[int, Mapping[str, float]]],
    clip_ids: tuple[str, ...],
    families: Sequence[str],
    expected_offset: int,
    gate: ActionGateConfig,
    *,
    min_informative: int,
    partition_clip_count: int,
    alpha: float,
    resamples: int,
    seed_offset: int,
    label: str,
) -> ExpectedOffsetPartitionFalsificationReport:
    """Reject a registered physical offset only when evidence contradicts it.

    Adjacent frame offsets are often statistically indistinguishable. The
    formal contract therefore comes from the cache/action construction; state
    and flow are independent falsification checks, not a second offset picker.
    """

    expected_index = CANDIDATE_OFFSETS.index(expected_offset)
    results: dict[str, ExpectedOffsetFamilyFalsificationReport] = {}
    for family_index, family in enumerate(families):
        family_clip_ids = _usable_clip_ids(scores, clip_ids, family)
        if len(family_clip_ids) < min_informative:
            raise ActionContractEvidenceError(
                f"{label} requires at least {min_informative} informative "
                f"{family} clips ({min_informative}/{partition_clip_count}), "
                f"got {len(family_clip_ids)}"
            )
        matrix = _v2_matrix(scores, family_clip_ids, family)
        means = matrix.mean(axis=0)
        best_offset = CANDIDATE_OFFSETS[int(np.argmax(means))]
        effects: dict[int, float] = {}
        raw_p: dict[int, float] = {}
        rng = np.random.default_rng(
            gate.seed + seed_offset + family_index * 100
        )
        for challenger_index, challenger in enumerate(CANDIDATE_OFFSETS):
            if challenger_index == expected_index:
                continue
            difference = matrix[:, challenger_index] - matrix[:, expected_index]
            effects[challenger] = _paired_dz(difference)
            raw_p[challenger] = _sign_flip_p_value(
                difference,
                resamples=resamples,
                rng=rng,
            )
        holm = _holm_adjust(raw_p)
        conflicts = tuple(
            challenger
            for challenger in CANDIDATE_OFFSETS
            if challenger != expected_offset
            and effects[challenger] >= gate.min_dz
            and holm[challenger] < alpha
        )
        results[family] = ExpectedOffsetFamilyFalsificationReport(
            best_offset=best_offset,
            clip_ids=family_clip_ids,
            clip_count=len(family_clip_ids),
            mean_score_by_offset={
                offset: float(means[index])
                for index, offset in enumerate(CANDIDATE_OFFSETS)
            },
            challenger_over_expected_dz=effects,
            raw_p_by_challenger=raw_p,
            holm_p_by_challenger=holm,
            conflicting_challengers=conflicts,
        )
        if conflicts:
            raise ActionContractEvidenceError(
                f"{label} falsified expected offset {expected_offset}: "
                f"family={family} conflicting_challengers={conflicts}",
                code="EXPECTED_OFFSET_FALSIFIED",
                context={
                    "partition": label,
                    "family": family,
                    "expected_offset": expected_offset,
                    "best_offset": best_offset,
                    "conflicting_challengers": list(conflicts),
                    "challenger_over_expected_dz": effects,
                    "holm_p_by_challenger": holm,
                },
            )
    return ExpectedOffsetPartitionFalsificationReport(
        expected_offset=expected_offset,
        clip_ids=clip_ids,
        by_family=results,
    )


def _confirmation_diagnostic(
    scores: Mapping[str, Mapping[int, Mapping[str, float]]],
    clip_ids: tuple[str, ...],
    selected: int,
    family: str,
    gate: ActionGateConfig,
) -> ConfirmationFamilyReport:
    """Summarize a visual diagnostic without making it a binding gate."""
    family_clip_ids = _usable_clip_ids(scores, clip_ids, family)
    matrix = _v2_matrix(scores, family_clip_ids, family)
    selected_index = CANDIDATE_OFFSETS.index(selected)
    effects: dict[int, float] = {}
    raw_p: dict[int, float] = {}
    rng = np.random.default_rng(gate.seed + 1000)
    for index, challenger in enumerate(CANDIDATE_OFFSETS):
        if index == selected_index:
            continue
        difference = matrix[:, selected_index] - matrix[:, index]
        effects[challenger] = _paired_dz(difference)
        raw_p[challenger] = _sign_flip_p_value(
            difference,
            resamples=gate.sign_flip_resamples,
            rng=rng,
        )
    return ConfirmationFamilyReport(
        clip_ids=family_clip_ids,
        clip_count=len(family_clip_ids),
        dz_by_challenger=effects,
        raw_p_by_challenger=raw_p,
        holm_p_by_challenger=_holm_adjust(raw_p),
    )


def _confirmation_visual_agreement(
    scores: Mapping[str, Mapping[int, Mapping[str, float]]],
    clip_ids: tuple[str, ...],
    selected: int,
    eligible: tuple[str, ...],
) -> bool | None:
    if not {"flow", "geometry"}.issubset(eligible):
        return None
    selected_index = CANDIDATE_OFFSETS.index(selected)
    for family in ("flow", "geometry"):
        family_clip_ids = _usable_clip_ids(scores, clip_ids, family)
        if not family_clip_ids:
            raise ActionContractEvidenceError(
                "eligible flow and geometry agreement cannot be checked on "
                f"confirmation: family={family} has no informative clips"
            )
        matrix = _v2_matrix(scores, family_clip_ids, family)
        for index, challenger in enumerate(CANDIDATE_OFFSETS):
            if index == selected_index:
                continue
            if float((matrix[:, selected_index] - matrix[:, index]).mean()) <= 0:
                raise ActionContractEvidenceError(
                    "eligible flow and geometry disagree at confirmation fixed "
                    f"offset: family={family} challenger={challenger}"
                )
    return True


def evaluate_action_contract_v2(
    rows: Sequence[ClipOffsetEvidence],
    gate: ActionGateConfig,
    *,
    contract_key: str = "",
    qualification_clip_ids: Sequence[str] | None = None,
    confirmation_clip_ids: Sequence[str] | None = None,
    split_artifact_sha256: str | None = None,
) -> ActionContractV2Report:
    if not isinstance(gate, ActionGateConfig):
        raise ActionContractEvidenceError(
            "v2 evaluation requires an immutable ActionGateConfig"
        )
    scores, source_by_family, families, diagnostic = _prepare_v2_scores(rows)
    qualification_ids, confirmation_ids, partition_sha = _resolve_split(
        tuple(scores),
        gate,
        contract_key=contract_key,
        qualification_clip_ids=qualification_clip_ids,
        confirmation_clip_ids=confirmation_clip_ids,
        split_artifact_sha256=split_artifact_sha256,
    )
    eligible = _qualification_eligible_families(
        scores,
        qualification_ids,
        families,
    )
    if "state" not in eligible:
        state_count = len(_usable_clip_ids(scores, qualification_ids, "state"))
        raise ActionContractEvidenceError(
            "state family is required with at least 24/32 informative "
            f"qualification clips, got {state_count}"
        )
    visual = "flow" if "flow" in eligible else "geometry"
    if visual not in eligible:
        sources = sorted({source_by_family[family] for family in eligible})
        raise ActionContractEvidenceError(
            "state plus an informative visual family (flow preferred, geometry "
            "fallback) is required with at least 24/32 qualification clips; "
            f"eligible source classes={sources}"
        )
    required = ("state", visual)
    source_classes = (
        source_by_family["state"],
        source_by_family[visual],
    )
    if len(set(source_classes)) != 2:
        raise ActionContractEvidenceError(
            "required evidence must span independent proprioceptive and "
            f"exteroceptive source classes, got {source_classes}"
        )
    binding_families = tuple(
        dict.fromkeys(
            required
            + tuple(
                family
                for family in ("flow", "geometry")
                if family in eligible
            )
        )
    )
    qualification = _qualification_v2(
        scores,
        qualification_ids,
        required,
        families,
        eligible,
        gate,
        binding_families=binding_families,
    )
    confirmation_families = tuple(
        dict.fromkeys(
            required
            + tuple(
                family
                for family in ("flow", "geometry")
                if family in eligible
            )
        )
    )
    confirmation = _confirmation_v2(
        scores,
        confirmation_ids,
        qualification.selected_offset,
        confirmation_families,
        gate,
    )
    agreement = _confirmation_visual_agreement(
        scores,
        confirmation_ids,
        qualification.selected_offset,
        eligible,
    )
    return ActionContractV2Report(
        selected_offset=qualification.selected_offset,
        clip_count=64,
        required_families=required,
        required_source_classes=source_classes,
        eligible_families=eligible,
        binding_families=binding_families,
        diagnostic_families=diagnostic,
        split_artifact_sha256=split_artifact_sha256,
        split_partition_sha256=partition_sha,
        qualification=qualification,
        confirmation=confirmation,
        flow_geometry_agree=agreement,
    )

def _namespaced_clip_id(contract_key: str, clip_id: str) -> str:
    return f"{contract_key}\0{clip_id}"


def _evaluate_oxe_subset_falsification(
    scores: Mapping[str, Mapping[int, Mapping[str, float]]],
    source_by_family: Mapping[str, str],
    families: tuple[str, ...],
    qualification_ids: tuple[str, ...],
    confirmation_ids: tuple[str, ...],
    gate: ActionGateConfig,
    *,
    member_contract_keys: tuple[str, ...],
    min_informative: int,
    label: str,
) -> CohortSubsetFalsificationReport:
    eligible = _qualification_eligible_families(
        scores,
        qualification_ids,
        families,
        min_informative=min_informative,
    )
    if "state" not in eligible:
        raise ActionContractEvidenceError(
            f"{label} state qualification evidence is below "
            f"{min_informative}/{len(qualification_ids)}"
        )
    if "flow" not in eligible:
        raise ActionContractEvidenceError(
            f"{label} requires eligible flow with at least "
            f"{min_informative}/{len(qualification_ids)} clips",
            code="OXE_FLOW_REQUIRED",
        )
    visual = "flow"
    required = ("state", visual)
    required_sources = (
        source_by_family["state"],
        source_by_family[visual],
    )
    if required_sources != ("proprioceptive", "exteroceptive"):
        raise ActionContractEvidenceError(
            f"{label} lacks independent source classes"
        )
    binding = required
    qualification = _expected_offset_falsification(
        scores,
        qualification_ids,
        binding,
        -2,
        gate,
        min_informative=min_informative,
        partition_clip_count=len(qualification_ids),
        alpha=gate.qualification_alpha,
        resamples=gate.permutation_resamples,
        seed_offset=1000,
        label=f"{label} qualification",
    )
    confirmation = _expected_offset_falsification(
        scores,
        confirmation_ids,
        binding,
        -2,
        gate,
        min_informative=min_informative,
        partition_clip_count=len(confirmation_ids),
        alpha=gate.confirmation_alpha,
        resamples=gate.sign_flip_resamples,
        seed_offset=2000,
        label=f"{label} confirmation",
    )
    return CohortSubsetFalsificationReport(
        selected_offset=-2,
        member_contract_keys=member_contract_keys,
        frozen_qualification_clip_count=len(qualification_ids),
        frozen_confirmation_clip_count=len(confirmation_ids),
        eligible_families=eligible,
        binding_families=binding,
        qualification=qualification,
        confirmation=confirmation,
    )


def _wrap_oxe_falsification_failure(
    error: ActionContractEvidenceError,
    *,
    code: str,
    context: Mapping[str, Any],
    label: str,
) -> ActionContractEvidenceError:
    machine_code = (
        "OXE_GEOMETRY_CONFLICT"
        if "geometry" in str(error).lower()
        else code
    )
    return ActionContractEvidenceError(
        f"{label}: {error}",
        code=machine_code,
        context=context,
    )

def evaluate_pooled_oxe_action_contract(
    rows_by_contract_key: Mapping[str, Sequence[ClipOffsetEvidence]],
    gate: ActionGateConfig,
    *,
    split_clip_ids_by_contract_key: Mapping[
        str, tuple[Sequence[str], Sequence[str]]
    ],
    split_artifact_sha256: str,
) -> PooledOXEActionContractReport:
    """Evaluate the preregistered five-domain OXE physical-offset cohort."""

    if set(rows_by_contract_key) != set(FORMAL_OXE_CONTRACT_KEYS):
        raise ActionContractEvidenceError(
            "formal pooled OXE cohort members are not exact: "
            f"expected={list(FORMAL_OXE_CONTRACT_KEYS)} "
            f"got={sorted(rows_by_contract_key)}"
        )
    if set(split_clip_ids_by_contract_key) != set(FORMAL_OXE_CONTRACT_KEYS):
        raise ActionContractEvidenceError(
            "formal pooled OXE split members are not exact"
        )

    merged_scores: dict[str, dict[int, dict[str, float]]] = {}
    source_by_family: dict[str, str] = {}
    family_names: set[str] = set()
    diagnostic_names: set[str] = set()
    member_partition_sha256: dict[str, str] = {}
    member_falsification: dict[str, CohortSubsetFalsificationReport] = {}
    qualification_ids: list[str] = []
    confirmation_ids: list[str] = []

    for contract_key in FORMAL_OXE_CONTRACT_KEYS:
        scores, sources, families, diagnostic = _prepare_v2_scores(
            rows_by_contract_key[contract_key]
        )
        raw_qualification, raw_confirmation = (
            split_clip_ids_by_contract_key[contract_key]
        )
        qualification, confirmation, partition_sha = _resolve_split(
            tuple(scores),
            gate,
            contract_key=contract_key,
            qualification_clip_ids=raw_qualification,
            confirmation_clip_ids=raw_confirmation,
            split_artifact_sha256=split_artifact_sha256,
        )
        member_partition_sha256[contract_key] = partition_sha
        try:
            member_report = _evaluate_oxe_subset_falsification(
                scores,
                sources,
                tuple(sorted(families)),
                qualification,
                confirmation,
                gate,
                member_contract_keys=(contract_key,),
                min_informative=24,
                label=f"OXE member {contract_key}",
            )
        except ActionContractEvidenceError as error:
            raise _wrap_oxe_falsification_failure(
                error,
                code="OXE_MEMBER_FALSIFICATION_FAILURE",
                context={"contract_key": contract_key},
                label=f"OXE member {contract_key}",
            ) from error
        if member_report.selected_offset != -2:
            raise ActionContractEvidenceError(
                f"OXE member {contract_key} did not independently select -2: "
                f"{member_report.selected_offset}",
                code="OXE_MEMBER_OFFSET_CONFLICT",
                context={
                    "contract_key": contract_key,
                    "selected_offset": member_report.selected_offset,
                },
            )
        member_falsification[contract_key] = member_report
        family_names.update(families)
        diagnostic_names.update(diagnostic)
        for family, source in sources.items():
            previous = source_by_family.setdefault(family, source)
            if previous != source:
                raise ActionContractEvidenceError(
                    "pooled OXE family source class differs across members: "
                    f"family={family} {previous!r}!={source!r}"
                )
        for clip_id, by_offset in scores.items():
            namespaced = _namespaced_clip_id(contract_key, clip_id)
            if namespaced in merged_scores:
                raise ActionContractEvidenceError(
                    f"duplicate pooled OXE clip identity: {namespaced}"
                )
            merged_scores[namespaced] = by_offset
        qualification_ids.extend(
            _namespaced_clip_id(contract_key, clip_id)
            for clip_id in qualification
        )
        confirmation_ids.extend(
            _namespaced_clip_id(contract_key, clip_id)
            for clip_id in confirmation
        )

    frozen_partition_count = 32 * len(FORMAL_OXE_CONTRACT_KEYS)
    min_informative = 24 * len(FORMAL_OXE_CONTRACT_KEYS)
    families = tuple(sorted(family_names))
    qualification_tuple = tuple(qualification_ids)
    confirmation_tuple = tuple(confirmation_ids)
    eligible = _qualification_eligible_families(
        merged_scores,
        qualification_tuple,
        families,
        min_informative=min_informative,
    )
    if "state" not in eligible:
        raise ActionContractEvidenceError(
            "pooled OXE state qualification evidence is below "
            f"{min_informative}/{frozen_partition_count}"
        )
    if "flow" not in eligible:
        raise ActionContractEvidenceError(
            "pooled OXE requires eligible flow with at least "
            f"{min_informative}/{frozen_partition_count} clips",
            code="OXE_FLOW_REQUIRED",
        )
    visual = "flow"
    required = ("state", visual)
    required_sources = (
        source_by_family["state"],
        source_by_family[visual],
    )
    if required_sources != ("proprioceptive", "exteroceptive"):
        raise ActionContractEvidenceError(
            "pooled OXE required families lack independent source classes"
        )

    binding_families = required
    qualification = _expected_offset_falsification(
        merged_scores,
        qualification_tuple,
        binding_families,
        -2,
        gate,
        min_informative=min_informative,
        partition_clip_count=frozen_partition_count,
        alpha=gate.qualification_alpha,
        resamples=gate.permutation_resamples,
        seed_offset=3000,
        label="pooled OXE qualification",
    )
    confirmation = _expected_offset_falsification(
        merged_scores,
        confirmation_tuple,
        binding_families,
        -2,
        gate,
        min_informative=min_informative,
        partition_clip_count=frozen_partition_count,
        alpha=gate.confirmation_alpha,
        resamples=gate.sign_flip_resamples,
        seed_offset=4000,
        label="pooled OXE confirmation",
    )
    if visual == "flow" and "geometry" in eligible:
        diagnostic_names.add("geometry")
    cohort_partition = hashlib.sha256()
    cohort_partition.update(FORMAL_OXE_COHORT_ID.encode("utf-8"))
    cohort_partition.update(b"\0")
    cohort_partition.update(str(split_artifact_sha256).encode("ascii"))
    for contract_key in FORMAL_OXE_CONTRACT_KEYS:
        cohort_partition.update(b"\0")
        cohort_partition.update(contract_key.encode("utf-8"))
        cohort_partition.update(b"\0")
        cohort_partition.update(
            member_partition_sha256[contract_key].encode("ascii")
        )

    return PooledOXEActionContractReport(
        selected_offset=-2,
        clip_count=len(merged_scores),
        frozen_qualification_clip_count=frozen_partition_count,
        frozen_confirmation_clip_count=frozen_partition_count,
        required_families=required,
        required_source_classes=required_sources,
        eligible_families=eligible,
        binding_families=binding_families,
        diagnostic_families=tuple(sorted(diagnostic_names)),
        split_artifact_sha256=split_artifact_sha256,
        split_partition_sha256=cohort_partition.hexdigest(),
        member_split_partition_sha256=member_partition_sha256,
        member_falsification=member_falsification,
        qualification=qualification,
        confirmation=confirmation,
        geometry_policy=(
            "diagnostic_only_never_binding"
        ),
    )


evaluate_pooled_oxe_action_contract_diagnostic_only = (
    evaluate_pooled_oxe_action_contract
)


def evaluate_exact_derived_droid_action_contract(
    rows: Sequence[ClipOffsetEvidence],
    gate: ActionGateConfig,
    *,
    contract_key: str,
    qualification_clip_ids: Sequence[str],
    confirmation_clip_ids: Sequence[str],
    split_artifact_sha256: str,
) -> ExactDerivedDroidReport:
    if contract_key != FORMAL_DROID_CONTRACT_KEY:
        raise ActionContractEvidenceError(
            f"unexpected exact-derived DROID contract key: {contract_key}"
        )
    scores, source_by_family, families, _ = _prepare_v2_scores(rows)
    for family in ("state", "flow"):
        if family not in families:
            raise ActionContractEvidenceError(
                f"DROID exact-derived method requires {family} evidence"
            )
    source_classes = (
        source_by_family["state"],
        source_by_family["flow"],
    )
    if source_classes != ("proprioceptive", "exteroceptive"):
        raise ActionContractEvidenceError(
            "DROID exact-derived sanity check lacks independent source classes"
        )
    qualification, confirmation, partition_sha = _resolve_split(
        tuple(scores),
        gate,
        contract_key=contract_key,
        qualification_clip_ids=qualification_clip_ids,
        confirmation_clip_ids=confirmation_clip_ids,
        split_artifact_sha256=split_artifact_sha256,
    )
    families = ("state", "flow")
    min_informative = 24
    eligible = _qualification_eligible_families(
        scores,
        qualification,
        families,
        min_informative=min_informative,
    )
    if set(eligible) != set(families):
        raise ActionContractEvidenceError(
            "DROID exact-derived formal gate requires state and flow to be "
            f"informative on qualification: eligible={eligible}"
        )
    qualification_falsification = _expected_offset_falsification(
        scores,
        qualification,
        families,
        -1,
        gate,
        min_informative=min_informative,
        partition_clip_count=len(qualification),
        alpha=gate.qualification_alpha,
        resamples=gate.permutation_resamples,
        seed_offset=5000,
        label="DROID qualification",
    )
    confirmation_falsification = _expected_offset_falsification(
        scores,
        confirmation,
        families,
        -1,
        gate,
        min_informative=min_informative,
        partition_clip_count=len(confirmation),
        alpha=gate.confirmation_alpha,
        resamples=gate.sign_flip_resamples,
        seed_offset=6000,
        label="DROID confirmation",
    )
    return ExactDerivedDroidReport(
        selected_offset=-1,
        clip_count=64,
        required_families=families,
        required_source_classes=source_classes,
        split_artifact_sha256=split_artifact_sha256,
        split_partition_sha256=partition_sha,
        qualification=qualification_falsification,
        confirmation=confirmation_falsification,
        separation_basis=(
            "exact_n_minus_one_interval_action_construction"
        ),
    )
