"""Collective Stage1 timing, memory aggregation, and stop decisions."""

from __future__ import annotations

import math
import time
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Mapping

import torch
import torch.distributed as dist

from wm3d_v3.stage1.runtime import Stage1Phase, phase_for_update


TIMING_COMPONENTS = (
    "data",
    "wm",
    "wan",
    "adapter",
    "vae",
    "communication",
    "checkpoint",
)
FORMAL_G5_UPDATE_IDS = frozenset({512, 1024, 2560, 5120, 9984})
_BASE_G5_TIMINGS = tuple(
    component for component in TIMING_COMPONENTS if component != "checkpoint"
)
G5_REQUIRED_TIMINGS_BY_PHASE = {phase: _BASE_G5_TIMINGS for phase in Stage1Phase}
CUDA_GATE_TIMING_COMPONENTS = (
    "wm",
    "wan",
    "adapter",
    "vae",
    "communication",
)
COLLECTIVE_STOP_REASONS = (
    "contract_changed",
    "source_digest_changed",
    "source_active",
    "forbidden_parameters",
    "memory_over_85_percent",
    "repeated_compute_errors",
    "repeated_checkpoint_errors",
    "shared_free_space_below_500_gb",
    "hard_gate_failure",
    "target_dependence",
    "step_stall_over_120_seconds",
    "performance_violation_after_retry",
    "average_step_over_1_25x_normal",
    "rolling_step_over_6x_normal",
    "unsafe_cuda_gate_timing",
    "missing_measurement",
)
MEMORY_LIMIT_FRACTION = 0.85
STALL_LIMIT_SECONDS = 120.0
MIN_SHARED_FREE_BYTES = 500_000_000_000
REPEATED_ERROR_LIMIT = 2
AVERAGE_STEP_RATIO_LIMIT = 1.25
ROLLING_STEP_RATIO_LIMIT = 6.0

_DEFAULT_DISTRIBUTED = object()
_SAFE_TIMING_SOURCES = frozenset({"cuda_event", "synchronized_boundary"})


class TelemetryCollectiveError(RuntimeError):
    """A process-group failure that requires the distributed job to abort."""


def _non_negative_seconds(value: float, *, name: str) -> float:
    seconds = float(value)
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError(f"{name} must be a finite non-negative duration")
    return seconds


def _byte_count(value: int, *, name: str, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer byte count")
    if value < 0 or (positive and value == 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be a {qualifier} integer byte count")
    return value


def _non_negative_int(value: int, *, name: str, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < 0 or (positive and value == 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be {qualifier}")
    return value


def _torch_device(device: torch.device | str | int) -> torch.device:
    if isinstance(device, int):
        return torch.device("cuda", device)
    return torch.device(device)


class UpdateTelemetry:
    """Accumulate local metrics and publish one collective verdict snapshot."""

    def __init__(
        self,
        update_id: int = 1,
        phase: Stage1Phase | str | None = None,
        *,
        clock: Callable[[], float] = time.perf_counter,
        memory_limit_fraction: float = MEMORY_LIMIT_FRACTION,
    ) -> None:
        if (
            isinstance(update_id, bool)
            or not isinstance(update_id, int)
            or update_id < 1
        ):
            raise ValueError("update_id must be a positive integer")
        self.update_id = update_id
        self.phase = (
            phase_for_update(update_id) if phase is None else Stage1Phase(phase)
        )
        self.clock = clock
        self.memory_limit_fraction = float(memory_limit_fraction)
        if (
            not math.isfinite(self.memory_limit_fraction)
            or not 0 < self.memory_limit_fraction <= 1
        ):
            raise ValueError("memory_limit_fraction must be in (0, 1]")

        self.timings_s = {component: 0.0 for component in TIMING_COMPONENTS}
        self._timing_sources: dict[str, list[str]] = {
            component: [] for component in TIMING_COMPONENTS
        }
        self.all_rank_max_timings_s = dict(self.timings_s)
        self.local_memory_reserved_bytes = 0
        self.local_total_memory_bytes = 0
        self.local_memory_fraction = 0.0
        self._memory_recorded = False
        self.all_rank_max_memory_reserved_bytes = 0
        self.all_rank_max_memory_fraction = 0.0
        self.all_rank_max_step_time_s: float | None = None
        self.all_rank_max_compute_error_count = 0
        self.all_rank_max_checkpoint_error_count = 0
        self.all_rank_min_shared_free_bytes: int | None = None
        self.expected_world_size: int | None = None
        self.observed_world_size: int | None = None
        self.step_time_s: float | None = None
        self._step_timing_synchronized = False
        self._step_started_at: float | None = None
        self._step_timer_device: torch.device | None = None
        self._recording_generation = 0
        self._collective_generation: int | None = None
        self._collective_stop_reasons: tuple[str, ...] = ()
        self._g5_measurements_required = False
        self._g5_checkpoint_expected = False

    @property
    def timings(self) -> Mapping[str, float]:
        return self.timings_s

    @property
    def timing_sources(self) -> Mapping[str, tuple[str, ...]]:
        return {
            component: tuple(sources)
            for component, sources in self._timing_sources.items()
        }

    @property
    def collective_fresh(self) -> bool:
        return self._collective_generation == self._recording_generation

    def _invalidate_collective(self) -> None:
        self._recording_generation += 1
        self._collective_generation = None
        self._collective_stop_reasons = ()

    @contextmanager
    def measure(self, component: str) -> Iterator[None]:
        """Measure CPU work only; CUDA gate work must use measure_cuda."""

        self._require_component(component)
        started_at = self.clock()
        try:
            yield
        finally:
            self._record_timing(
                component,
                self.clock() - started_at,
                source="wall_clock",
            )

    time_component = measure

    @contextmanager
    def measure_cuda(
        self,
        component: str,
        *,
        device: torch.device | str | int | None = None,
        stream: torch.cuda.Stream | None = None,
    ) -> Iterator[None]:
        """Measure CUDA work with stream events and a synchronized end boundary."""

        self._require_component(component)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for CUDA event timing")
        resolved_device = (
            torch.device("cuda", torch.cuda.current_device())
            if device is None
            else _torch_device(device)
        )
        if resolved_device.type != "cuda":
            raise ValueError("measure_cuda requires a CUDA device")

        with torch.cuda.device(resolved_device):
            active_stream = (
                torch.cuda.current_stream(resolved_device) if stream is None else stream
            )
            started = torch.cuda.Event(enable_timing=True)
            finished = torch.cuda.Event(enable_timing=True)
            started.record(active_stream)
            try:
                yield
            finally:
                finished.record(active_stream)
                finished.synchronize()
                self._record_timing(
                    component,
                    started.elapsed_time(finished) / 1000.0,
                    source="cuda_event",
                )

    def _require_component(self, component: str) -> None:
        if component not in self.timings_s:
            allowed = ", ".join(TIMING_COMPONENTS)
            raise ValueError(
                f"unknown timing component {component!r}; expected one of {allowed}"
            )

    def _record_timing(
        self,
        component: str,
        seconds: float,
        *,
        source: str,
    ) -> None:
        self._require_component(component)
        duration = _non_negative_seconds(seconds, name=f"{component} timing")
        self.timings_s[component] += duration
        self._timing_sources[component].append(source)
        self._invalidate_collective()

    def record_timing(
        self,
        component: str,
        seconds: float,
        *,
        synchronized: bool = False,
    ) -> None:
        source = "synchronized_boundary" if synchronized else "wall_clock"
        self._record_timing(component, seconds, source=source)

    def start_step(
        self,
        *,
        device: torch.device | str | int | None = None,
    ) -> None:
        if self._step_started_at is not None:
            raise RuntimeError("step timer is already running")
        self._invalidate_collective()
        if torch.cuda.is_available():
            resolved_device = (
                torch.device("cuda", torch.cuda.current_device())
                if device is None
                else _torch_device(device)
            )
            if resolved_device.type == "cuda":
                torch.cuda.synchronize(resolved_device)
                self._step_timer_device = resolved_device
            else:
                self._step_timer_device = None
        else:
            self._step_timer_device = None
        self._step_started_at = self.clock()

    def finish_step(self) -> float:
        if self._step_started_at is None:
            raise RuntimeError("step timer was not started")
        if self._step_timer_device is not None:
            torch.cuda.synchronize(self._step_timer_device)
        elapsed = self.clock() - self._step_started_at
        self._step_started_at = None
        self._step_timer_device = None
        self.set_step_time(elapsed, synchronized=True)
        assert self.step_time_s is not None
        return self.step_time_s

    def set_step_time(
        self,
        seconds: float,
        *,
        synchronized: bool = False,
    ) -> None:
        self.step_time_s = _non_negative_seconds(seconds, name="step_time_s")
        self._step_timing_synchronized = bool(synchronized)
        self._invalidate_collective()

    def reset_cuda_peak_memory(
        self,
        device: torch.device | str | int | None = None,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required to reset peak memory telemetry")
        torch.cuda.reset_peak_memory_stats(device)

    def capture_cuda_memory(
        self,
        device: torch.device | str | int | None = None,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required to capture memory telemetry")
        reserved_bytes = int(torch.cuda.max_memory_reserved(device))
        properties = torch.cuda.get_device_properties(device)
        self.record_local_memory(
            reserved_bytes=reserved_bytes,
            total_bytes=int(properties.total_memory),
        )

    capture_memory = capture_cuda_memory

    def record_local_memory(
        self,
        *,
        reserved_bytes: int,
        total_bytes: int,
    ) -> None:
        reserved = _byte_count(reserved_bytes, name="reserved_bytes")
        total = _byte_count(total_bytes, name="total_bytes", positive=True)
        if reserved > total:
            raise ValueError("reserved_bytes cannot exceed total_bytes")
        self.local_memory_reserved_bytes = reserved
        self.local_total_memory_bytes = total
        self.local_memory_fraction = reserved / total
        self._memory_recorded = True
        self._invalidate_collective()

    @staticmethod
    def _collective_device(
        distributed: Any,
        process_group: Any,
        device: torch.device | str | int | None,
    ) -> torch.device:
        if device is not None:
            return _torch_device(device)
        get_backend = getattr(distributed, "get_backend", None)
        if callable(get_backend):
            try:
                backend = str(get_backend(process_group)).lower()
            except (RuntimeError, TypeError):
                backend = ""
            if "nccl" in backend:
                if not torch.cuda.is_available():
                    raise RuntimeError("NCCL telemetry reduction requires CUDA")
                return torch.device("cuda", torch.cuda.current_device())
        return torch.device("cpu")

    def _has_unsafe_cuda_gate_timing(self) -> bool:
        for component in CUDA_GATE_TIMING_COMPONENTS:
            sources = self._timing_sources[component]
            if sources and any(
                source not in _SAFE_TIMING_SOURCES for source in sources
            ):
                return True
        return self.step_time_s is not None and not self._step_timing_synchronized

    def _missing_required_measurement(
        self,
        *,
        require_g5_measurements: bool,
        checkpoint_expected: bool,
    ) -> bool:
        if not require_g5_measurements:
            return False
        required = list(G5_REQUIRED_TIMINGS_BY_PHASE[self.phase])
        if checkpoint_expected:
            required.append("checkpoint")
        return (
            self.step_time_s is None
            or not self._memory_recorded
            or any(not self._timing_sources[component] for component in required)
        )

    def _local_stop_flags(
        self,
        *,
        require_g5_measurements: bool,
        checkpoint_expected: bool,
        compute_error_count: int,
        checkpoint_error_count: int,
        repeated_error_limit: int,
        shared_free_bytes: int | None,
        min_shared_free_bytes: int,
        contract_changed: bool,
        source_digest_changed: bool,
        source_active: bool,
        forbidden_parameters: bool,
        hard_gate_failed: bool,
        target_dependent: bool,
        normal_step_baseline_s: float | None,
        average_step_time_s: float | None,
        profiler_retry_confirmed: bool,
        performance_violation: bool,
    ) -> dict[str, bool]:
        average_violation = False
        rolling_violation = False
        if profiler_retry_confirmed and normal_step_baseline_s is not None:
            baseline = _non_negative_seconds(
                normal_step_baseline_s,
                name="normal_step_baseline_s",
            )
            if baseline == 0:
                raise ValueError("normal_step_baseline_s must be positive")
            if average_step_time_s is not None:
                average = _non_negative_seconds(
                    average_step_time_s,
                    name="average_step_time_s",
                )
                average_violation = average > AVERAGE_STEP_RATIO_LIMIT * baseline
            rolling_violation = (
                self.phase is Stage1Phase.ROLLING
                and self.step_time_s is not None
                and self.step_time_s > ROLLING_STEP_RATIO_LIMIT * baseline
            )

        return {
            "contract_changed": bool(contract_changed),
            "source_digest_changed": bool(source_digest_changed),
            "source_active": bool(source_active),
            "forbidden_parameters": bool(forbidden_parameters),
            "memory_over_85_percent": (
                self.local_memory_fraction > self.memory_limit_fraction
            ),
            "repeated_compute_errors": (compute_error_count >= repeated_error_limit),
            "repeated_checkpoint_errors": (
                checkpoint_error_count >= repeated_error_limit
            ),
            "shared_free_space_below_500_gb": (
                shared_free_bytes is not None
                and shared_free_bytes < min_shared_free_bytes
            ),
            "hard_gate_failure": bool(hard_gate_failed),
            "target_dependence": bool(target_dependent),
            "step_stall_over_120_seconds": (
                self.step_time_s is not None and self.step_time_s > STALL_LIMIT_SECONDS
            ),
            "performance_violation_after_retry": (
                bool(performance_violation) and profiler_retry_confirmed
            ),
            "average_step_over_1_25x_normal": average_violation,
            "rolling_step_over_6x_normal": rolling_violation,
            "unsafe_cuda_gate_timing": self._has_unsafe_cuda_gate_timing(),
            "missing_measurement": self._missing_required_measurement(
                require_g5_measurements=require_g5_measurements,
                checkpoint_expected=checkpoint_expected,
            ),
        }

    def _all_reduce_or_abort(
        self,
        backend: Any,
        payload: torch.Tensor,
        *,
        op: Any,
        process_group: Any,
        stage: str,
    ) -> None:
        try:
            backend.all_reduce(payload, op=op, group=process_group)
        except Exception as exc:
            self._collective_generation = None
            self._collective_stop_reasons = ()
            raise TelemetryCollectiveError(
                f"Stage1 telemetry collective {stage} failed; hard abort "
                "required so torchrun/process-group timeout can terminate "
                "surviving ranks"
            ) from exc

    def _reduce_rank_maxima(
        self,
        *,
        expected_world_size: int,
        distributed: Any = _DEFAULT_DISTRIBUTED,
        process_group: Any = None,
        device: torch.device | str | int | None = None,
        compute_error_count: int = 0,
        checkpoint_error_count: int = 0,
        repeated_error_limit: int = REPEATED_ERROR_LIMIT,
        shared_free_bytes: int | None = None,
        min_shared_free_bytes: int = MIN_SHARED_FREE_BYTES,
        contract_changed: bool = False,
        source_digest_changed: bool = False,
        source_active: bool = False,
        forbidden_parameters: bool = False,
        hard_gate_failed: bool = False,
        target_dependent: bool = False,
        normal_step_baseline_s: float | None = None,
        average_step_time_s: float | None = None,
        profiler_retry_confirmed: bool = False,
        performance_violation: bool = False,
        require_g5_measurements: bool | None = None,
        checkpoint_expected_override: bool | None = None,
    ) -> None:
        """Collect all rank metrics and hard-failure bits in a fixed order."""

        expected = _non_negative_int(
            expected_world_size,
            name="expected_world_size",
            positive=True,
        )
        compute_errors = _non_negative_int(
            compute_error_count,
            name="compute_error_count",
        )
        checkpoint_errors = _non_negative_int(
            checkpoint_error_count,
            name="checkpoint_error_count",
        )
        error_limit = _non_negative_int(
            repeated_error_limit,
            name="repeated_error_limit",
            positive=True,
        )
        min_free = _byte_count(
            min_shared_free_bytes,
            name="min_shared_free_bytes",
            positive=True,
        )
        local_free = (
            None
            if shared_free_bytes is None
            else _byte_count(shared_free_bytes, name="shared_free_bytes")
        )
        formal_g5_update = self.update_id in FORMAL_G5_UPDATE_IDS
        g5_required = formal_g5_update or bool(require_g5_measurements)
        checkpoint_required = (
            formal_g5_update
            if checkpoint_expected_override is None
            else bool(checkpoint_expected_override)
        )
        self._g5_measurements_required = g5_required
        self._g5_checkpoint_expected = checkpoint_required

        backend = dist if distributed is _DEFAULT_DISTRIBUTED else distributed
        initialized = (
            backend is not None
            and bool(backend.is_available())
            and bool(backend.is_initialized())
        )
        if not initialized:
            raise TelemetryCollectiveError(
                "an initialized distributed collective is required for "
                "Stage1 telemetry verdicts; hard abort required"
            )

        reduction_device = self._collective_device(
            backend,
            process_group,
            device,
        )
        self._collective_generation = None
        self._collective_stop_reasons = ()

        participation = torch.ones(
            1,
            dtype=torch.float64,
            device=reduction_device,
        )
        self._all_reduce_or_abort(
            backend,
            participation,
            op=backend.ReduceOp.SUM,
            process_group=process_group,
            stage="participation-token",
        )

        local_flags = self._local_stop_flags(
            require_g5_measurements=g5_required,
            checkpoint_expected=checkpoint_required,
            compute_error_count=compute_errors,
            checkpoint_error_count=checkpoint_errors,
            repeated_error_limit=error_limit,
            shared_free_bytes=local_free,
            min_shared_free_bytes=min_free,
            contract_changed=contract_changed,
            source_digest_changed=source_digest_changed,
            source_active=source_active,
            forbidden_parameters=forbidden_parameters,
            hard_gate_failed=hard_gate_failed,
            target_dependent=target_dependent,
            normal_step_baseline_s=normal_step_baseline_s,
            average_step_time_s=average_step_time_s,
            profiler_retry_confirmed=profiler_retry_confirmed,
            performance_violation=performance_violation,
        )
        max_values = [self.timings_s[component] for component in TIMING_COMPONENTS]
        max_values.extend(
            [
                float(self.local_memory_reserved_bytes),
                self.local_memory_fraction,
                -1.0 if self.step_time_s is None else self.step_time_s,
                float(compute_errors),
                float(checkpoint_errors),
                float(expected),
                self.memory_limit_fraction,
                float(error_limit),
                float(min_free),
            ]
        )
        max_values.extend(
            float(local_flags[reason]) for reason in COLLECTIVE_STOP_REASONS
        )
        maxima = torch.tensor(
            max_values,
            dtype=torch.float64,
            device=reduction_device,
        )
        self._all_reduce_or_abort(
            backend,
            maxima,
            op=backend.ReduceOp.MAX,
            process_group=process_group,
            stage="rank-maxima-and-stop-bits",
        )

        minima = torch.tensor(
            [
                math.inf if local_free is None else float(local_free),
                float(expected),
                self.memory_limit_fraction,
                float(error_limit),
                float(min_free),
            ],
            dtype=torch.float64,
            device=reduction_device,
        )
        self._all_reduce_or_abort(
            backend,
            minima,
            op=backend.ReduceOp.MIN,
            process_group=process_group,
            stage="rank-minima",
        )

        participation_value = float(participation.item())
        observed = int(round(participation_value))
        reduced_max = maxima.cpu().tolist()
        reduced_min = minima.cpu().tolist()
        timing_count = len(TIMING_COMPONENTS)
        self.all_rank_max_timings_s = dict(
            zip(TIMING_COMPONENTS, reduced_max[:timing_count])
        )
        prefix = timing_count
        self.all_rank_max_memory_reserved_bytes = int(round(reduced_max[prefix]))
        self.all_rank_max_memory_fraction = float(reduced_max[prefix + 1])
        max_step = float(reduced_max[prefix + 2])
        self.all_rank_max_step_time_s = None if max_step < 0 else max_step
        self.all_rank_max_compute_error_count = int(round(reduced_max[prefix + 3]))
        self.all_rank_max_checkpoint_error_count = int(round(reduced_max[prefix + 4]))
        max_expected = int(round(reduced_max[prefix + 5]))
        max_memory_limit = float(reduced_max[prefix + 6])
        max_error_limit = int(round(reduced_max[prefix + 7]))
        max_min_free = int(round(reduced_max[prefix + 8]))
        reason_start = prefix + 9
        global_flags = {
            reason: bool(round(reduced_max[reason_start + index]))
            for index, reason in enumerate(COLLECTIVE_STOP_REASONS)
        }

        min_shared_free = float(reduced_min[0])
        min_expected = int(round(reduced_min[1]))
        min_memory_limit = float(reduced_min[2])
        min_error_limit = int(round(reduced_min[3]))
        min_min_free = int(round(reduced_min[4]))
        self.all_rank_min_shared_free_bytes = (
            None if math.isinf(min_shared_free) else int(round(min_shared_free))
        )
        self.expected_world_size = max_expected
        self.observed_world_size = observed
        global_flags["memory_over_85_percent"] |= (
            self.all_rank_max_memory_fraction > min_memory_limit
        )
        global_flags["repeated_compute_errors"] |= (
            self.all_rank_max_compute_error_count >= min_error_limit
        )
        global_flags["repeated_checkpoint_errors"] |= (
            self.all_rank_max_checkpoint_error_count >= min_error_limit
        )
        global_flags["shared_free_space_below_500_gb"] |= (
            not math.isinf(min_shared_free) and min_shared_free < max_min_free
        )
        global_flags["step_stall_over_120_seconds"] |= (
            self.all_rank_max_step_time_s is not None
            and self.all_rank_max_step_time_s > STALL_LIMIT_SECONDS
        )

        if (
            not math.isclose(participation_value, float(observed))
            or observed != max_expected
        ):
            self._collective_generation = None
            self._collective_stop_reasons = ()
            raise TelemetryCollectiveError(
                "Stage1 telemetry participation-token integrity failure: "
                f"observed={participation_value}, expected={max_expected}; "
                "hard abort required"
            )

        derived_reasons: list[str] = []
        if min_expected != max_expected:
            derived_reasons.append("expected_world_size_mismatch")
        if (
            not math.isclose(min_memory_limit, max_memory_limit)
            or min_error_limit != max_error_limit
            or min_min_free != max_min_free
        ):
            derived_reasons.append("collective_config_mismatch")
        collective_reasons = [
            reason for reason in COLLECTIVE_STOP_REASONS if global_flags[reason]
        ]
        self._collective_stop_reasons = tuple([*derived_reasons, *collective_reasons])
        self._collective_generation = self._recording_generation

    def prepare_checkpoint_finalize(
        self,
        *,
        expected_world_size: int,
        distributed: Any = _DEFAULT_DISTRIBUTED,
        process_group: Any = None,
        device: torch.device | str | int | None = None,
        compute_error_count: int = 0,
        checkpoint_error_count: int = 0,
        repeated_error_limit: int = REPEATED_ERROR_LIMIT,
        shared_free_bytes: int | None = None,
        min_shared_free_bytes: int = MIN_SHARED_FREE_BYTES,
        contract_changed: bool = False,
        source_digest_changed: bool = False,
        source_active: bool = False,
        forbidden_parameters: bool = False,
        hard_gate_failed: bool = False,
        target_dependent: bool = False,
        normal_step_baseline_s: float | None = None,
        average_step_time_s: float | None = None,
        profiler_retry_confirmed: bool = False,
        performance_violation: bool = False,
        require_g5_measurements: bool | None = None,
    ) -> None:
        """Publish a fresh pre-checkpoint snapshot without requiring checkpoint timing."""

        self._reduce_rank_maxima(
            expected_world_size=expected_world_size,
            distributed=distributed,
            process_group=process_group,
            device=device,
            compute_error_count=compute_error_count,
            checkpoint_error_count=checkpoint_error_count,
            repeated_error_limit=repeated_error_limit,
            shared_free_bytes=shared_free_bytes,
            min_shared_free_bytes=min_shared_free_bytes,
            contract_changed=contract_changed,
            source_digest_changed=source_digest_changed,
            source_active=source_active,
            forbidden_parameters=forbidden_parameters,
            hard_gate_failed=hard_gate_failed,
            target_dependent=target_dependent,
            normal_step_baseline_s=normal_step_baseline_s,
            average_step_time_s=average_step_time_s,
            profiler_retry_confirmed=profiler_retry_confirmed,
            performance_violation=performance_violation,
            require_g5_measurements=require_g5_measurements,
            checkpoint_expected_override=False,
        )

    def reduce_rank_maxima(
        self,
        *,
        expected_world_size: int,
        distributed: Any = _DEFAULT_DISTRIBUTED,
        process_group: Any = None,
        device: torch.device | str | int | None = None,
        compute_error_count: int = 0,
        checkpoint_error_count: int = 0,
        repeated_error_limit: int = REPEATED_ERROR_LIMIT,
        shared_free_bytes: int | None = None,
        min_shared_free_bytes: int = MIN_SHARED_FREE_BYTES,
        contract_changed: bool = False,
        source_digest_changed: bool = False,
        source_active: bool = False,
        forbidden_parameters: bool = False,
        hard_gate_failed: bool = False,
        target_dependent: bool = False,
        normal_step_baseline_s: float | None = None,
        average_step_time_s: float | None = None,
        profiler_retry_confirmed: bool = False,
        performance_violation: bool = False,
        require_g5_measurements: bool | None = None,
    ) -> None:
        self._reduce_rank_maxima(
            expected_world_size=expected_world_size,
            distributed=distributed,
            process_group=process_group,
            device=device,
            compute_error_count=compute_error_count,
            checkpoint_error_count=checkpoint_error_count,
            repeated_error_limit=repeated_error_limit,
            shared_free_bytes=shared_free_bytes,
            min_shared_free_bytes=min_shared_free_bytes,
            contract_changed=contract_changed,
            source_digest_changed=source_digest_changed,
            source_active=source_active,
            forbidden_parameters=forbidden_parameters,
            hard_gate_failed=hard_gate_failed,
            target_dependent=target_dependent,
            normal_step_baseline_s=normal_step_baseline_s,
            average_step_time_s=average_step_time_s,
            profiler_retry_confirmed=profiler_retry_confirmed,
            performance_violation=performance_violation,
            require_g5_measurements=require_g5_measurements,
        )

    def _require_fresh_collective(self) -> None:
        if not self.collective_fresh:
            raise RuntimeError(
                "a fresh all-rank telemetry reduction is required before "
                "reading a Stage1 verdict"
            )

    @property
    def max_memory_reserved_bytes(self) -> int:
        self._require_fresh_collective()
        return self.all_rank_max_memory_reserved_bytes

    @property
    def max_memory_fraction(self) -> float:
        self._require_fresh_collective()
        return self.all_rank_max_memory_fraction

    def evaluate_stop_conditions(self) -> tuple[str, ...]:
        """Return the identical hard-stop verdict from the fresh snapshot."""

        self._require_fresh_collective()
        return self._collective_stop_reasons

    def should_stop(self) -> bool:
        self._require_fresh_collective()
        return bool(self._collective_stop_reasons)

    def as_dict(self) -> dict[str, Any]:
        self._require_fresh_collective()
        return {
            "update_id": self.update_id,
            "phase": self.phase.value,
            "timings_s": dict(self.timings_s),
            "timing_sources": dict(self.timing_sources),
            "all_rank_max_timings_s": dict(self.all_rank_max_timings_s),
            "step_time_s": self.step_time_s,
            "all_rank_max_step_time_s": self.all_rank_max_step_time_s,
            "local_memory_reserved_bytes": self.local_memory_reserved_bytes,
            "local_total_memory_bytes": self.local_total_memory_bytes,
            "all_rank_max_memory_reserved_bytes": (
                self.all_rank_max_memory_reserved_bytes
            ),
            "all_rank_max_memory_fraction": (self.all_rank_max_memory_fraction),
            "all_rank_max_compute_error_count": (self.all_rank_max_compute_error_count),
            "all_rank_max_checkpoint_error_count": (
                self.all_rank_max_checkpoint_error_count
            ),
            "all_rank_min_shared_free_bytes": (self.all_rank_min_shared_free_bytes),
            "expected_world_size": self.expected_world_size,
            "observed_world_size": self.observed_world_size,
            "collective_stop_reasons": self._collective_stop_reasons,
            "collective_fresh": self.collective_fresh,
            "g5_measurements_required": self._g5_measurements_required,
            "g5_checkpoint_expected": self._g5_checkpoint_expected,
        }


__all__ = [
    "AVERAGE_STEP_RATIO_LIMIT",
    "COLLECTIVE_STOP_REASONS",
    "CUDA_GATE_TIMING_COMPONENTS",
    "FORMAL_G5_UPDATE_IDS",
    "G5_REQUIRED_TIMINGS_BY_PHASE",
    "MEMORY_LIMIT_FRACTION",
    "MIN_SHARED_FREE_BYTES",
    "REPEATED_ERROR_LIMIT",
    "ROLLING_STEP_RATIO_LIMIT",
    "STALL_LIMIT_SECONDS",
    "TIMING_COMPONENTS",
    "TelemetryCollectiveError",
    "UpdateTelemetry",
]
