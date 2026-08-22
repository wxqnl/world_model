"""Bounded one-batch CUDA pipeline for frozen input adapters."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import threading
import time
from typing import Any, Callable, Iterator, Mapping

import torch


def _record_stream(value: Any, stream: torch.cuda.Stream) -> None:
    if isinstance(value, torch.Tensor):
        if value.device.type == "cuda":
            value.record_stream(stream)
        return
    if isinstance(value, Mapping):
        for item in value.values():
            _record_stream(item, stream)
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            _record_stream(item, stream)


class AsyncCudaInputPipeline:
    """Overlap next-batch frozen teacher work with current model training.

    Exactly one batch may be pending.  The iterator and adapter are both owned
    by one worker thread, which prevents reordering and preserves the sealed
    step-addressed sampler.  CUDA events make cross-stream ownership explicit;
    no model/data/loss value is changed.
    """

    def __init__(
        self,
        *,
        iterator: Iterator[Any],
        adapter: Any,
        transfer: Callable[[Mapping[str, Any], torch.device], dict[str, Any]],
        device: torch.device,
    ) -> None:
        if device.type != "cuda":
            raise ValueError("async input pipeline requires a CUDA device")
        self.iterator = iterator
        self.adapter = adapter
        self.transfer = transfer
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        self.stream.wait_stream(torch.cuda.current_stream(device))
        self._adapter_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="wm3d-direct-cuda-input",
        )
        self._future: Future[
            tuple[dict[str, Any], torch.cuda.Event, torch.cuda.Event]
        ] | None = None
        self.submitted = 0
        self.consumed = 0
        self.data_wait_seconds = 0.0
        self.consume_wait_seconds = 0.0
        self.cuda_stage_seconds = 0.0

    @property
    def pending(self) -> bool:
        return self._future is not None

    @property
    def metrics(self) -> Mapping[str, float | int]:
        return {
            "submitted": self.submitted,
            "consumed": self.consumed,
            "pending": int(self.pending),
            "data_wait_seconds": self.data_wait_seconds,
            "consume_wait_seconds": self.consume_wait_seconds,
            "cuda_stage_seconds": self.cuda_stage_seconds,
        }

    def _prepare_one(
        self,
    ) -> tuple[dict[str, Any], torch.cuda.Event, torch.cuda.Event]:
        with torch.cuda.device(self.device):
            started = time.perf_counter()
            cpu_batch = next(self.iterator)
            self.data_wait_seconds += time.perf_counter() - started
            with self._adapter_lock, torch.cuda.stream(self.stream):
                stage_start = torch.cuda.Event(enable_timing=True)
                stage_end = torch.cuda.Event(enable_timing=True)
                stage_start.record(self.stream)
                materialized = self.adapter.materialize(cpu_batch)
                batch = self.transfer(materialized, self.device)
                stage_end.record(self.stream)
            return batch, stage_start, stage_end

    def submit(self) -> None:
        if self._future is not None:
            raise RuntimeError("async input pipeline already has a pending batch")
        self._future = self._executor.submit(self._prepare_one)
        self.submitted += 1

    def consume(self) -> dict[str, Any]:
        future = self._future
        if future is None:
            raise RuntimeError("async input pipeline has no pending batch")
        self._future = None
        started = time.perf_counter()
        batch, stage_start, stage_end = future.result()
        stage_end.synchronize()
        self.consume_wait_seconds += time.perf_counter() - started
        self.cuda_stage_seconds += stage_start.elapsed_time(stage_end) / 1000.0
        current = torch.cuda.current_stream(self.device)
        current.wait_event(stage_end)
        _record_stream(batch, current)
        self.consumed += 1
        return batch

    def materialize(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        """Serialized synchronous adapter entry used by validation boundaries."""

        if self._future is not None:
            raise RuntimeError(
                "validation cannot share an adapter with a pending async batch"
            )
        with self._adapter_lock:
            return self.adapter.materialize(batch)

    def close(self) -> None:
        if self._future is not None:
            try:
                batch, _stage_start, stage_end = self._future.result()
                stage_end.synchronize()
                del batch
            except Exception:
                # Teardown must not hide the training exception that caused it.
                pass
            finally:
                self._future = None
        self._executor.shutdown(wait=True, cancel_futures=True)
