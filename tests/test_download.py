from __future__ import annotations

import runpy

import pytest


def test_network_retry_recovers_and_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = runpy.run_path("scripts/data/download_raw_snapshots.py")
    calls = 0
    delays: list[float] = []

    def flaky() -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise module["httpx"].ConnectError("temporary network failure")
        return "ok"

    monkeypatch.setattr(module["time"], "sleep", delays.append)
    assert module["_retry_network"](
        flaky,
        label="test",
        attempts=4,
        initial_delay=0.5,
    ) == "ok"
    assert calls == 3
    assert delays == [0.5, 1.0]

    calls = 0
    with pytest.raises(module["httpx"].ConnectError):
        module["_retry_network"](
            flaky,
            label="test",
            attempts=2,
            initial_delay=0.5,
        )
    assert calls == 2


def test_network_retry_does_not_hide_programming_errors() -> None:
    module = runpy.run_path("scripts/data/download_raw_snapshots.py")

    def invalid() -> None:
        raise ValueError("invalid source contract")

    with pytest.raises(ValueError, match="invalid source contract"):
        module["_retry_network"](
            invalid, label="test", attempts=8, initial_delay=0
        )
