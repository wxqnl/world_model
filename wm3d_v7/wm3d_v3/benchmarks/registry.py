"""External benchmark availability registry.

This module intentionally separates "benchmark interface exists" from
"benchmark is installed and runnable". The system report should not imply that
LIBERO, CALVIN, SimplerEnv, or WorldArena have been evaluated until the relevant
package/entrypoint is present and a run writes metrics.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class BenchmarkProbe:
    name: str
    status: str
    reason: str
    import_names: list[str]
    env_hints: list[str]
    root_hints: list[str]
    purpose: str


BENCHMARKS: dict[str, dict[str, object]] = {
    "libero": {
        "import_names": ["libero", "libero.libero"],
        "env_hints": ["LIBERO_ROOT"],
        "root_hints": ["/data/Minko/benchmarks/LIBERO"],
        "purpose": "language-conditioned manipulation success-rate benchmark",
    },
    "calvin": {
        "import_names": ["calvin_agent", "calvin_env", "calvin_models"],
        "env_hints": ["CALVIN_ROOT"],
        "root_hints": [],
        "purpose": "long-horizon language-conditioned manipulation benchmark",
    },
    "simpler_env": {
        "import_names": ["simpler_env", "simpler_env.main_inference"],
        "env_hints": ["SIMPLERENV_ROOT", "SIMPLER_ENV_ROOT"],
        "root_hints": [],
        "purpose": "simulation benchmark for real-robot policy generalization",
    },
    "worldarena": {
        "import_names": ["worldarena", "world_arena"],
        "env_hints": ["WORLD_ARENA_ROOT", "WORLDARENA_ROOT"],
        "root_hints": [],
        "purpose": "embodied world-model functional utility benchmark",
    },
}


def _first_importable(names: list[str]) -> str | None:
    for name in names:
        try:
            if importlib.util.find_spec(name) is not None:
                return name
        except (ImportError, ModuleNotFoundError, ValueError):
            continue
    return None


def _existing_env_paths(names: list[str]) -> list[str]:
    found: list[str] = []
    for name in names:
        value = os.environ.get(name)
        if value and Path(value).exists():
            found.append(f"{name}={value}")
    return found


def _existing_root_hints(paths: list[str]) -> list[str]:
    return [str(Path(path)) for path in paths if Path(path).exists()]


def probe_benchmarks() -> dict[str, BenchmarkProbe]:
    probes: dict[str, BenchmarkProbe] = {}
    for name, spec in BENCHMARKS.items():
        import_names = list(spec["import_names"])  # type: ignore[index]
        env_hints = list(spec["env_hints"])  # type: ignore[index]
        root_hints = list(spec.get("root_hints", []))  # type: ignore[union-attr]
        purpose = str(spec["purpose"])
        imported = _first_importable(import_names)
        env_paths = _existing_env_paths(env_hints)
        root_paths = _existing_root_hints(root_hints)
        if imported is not None:
            probes[name] = BenchmarkProbe(
                name=name,
                status="available",
                reason=f"importable module: {imported}",
                import_names=import_names,
                env_hints=env_hints,
                root_hints=root_hints,
                purpose=purpose,
            )
        elif env_paths or root_paths:
            reason_parts = []
            if env_paths:
                reason_parts.append(f"configured root path exists: {', '.join(env_paths)}")
            if root_paths:
                reason_parts.append(f"default source checkout exists: {', '.join(root_paths)}")
            probes[name] = BenchmarkProbe(
                name=name,
                status="partial",
                reason="; ".join(reason_parts) + "; Python package/runtime not fully importable",
                import_names=import_names,
                env_hints=env_hints,
                root_hints=root_hints,
                purpose=purpose,
            )
        else:
            probes[name] = BenchmarkProbe(
                name=name,
                status="unavailable",
                reason="no importable package or configured root path found",
                import_names=import_names,
                env_hints=env_hints,
                root_hints=root_hints,
                purpose=purpose,
            )
    return probes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    report = {name: asdict(probe) for name, probe in probe_benchmarks().items()}
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"wrote {args.out}")
    print(text)


if __name__ == "__main__":
    main()
