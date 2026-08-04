#!/usr/bin/env python3
"""Distributed fail-closed preflight for a materialized WM3D run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import resource
import socket
import subprocess
import time
from typing import Any

import torch
import torch.distributed as dist
import yaml

from wm3d.data.contracts import (
    atomic_write_json,
    canonical_sha256,
    verify_dataset_seal,
)
from wm3d.training.config import (
    TRAIN_CONFIG_SCHEMA,
    training_contract_sha256,
    verify_code_receipt,
)
from wm3d.training.environment import (
    load_environment_contract,
    verify_environment_receipt,
)
from wm3d.training.runtime import (
    destroy_distributed,
    initialize_distributed,
)


PREFLIGHT_SCHEMA = "wm3d_v7_cluster_preflight_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def _command(*command: str) -> str:
    result = subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def _strict_integer_field(value: str, name: str) -> int:
    cleaned = value.strip()
    if not cleaned.isdigit():
        raise RuntimeError(f"nvidia-smi {name} is unavailable/non-integer: {value!r}")
    return int(cleaned)


def _gpu_report(local_rank: int) -> dict[str, Any]:
    query = ",".join(
        (
            "name",
            "uuid",
            "memory.total",
            "ecc.errors.uncorrected.volatile.total",
            "ecc.errors.uncorrected.aggregate.total",
            "driver_version",
        )
    )
    values = [
        value.strip()
        for value in _command(
            "nvidia-smi",
            f"--id={local_rank}",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ).split(",")
    ]
    if len(values) != 6:
        raise RuntimeError(f"unexpected nvidia-smi result: {values}")
    compute = _command(
        "nvidia-smi",
        f"--id={local_rank}",
        "--query-compute-apps=pid",
        "--format=csv,noheader,nounits",
    )
    pids = sorted(
        int(line.strip()) for line in compute.splitlines() if line.strip().isdigit()
    )
    return {
        "name": values[0],
        "uuid": values[1],
        "memory_total_mib": _strict_integer_field(values[2], "memory.total"),
        "uncorrected_volatile": _strict_integer_field(
            values[3],
            "ecc.errors.uncorrected.volatile.total",
        ),
        "uncorrected_aggregate": _strict_integer_field(
            values[4],
            "ecc.errors.uncorrected.aggregate.total",
        ),
        "driver_version": values[5],
        "compute_pids": pids,
    }


def _ib_report() -> list[dict[str, Any]]:
    root = Path("/sys/class/infiniband")
    result = []
    if root.is_dir():
        for device in sorted(root.iterdir()):
            for port in sorted((device / "ports").iterdir()):
                result.append(
                    {
                        "device": device.name,
                        "port": port.name,
                        "state": (port / "state").read_text().strip(),
                        "rate": (port / "rate").read_text().strip(),
                    }
                )
    return result


def _parse_ib_rate_gbps(value: str) -> float:
    match = re.fullmatch(
        r"\s*([0-9]+(?:\.[0-9]+)?)\s+Gb/sec(?:\s+\(.*\))?\s*",
        str(value),
    )
    if match is None:
        raise RuntimeError(f"unrecognized InfiniBand link rate: {value!r}")
    rate = float(match.group(1))
    if rate <= 0:
        raise RuntimeError(f"non-positive InfiniBand link rate: {value!r}")
    return rate


def _parse_nvlink_topology(
    output: str,
    expected_gpus: int = 8,
) -> dict[str, Any]:
    """Parse ``nvidia-smi topo -m`` into a strict eight-GPU clique."""

    rows: dict[int, list[str]] = {}
    for line in output.splitlines():
        fields = line.split()
        if not fields or re.fullmatch(r"GPU[0-9]+", fields[0]) is None:
            continue
        gpu_id = int(fields[0][3:])
        candidate = fields[1 : 1 + expected_gpus]
        # The header itself starts with GPU0 on some nvidia-smi versions.
        if len(candidate) != expected_gpus or candidate.count("X") != 1:
            continue
        rows[gpu_id] = candidate
    if set(rows) != set(range(expected_gpus)):
        raise RuntimeError(
            f"topology GPU set mismatch for {expected_gpus} devices: {sorted(rows)}"
        )
    bad_links = []
    for source, links in sorted(rows.items()):
        for target, link in enumerate(links):
            if source == target:
                if link != "X":
                    bad_links.append(f"GPU{source}->GPU{target}={link}")
            elif not link.startswith("NV"):
                bad_links.append(f"GPU{source}->GPU{target}={link}")
    if bad_links:
        raise RuntimeError(
            "eight-GPU NVLink clique is incomplete: " + ", ".join(bad_links)
        )
    matrix = {f"GPU{index}": links for index, links in sorted(rows.items())}
    return {
        "pass": True,
        "matrix": matrix,
        "matrix_sha256": canonical_sha256(matrix),
    }


def _nvlink_report(expected_gpus: int) -> dict[str, Any]:
    """Require a complete eight-GPU NVLink/NVSwitch clique on each host."""

    return _parse_nvlink_topology(
        _command("nvidia-smi", "topo", "-m"),
        expected_gpus=expected_gpus,
    )


def _available_bytes(path: Path) -> int:
    value = os.statvfs(path)
    return int(value.f_bavail) * int(value.f_frsize)


def _all_reduce_gbps(device: torch.device, megabytes: int = 64) -> float:
    elements = megabytes * 1024 * 1024 // 4
    tensor = torch.ones(elements, dtype=torch.float32, device=device)
    for _ in range(2):
        dist.all_reduce(tensor)
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    iterations = 4
    for _ in range(iterations):
        dist.all_reduce(tensor)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    algorithm_bytes = tensor.numel() * tensor.element_size() * iterations
    return algorithm_bytes / elapsed / 1.0e9


def _local_report(config: dict[str, Any], context: Any) -> dict[str, Any]:
    errors: list[str] = []
    environment: dict[str, Any] = {}
    gpu: dict[str, Any] = {}
    nvlink: dict[str, Any] = {}
    ib: list[dict[str, Any]] = []
    soft_memlock = -1
    soft_nofile = -1
    shm_free = -1
    dataset_free = -1
    output_free = -1
    try:
        environment_contract = load_environment_contract(
            Path(config["run"]["environment_contract_path"])
        )
        environment_contract_sha = canonical_sha256(environment_contract)
        if environment_contract_sha != str(
            config["run"]["environment_contract_sha256"]
        ):
            raise RuntimeError(
                "configured environment contract SHA does not match file"
            )
        environment_receipt = verify_environment_receipt(
            Path(config["run"]["environment_receipt_path"]),
            expected_sha256=str(config["run"]["environment_receipt_sha256"]),
            contract_path=Path(config["run"]["environment_contract_path"]),
            check_current=True,
        )
        environment = {
            "contract_sha256": environment_contract_sha,
            "receipt_sha256": canonical_sha256(environment_receipt),
            "fingerprint_sha256": environment_receipt["environment"][
                "fingerprint_sha256"
            ],
        }
    except Exception as exc:
        errors.append(f"environment qualification failed: {exc}")
    hardware = config["hardware"]
    expected_gpus = int(hardware["gpus_per_node"])
    try:
        visible_gpus = torch.cuda.device_count()
        if visible_gpus != expected_gpus:
            errors.append(
                f"visible GPU count {visible_gpus} != configured {expected_gpus}"
            )
        gpu = _gpu_report(context.local_rank)
        required_name = str(hardware["accelerator_name_contains"]).upper()
        if required_name and required_name not in gpu["name"].upper():
            errors.append(
                f"GPU name {gpu['name']!r} does not contain {required_name!r}"
            )
        minimum_memory = int(hardware["minimum_accelerator_memory_mib"])
        if gpu["memory_total_mib"] < minimum_memory:
            errors.append(
                f"GPU HBM {gpu['memory_total_mib']} MiB below {minimum_memory} MiB"
            )
        if gpu["uncorrected_volatile"] or gpu["uncorrected_aggregate"]:
            errors.append(f"uncorrected ECC is non-zero: {gpu}")
        external_pids = [pid for pid in gpu["compute_pids"] if pid != os.getpid()]
        if external_pids:
            errors.append(f"GPU has external compute PIDs {external_pids}")
    except Exception as exc:
        errors.append(f"GPU qualification failed: {exc}")
    if bool(hardware["require_nvlink_clique"]):
        try:
            nvlink = _nvlink_report(expected_gpus)
        except Exception as exc:
            errors.append(f"NVLink topology qualification failed: {exc}")
    else:
        nvlink = {"pass": True, "skipped": True}
    try:
        ib = _ib_report()
        active_ib = [item for item in ib if "ACTIVE" in str(item["state"]).upper()]
        if not active_ib:
            errors.append("no active InfiniBand port")
        else:
            minimum_ib_rate = float(config["distributed"]["minimum_ib_rate_gbps"])
            active_rates: list[float] = []
            for item in active_ib:
                try:
                    active_rates.append(_parse_ib_rate_gbps(item["rate"]))
                except RuntimeError as exc:
                    errors.append(
                        f"unable to verify InfiniBand rate for "
                        f"{item['device']} port {item['port']}: {exc}"
                    )
            if active_rates and max(active_rates) < minimum_ib_rate:
                errors.append(
                    "no active InfiniBand port meets minimum rate "
                    f"{minimum_ib_rate:g} Gb/s: {active_rates}"
                )
    except Exception as exc:
        errors.append(f"InfiniBand qualification failed: {exc}")
    if os.environ.get("NCCL_IB_DISABLE", "0") == "1":
        errors.append("NCCL_IB_DISABLE=1 is forbidden")
    try:
        soft_memlock, _hard_memlock = resource.getrlimit(resource.RLIMIT_MEMLOCK)
        soft_nofile, _hard_nofile = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft_memlock != resource.RLIM_INFINITY and soft_memlock < 1 << 30:
            errors.append(f"memlock ulimit is too low: {soft_memlock}")
        if soft_nofile < 1_048_576:
            errors.append(f"nofile ulimit is below 1048576: {soft_nofile}")
    except Exception as exc:
        errors.append(f"ulimit qualification failed: {exc}")
    try:
        shm_free = _available_bytes(Path("/dev/shm"))
        if shm_free < int(config["distributed"]["minimum_shm_bytes"]):
            errors.append(f"/dev/shm free bytes too low: {shm_free}")
    except Exception as exc:
        errors.append(f"/dev/shm qualification failed: {exc}")
    try:
        data_root = Path(config["data"]["root"])
        if not data_root.is_dir():
            errors.append(f"dataset root does not exist: {data_root}")
            dataset_free = 0
        else:
            dataset_free = _available_bytes(data_root)
            minimum_data_free = int(config["data"]["minimum_free_bytes"])
            if dataset_free < minimum_data_free:
                errors.append(f"dataset free bytes too low: {dataset_free}")
    except Exception as exc:
        errors.append(f"dataset-volume qualification failed: {exc}")
    try:
        output_parent = Path(config["run"]["output_root"]).parent
        if not output_parent.exists():
            errors.append(f"output parent does not exist: {output_parent}")
            output_free = 0
        else:
            output_free = _available_bytes(output_parent)
            minimum_free = int(config["distributed"]["minimum_output_free_bytes"])
            if output_free < minimum_free:
                errors.append(f"output free bytes too low: {output_free}")
    except Exception as exc:
        errors.append(f"output-volume qualification failed: {exc}")

    # Every surviving rank must reach this collective, even when a preceding
    # node-local probe failed.  Otherwise one bad nvidia-smi/sysfs response can
    # strand the remaining ranks inside all_reduce.
    bandwidth = -1.0
    try:
        bandwidth = _all_reduce_gbps(context.device)
        if bandwidth < float(config["distributed"]["minimum_allreduce_gbps"]):
            errors.append(f"all-reduce throughput too low: {bandwidth:.3f} GB/s")
    except Exception as exc:
        errors.append(f"all-reduce qualification failed: {exc}")
    return {
        "rank": context.rank,
        "local_rank": context.local_rank,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "environment": environment,
        "gpu": gpu,
        "nvlink": nvlink,
        "ib": ib,
        "ulimit_memlock": soft_memlock,
        "ulimit_nofile": soft_nofile,
        "shm_free_bytes": shm_free,
        "dataset_free_bytes": dataset_free,
        "output_free_bytes": output_free,
        "allreduce_gbps": bandwidth,
        "errors": errors,
    }


def _aggregate_reports(
    *,
    config: dict[str, Any],
    context: Any,
    gathered: list[Any],
) -> dict[str, Any]:
    errors: list[str] = []
    host_to_ranks: dict[str, list[int]] = {}
    valid_reports: list[dict[str, Any]] = []
    reports_by_rank: dict[int, dict[str, Any]] = {}
    for index, value in enumerate(gathered):
        if not isinstance(value, dict):
            errors.append(f"rank slot {index} returned a malformed report")
            continue
        valid_reports.append(value)
        rank = int(value.get("rank", index))
        if rank in reports_by_rank:
            errors.append(f"duplicate rank identity {rank} in preflight reports")
        reports_by_rank[rank] = value
        for error in value.get("errors", ()):
            errors.append(f"rank {rank}: {error}")
        hostname = str(value.get("hostname", ""))
        if not hostname:
            errors.append(f"rank {rank}: hostname is missing")
        else:
            host_to_ranks.setdefault(hostname, []).append(rank)

    gpus_per_node = int(config["hardware"]["gpus_per_node"])
    if context.world_size % gpus_per_node:
        errors.append("world size is not divisible by configured GPUs per node")
        expected_nodes = -1
    else:
        expected_nodes = context.world_size // gpus_per_node
    if len(host_to_ranks) != expected_nodes:
        errors.append(f"host count {len(host_to_ranks)} != expected {expected_nodes}")
    for hostname, ranks in host_to_ranks.items():
        if len(ranks) != gpus_per_node:
            errors.append(
                f"{hostname} has {len(ranks)} ranks, not {gpus_per_node}"
            )

    uuids = [str(report.get("gpu", {}).get("uuid", "")) for report in valid_reports]
    if any(not value for value in uuids):
        errors.append("one or more GPU UUIDs are missing")
    elif len(set(uuids)) != len(uuids):
        errors.append("GPU UUIDs are not globally unique")

    environment_fingerprints = {
        str(report.get("environment", {}).get("fingerprint_sha256", ""))
        for report in valid_reports
        if report.get("environment", {}).get("fingerprint_sha256")
    }
    if len(environment_fingerprints) != 1:
        errors.append("software environment fingerprints differ across ranks")

    if bool(config["hardware"]["require_nvlink_clique"]):
        for hostname, ranks in host_to_ranks.items():
            topology_digests = {
                str(
                    reports_by_rank.get(rank, {})
                    .get("nvlink", {})
                    .get("matrix_sha256", "")
                )
                for rank in ranks
            }
            if len(topology_digests) != 1 or "" in topology_digests:
                errors.append(f"{hostname}: NVLink topology reports disagree")

    dataset_receipt_sha: str | None = None
    try:
        data_root = Path(config["data"]["root"]).resolve(strict=True)
        dataset_report = verify_dataset_seal(
            data_root,
            str(config["data"]["seal_receipt"]),
        )
        dataset_receipt_sha = dataset_report.get("receipt_sha256")
        if not dataset_report["pass"]:
            errors.extend(f"dataset: {error}" for error in dataset_report["errors"])
    except Exception as exc:
        errors.append(f"dataset seal qualification failed: {exc}")

    code_receipt_sha: str | None = None
    try:
        code_receipt = verify_code_receipt(
            Path(config["run"]["code_receipt_path"]),
            expected_sha256=str(config["run"]["code_receipt_sha256"]),
            repo_root=Path(__file__).resolve().parents[2],
        )
        code_receipt_sha = canonical_sha256(code_receipt)
    except Exception as exc:
        errors.append(f"code receipt qualification failed: {exc}")

    contract_sha: str | None = None
    try:
        contract_sha = training_contract_sha256(config)
        if contract_sha != config["run"]["training_contract_sha256"]:
            errors.append("training contract SHA mismatch")
    except Exception as exc:
        errors.append(f"training contract qualification failed: {exc}")

    return {
        "schema": PREFLIGHT_SCHEMA,
        "pass": not errors,
        "errors": errors,
        "world_size": context.world_size,
        "nodes": host_to_ranks,
        "training_contract_sha256": contract_sha,
        "dataset_receipt_sha256": dataset_receipt_sha,
        "code_receipt_sha256": code_receipt_sha,
        "environment_receipt_sha256": config["run"].get("environment_receipt_sha256"),
        "environment_fingerprint_sha256": next(iter(environment_fingerprints), None),
        "ranks": gathered,
    }


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve(strict=True)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema") != TRAIN_CONFIG_SCHEMA:
        raise ValueError("preflight config schema mismatch")
    context = initialize_distributed()
    try:
        local = _local_report(config, context)
        expected_world_size = int(config["distributed"]["expected_world_size"])
        if context.world_size != expected_world_size:
            local["errors"].append(
                f"world size {context.world_size} != {expected_world_size}"
            )
        gathered: list[Any] = [None] * context.world_size
        dist.all_gather_object(gathered, local)
        final: list[Any] = [None]
        if context.is_rank0:
            try:
                final[0] = _aggregate_reports(
                    config=config,
                    context=context,
                    gathered=gathered,
                )
            except Exception as exc:
                final[0] = {
                    "schema": PREFLIGHT_SCHEMA,
                    "pass": False,
                    "errors": [f"rank0 report aggregation failed: {exc}"],
                    "world_size": context.world_size,
                    "ranks": gathered,
                }
            try:
                report_path = args.report.resolve()
                report_path.parent.mkdir(parents=True, exist_ok=True)
                atomic_write_json(report_path, final[0], exclusive=True)
            except Exception as exc:
                final[0]["pass"] = False
                final[0].setdefault("errors", []).append(
                    f"preflight report publication failed: {exc}"
                )
        dist.broadcast_object_list(final, src=0)
        report = final[0]
        if not isinstance(report, dict):
            raise RuntimeError("cluster preflight returned no final report")
        if not report.get("pass"):
            raise RuntimeError(
                "cluster preflight failed:\n"
                + "\n".join(str(error) for error in report.get("errors", ()))
            )
        if context.is_rank0:
            print(json.dumps(report, sort_keys=True))
    finally:
        destroy_distributed()


if __name__ == "__main__":
    main()
