"""Distributed resource qualification shared by every WM3D V8 model profile."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import socket
import subprocess
import time
from typing import Any, Mapping

import torch
import torch.distributed as dist


RESOURCE_PREFLIGHT_SCHEMA = "wm3d_v8_resource_preflight_v1"


class ResourcePreflightError(RuntimeError):
    pass


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _command(*command: str) -> str:
    return subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()


def _strict_integer(value: str, field: str) -> int:
    text = value.strip()
    if not text.isdigit():
        raise ResourcePreflightError(
            f"nvidia-smi {field} is unavailable or non-integer: {value!r}"
        )
    return int(text)


def _visible_gpu_identifier(local_rank: int) -> str:
    if not 0 <= int(local_rank) < int(torch.cuda.device_count()):
        raise ResourcePreflightError(
            f"local CUDA rank {local_rank} is outside the visible device set"
        )
    raw = str(torch.cuda.get_device_properties(int(local_rank)).uuid).strip()
    if not raw:
        raise ResourcePreflightError("PyTorch returned an empty CUDA device UUID")
    return raw if raw.startswith(("GPU-", "MIG-")) else f"GPU-{raw}"


def _physical_gpu_index(identifier: str) -> int:
    return _strict_integer(
        _command(
            "nvidia-smi",
            f"--id={identifier}",
            "--query-gpu=index",
            "--format=csv,noheader,nounits",
        ),
        "index",
    )


def _gpu_report(local_rank: int) -> dict[str, Any]:
    identifier = _visible_gpu_identifier(local_rank)
    fields = (
        "index",
        "name",
        "uuid",
        "memory.total",
        "ecc.errors.uncorrected.volatile.total",
        "ecc.errors.uncorrected.aggregate.total",
        "driver_version",
    )
    values = [
        value.strip()
        for value in _command(
            "nvidia-smi",
            f"--id={identifier}",
            f"--query-gpu={','.join(fields)}",
            "--format=csv,noheader,nounits",
        ).split(",")
    ]
    if len(values) != len(fields):
        raise ResourcePreflightError(f"unexpected nvidia-smi result: {values}")
    if values[2] != identifier:
        raise ResourcePreflightError(
            f"PyTorch/nvidia-smi GPU UUID mismatch: {identifier} != {values[2]}"
        )
    compute = _command(
        "nvidia-smi",
        f"--id={identifier}",
        "--query-compute-apps=pid",
        "--format=csv,noheader,nounits",
    )
    return {
        "physical_index": _strict_integer(values[0], "index"),
        "name": values[1],
        "uuid": values[2],
        "memory_total_mib": _strict_integer(values[3], "memory.total"),
        "uncorrected_volatile": _strict_integer(
            values[4], "ecc.errors.uncorrected.volatile.total"
        ),
        "uncorrected_aggregate": _strict_integer(
            values[5], "ecc.errors.uncorrected.aggregate.total"
        ),
        "driver_version": values[6],
        "compute_pids": sorted(
            int(line.strip())
            for line in compute.splitlines()
            if line.strip().isdigit() and int(line.strip()) != os.getpid()
        ),
    }


def _ib_report() -> list[dict[str, str]]:
    root = Path("/sys/class/infiniband")
    result: list[dict[str, str]] = []
    if not root.is_dir():
        return result
    for device in sorted(root.iterdir()):
        ports = device / "ports"
        if not ports.is_dir():
            continue
        for port in sorted(ports.iterdir()):
            result.append(
                {
                    "device": device.name,
                    "port": port.name,
                    "state": (port / "state").read_text(encoding="utf-8").strip(),
                    "rate": (port / "rate").read_text(encoding="utf-8").strip(),
                }
            )
    return result


def _parse_ib_rate_gbps(value: str) -> float:
    match = re.fullmatch(
        r"\s*([0-9]+(?:\.[0-9]+)?)\s+Gb/sec(?:\s+\(.*\))?\s*", str(value)
    )
    if match is None or float(match.group(1)) <= 0:
        raise ResourcePreflightError(f"unrecognized InfiniBand rate: {value!r}")
    return float(match.group(1))


def _parse_nvlink_topology(
    output: str,
    *,
    local_world_size: int | None = None,
    physical_gpu_indices: list[int] | None = None,
) -> dict[str, Any]:
    output = re.sub(r"\x1b\[[0-9;]*m", "", output)
    if physical_gpu_indices is None:
        if local_world_size is None:
            raise ResourcePreflightError("NVLink topology requires a local GPU set")
        physical_gpu_indices = list(range(int(local_world_size)))
    selected = [int(value) for value in physical_gpu_indices]
    if not selected or len(selected) != len(set(selected)):
        raise ResourcePreflightError("visible physical GPU indices are empty or duplicated")
    header: list[int] | None = None
    for line in output.splitlines():
        fields = line.split()
        leading = []
        for field in fields:
            if re.fullmatch(r"GPU[0-9]+", field) is None:
                break
            leading.append(int(field[3:]))
        if len(leading) >= len(selected) and "X" not in fields:
            header = leading
            break
    if header is None or not set(selected).issubset(header):
        raise ResourcePreflightError(
            f"NVLink topology header lacks selected GPUs: {selected}"
        )
    all_rows: dict[int, dict[int, str]] = {}
    for line in output.splitlines():
        fields = line.split()
        if not fields or re.fullmatch(r"GPU[0-9]+", fields[0]) is None:
            continue
        gpu_id = int(fields[0][3:])
        candidate = fields[1 : 1 + len(header)]
        if len(candidate) == len(header) and candidate.count("X") == 1:
            all_rows[gpu_id] = dict(zip(header, candidate))
    if not set(selected).issubset(all_rows):
        raise ResourcePreflightError(
            "NVLink topology does not contain the exact local GPU set: "
            f"expected={selected} actual={sorted(all_rows)}"
        )
    bad: list[str] = []
    for source in selected:
        for target in selected:
            link = all_rows[source][target]
            if source == target and link != "X":
                bad.append(f"GPU{source}->GPU{target}={link}")
            elif source != target and not link.startswith("NV"):
                bad.append(f"GPU{source}->GPU{target}={link}")
    if bad:
        raise ResourcePreflightError(
            "local GPU NVLink clique is incomplete: " + ", ".join(bad)
        )
    matrix = {
        f"GPU{source}": [all_rows[source][target] for target in selected]
        for source in selected
    }
    return {"matrix": matrix, "matrix_sha256": _canonical_sha256(matrix)}


def _available_bytes(path: Path) -> int:
    value = os.statvfs(path)
    return int(value.f_bavail) * int(value.f_frsize)


def _all_reduce_gbps(device: torch.device, *, megabytes: int = 64) -> float:
    tensor = torch.ones(
        megabytes * 1024 * 1024 // 4, dtype=torch.float32, device=device
    )
    for _ in range(2):
        dist.all_reduce(tensor)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    iterations = 4
    for _ in range(iterations):
        dist.all_reduce(tensor)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    algorithm_bytes = tensor.numel() * tensor.element_size() * iterations
    return algorithm_bytes / elapsed / 1.0e9


def _local_report(
    *,
    resources: Mapping[str, Any],
    rank: int,
    local_rank: int,
    local_world_size: int,
    device: torch.device,
    cache_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    gpu: dict[str, Any] = {}
    ib: list[dict[str, str]] = []
    nvlink: dict[str, Any] = {}
    shm_free = data_free = output_free = -1
    memlock = nofile = -1

    try:
        gpu = _gpu_report(local_rank)
        required_name = str(resources["gpu_name_substring"]).upper()
        if required_name not in str(gpu["name"]).upper():
            errors.append(f"GPU name does not contain {required_name!r}: {gpu['name']}")
        if int(gpu["memory_total_mib"]) < int(resources["minimum_gpu_memory_mib"]):
            errors.append(f"GPU memory is below minimum: {gpu['memory_total_mib']} MiB")
        if bool(resources["require_zero_uncorrected_ecc"]) and (
            int(gpu["uncorrected_volatile"]) or int(gpu["uncorrected_aggregate"])
        ):
            errors.append(f"uncorrected ECC is non-zero: {gpu}")
        if bool(resources["require_idle_gpu"]) and gpu["compute_pids"]:
            errors.append(f"GPU has external compute PIDs: {gpu['compute_pids']}")
    except Exception as exc:
        errors.append(f"GPU qualification failed: {exc}")

    if bool(resources["require_full_local_nvlink_clique"]):
        try:
            if int(torch.cuda.device_count()) != int(local_world_size):
                raise ResourcePreflightError(
                    "visible CUDA device count differs from LOCAL_WORLD_SIZE: "
                    f"{torch.cuda.device_count()} != {local_world_size}"
                )
            physical_gpu_indices = [
                _physical_gpu_index(_visible_gpu_identifier(index))
                for index in range(int(local_world_size))
            ]
            nvlink = _parse_nvlink_topology(
                _command("nvidia-smi", "topo", "-m"),
                physical_gpu_indices=physical_gpu_indices,
            )
        except Exception as exc:
            errors.append(f"NVLink qualification failed: {exc}")

    try:
        ib = _ib_report()
        active = [item for item in ib if "ACTIVE" in item["state"].upper()]
        if not active:
            errors.append("no active InfiniBand port")
        else:
            rates = [_parse_ib_rate_gbps(item["rate"]) for item in active]
            if max(rates) < float(resources["minimum_ib_rate_gbps"]):
                errors.append(
                    "no active InfiniBand port meets minimum rate: "
                    f"required={resources['minimum_ib_rate_gbps']} actual={rates}"
                )
    except Exception as exc:
        errors.append(f"InfiniBand qualification failed: {exc}")
    if bool(resources["forbid_nccl_ib_disable"]) and os.environ.get(
        "NCCL_IB_DISABLE", "0"
    ) == "1":
        errors.append("NCCL_IB_DISABLE=1 is forbidden")

    try:
        memlock = int(resource.getrlimit(resource.RLIMIT_MEMLOCK)[0])
        nofile = int(resource.getrlimit(resource.RLIMIT_NOFILE)[0])
        if memlock != resource.RLIM_INFINITY and memlock < int(
            resources["minimum_memlock_bytes"]
        ):
            errors.append(f"memlock ulimit is too low: {memlock}")
        if nofile < int(resources["minimum_nofile"]):
            errors.append(f"nofile ulimit is too low: {nofile}")
    except Exception as exc:
        errors.append(f"ulimit qualification failed: {exc}")

    for label, path, minimum in (
        ("/dev/shm", Path("/dev/shm"), int(resources["minimum_shm_bytes"])),
        ("data", cache_root, int(resources["minimum_data_free_bytes"])),
        ("output", output_root.parent, int(resources["minimum_output_free_bytes"])),
    ):
        try:
            if not path.is_dir():
                raise ResourcePreflightError(f"directory does not exist: {path}")
            available = _available_bytes(path)
            if label == "/dev/shm":
                shm_free = available
            elif label == "data":
                data_free = available
            else:
                output_free = available
            if available < minimum:
                errors.append(
                    f"{label} free bytes below minimum: {available} < {minimum}"
                )
        except Exception as exc:
            errors.append(f"{label} volume qualification failed: {exc}")

    bandwidth = -1.0
    try:
        bandwidth = _all_reduce_gbps(device)
        if bandwidth < float(resources["minimum_allreduce_gbps"]):
            errors.append(f"all-reduce throughput too low: {bandwidth:.3f} GB/s")
    except Exception as exc:
        errors.append(f"all-reduce qualification failed: {exc}")

    return {
        "rank": rank,
        "local_rank": local_rank,
        "hostname": socket.gethostname(),
        "gpu": gpu,
        "nvlink": nvlink,
        "infiniband": ib,
        "memlock": memlock,
        "nofile": nofile,
        "shm_free_bytes": shm_free,
        "data_free_bytes": data_free,
        "output_free_bytes": output_free,
        "allreduce_gbps": bandwidth,
        "errors": errors,
    }


def run_resource_preflight(
    *,
    resources: Mapping[str, Any],
    context: Any,
    runtime_config_sha256: str,
    cache_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Run all local probes, gather every rank, and return one bound receipt."""

    local = _local_report(
        resources=resources,
        rank=int(context.rank),
        local_rank=int(context.local_rank),
        local_world_size=int(context.local_world_size),
        device=context.device,
        cache_root=Path(cache_root),
        output_root=Path(output_root),
    )
    gathered: list[Any] = [None] * int(context.world_size)
    dist.all_gather_object(gathered, local)
    errors: list[str] = []
    ranks: set[int] = set()
    host_slots: set[tuple[str, int]] = set()
    host_gpus: set[tuple[str, str]] = set()
    for slot, report in enumerate(gathered):
        if not isinstance(report, dict):
            errors.append(f"rank slot {slot} returned a malformed report")
            continue
        rank = int(report.get("rank", -1))
        if rank in ranks:
            errors.append(f"duplicate rank report: {rank}")
        ranks.add(rank)
        hostname = str(report.get("hostname", ""))
        local_rank = int(report.get("local_rank", -1))
        gpu = report.get("gpu")
        gpu_uuid = str(gpu.get("uuid", "")) if isinstance(gpu, dict) else ""
        if (hostname, local_rank) in host_slots:
            errors.append(f"duplicate host/local rank report: {hostname}/{local_rank}")
        host_slots.add((hostname, local_rank))
        if not gpu_uuid:
            errors.append(f"rank {rank}: empty GPU UUID")
        elif (hostname, gpu_uuid) in host_gpus:
            errors.append(f"duplicate host/GPU UUID report: {hostname}/{gpu_uuid}")
        host_gpus.add((hostname, gpu_uuid))
        errors.extend(f"rank {rank}: {message}" for message in report.get("errors", ()))
    if ranks != set(range(int(context.world_size))):
        errors.append(f"rank report closure mismatch: {sorted(ranks)}")
    hostnames = sorted(
        {
            str(report.get("hostname", ""))
            for report in gathered
            if isinstance(report, dict)
        }
    )
    if any(not hostname for hostname in hostnames):
        errors.append("resource reports contain an empty hostname")
    if len(hostnames) * int(context.local_world_size) != int(context.world_size):
        errors.append(
            "host/local-world closure mismatch: "
            f"hosts={hostnames} local_world_size={context.local_world_size} "
            f"world_size={context.world_size}"
        )
    created_unix_ns = time.time_ns()
    created_values: list[Any] = [created_unix_ns if int(context.rank) == 0 else None]
    dist.broadcast_object_list(created_values, src=0)
    created_unix_ns = int(created_values[0])
    receipt = {
        "schema": RESOURCE_PREFLIGHT_SCHEMA,
        "created_unix_ns": created_unix_ns,
        "runtime_config_sha256": runtime_config_sha256,
        "world_size": int(context.world_size),
        "local_world_size": int(context.local_world_size),
        "hostnames": hostnames,
        "resource_contract_sha256": _canonical_sha256(dict(resources)),
        "reports": gathered,
        "passed": not errors,
        "errors": errors,
    }
    return receipt


def validate_resource_receipt(
    receipt: Mapping[str, Any],
    *,
    resources: Mapping[str, Any],
    runtime_config_sha256: str,
    world_size: int,
    now_unix_ns: int | None = None,
) -> int:
    """Validate one persisted receipt and return its creation timestamp."""

    required = {
        "schema",
        "created_unix_ns",
        "runtime_config_sha256",
        "world_size",
        "local_world_size",
        "hostnames",
        "resource_contract_sha256",
        "reports",
        "passed",
        "errors",
    }
    if not isinstance(receipt, dict) or set(receipt) != required:
        raise ResourcePreflightError("resource receipt fields mismatch")
    if receipt.get("schema") != RESOURCE_PREFLIGHT_SCHEMA:
        raise ResourcePreflightError("resource receipt schema mismatch")
    if receipt.get("passed") is not True or receipt.get("errors") != []:
        raise ResourcePreflightError("resource receipt is not a clean pass")
    if receipt.get("runtime_config_sha256") != runtime_config_sha256:
        raise ResourcePreflightError("resource receipt runtime SHA mismatch")
    if receipt.get("resource_contract_sha256") != _canonical_sha256(dict(resources)):
        raise ResourcePreflightError("resource receipt contract SHA mismatch")
    if isinstance(receipt.get("world_size"), bool) or int(
        receipt.get("world_size", -1)
    ) != int(world_size):
        raise ResourcePreflightError("resource receipt world size mismatch")
    local_world_size = receipt.get("local_world_size")
    if isinstance(local_world_size, bool) or int(local_world_size) <= 0:
        raise ResourcePreflightError("resource receipt local world size is invalid")
    hostnames = receipt.get("hostnames")
    reports = receipt.get("reports")
    if (
        not isinstance(hostnames, list)
        or not hostnames
        or len(hostnames) != len(set(str(value) for value in hostnames))
        or any(not isinstance(value, str) or not value for value in hostnames)
    ):
        raise ResourcePreflightError("resource receipt host closure is invalid")
    if len(hostnames) * int(local_world_size) != int(world_size):
        raise ResourcePreflightError("resource receipt host/local-world closure mismatch")
    if not isinstance(reports, list) or len(reports) != int(world_size):
        raise ResourcePreflightError("resource receipt rank report count mismatch")
    ranks: set[int] = set()
    report_hosts: set[str] = set()
    host_slots: set[tuple[str, int]] = set()
    host_gpus: set[tuple[str, str]] = set()
    for report in reports:
        if not isinstance(report, dict) or report.get("errors") != []:
            raise ResourcePreflightError("resource receipt contains a failed rank report")
        _validate_persisted_rank_report(report, resources=resources)
        rank = report.get("rank")
        local_rank = report.get("local_rank")
        if (
            isinstance(rank, bool)
            or isinstance(local_rank, bool)
            or not 0 <= int(rank) < int(world_size)
            or not 0 <= int(local_rank) < int(local_world_size)
        ):
            raise ResourcePreflightError("resource receipt rank identity is invalid")
        ranks.add(int(rank))
        hostname = str(report.get("hostname", ""))
        gpu = report.get("gpu")
        gpu_uuid = str(gpu.get("uuid", "")) if isinstance(gpu, dict) else ""
        if not gpu_uuid:
            raise ResourcePreflightError("resource receipt GPU UUID is invalid")
        slot = (hostname, int(local_rank))
        gpu_slot = (hostname, gpu_uuid)
        if slot in host_slots or gpu_slot in host_gpus:
            raise ResourcePreflightError("resource receipt rank/GPU identity is duplicated")
        host_slots.add(slot)
        host_gpus.add(gpu_slot)
        report_hosts.add(hostname)
    if ranks != set(range(int(world_size))) or report_hosts != set(hostnames):
        raise ResourcePreflightError("resource receipt rank/host closure mismatch")
    created_ns = receipt.get("created_unix_ns")
    if isinstance(created_ns, bool) or int(created_ns) <= 0:
        raise ResourcePreflightError("resource receipt timestamp is invalid")
    now_ns = time.time_ns() if now_unix_ns is None else int(now_unix_ns)
    age_ns = now_ns - int(created_ns)
    maximum_age_ns = int(resources["maximum_preflight_age_seconds"]) * 1_000_000_000
    if age_ns < 0 or age_ns > maximum_age_ns:
        raise ResourcePreflightError("resource receipt is stale")
    return int(created_ns)


def _exact_integer(value: object, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ResourcePreflightError(f"resource receipt {field} is invalid")
    return value


def _exact_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ResourcePreflightError(f"resource receipt {field} is invalid")
    result = float(value)
    if not math.isfinite(result):
        raise ResourcePreflightError(f"resource receipt {field} is non-finite")
    return result


def _validate_persisted_rank_report(
    report: Mapping[str, Any], *, resources: Mapping[str, Any]
) -> None:
    expected = {
        "rank",
        "local_rank",
        "hostname",
        "gpu",
        "nvlink",
        "infiniband",
        "memlock",
        "nofile",
        "shm_free_bytes",
        "data_free_bytes",
        "output_free_bytes",
        "allreduce_gbps",
        "errors",
    }
    if set(report) != expected:
        raise ResourcePreflightError("resource receipt rank report fields mismatch")
    if not isinstance(report.get("hostname"), str) or not report["hostname"]:
        raise ResourcePreflightError("resource receipt rank hostname is invalid")
    _exact_integer(report.get("rank"), "rank")
    _exact_integer(report.get("local_rank"), "local_rank")

    gpu = report.get("gpu")
    expected_gpu = {
        "physical_index",
        "name",
        "uuid",
        "memory_total_mib",
        "uncorrected_volatile",
        "uncorrected_aggregate",
        "driver_version",
        "compute_pids",
    }
    if not isinstance(gpu, dict) or set(gpu) != expected_gpu:
        raise ResourcePreflightError("resource receipt GPU report fields mismatch")
    _exact_integer(gpu.get("physical_index"), "GPU physical index")
    if (
        not isinstance(gpu.get("name"), str)
        or str(resources["gpu_name_substring"]).upper() not in gpu["name"].upper()
    ):
        raise ResourcePreflightError("resource receipt GPU model is invalid")
    if not isinstance(gpu.get("uuid"), str) or not gpu["uuid"].startswith("GPU-"):
        raise ResourcePreflightError("resource receipt GPU UUID is invalid")
    if not isinstance(gpu.get("driver_version"), str) or not gpu["driver_version"]:
        raise ResourcePreflightError("resource receipt GPU driver is invalid")
    if _exact_integer(gpu.get("memory_total_mib"), "GPU memory", minimum=1) < int(
        resources["minimum_gpu_memory_mib"]
    ):
        raise ResourcePreflightError("resource receipt GPU memory is below minimum")
    volatile = _exact_integer(gpu.get("uncorrected_volatile"), "volatile ECC")
    aggregate = _exact_integer(gpu.get("uncorrected_aggregate"), "aggregate ECC")
    if bool(resources["require_zero_uncorrected_ecc"]) and (volatile or aggregate):
        raise ResourcePreflightError("resource receipt uncorrected ECC is non-zero")
    compute_pids = gpu.get("compute_pids")
    if (
        not isinstance(compute_pids, list)
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in compute_pids
        )
        or compute_pids != sorted(set(compute_pids))
    ):
        raise ResourcePreflightError("resource receipt compute PID list is invalid")
    if bool(resources["require_idle_gpu"]) and compute_pids:
        raise ResourcePreflightError("resource receipt GPU is not idle")

    nvlink = report.get("nvlink")
    if bool(resources["require_full_local_nvlink_clique"]):
        if not isinstance(nvlink, dict) or set(nvlink) != {"matrix", "matrix_sha256"}:
            raise ResourcePreflightError("resource receipt NVLink report is malformed")
        matrix = nvlink.get("matrix")
        if (
            not isinstance(matrix, dict)
            or not matrix
            or nvlink.get("matrix_sha256") != _canonical_sha256(matrix)
        ):
            raise ResourcePreflightError("resource receipt NVLink matrix SHA mismatch")
        keys = sorted(matrix, key=lambda key: int(str(key)[3:]))
        if list(matrix) != keys:
            raise ResourcePreflightError("resource receipt NVLink GPU order is invalid")
        width = len(keys)
        for source_index, key in enumerate(keys):
            row = matrix[key]
            if (
                re.fullmatch(r"GPU[0-9]+", str(key)) is None
                or not isinstance(row, list)
                or len(row) != width
                or row[source_index] != "X"
                or row.count("X") != 1
                or any(value != "X" and not str(value).startswith("NV") for value in row)
            ):
                raise ResourcePreflightError("resource receipt NVLink clique is invalid")
        for source_index in range(width):
            for target_index in range(width):
                if matrix[keys[source_index]][target_index] != matrix[keys[target_index]][source_index]:
                    raise ResourcePreflightError("resource receipt NVLink matrix is asymmetric")
    elif nvlink != {}:
        raise ResourcePreflightError("resource receipt has undeclared NVLink data")

    infiniband = report.get("infiniband")
    if not isinstance(infiniband, list) or not infiniband:
        raise ResourcePreflightError("resource receipt has no InfiniBand reports")
    active_rates: list[float] = []
    for port in infiniband:
        if not isinstance(port, dict) or set(port) != {"device", "port", "state", "rate"}:
            raise ResourcePreflightError("resource receipt InfiniBand fields mismatch")
        if any(not isinstance(port[field], str) or not port[field] for field in port):
            raise ResourcePreflightError("resource receipt InfiniBand value is invalid")
        if "ACTIVE" in port["state"].upper():
            active_rates.append(_parse_ib_rate_gbps(port["rate"]))
    if not active_rates or max(active_rates) < float(resources["minimum_ib_rate_gbps"]):
        raise ResourcePreflightError("resource receipt InfiniBand rate is below minimum")

    memlock = report.get("memlock")
    if isinstance(memlock, bool) or not isinstance(memlock, int):
        raise ResourcePreflightError("resource receipt memlock is invalid")
    if memlock != resource.RLIM_INFINITY and memlock < int(
        resources["minimum_memlock_bytes"]
    ):
        raise ResourcePreflightError("resource receipt memlock is below minimum")
    if _exact_integer(report.get("nofile"), "nofile", minimum=1) < int(
        resources["minimum_nofile"]
    ):
        raise ResourcePreflightError("resource receipt nofile is below minimum")
    for field, required in (
        ("shm_free_bytes", "minimum_shm_bytes"),
        ("data_free_bytes", "minimum_data_free_bytes"),
        ("output_free_bytes", "minimum_output_free_bytes"),
    ):
        if _exact_integer(report.get(field), field) < int(resources[required]):
            raise ResourcePreflightError(f"resource receipt {field} is below minimum")
    if _exact_number(report.get("allreduce_gbps"), "allreduce_gbps") < float(
        resources["minimum_allreduce_gbps"]
    ):
        raise ResourcePreflightError("resource receipt all-reduce rate is below minimum")


def current_rank_identity(local_rank: int) -> dict[str, Any]:
    """Return the inexpensive launch-time identity subset of the full probe."""

    gpu = _gpu_report(int(local_rank))
    return {
        "hostname": socket.gethostname(),
        "local_rank": int(local_rank),
        "gpu_uuid": str(gpu["uuid"]),
    }


def validate_current_rank_identities(
    receipt: Mapping[str, Any], identities: list[Any]
) -> None:
    reports = receipt.get("reports")
    if not isinstance(reports, list) or len(reports) != len(identities):
        raise ResourcePreflightError("launch identity/rank report count mismatch")
    expected: dict[int, tuple[str, int, str]] = {}
    for report in reports:
        if not isinstance(report, dict) or not isinstance(report.get("gpu"), dict):
            raise ResourcePreflightError("resource receipt GPU identity is malformed")
        rank = int(report.get("rank", -1))
        expected[rank] = (
            str(report.get("hostname", "")),
            int(report.get("local_rank", -1)),
            str(report["gpu"].get("uuid", "")),
        )
    actual: dict[int, tuple[str, int, str]] = {}
    for rank, identity in enumerate(identities):
        if not isinstance(identity, dict):
            raise ResourcePreflightError("current rank identity is malformed")
        actual[rank] = (
            str(identity.get("hostname", "")),
            int(identity.get("local_rank", -1)),
            str(identity.get("gpu_uuid", "")),
        )
    if expected != actual:
        raise ResourcePreflightError(
            f"current rank/GPU identity differs from preflight: {actual} != {expected}"
        )
