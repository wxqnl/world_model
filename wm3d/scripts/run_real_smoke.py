#!/usr/bin/env python3
"""从空目录执行真实公开双臂 Stage0 端到端验收。

本文件只编排 ``run_wm3d.sh`` 已有的统一子命令，不实现第二套数据、训练或评测
逻辑。每一步成功后写不可覆盖的 receipt；重入时先逐字节验证该步输出，再决定
跳过。任何输入、命令、产物或代码 commit 漂移都会 fail closed。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from typing import Any, Callable, Mapping, Sequence


SCHEMA = "wm3d_v8_real_smoke_acceptance_v1"
STEP_SCHEMA = "wm3d_v8_real_smoke_step_v1"
PLAN_SCHEMA = "wm3d_v8_real_smoke_plan_v1"
ASSET_SCHEMA = "wm3d_v8_real_smoke_assets_v1"
DATASET_REVISION = "cc571a3c661df81b566dbfde3d5c1e85fcdf7884"
QWEN_REVISION = "9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda"
VGGT_MODEL_REVISION = "860abec7937da0a4c03c41d3c269c366e82abdf9"
VGGT_SOURCE_REVISION = "a288dd0f14786c93483e45524328726ab7b1b4ce"
VGGT_SOURCE_FILE_SHA256 = (
    "dbc92d6214882161d9042b1bc60cc932a9fd6bb609fc913d14906328f7833607"
)
SMOKE_SPLIT_SEED = 3407
SMOKE_TRAIN_FRACTION = 0.5
SMOKE_VALIDATION_FRACTION = 0.49
LICENSE_LITERAL = "YES_I_HAVE_ACCEPTED_THE_UPSTREAM_LICENSES"
ADAPTER_LITERAL = "I_VERIFIED_FIELDS_UNITS_FRAMES_GRIPPER_GROUPS_AND_NATIVE_CLOCKS"
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")


class SmokeError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()


def _atomic_no_clobber(path: Path, value: Mapping[str, Any]) -> None:
    payload = (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_file() and not path.is_symlink() and path.read_bytes() == payload:
            return
        raise SmokeError(f"拒绝覆盖不一致产物：{path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError as error:
        raise SmokeError(f"产物发布竞态：{path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def _regular(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise SmokeError(f"{label} 必须是普通文件：{path}")
    return path.resolve(strict=True)


def _tree_manifest(root: Path) -> dict[str, Any]:
    if root.is_symlink() or not root.is_dir():
        raise SmokeError(f"目录产物无效：{root}")
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(root).as_posix()
        if any(part in {".cache", "__pycache__"} for part in path.relative_to(root).parts):
            continue
        if path.is_symlink():
            target = path.resolve(strict=True)
            if not target.is_file():
                raise SmokeError(f"只允许指向普通文件的资产 symlink：{path}")
            rows.append(
                {
                    "path": relative,
                    "kind": "symlink_file",
                    "link": os.readlink(path),
                    "size": target.stat().st_size,
                    "sha256": _sha256(target),
                }
            )
        elif path.is_file():
            rows.append(
                {
                    "path": relative,
                    "kind": "file",
                    "size": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
        elif not path.is_dir():
            raise SmokeError(f"目录含设备或其他特殊节点：{path}")
    if not rows:
        raise SmokeError(f"目录为空：{root}")
    return {
        "root": str(root.resolve(strict=True)),
        "file_count": len(rows),
        "total_bytes": sum(int(row["size"]) for row in rows),
        "content_sha256": _canonical_sha(rows),
    }


def _evidence(path: Path, kind: str) -> dict[str, Any]:
    if kind == "file":
        safe = _regular(path, "输出")
        return {
            "path": str(safe),
            "kind": kind,
            "size": safe.stat().st_size,
            "sha256": _sha256(safe),
        }
    if kind == "tree":
        value = _tree_manifest(path.resolve(strict=True))
        value["kind"] = kind
        return value
    raise SmokeError(f"未知 evidence kind：{kind}")


def _matches_evidence(path: Path, kind: str, sealed: Mapping[str, Any]) -> bool:
    """Verify a sealed artifact without re-hashing immutable giant model trees.

    Ordinary smoke outputs are small and always use a full content digest.  The
    model asset receipt is the single authority for multi-GB snapshot closure;
    on re-entry we validate its exact recorded file set, type and size before
    encoder load.  The encoder then reads every required shard locally.  This
    avoids hashing 10+ GB again merely to resume a completed smoke command.
    """

    return _evidence(path, kind) == dict(sealed)


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *arguments], text=True, stderr=subprocess.STDOUT
    ).strip()


def _run(
    command: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    log: Path,
) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    print("+", " ".join(command), flush=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write("\n+ " + " ".join(command) + "\n")
        handle.flush()
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            env=dict(environment),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line)
        return_code = process.wait()
        handle.flush()
        os.fsync(handle.fileno())
    if return_code != 0:
        raise SmokeError(f"命令失败（exit={return_code}）：{' '.join(command)}")


def _parse_gpus(value: str) -> tuple[int, int]:
    parts = value.split(",")
    if len(parts) != 2 or any(re.fullmatch(r"[0-9]+", part) is None for part in parts):
        raise argparse.ArgumentTypeError("--gpus 必须恰为两个编号，例如 0,1")
    result = (int(parts[0]), int(parts[1]))
    if result[0] == result[1]:
        raise argparse.ArgumentTypeError("--gpus 不能重复")
    return result


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="真实 ALOHA 双臂小样本：从环境到 exact-resume/eval 的一键验收"
    )
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--operator", required=True)
    parser.add_argument("--gpus", type=_parse_gpus, default=(0, 1))
    parser.add_argument("--accept-dataset-license", action="store_true")
    parser.add_argument("--confirm-adapter-semantics", action="store_true")
    parser.add_argument("--token-file", type=Path)
    parser.add_argument("--environment-dir", type=Path)
    parser.add_argument("--vggt-source-root", type=Path)
    parser.add_argument("--vggt-model-snapshot", type=Path)
    parser.add_argument("--qwen-model-snapshot", type=Path)
    parser.add_argument("--model-cache-root", type=Path)
    parser.add_argument(
        "--hf-endpoint",
        default=os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
        help="Hugging Face compatible endpoint；revision 与 file-list 仍严格锁定。",
    )
    parser.add_argument("--max-download-workers", type=int, default=8)
    parser.add_argument("--batch-frames", type=int, default=8)
    parser.add_argument("--master-port", type=int, default=29631)
    parser.add_argument("--system-python", default=os.environ.get("SYSTEM_PYTHON", "python3.10"))
    parser.add_argument("--_bootstrapped", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(arguments)


def _validate_args(args: argparse.Namespace, repo: Path) -> None:
    if not args.operator.strip():
        raise SmokeError("--operator 不能为空")
    if not args.accept_dataset_license:
        raise SmokeError("先阅读并接受公开数据集许可，再传 --accept-dataset-license")
    if not args.confirm_adapter_semantics:
        raise SmokeError(
            "先审阅双臂字段、group、时钟与 opaque gripper 合同，再传 "
            "--confirm-adapter-semantics"
        )
    if not args.work_root.is_absolute() or args.work_root.is_symlink():
        raise SmokeError("--work-root 必须是绝对路径且不能是 symlink")
    if args.work_root.resolve() == repo.resolve():
        raise SmokeError("--work-root 不能是源码目录")
    if args.max_download_workers <= 0 or args.batch_frames <= 0:
        raise SmokeError("worker/batch 参数必须为正数")
    if not (1024 <= args.master_port <= 65535):
        raise SmokeError("--master-port 超出范围")
    if not re.fullmatch(r"https://[^\s/]+(?:/[^\s]*)?", args.hf_endpoint):
        raise SmokeError("--hf-endpoint 必须是 https URL")
    for name in ("environment_dir", "vggt_source_root", "vggt_model_snapshot", "qwen_model_snapshot", "model_cache_root"):
        value = getattr(args, name)
        if value is not None and not value.is_absolute():
            raise SmokeError(f"--{name.replace('_', '-')} 必须是绝对路径")
    if args.token_file is not None:
        _regular(args.token_file, "token file")


def _bootstrap_environment(args: argparse.Namespace, repo: Path, work: Path) -> Path:
    environment_dir = (args.environment_dir or work / "environment").absolute()
    if environment_dir.is_symlink():
        raise SmokeError("环境目录不能是 symlink")
    environment = os.environ.copy()
    environment["ENV_DIR"] = str(environment_dir)
    environment["SYSTEM_PYTHON"] = args.system_python
    _run(
        ["bash", "environments/bootstrap_environment.sh"],
        cwd=repo,
        environment=environment,
        log=work / "logs" / "00_environment.log",
    )
    python = environment_dir / "bin" / "python"
    _regular(environment_dir / "environment_receipt.json", "environment receipt")
    if not python.is_file() or not os.access(python, os.X_OK):
        raise SmokeError(f"环境 Python 不可执行：{python}")
    return python


def _reexec_in_environment(args: argparse.Namespace, python: Path) -> None:
    if args._bootstrapped:
        if Path(sys.executable).resolve(strict=True) != python.resolve(strict=True):
            raise SmokeError("--_bootstrapped 与实际解释器不一致")
        return
    argv = [str(python), str(Path(__file__).resolve(strict=True)), *sys.argv[1:], "--_bootstrapped"]
    os.execve(str(python), argv, os.environ.copy())


def _safe_extract_vggt(archive: Path, destination: Path) -> None:
    prefix = f"vggt-{VGGT_SOURCE_REVISION}/"
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    allowed_roots = ("vggt/", "pyproject.toml", "LICENSE.txt")
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            name = member.name
            if not name.startswith(prefix):
                raise SmokeError(f"VGGT archive 越界成员：{name}")
            relative = name[len(prefix) :]
            if not relative:
                continue
            pure = PurePosixPath(relative)
            if pure.is_absolute() or ".." in pure.parts:
                raise SmokeError(f"VGGT archive 非法路径：{relative}")
            if not any(relative == root or relative.startswith(root) for root in allowed_roots):
                continue
            target = temporary.joinpath(*pure.parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile() or member.issym() or member.islnk():
                raise SmokeError(f"VGGT source 禁止 link/设备成员：{relative}")
            target.parent.mkdir(parents=True, exist_ok=True)
            source = handle.extractfile(member)
            if source is None:
                raise SmokeError(f"无法读取 archive 成员：{relative}")
            with target.open("xb") as output:
                shutil.copyfileobj(source, output)
    model_file = temporary / "vggt" / "models" / "vggt.py"
    if _sha256(_regular(model_file, "VGGT source file")) != VGGT_SOURCE_FILE_SHA256:
        raise SmokeError("VGGT source commit 的关键文件 SHA 与封存值不符")
    try:
        os.rename(temporary, destination)
    except FileExistsError:
        shutil.rmtree(temporary)


def _resolve_vggt_source(args: argparse.Namespace, work: Path) -> Path:
    if args.vggt_source_root is not None:
        source = args.vggt_source_root.resolve(strict=True)
    else:
        source = work / "assets" / "vggt_source" / VGGT_SOURCE_REVISION
        if not source.exists():
            source.parent.mkdir(parents=True, exist_ok=True)
            url = f"https://codeload.github.com/facebookresearch/vggt/tar.gz/{VGGT_SOURCE_REVISION}"
            with tempfile.NamedTemporaryFile(
                prefix="vggt-source-", suffix=".tar.gz", dir=source.parent, delete=False
            ) as handle:
                archive = Path(handle.name)
            try:
                with urllib.request.urlopen(url, timeout=120) as response, archive.open("wb") as output:
                    shutil.copyfileobj(response, output)
                _safe_extract_vggt(archive, source)
            finally:
                archive.unlink(missing_ok=True)
    if _sha256(_regular(source / "vggt" / "models" / "vggt.py", "VGGT source")) != VGGT_SOURCE_FILE_SHA256:
        raise SmokeError("显式 VGGT source 不是已审计 commit")
    return source


def _snapshot(
    *,
    explicit: Path | None,
    repo_id: str,
    revision: str,
    cache_root: Path,
    marker: str,
) -> Path:
    if explicit is not None:
        result = explicit.resolve(strict=True)
    else:
        from huggingface_hub import snapshot_download

        result = Path(
            snapshot_download(
                repo_id=repo_id,
                revision=revision,
                cache_dir=str(cache_root / "hub"),
            )
        ).resolve(strict=True)
    if result.name != revision or not (result / marker).is_file():
        raise SmokeError(f"模型 snapshot revision/marker 不匹配：{result}")
    return result


def _tree_shape(root: Path) -> dict[str, Any]:
    if root.is_symlink() or not root.is_dir():
        raise SmokeError(f"模型资产目录无效：{root}")
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if any(part in {".cache", "__pycache__"} for part in path.relative_to(root).parts):
            continue
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            target = path.resolve(strict=True)
            if not target.is_file():
                raise SmokeError(f"模型资产 symlink 目标无效：{path}")
            rows.append({"path": relative, "kind": "symlink_file", "link": os.readlink(path), "size": target.stat().st_size})
        elif path.is_file():
            rows.append({"path": relative, "kind": "file", "size": path.stat().st_size})
        elif not path.is_dir():
            raise SmokeError(f"模型资产包含特殊节点：{path}")
    if not rows:
        raise SmokeError(f"模型资产目录为空：{root}")
    return {"root": str(root.resolve(strict=True)), "file_count": len(rows), "total_bytes": sum(int(row["size"]) for row in rows), "shape_sha256": _canonical_sha(rows)}


def _asset_tree(root: Path, *, hash_content: bool) -> dict[str, Any]:
    if hash_content:
        value = _tree_manifest(root)
        value["verification"] = "full_content_sha256"
        return value
    value = _tree_shape(root)
    value["verification"] = "sealed_snapshot_shape_plus_encoder_full_load"
    return value


def _resolve_assets(args: argparse.Namespace, work: Path) -> tuple[Path, Path, Path, Path]:
    receipt_path = work / "receipts" / "01_model_assets.json"
    source = _resolve_vggt_source(args, work)
    cache_root = (args.model_cache_root or work / "assets" / "huggingface").absolute()
    cache_root.mkdir(parents=True, exist_ok=True)
    qwen = _snapshot(
        explicit=args.qwen_model_snapshot,
        repo_id="Qwen/Qwen3-VL-Embedding-2B",
        revision=QWEN_REVISION,
        cache_root=cache_root,
        marker="modules.json",
    )
    vggt = _snapshot(
        explicit=args.vggt_model_snapshot,
        repo_id="facebook/VGGT-1B",
        revision=VGGT_MODEL_REVISION,
        cache_root=cache_root,
        marker="config.json",
    )
    if receipt_path.exists() or receipt_path.is_symlink():
        sealed = json.loads(_regular(receipt_path, "model asset receipt").read_text())
        if (
            sealed.get("schema") != ASSET_SCHEMA
            or sealed.get("vggt_source_revision") != VGGT_SOURCE_REVISION
            or sealed.get("vggt_source_file_sha256") != VGGT_SOURCE_FILE_SHA256
            or sealed.get("vggt_model_revision") != VGGT_MODEL_REVISION
            or sealed.get("qwen_model_revision") != QWEN_REVISION
            or sealed.get("vggt_source") != _asset_tree(source, hash_content=True)
            or sealed.get("vggt_model") != _asset_tree(vggt, hash_content=False)
            or sealed.get("qwen_model") != _asset_tree(qwen, hash_content=False)
        ):
            raise SmokeError("模型资产 receipt 与当前 file-set/size/identity 不一致")
        return source, vggt, qwen, receipt_path
    value = {
        "schema": ASSET_SCHEMA,
        "vggt_source_revision": VGGT_SOURCE_REVISION,
        "vggt_source_file_sha256": VGGT_SOURCE_FILE_SHA256,
        "vggt_source": _asset_tree(source, hash_content=True),
        "vggt_model_revision": VGGT_MODEL_REVISION,
        "vggt_model": _asset_tree(vggt, hash_content=False),
        "qwen_model_revision": QWEN_REVISION,
        "qwen_model": _asset_tree(qwen, hash_content=False),
    }
    _atomic_no_clobber(receipt_path, value)
    return source, vggt, qwen, receipt_path


class Orchestrator:
    def __init__(
        self,
        *,
        repo: Path,
        work: Path,
        python: Path,
        args: argparse.Namespace,
        plan_sha: str,
        environment: Mapping[str, str],
    ) -> None:
        self.repo = repo
        self.work = work
        self.python = python
        self.args = args
        self.plan_sha = plan_sha
        self.environment = dict(environment)
        self.run_wm3d = str(repo / "run_wm3d.sh")

    def step(
        self,
        ordinal: int,
        name: str,
        subcommand: Sequence[str],
        outputs: Sequence[tuple[Path, str]],
        *,
        inputs: Sequence[Path] = (),
        environment: Mapping[str, str] | None = None,
        recover_existing: Callable[[], None] | None = None,
    ) -> Path:
        receipt = self.work / "receipts" / f"{ordinal:02d}_{name}.json"
        command = [self.run_wm3d, *map(str, subcommand)]
        definition = {
            "plan_sha256": self.plan_sha,
            "command": command,
            "input_sha256": {
                str(_regular(path, "step input")): _sha256(path.resolve(strict=True))
                for path in inputs
            },
        }
        if receipt.exists() or receipt.is_symlink():
            sealed = json.loads(_regular(receipt, "step receipt").read_text())
            if sealed.get("schema") != STEP_SCHEMA or sealed.get("definition") != definition:
                raise SmokeError(f"步骤 receipt 定义漂移：{receipt}")
            current = [_evidence(path, kind) for path, kind in outputs]
            if sealed.get("outputs") != current:
                raise SmokeError(f"已完成步骤产物发生漂移：{name}")
            print(f"[verified-skip] {name}", flush=True)
            return receipt
        if recover_existing is not None and all(path.exists() for path, _kind in outputs):
            # 训练进程可能在 committed DCP 原子发布后、步骤 receipt 发布前被中断。
            # 只有调用方完成全 manifest/SHA/lineage/runtime 校验后才能恢复 receipt；
            # 不能因为目录“看起来存在”就跳过。
            recover_existing()
            value = {
                "schema": STEP_SCHEMA,
                "name": name,
                "definition": definition,
                "outputs": [_evidence(path, kind) for path, kind in outputs],
                "passed": True,
                "recovered_after_committed_output": True,
            }
            _atomic_no_clobber(receipt, value)
            print(f"[verified-recovery] {name}", flush=True)
            return receipt
        merged = self.environment.copy()
        if environment:
            merged.update(environment)
        _run(
            command,
            cwd=self.repo,
            environment=merged,
            log=self.work / "logs" / f"{ordinal:02d}_{name}.log",
        )
        value = {
            "schema": STEP_SCHEMA,
            "name": name,
            "definition": definition,
            "outputs": [_evidence(path, kind) for path, kind in outputs],
            "passed": True,
        }
        _atomic_no_clobber(receipt, value)
        return receipt


def _gpu_guard(gpus: tuple[int, int]) -> None:
    query = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        text=True,
    )
    uuid_by_index: dict[int, str] = {}
    for line in query.splitlines():
        index, uuid = [item.strip() for item in line.split(",", 1)]
        uuid_by_index[int(index)] = uuid
    missing = [index for index in gpus if index not in uuid_by_index]
    if missing:
        raise SmokeError(f"GPU 编号不存在：{missing}")
    processes = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    selected = {uuid_by_index[index] for index in gpus}
    conflicts = [line for line in processes.splitlines() if line.split(",", 1)[0].strip() in selected]
    if conflicts:
        raise SmokeError(f"目标 GPU 已有 compute process，禁止挤占：{conflicts}")


def _validate_checkpoint(path: Path, expected_step: int, runtime_sha: str, lineage: str) -> dict[str, Any]:
    if path.name != f"step_{expected_step:08d}" or path.is_symlink() or not path.is_dir():
        raise SmokeError(f"checkpoint 路径/编号错误：{path}")
    manifest_path = _regular(path / "MANIFEST.json", "checkpoint manifest")
    committed_path = _regular(path / "COMMITTED.json", "checkpoint commit")
    metadata_path = _regular(path / "metadata.json", "checkpoint metadata")
    manifest = json.loads(manifest_path.read_text())
    commit = json.loads(committed_path.read_text())
    metadata = json.loads(metadata_path.read_text())
    if (
        manifest.get("schema") != "wm3d_v8_distributed_checkpoint_v2"
        or int(manifest.get("step", -1)) != expected_step
        or commit.get("schema") != "wm3d_v8_distributed_checkpoint_commit_v2"
        or int(commit.get("step", -1)) != expected_step
        or commit.get("manifest_sha256") != _sha256(manifest_path)
        or commit.get("manifest_content_sha256") != _canonical_sha(manifest)
        or commit.get("metadata_sha256") != _sha256(metadata_path)
        or commit.get("run_lineage") != lineage
        or metadata.get("runtime_config_sha256") != runtime_sha
        or metadata.get("run_lineage") != lineage
        or int(metadata.get("step", -1)) != expected_step
        or int(metadata.get("sampler_progress", {}).get("next_optimizer_step", -1)) != expected_step
        or int(metadata.get("world_size", -1)) != 2
        or metadata.get("distributed_strategy") != "fsdp2"
        or not metadata.get("gradient_ownership", {}).get("passed")
    ):
        raise SmokeError(f"checkpoint identity/contract 不成立：{path}")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise SmokeError("checkpoint manifest files 为空")
    for relative, evidence in files.items():
        file_path = _regular(path / relative, "checkpoint shard")
        if (
            set(evidence) != {"sha256", "size"}
            or file_path.stat().st_size != int(evidence["size"])
            or _sha256(file_path) != evidence["sha256"]
        ):
            raise SmokeError(f"checkpoint shard SHA/size 不一致：{file_path}")
    return {"manifest": _sha256(manifest_path), "committed": _sha256(committed_path), "metadata": metadata}


def _validate_eval(path: Path, runtime_sha: str, lineage: str) -> dict[str, Any]:
    value = json.loads(_regular(path, "offline eval receipt").read_text())
    metrics = value.get("metrics")
    if (
        value.get("schema") != "wm3d_v8_unified_offline_eval_v2"
        or value.get("runtime_sha256") != runtime_sha
        or int(value.get("checkpoint_step", -1)) != 2
        or int(value.get("world_size", -1)) != 2
        or value.get("evaluated_split") != "val"
        or value.get("checkpoint_metadata", {}).get("run_lineage") != lineage
        or value.get("all_metrics_finite") is not True
        or not isinstance(metrics, dict)
        or not metrics
        or any(not isinstance(number, (int, float)) or not math.isfinite(number) for number in metrics.values())
        or float(metrics.get("fine_supervised_dimensions", 0.0)) <= 0
    ):
        raise SmokeError("offline eval receipt 不满足真实双臂 smoke 合同")
    return value


def _validate_smoke_inventory(manifest: Path, receipt: Path) -> dict[str, Any]:
    receipt_value = json.loads(_regular(receipt, "inventory receipt").read_text())
    if (
        receipt_value.get("selection", {}).get("episode_indices") != [0, 30]
        or receipt_value.get("split_count")
        != {"test": 0, "train": 1, "val": 1}
    ):
        raise SmokeError("真实 smoke episode/split receipt 与固定合同不一致")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        _regular(manifest, "inventory manifest").read_text().splitlines(), 1
    ):
        if not line.strip():
            raise SmokeError(f"inventory manifest 第 {line_number} 行为空")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise SmokeError(f"inventory manifest 第 {line_number} 行不是对象")
        rows.append(value)
    actual = {str(row.get("episode_id")): row.get("split") for row in rows}
    expected = {
        "aloha_smoke:000000000": "train",
        "aloha_smoke:000000030": "val",
    }
    if len(rows) != 2 or actual != expected:
        raise SmokeError(
            f"真实 smoke manifest 必须精确为 episode 0=train、30=val，实际 {actual}"
        )
    return receipt_value


def _plan(args: argparse.Namespace, repo: Path, work: Path, code_commit: str) -> tuple[dict[str, Any], str]:
    config_paths = [
        repo / "configs/sources/smoke_aloha.lock.yaml",
        repo / "configs/data/smoke_aloha_bimanual.template.yaml",
        repo / "configs/data/smoke_aloha_episode_indices.txt",
        repo / "configs/adapters/aloha_sim_insertion_human.yaml",
        repo / "configs/encoder/vggt_native_p144_appearance_p256.yaml",
        repo / "configs/encoder/task_qwen3_vl_embedding_2b.yaml",
        repo / "configs/model/native_1b_dual_path.yaml",
        repo / "configs/objective/stage0_native_dual_path.yaml",
        repo / "configs/runtime/smoke_2gpu_fsdp2.yaml",
    ]
    value = {
        "schema": PLAN_SCHEMA,
        "code_commit": code_commit,
        "work_root": str(work),
        "dataset": "lerobot/aloha_sim_insertion_human",
        "dataset_revision": DATASET_REVISION,
        "episode_indices": [0, 30],
        "split_contract": {
            "seed": SMOKE_SPLIT_SEED,
            "train_fraction": SMOKE_TRAIN_FRACTION,
            "validation_fraction": SMOKE_VALIDATION_FRACTION,
            "expected": {
                "aloha_smoke:000000000": "train",
                "aloha_smoke:000000030": "val",
            },
        },
        "gpus": list(args.gpus),
        "operator": args.operator,
        "master_port": args.master_port,
        "max_download_workers": args.max_download_workers,
        "batch_frames": args.batch_frames,
        "hf_endpoint": args.hf_endpoint.rstrip("/"),
        "config_sha256": {str(path.relative_to(repo)): _sha256(_regular(path, "config")) for path in config_paths},
        "external_asset_paths": {
            "environment_dir": str((args.environment_dir or work / "environment").absolute()),
            "vggt_source_root": str(args.vggt_source_root) if args.vggt_source_root else None,
            "vggt_model_snapshot": str(args.vggt_model_snapshot) if args.vggt_model_snapshot else None,
            "qwen_model_snapshot": str(args.qwen_model_snapshot) if args.qwen_model_snapshot else None,
            "model_cache_root": str(args.model_cache_root) if args.model_cache_root else None,
        },
    }
    return value, _canonical_sha(value)


def _make_runtime_environment(
    *,
    repo: Path,
    python: Path,
    work: Path,
    source: Path,
    vggt: Path,
    qwen: Path,
    gpus: tuple[int, int],
    hf_endpoint: str,
) -> dict[str, str]:
    value = os.environ.copy()
    value.update(
        {
            "PYTHON_BIN": str(python),
            "PYTHONPATH": f"{repo}:{source}" + (f":{value['PYTHONPATH']}" if value.get("PYTHONPATH") else ""),
            "PYTHONDONTWRITEBYTECODE": "1",
            "HF_HOME": str(work / "assets" / "huggingface"),
            "HF_ENDPOINT": hf_endpoint.rstrip("/"),
            "WM3D_VGGT_SOURCE_ROOT": str(source),
            "WM3D_VGGT_MODEL_SNAPSHOT": str(vggt),
            "QWEN3_VL_EMBEDDING_PATH": str(qwen),
            "CUDA_VISIBLE_DEVICES": f"{gpus[0]},{gpus[1]}",
        }
    )
    return value


def main(arguments: Sequence[str] | None = None) -> None:
    args = parse_args(arguments)
    repo = Path(__file__).resolve(strict=True).parents[1]
    _validate_args(args, repo)
    status = _git(repo, "status", "--porcelain")
    if status:
        raise SmokeError("smoke-real 只接受 clean commit；请先提交或切到发布 worktree")
    code_commit = _git(repo, "rev-parse", "HEAD")
    if HEX40.fullmatch(code_commit) is None:
        raise SmokeError("当前代码 commit 无效")
    work = args.work_root.absolute()
    if not work.exists():
        work.mkdir(parents=True)
    if work.is_symlink() or not work.is_dir():
        raise SmokeError("work-root 必须是实际目录")
    plan, plan_sha = _plan(args, repo, work, code_commit)
    _atomic_no_clobber(work / "smoke_plan.json", plan)

    python = _bootstrap_environment(args, repo, work)
    _reexec_in_environment(args, python)
    source, vggt, qwen, asset_receipt = _resolve_assets(args, work)
    environment = _make_runtime_environment(
        repo=repo,
        python=python,
        work=work,
        source=source,
        vggt=vggt,
        qwen=qwen,
        gpus=args.gpus,
        hf_endpoint=args.hf_endpoint,
    )
    flow = Orchestrator(
        repo=repo, work=work, python=python, args=args, plan_sha=plan_sha, environment=environment
    )

    source_lock = work / "source" / "smoke_aloha.lock.yaml"
    raw_root = work / "raw"
    raw_source = raw_root / "aloha_smoke"
    download_receipt = raw_root / "receipts" / "aloha_smoke.json"
    schema = work / "audit" / "aloha.schema.json"
    candidate = work / "audit" / "aloha.candidate.json"
    adapter_receipt = work / "audit" / "aloha.adapter.json"
    inventory = work / "data" / "aloha.inventory.jsonl"
    inventory_receipt = work / "data" / "aloha.inventory.receipt.json"
    data_profile = work / "data" / "aloha.profile.yaml"
    data_receipt = work / "data" / "aloha.profile.receipt.json"
    task_bank = work / "task_bank"
    task_manifest = work / "cache" / "tasks.jsonl"
    cache = work / "cache"
    episode_index = cache / "episode_index.jsonl"
    episode_seal = cache / "episode_seal.json"
    window_index = cache / "window_1b.jsonl"
    window_seal = cache / "window_1b.seal.json"
    normalization = cache / "grouped_normalization.json"
    runtime = work / "runtime_1b_smoke.yaml"
    training = work / "training"
    lineage = f"wm3d_real_aloha_bimanual_smoke_{plan_sha[:12]}"

    token_arguments: list[str] = []
    if args.token_file is not None:
        token_arguments = ["--token-file", str(args.token_file)]
    flow.step(
        2,
        "source_lock",
        [
            "lock-resolve", "--template", repo / "configs/sources/smoke_aloha.lock.yaml",
            "--output", source_lock, "--confirm-licenses", LICENSE_LITERAL,
            "--revision", f"aloha_smoke={DATASET_REVISION}", *token_arguments,
        ],
        [(source_lock, "file"), (Path(str(source_lock) + ".receipt.json"), "file"), (Path(str(source_lock) + ".file_lists"), "tree")],
        inputs=[repo / "configs/sources/smoke_aloha.lock.yaml"],
    )
    flow.step(
        3,
        "download",
        ["download", "--lock", source_lock, "--raw-root", raw_root, "--source", "aloha_smoke", "--max-workers", str(args.max_download_workers), *token_arguments],
        [(raw_source, "tree"), (download_receipt, "file")],
        inputs=[source_lock, Path(str(source_lock) + ".receipt.json")],
    )
    flow.step(
        4,
        "schema_audit",
        ["schema-audit", "--root", raw_source, "--max-data-files", "2", "--max-video-files", "8", "--require-homogeneous", "--upstream-receipt", download_receipt, "--candidate-output", candidate, "--output", schema],
        [(candidate, "file"), (schema, "file")],
        inputs=[download_receipt],
    )
    adapter = repo / "configs/adapters/aloha_sim_insertion_human.yaml"
    template = repo / "configs/data/smoke_aloha_bimanual.template.yaml"
    adapter_sha = _sha256(adapter)
    flow.step(
        5,
        "adapter_audit",
        ["adapter-audit", "--schema-audit", schema, "--adapter-candidate", candidate, "--adapter-contract", adapter, "--adapter-contract-sha256", adapter_sha, "--data-template", template, "--source", "aloha_smoke", "--operator", args.operator, "--confirm", ADAPTER_LITERAL, "--output", adapter_receipt],
        [(adapter_receipt, "file")],
        inputs=[schema, candidate, adapter, template],
    )
    flow.step(
        6,
        "inventory",
        ["inventory", "--data-template", template, "--source", "aloha_smoke", "--raw-root", raw_source, "--adapter-contract", adapter, "--adapter-contract-sha256", adapter_sha, "--adapter-audit-receipt", adapter_receipt, "--output-manifest", inventory, "--output-receipt", inventory_receipt, "--episode-index-file", repo / "configs/data/smoke_aloha_episode_indices.txt", "--split-seed", str(SMOKE_SPLIT_SEED), "--train-fraction", str(SMOKE_TRAIN_FRACTION), "--validation-fraction", str(SMOKE_VALIDATION_FRACTION)],
        [(inventory, "file"), (inventory_receipt, "file")],
        inputs=[adapter_receipt, adapter, template, repo / "configs/data/smoke_aloha_episode_indices.txt"],
    )
    flow.step(
        7,
        "data_profile",
        ["data-profile", "--template", template, "--inventory", f"aloha_smoke={inventory_receipt}", "--output", data_profile, "--receipt", data_receipt],
        [(data_profile, "file"), (data_receipt, "file")],
        inputs=[template, inventory_receipt, inventory],
    )
    _gpu_guard(args.gpus)
    flow.step(
        8,
        "task_bank",
        ["task-bank", "--data-profile", data_profile, "--encoder-contract", repo / "configs/encoder/task_qwen3_vl_embedding_2b.yaml", "--output-root", task_bank, "--device", "cuda:0"],
        [(task_bank, "tree")],
        inputs=[data_profile, repo / "configs/encoder/task_qwen3_vl_embedding_2b.yaml", asset_receipt],
    )
    task_bank_sha = _sha256(task_bank / "index.jsonl")
    flow.step(
        9,
        "cache_plan",
        ["cache-plan", "--data-profile", data_profile, "--encoder-contract", repo / "configs/encoder/vggt_native_p144_appearance_p256.yaml", "--task-encoder-contract", repo / "configs/encoder/task_qwen3_vl_embedding_2b.yaml", "--task-bank-index", task_bank / "index.jsonl", "--output", task_manifest],
        [(task_manifest, "file")],
        inputs=[data_profile, task_bank / "index.jsonl", repo / "configs/encoder/vggt_native_p144_appearance_p256.yaml", repo / "configs/encoder/task_qwen3_vl_embedding_2b.yaml"],
    )
    worker_outputs = [(cache / "payload", "tree"), (cache / "receipts", "tree"), (cache / "episode_index_fragments", "tree")]
    worker_receipt = work / "receipts" / "10_cache_workers.json"
    if worker_receipt.exists():
        # 使用同一验证逻辑，不重新实例化两个 1B encoder。
        flow.step(10, "cache_workers", ["cache-worker-pair", "receipt-only"], worker_outputs, inputs=[task_manifest, data_profile, task_bank / "index.jsonl", asset_receipt])
    else:
        definition_command = [flow.run_wm3d, "cache-worker-pair", "receipt-only"]
        definition = {
            "plan_sha256": plan_sha,
            "command": definition_command,
            "input_sha256": {str(_regular(path, "worker input")): _sha256(path) for path in [task_manifest, data_profile, task_bank / "index.jsonl", asset_receipt]},
        }
        commands = []
        for worker in range(2):
            commands.append(
                [flow.run_wm3d, "cache-worker", "--task-manifest", str(task_manifest), "--data-profile", str(data_profile), "--encoder-contract", str(repo / "configs/encoder/vggt_native_p144_appearance_p256.yaml"), "--task-bank-root", str(task_bank), "--task-bank-index-sha256", task_bank_sha, "--cache-root", str(cache), "--worker-index", str(worker), "--worker-count", "2", "--device", f"cuda:{worker}", "--batch-frames", str(args.batch_frames), "--fail-fast"]
            )
        processes = []
        logs = []
        try:
            for worker, command in enumerate(commands):
                log_path = work / "logs" / f"10_cache_worker_{worker}.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                handle = log_path.open("a", encoding="utf-8")
                logs.append(handle)
                print("+", " ".join(command), flush=True)
                processes.append(subprocess.Popen(command, cwd=repo, env=environment, stdout=handle, stderr=subprocess.STDOUT, text=True))
            codes = [process.wait() for process in processes]
        finally:
            for handle in logs:
                handle.close()
        if codes != [0, 0]:
            raise SmokeError(f"cache workers 失败：{codes}")
        _atomic_no_clobber(worker_receipt, {"schema": STEP_SCHEMA, "name": "cache_workers", "definition": definition, "outputs": [_evidence(path, kind) for path, kind in worker_outputs], "passed": True})
    flow.step(
        11,
        "cache_seal",
        ["cache-seal", "--task-manifest", task_manifest, "--receipt-root", cache / "receipts", "--episode-index-fragment-root", cache / "episode_index_fragments", "--output-index", episode_index, "--output-seal", episode_seal],
        [(episode_index, "file"), (episode_seal, "file")],
        inputs=[task_manifest],
    )
    flow.step(
        12,
        "window",
        ["window", "--episode-index", episode_index, "--episode-seal", episode_seal, "--cache-root", cache, "--data-profile", data_profile, "--model-profile", repo / "configs/model/native_1b_dual_path.yaml", "--output-index", window_index, "--output-seal", window_seal],
        [(window_index, "file"), (window_seal, "file")],
        inputs=[episode_index, episode_seal, data_profile, repo / "configs/model/native_1b_dual_path.yaml"],
    )
    flow.step(
        13,
        "normalization",
        ["normalization", "--data-profile", data_profile, "--model-profile", repo / "configs/model/native_1b_dual_path.yaml", "--window-index", window_index, "--window-index-sha256", _sha256(window_index), "--cache-root", cache, "--output", normalization],
        [(normalization, "file")],
        inputs=[data_profile, repo / "configs/model/native_1b_dual_path.yaml", window_index, episode_seal],
    )
    flow.step(
        14,
        "runtime",
        ["runtime", "--model", repo / "configs/model/native_1b_dual_path.yaml", "--data", data_profile, "--runtime", repo / "configs/runtime/smoke_2gpu_fsdp2.yaml", "--objective", repo / "configs/objective/stage0_native_dual_path.yaml", "--cache-root", cache, "--episode-cache-index", episode_index, "--episode-cache-seal", episode_seal, "--cache-index", window_index, "--cache-seal", window_seal, "--grouped-normalization", normalization, "--environment-lock", (args.environment_dir or work / "environment") / "environment_receipt.json", "--run-name", "wm3d_real_aloha_1b_smoke", "--run-lineage", lineage, "--output-root", training, "--output", runtime],
        [(runtime, "file")],
        inputs=[data_profile, episode_index, episode_seal, window_index, window_seal, normalization, repo / "configs/model/native_1b_dual_path.yaml", repo / "configs/runtime/smoke_2gpu_fsdp2.yaml", repo / "configs/objective/stage0_native_dual_path.yaml"],
    )
    runtime_sha = _sha256(runtime)
    torch_args = ["--nnodes=1", "--nproc_per_node=2", "--node_rank=0", "--master_addr=127.0.0.1"]
    _gpu_guard(args.gpus)
    flow.step(
        15,
        "preflight",
        ["preflight", *torch_args, f"--master_port={args.master_port}", "--", "--runtime", runtime],
        [],
        inputs=[runtime, asset_receipt],
    )
    checkpoint1 = training / "checkpoints" / "step_00000001"
    flow.step(
        16,
        "train_0_to_1",
        ["train", *torch_args, f"--master_port={args.master_port + 1}", "--", "--runtime", runtime, "--stop-after-step", "1"],
        [(checkpoint1, "tree")],
        inputs=[runtime],
        recover_existing=lambda: _validate_checkpoint(
            checkpoint1, 1, runtime_sha, lineage
        ),
    )
    _validate_checkpoint(checkpoint1, 1, runtime_sha, lineage)
    checkpoint2 = training / "checkpoints" / "step_00000002"
    flow.step(
        17,
        "resume_1_to_2",
        ["train", *torch_args, f"--master_port={args.master_port + 2}", "--", "--runtime", runtime, "--resume", checkpoint1, "--stop-after-step", "2"],
        [(checkpoint2, "tree")],
        inputs=[runtime, checkpoint1 / "COMMITTED.json", checkpoint1 / "MANIFEST.json"],
        recover_existing=lambda: _validate_checkpoint(
            checkpoint2, 2, runtime_sha, lineage
        ),
    )
    step1 = _validate_checkpoint(checkpoint1, 1, runtime_sha, lineage)
    step2 = _validate_checkpoint(checkpoint2, 2, runtime_sha, lineage)
    eval_path = work / "eval_step2.json"
    flow.step(
        18,
        "offline_eval",
        ["eval", *torch_args, f"--master_port={args.master_port + 3}", "--", "--runtime", runtime, "--checkpoint", checkpoint2, "--output", eval_path],
        [(eval_path, "file")],
        inputs=[runtime, checkpoint2 / "COMMITTED.json", checkpoint2 / "MANIFEST.json"],
    )
    evaluation = _validate_eval(eval_path, runtime_sha, lineage)
    _validate_smoke_inventory(inventory, inventory_receipt)
    final = {
        "schema": SCHEMA,
        "passed": True,
        "code_commit": code_commit,
        "plan_sha256": plan_sha,
        "work_root": str(work),
        "dataset": {"repo_id": "lerobot/aloha_sim_insertion_human", "revision": DATASET_REVISION, "episode_indices": [0, 30], "inventory_receipt_sha256": _sha256(inventory_receipt)},
        "distributed": {"world_size": 2, "strategy": "fsdp2", "gpus": list(args.gpus), "fresh_process_boundaries": ["0_to_1", "resume_1_to_2", "offline_eval"]},
        "runtime_path": str(runtime),
        "runtime_sha256": runtime_sha,
        "run_lineage": lineage,
        "checkpoint_1": {"path": str(checkpoint1), "manifest_sha256": step1["manifest"], "committed_sha256": step1["committed"], "next_optimizer_step": step1["metadata"]["sampler_progress"]["next_optimizer_step"]},
        "checkpoint_2": {"path": str(checkpoint2), "manifest_sha256": step2["manifest"], "committed_sha256": step2["committed"], "next_optimizer_step": step2["metadata"]["sampler_progress"]["next_optimizer_step"]},
        "offline_eval": {"path": str(eval_path), "sha256": _sha256(eval_path), "evaluated_split": evaluation["evaluated_split"], "all_metrics_finite": evaluation["all_metrics_finite"], "metrics": evaluation["metrics"]},
        "artifact_sha256": {
            "environment": _sha256((args.environment_dir or work / "environment") / "environment_receipt.json"),
            "model_assets": _sha256(asset_receipt),
            "source_lock": _sha256(source_lock),
            "download": _sha256(download_receipt),
            "schema_audit": _sha256(schema),
            "adapter_audit": _sha256(adapter_receipt),
            "data_profile": _sha256(data_profile),
            "task_bank": _sha256(task_bank / "receipt.json"),
            "episode_seal": _sha256(episode_seal),
            "window_seal": _sha256(window_seal),
            "normalization": _sha256(normalization),
        },
        "step_receipts_sha256": {path.name: _sha256(path) for path in sorted((work / "receipts").glob("[0-9][0-9]_*.json"))},
    }
    acceptance = work / "SMOKE_REAL_ACCEPTANCE.json"
    _atomic_no_clobber(acceptance, final)
    print(json.dumps({"passed": True, "acceptance": str(acceptance), "sha256": _sha256(acceptance)}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (SmokeError, OSError, subprocess.SubprocessError) as error:
        raise SystemExit(f"smoke-real FAILED: {error}") from error
