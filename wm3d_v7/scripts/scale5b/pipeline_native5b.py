#!/usr/bin/env python3
"""WM3D-V7 Native 5B 集群训练流水线。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import stat
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

import yaml

from wm3d_v3.data.scale5b_contracts import atomic_write_json


LICENSE_CONFIRM = "YES_I_HAVE_ACCEPTED_THE_UPSTREAM_LICENSES"
YES = "YES"
STEP_RE = re.compile(r"^step_([0-9]{8})$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "doctor",
            "data",
            "lock",
            "download",
            "prepare",
            "cache",
            "canary",
            "eval",
            "train",
            "all",
            "status",
        ),
    )
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


class PipelineError(RuntimeError):
    pass


def _environment(name: str, *, default: str | None = None) -> str:
    value = os.environ.get(name, default)
    if value is None or not str(value).strip():
        raise PipelineError(f"site.env 缺少 {name}")
    return str(value)


def _integer(name: str, *, minimum: int = 1) -> int:
    try:
        value = int(_environment(name))
    except ValueError as exc:
        raise PipelineError(f"{name} 必须是整数") from exc
    if value < minimum:
        raise PipelineError(f"{name} 必须 >= {minimum}")
    return value


def _absolute(name: str, *, must_exist: bool = False) -> Path:
    value = Path(_environment(name))
    if not value.is_absolute():
        raise PipelineError(f"{name} 必须是绝对路径：{value}")
    if must_exist:
        value = value.resolve(strict=True)
    else:
        value = Path(os.path.abspath(value))
    return value


def _regular(path: Path, label: str) -> Path:
    info = os.lstat(path)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise PipelineError(f"{label} 必须是普通文件：{path}")
    return path.resolve(strict=True)


def _safe_token(path: Path) -> str:
    safe = _regular(path, "HF_TOKEN_FILE")
    if stat.S_IMODE(safe.stat().st_mode) & 0o077:
        raise PipelineError("HF_TOKEN_FILE 权限必须不高于 0600")
    token = safe.read_text(encoding="utf-8").strip()
    if not token or any(character.isspace() for character in token):
        raise PipelineError("HF_TOKEN_FILE 必须只含一行 token")
    return token


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(_regular(path, str(path)).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PipelineError(f"JSON 不是 object：{path}")
    return value


def _shell(command: Sequence[str]) -> str:
    return shlex.join([str(item) for item in command])


def _array_spec(indices: Iterable[int], concurrency: int) -> str:
    values = sorted(set(int(value) for value in indices))
    if not values:
        raise PipelineError("array index 为空")
    groups: list[str] = []
    start = previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        groups.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    groups.append(str(start) if start == previous else f"{start}-{previous}")
    return f"{','.join(groups)}%{concurrency}"


class Pipeline:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.dry_run = bool(args.dry_run or os.environ.get("PIPELINE_DRY_RUN") == "1")
        self.repo = _absolute("REPO_ROOT", must_exist=True)
        self.python = _absolute("PYTHON_BIN", must_exist=not self.dry_run)
        self.converter_python = _absolute(
            "CONVERTER_PYTHON_BIN", must_exist=not self.dry_run
        )
        self.raw = _absolute("RAW_ROOT")
        self.release = _absolute("RELEASE_ROOT")
        self.dataset = _absolute("DATASET_ROOT")
        self.assets = _absolute("ASSET_ROOT")
        self.runs = _absolute("RUNS_ROOT")
        self.logs = _absolute("LOG_ROOT")
        self.staging = _absolute("STAGING_ROOT")
        self.token_file = _absolute("HF_TOKEN_FILE")
        self.raw_lock = self.release / "raw_sources.lock.yaml"
        self.snapshots = self.raw / "snapshots"
        self.materialized = self.raw / "materialized"
        self.bootstrap = self.release / "dataset_bootstrap"
        self.world_size = _integer("TRAIN_NODES") * _integer("TRAIN_GPUS_PER_NODE")
        self.site = args.site.resolve(strict=True)
        self.submissions = self.release / "submissions"
        self._secret_env: dict[str, str] | None = None

    def banner(self, name: str) -> None:
        print(f"\n===== WM3D-V7: {name} =====", flush=True)

    def command(
        self,
        command: Sequence[str | Path],
        *,
        env: Mapping[str, str] | None = None,
        cwd: Path | None = None,
        capture: bool = False,
    ) -> str:
        values = [str(item) for item in command]
        print(f"+ {_shell(values)}", flush=True)
        if self.dry_run:
            return "DRY_RUN"
        merged = os.environ.copy()
        if env:
            merged.update({str(key): str(value) for key, value in env.items()})
        result = subprocess.run(
            values,
            cwd=cwd or self.repo,
            env=merged,
            check=True,
            text=True,
            stdout=subprocess.PIPE if capture else None,
        )
        return result.stdout.strip() if capture else ""

    def python_command(
        self, script: str, *arguments: str | Path, converter: bool = False
    ) -> str:
        executable = self.converter_python if converter else self.python
        return self.command([executable, self.repo / script, *arguments])

    def secret_env(self) -> dict[str, str]:
        if self._secret_env is None:
            if self.dry_run:
                token = "DRY_RUN_TOKEN"
            else:
                token = _safe_token(self.token_file)
            self._secret_env = {
                "HF_TOKEN": token,
                "HF_HOME": str(self.staging / "huggingface-home"),
                "HF_HUB_ENABLE_HF_TRANSFER": "0",
            }
        return self._secret_env

    def _slurm_base(self, *, wait: bool, nodes: int | None = None) -> list[str]:
        command = ["sbatch", "--parsable"]
        if wait:
            command.append("--wait")
        partition = _environment("SLURM_PARTITION")
        command.extend(["--partition", partition])
        account = os.environ.get("SLURM_ACCOUNT", "").strip()
        qos = os.environ.get("SLURM_QOS", "").strip()
        if account:
            command.extend(["--account", account])
        if qos:
            command.extend(["--qos", qos])
        if nodes is not None:
            command.extend(["--nodes", str(nodes)])
        command.extend(shlex.split(os.environ.get("SBATCH_EXTRA_ARGS", "")))
        return command

    def _export(self, values: Mapping[str, str | Path | int]) -> str:
        items = []
        for name, value in values.items():
            text = str(value)
            if any(character in text for character in (",", "\n", "\r")):
                raise PipelineError(f"Slurm export 值含非法字符：{name}")
            items.append(f"{name}={text}")
        return "ALL," + ",".join(items)

    def sbatch_script(
        self,
        script: str,
        *,
        exports: Mapping[str, str | Path | int],
        wait: bool,
        nodes: int | None = None,
        array: str | None = None,
    ) -> str:
        command = self._slurm_base(wait=wait, nodes=nodes)
        if array is not None:
            command.extend(["--array", array])
        command.extend(["--export", self._export(exports), self.repo / script])
        return self.command(command, capture=True)

    def sbatch_wrap(
        self,
        shell_command: str,
        *,
        wait: bool,
        array: str | None = None,
        converter: bool = False,
    ) -> str:
        command = self._slurm_base(wait=wait)
        if array is not None:
            command.extend(["--array", array])
        environment = {
            "REPO_ROOT": self.repo,
            "PYTHONPATH": self.repo,
            "PYTHON_BIN": self.converter_python if converter else self.python,
        }
        command.extend(["--export", self._export(environment), "--wrap", shell_command])
        return self.command(command, capture=True)

    def _mkdirs(self) -> None:
        if self.dry_run:
            return
        for path in (
            self.raw,
            self.release,
            self.snapshots,
            self.materialized,
            self.runs,
            self.logs,
            self.staging,
            self.submissions,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def doctor(self) -> None:
        self.banner("doctor")
        if self.world_size not in (64, 128):
            raise PipelineError("正式拓扑只支持 64 或 128 GPU")
        if _integer("TRAIN_GPUS_PER_NODE") != 8:
            raise PipelineError("每节点必须是 8 张 H200")
        if _integer("SHARD_DEGREE") != 8:
            raise PipelineError("当前正式 FSDP2 shard degree 必须是 8")
        for command in ("git", "sbatch", "srun", "scontrol"):
            if not self.dry_run and shutil.which(command) is None:
                raise PipelineError(f"缺少命令：{command}")
        if not self.dry_run:
            _regular(self.site, "site.env")
            _safe_token(self.token_file)
            if _environment("ACCEPT_DATA_LICENSES") != YES:
                raise PipelineError(
                    "先接受上游数据许可，并设置 ACCEPT_DATA_LICENSES=YES"
                )
            self._mkdirs()
            free = shutil.disk_usage(self.release).free
            minimum = int(os.environ.get("MIN_WORK_FREE_BYTES", "100000000000000"))
            if free < minimum:
                raise PipelineError(
                    f"WORK_ROOT 可用空间 {free} B，低于站点门槛 {minimum} B"
                )
        report = {
            "pass": True,
            "dry_run": self.dry_run,
            "repo": str(self.repo),
            "site": str(self.site),
            "world_size": self.world_size,
            "train_nodes": _integer("TRAIN_NODES"),
            "data_plan_hours": 6106.4,
            "architecture": "WM3D-V7 native3d",
        }
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))

    def lock(self) -> None:
        self.banner("冻结上游版本")
        template_path = self.repo / "configs/scale5b/raw_sources.lock.template.yaml"
        if self.raw_lock.exists():
            value = yaml.safe_load(self.raw_lock.read_text(encoding="utf-8"))
            template = yaml.safe_load(template_path.read_text(encoding="utf-8"))
            if tuple(value.get("sources", ())) != tuple(template["sources"]):
                raise PipelineError("已有 raw lock 与当前公开数据清单不一致")
            for name, source in value["sources"].items():
                if not re.fullmatch(r"[0-9a-f]{40}", str(source.get("revision", ""))):
                    raise PipelineError(f"已有 lock revision 非法：{name}")
                expected = template["sources"][name]
                for field in ("repo_id", "repo_type", "target_subdir"):
                    if source.get(field) != expected.get(field):
                        raise PipelineError(
                            f"已有 raw lock 的 {name}.{field} 与模板不一致"
                        )
            print(f"已完成：{self.raw_lock}")
            return
        overrides = {
            "droid": _environment("DROID_REVISION", default="AUTO"),
            "bridge": _environment("BRIDGE_REVISION", default="AUTO"),
            "atomic": _environment("ROBOCASA_ATOMIC_REVISION", default="AUTO"),
            "composite": _environment(
                "ROBOCASA_COMPOSITE_REVISION", default="AUTO"
            ),
            "mg": _environment("ROBOCASA_MG_REVISION", default="AUTO"),
            "agibot_world_2026_snapshot": _environment(
                "AGIBOT_2026_REVISION", default="AUTO"
            ),
            "agibot_beta_snapshot": _environment(
                "AGIBOT_BETA_REVISION", default="AUTO"
            ),
            "agibot_alpha_converter_snapshot": _environment(
                "AGIBOT_ALPHA_REVISION", default="AUTO"
            ),
        }
        command: list[str | Path] = [
            self.python,
            self.repo / "scripts/scale5b/resolve_source_lock.py",
            "--template",
            template_path,
            "--output",
            self.raw_lock,
            "--token-file",
            self.token_file,
            "--confirm-licenses",
            LICENSE_CONFIRM,
        ]
        for name, revision in overrides.items():
            command.extend(["--revision", f"{name}={revision}"])
        self.command(command, env=self.secret_env())

    def download(self) -> None:
        self.banner("下载公开数据快照")
        if not self.raw_lock.exists() and not self.dry_run:
            self.lock()
        self.command(
            [
                self.python,
                self.repo / "scripts/scale5b/download_raw_snapshots.py",
                "--lock",
                self.raw_lock,
                "--raw-root",
                self.snapshots,
                "--resume",
            ],
            env=self.secret_env(),
        )

    def _extract_collection(self, source_dir: str, output_name: str) -> None:
        archive_root = self.snapshots / "agibot_world_2026_snapshot" / source_dir
        output = self.materialized / output_name
        final = output / ".wm3d_v7_collection_materialization_receipt.json"
        if final.exists():
            print(f"已完成 collection：{output_name}")
            return
        shards = _integer("EXTRACT_SHARDS")
        array = _array_spec(range(shards), _integer("EXTRACT_CONCURRENCY"))
        shell_command = (
            f"cd {shlex.quote(str(self.repo))} && "
            f"{shlex.quote(str(self.python))} "
            "scripts/scale5b/safe_extract_lerobot_collection.py "
            f"--archive-root {shlex.quote(str(archive_root))} "
            f"--output-root {shlex.quote(str(output))} "
            f"--num-shards {shards} "
            '--shard-id "${SLURM_ARRAY_TASK_ID}"'
        )
        self.sbatch_wrap(shell_command, wait=True, array=array)
        self.python_command(
            "scripts/scale5b/safe_extract_lerobot_collection.py",
            "--archive-root",
            archive_root,
            "--output-root",
            output,
            "--finalize",
            "--download-receipt",
            self.snapshots
            / "agibot_world_2026_snapshot/.wm3d_v7_download_receipt.json",
        )

    def _prepare_beta(self) -> None:
        snapshot = self.snapshots / "agibot_beta_snapshot"
        raw = self.materialized / "agibot_beta_raw"
        converted = self.materialized / "agibot_beta"
        task_list = self.release / "agibot_beta_task_ids.txt"
        converted_receipt = (
            converted / ".wm3d_v7_beta_conversion_collection_receipt.json"
        )
        converter_snapshot = self.snapshots / "agibot_alpha_converter_snapshot"
        converter = converter_snapshot / "scripts/convert_to_lerobot.py"
        converter_download = converter_snapshot / ".wm3d_v7_download_receipt.json"
        converter_receipt = Path(
            os.environ.get(
                "CONVERTER_ENV_RECEIPT",
                str(self.converter_python.parents[1] / "environment_receipt.json"),
            )
        )
        if not task_list.exists():
            self.python_command(
                "scripts/scale5b/list_agibot_beta_tasks.py",
                "--raw-root",
                snapshot,
                "--output",
                task_list,
            )
        raw_receipt = raw / ".wm3d_v7_beta_materialization_receipt.json"
        if not raw_receipt.exists():
            self.python_command(
                "scripts/scale5b/safe_materialize_agibot_beta.py",
                "prepare",
                "--snapshot-root",
                snapshot,
                "--output-root",
                raw,
            )
            shards = _integer("EXTRACT_SHARDS")
            array = _array_spec(range(shards), _integer("EXTRACT_CONCURRENCY"))
            shell_command = (
                f"cd {shlex.quote(str(self.repo))} && "
                f"{shlex.quote(str(self.python))} "
                "scripts/scale5b/safe_materialize_agibot_beta.py extract "
                f"--snapshot-root {shlex.quote(str(snapshot))} "
                f"--output-root {shlex.quote(str(raw))} --num-shards {shards} "
                '--shard-id "${SLURM_ARRAY_TASK_ID}"'
            )
            self.sbatch_wrap(shell_command, wait=True, array=array)
            self.python_command(
                "scripts/scale5b/safe_materialize_agibot_beta.py",
                "finalize",
                "--snapshot-root",
                snapshot,
                "--output-root",
                raw,
            )
        if converted_receipt.exists():
            return
        if self.dry_run:
            task_count = 1000
        else:
            task_count = len(
                [
                    line
                    for line in task_list.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            )
        if task_count <= 0:
            raise PipelineError("AgiBot Beta task list 为空")
        array = _array_spec(range(task_count), _integer("BETA_CONVERT_CONCURRENCY"))
        shell_command = (
            f"cd {shlex.quote(str(self.repo))} && "
            f"{shlex.quote(str(self.converter_python))} "
            "scripts/scale5b/convert_agibot_beta_task.py "
            f"--raw-root {shlex.quote(str(raw))} "
            f"--output-root {shlex.quote(str(converted))} "
            f"--vendor-converter {shlex.quote(str(converter))} "
            f"--converter-download-receipt {shlex.quote(str(converter_download))} "
            f"--converter-environment-receipt {shlex.quote(str(converter_receipt))} "
            f"--task-list {shlex.quote(str(task_list))} "
            '--array-index "${SLURM_ARRAY_TASK_ID}"'
        )
        self.sbatch_wrap(shell_command, wait=True, array=array, converter=True)
        self.command(
            [
                self.converter_python,
                self.repo / "scripts/scale5b/convert_agibot_beta_task.py",
                "--raw-root",
                raw,
                "--output-root",
                converted,
                "--vendor-converter",
                converter,
                "--converter-download-receipt",
                converter_download,
                "--converter-environment-receipt",
                converter_receipt,
                "--task-list",
                task_list,
                "--finalize",
            ]
        )

    def _schema_audits(self) -> None:
        inputs = {
            "droid": (self.snapshots / "droid", False),
            "bridge": (self.snapshots / "bridge", False),
            "robocasa_atomic": (self.snapshots / "atomic", False),
            "robocasa_composite": (self.snapshots / "composite", False),
            "robocasa_mg": (self.snapshots / "mg", False),
            "agibot2026_imitation": (
                self.materialized / "agibot2026_imitation",
                True,
            ),
            "agibot2026_rich": (self.materialized / "agibot2026_rich", True),
            "agibot2026_reinforcement": (
                self.materialized / "agibot2026_reinforcement",
                True,
            ),
            "agibot_beta": (self.materialized / "agibot_beta", True),
        }
        for name, (root, collection) in inputs.items():
            output = self.release / f"schema_{name}.json"
            if output.exists():
                continue
            command: list[str | Path] = [
                self.python,
                self.repo / "scripts/scale5b/inspect_lerobot_schema.py",
                "--root",
                root,
                "--output",
                output,
            ]
            if collection:
                command.extend(
                    ["--collection", "--max-roots", "1000000", "--require-homogeneous"]
                )
            self.command(command)

    def _prepare_assets(self) -> None:
        if (self.assets / "receipt.json").exists():
            self.python_command(
                "scripts/scale5b/verify_encoder_assets.py",
                "--asset-root",
                self.assets,
                "--deep",
            )
            return
        self.command(
            [
                self.python,
                self.repo / "scripts/scale5b/prepare_encoder_bundle.py",
                "--staging-root",
                self.staging / "encoder_assets",
                "--output-root",
                self.assets,
                "--token-file",
                self.token_file,
                "--vggt-source-commit",
                _environment("VGGT_SOURCE_COMMIT"),
                "--vggt-model-revision",
                _environment("VGGT_MODEL_REVISION"),
                "--task-model-revision",
                _environment("TASK_MODEL_REVISION", default="AUTO"),
            ],
            env=self.secret_env(),
        )

    def prepare(self) -> None:
        self.banner("数据转换、schema 审计与 episode plan")
        self._extract_collection("ImitationLearning", "agibot2026_imitation")
        self._extract_collection("RichInteraction", "agibot2026_rich")
        self._extract_collection("ReinforcementLearning", "agibot2026_reinforcement")
        self._prepare_beta()
        self._schema_audits()
        self._prepare_assets()
        contract = self.bootstrap / "dataset_contract.json"
        if not contract.exists():
            self.python_command(
                "scripts/scale5b/compile_dataset_contract.py",
                "--inventory",
                self.repo
                / "configs/scale5b/dataset_inventory_public6106h.template.yaml",
                "--output",
                contract,
            )
        source_receipt = self.dataset / "receipts/source_scan.json"
        if not source_receipt.exists():
            environment = {
                "DROID_ROOT": str(self.snapshots / "droid"),
                "BRIDGE_ROOT": str(self.snapshots / "bridge"),
                "ATOMIC_ROOT": str(self.snapshots / "atomic"),
                "COMPOSITE_ROOT": str(self.snapshots / "composite"),
                "MG_ROOT": str(self.snapshots / "mg"),
                "AGIBOT_2026_IMITATION_ROOT": str(
                    self.materialized / "agibot2026_imitation"
                ),
                "AGIBOT_2026_RICH_ROOT": str(self.materialized / "agibot2026_rich"),
                "AGIBOT_2026_REINFORCEMENT_ROOT": str(
                    self.materialized / "agibot2026_reinforcement"
                ),
                "AGIBOT_BETA_ROOT": str(self.materialized / "agibot_beta"),
            }
            self.command(
                [
                    self.python,
                    self.repo / "scripts/scale5b/scan_sources.py",
                    "--dataset-contract",
                    contract,
                    "--source-layouts",
                    self.repo
                    / "configs/scale5b/source_layouts_public6106h.template.json",
                    "--output-root",
                    self.dataset,
                ],
                env=environment,
            )

    def _asset_revisions(self) -> tuple[str, str]:
        if self.dry_run:
            return _environment("VGGT_MODEL_REVISION"), "0" * 40
        receipt = _json(self.assets / "receipt.json")
        return (
            str(receipt["assets"]["vggt_model"]["revision"]),
            str(receipt["assets"]["task_model"]["revision"]),
        )

    def cache(self) -> None:
        self.banner("action/task/VGGT cache、merge 与 seal")
        stats = self.dataset / "control/action_stats.json"
        stat_shards = _integer("ACTION_STATS_SHARDS")
        partial_root = self.dataset / "control/action_stats_partials"
        if not stats.exists():
            missing = [
                index
                for index in range(stat_shards)
                if not (partial_root / f"partial_{index:05d}.npz").exists()
            ]
            if missing:
                self.sbatch_script(
                    "scripts/scale5b/sbatch_action_stats_array.sh",
                    exports={
                        "REPO_ROOT": self.repo,
                        "DATASET_ROOT": self.dataset,
                        "NUM_SHARDS": stat_shards,
                        "GLOBAL_SAMPLE_BUDGET": 8_000_000,
                        "PYTHON_BIN": self.python,
                        "RESUME_ARRAY": 1,
                    },
                    wait=True,
                    array=_array_spec(missing, _integer("ACTION_STATS_CONCURRENCY")),
                )
            partials = (
                [
                    str(partial_root / f"partial_{index:05d}.npz")
                    for index in range(stat_shards)
                ]
                if self.dry_run
                else [str(path) for path in sorted(partial_root.glob("partial_*.npz"))]
            )
            self.command(
                [
                    self.python,
                    self.repo / "scripts/scale5b/build_action_stats.py",
                    "merge",
                    "--partials",
                    *partials,
                    "--output",
                    stats,
                    "--clip",
                    "5.0",
                ]
            )
        vggt_revision, task_revision = self._asset_revisions()
        task_index = self.dataset / "control/task_index.json"
        if not task_index.exists():
            self.sbatch_script(
                "scripts/scale5b/sbatch_task_bank.sh",
                exports={
                    "REPO_ROOT": self.repo,
                    "DATASET_ROOT": self.dataset,
                    "ENCODER_ASSET_ROOT": self.assets,
                    "TASK_MODEL_REVISION": task_revision,
                    "PYTHON_BIN": self.python,
                },
                wait=True,
            )
        encoder_shards = _integer("ENCODER_SHARDS")
        receipt_root = self.dataset / "receipts/encode_workers"
        missing = [
            index
            for index in range(encoder_shards)
            if not (receipt_root / f"worker_{index:05d}.json").exists()
        ]
        if missing:
            self.sbatch_script(
                "scripts/scale5b/sbatch_encode_array.sh",
                exports={
                    "REPO_ROOT": self.repo,
                    "DATASET_ROOT": self.dataset,
                    "ENCODER_ASSET_ROOT": self.assets,
                    "VGGT_REVISION": vggt_revision,
                    "NUM_SHARDS": encoder_shards,
                    "PYTHON_BIN": self.python,
                    "RESUME_ARRAY": 1,
                },
                wait=True,
                array=_array_spec(missing, _integer("ENCODER_CONCURRENCY")),
            )
        seal = self.dataset / "receipts/dataset_seal.json"
        if not seal.exists():
            self.python_command(
                "scripts/scale5b/merge_and_seal.py",
                "--dataset-root",
                self.dataset,
                "--num-encoder-shards",
                str(encoder_shards),
                "--index-rows-per-file",
                "1000000",
            )
        self.python_command(
            "scripts/scale5b/verify_dataset.py",
            "--dataset-root",
            self.dataset,
            "--mode",
            "control",
            "--sample-windows-per-source",
            "4",
        )
        self.python_command(
            "scripts/scale5b/verify_dataset.py",
            "--dataset-root",
            self.dataset,
            "--mode",
            "deep",
            "--sample-windows-per-source",
            "16",
        )

    def _environment_receipt(self) -> Path:
        return Path(
            os.environ.get(
                "TRAIN_ENV_RECEIPT",
                str(self.python.parents[1] / "environment_receipt.json"),
            )
        )

    def _code_receipt(self) -> Path:
        output = self.release / "code_receipt.json"
        if not output.exists():
            self.python_command(
                "scripts/scale5b/seal_code.py",
                "--repo-root",
                self.repo,
                "--output",
                output,
            )
        return output

    def _lineage(self, name: str) -> str:
        code = self._code_receipt()
        seal = self.dataset / "receipts/dataset_seal.json"
        if self.dry_run:
            seed = f"{name}:dry-run"
        else:
            seed = f"{name}:{_sha256_file(code)}:{_sha256_file(seal)}"
        return hashlib.sha256(seed.encode("utf-8")).hexdigest()

    def _template(self, *, canary: bool) -> Path:
        name = "CANARY_CONFIG_TEMPLATE" if canary else "FORMAL_CONFIG_TEMPLATE"
        path = Path(_environment(name))
        if not path.is_absolute():
            path = self.repo / path
        return _regular(path, name)

    def _total_steps(self, *, canary: bool) -> int:
        value = yaml.safe_load(
            self._template(canary=canary).read_text(encoding="utf-8")
        )
        try:
            steps = int(value["train"]["total_steps"])
        except (KeyError, TypeError, ValueError) as exc:
            raise PipelineError("配置缺少合法 train.total_steps") from exc
        if steps <= 0:
            raise PipelineError("train.total_steps 必须为正数")
        return steps

    def _materialize(self, *, canary: bool) -> tuple[Path, Path]:
        nodes = _integer("TRAIN_NODES")
        name = _environment("CANARY_RUN_NAME" if canary else "FORMAL_RUN_NAME")
        step = self._total_steps(canary=canary)
        kind = "canary" if canary else "formal"
        config = self.release / f"{kind}_step{step}_{self.world_size}gpu.yaml"
        output = self.runs / name
        if config.exists():
            return config, output
        template = self._template(canary=canary)
        self.python_command(
            "scripts/scale5b/materialize_config.py",
            "--template",
            template,
            "--dataset-root",
            self.dataset,
            "--code-receipt",
            self._code_receipt(),
            "--code-root",
            self.repo,
            "--environment-contract",
            self.repo / "environments/scale5b/environment_contract.json",
            "--environment-receipt",
            self._environment_receipt(),
            "--output-root",
            output,
            "--output-config",
            config,
            "--run-name",
            name,
            "--run-lineage",
            self._lineage(name),
            "--world-size",
            str(nodes * 8),
            "--shard-degree",
            _environment("SHARD_DEGREE"),
            "--global-batch-size",
            _environment("GLOBAL_BATCH_SIZE"),
            "--micro-batch-size",
            _environment("MICRO_BATCH_SIZE"),
        )
        return config, output

    def _highest_checkpoint(self, run_root: Path) -> Path | None:
        root = run_root / "checkpoints"
        if not root.is_dir():
            return None
        complete = []
        for path in root.iterdir():
            match = STEP_RE.fullmatch(path.name)
            if match and path.is_dir() and (path / "COMMITTED.json").is_file():
                complete.append((int(match.group(1)), path))
        return max(complete, default=(0, None))[1]

    def _training_job(self, config: Path, run_root: Path, *, wait: bool) -> str:
        resume = self._highest_checkpoint(run_root)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        kind = "canary" if "canary" in config.name else "formal"
        log_root = self.logs / f"{kind}_{timestamp}"
        rendezvous = f"wm3d-v7-native5b-{kind}-{self._lineage(run_root.name)[:16]}"
        exports: dict[str, str | Path | int] = {
            "CONFIG": config,
            "REPO_ROOT": self.repo,
            "LOG_ROOT": log_root,
            "RDZV_ID": rendezvous,
            "PYTHON_BIN": self.python,
            "TORCHRUN_BIN": _environment("TORCHRUN_BIN"),
        }
        if resume is not None:
            exports["RESUME_CHECKPOINT"] = resume
        return self.sbatch_script(
            "scripts/scale5b/sbatch_native5b_h200.sh",
            exports=exports,
            wait=wait,
            nodes=_integer("TRAIN_NODES"),
        )

    def _config_for_checkpoint(self, checkpoint: Path) -> Path:
        run_root = checkpoint.parent.parent.resolve()
        for path in sorted(self.release.glob("*.yaml")):
            value = yaml.safe_load(path.read_text(encoding="utf-8"))
            if (
                isinstance(value, dict)
                and Path(value.get("run", {}).get("output_root", "")) == run_root
            ):
                return path.resolve(strict=True)
        raise PipelineError(
            f"找不到绑定 checkpoint run 的 materialized config：{run_root}"
        )

    def evaluate(self, checkpoint: Path) -> Path:
        checkpoint = checkpoint.resolve(strict=not self.dry_run)
        if not self.dry_run:
            match = STEP_RE.fullmatch(checkpoint.name)
            if not match or not (checkpoint / "COMMITTED.json").is_file():
                raise PipelineError("eval 只接受完整编号 checkpoint")
            config = self._config_for_checkpoint(checkpoint)
        else:
            step = self._total_steps(canary=True)
            config = self.release / f"canary_step{step}_{self.world_size}gpu.yaml"
        output = (
            self.release / "eval" / f"{checkpoint.parent.parent.name}_{checkpoint.name}"
        )
        report = output / "report.json"
        if report.exists():
            if _json(report).get("pass") is not True:
                raise PipelineError(f"已有 eval report 未通过：{report}")
            return report
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.sbatch_script(
            "scripts/scale5b/sbatch_eval_native5b.sh",
            exports={
                "CONFIG": config,
                "CHECKPOINT": checkpoint,
                "EVAL_OUTPUT_ROOT": output,
                "REPO_ROOT": self.repo,
                "EVAL_LOG_ROOT": self.logs / f"eval_{checkpoint.name}_{timestamp}",
                "EVAL_STEPS": int(os.environ.get("EVAL_STEPS", "64")),
                "PYTHON_BIN": self.python,
                "TORCHRUN_BIN": _environment("TORCHRUN_BIN"),
            },
            wait=True,
            nodes=_integer("TRAIN_NODES"),
        )
        if not self.dry_run and _json(report).get("pass") is not True:
            raise PipelineError(f"eval 未通过：{report}")
        return report

    def canary(self) -> None:
        step = self._total_steps(canary=True)
        self.banner(f"{step}-step canary 与显式 RGB/depth/point/action eval")
        config, output = self._materialize(canary=True)
        checkpoint = output / f"checkpoints/step_{step:08d}"
        if not (checkpoint / "COMMITTED.json").exists():
            self._training_job(config, output, wait=True)
        report = self.evaluate(checkpoint)
        print(f"canary eval PASS：{report}")

    def train(self) -> None:
        final_step = self._total_steps(canary=False)
        canary_step = self._total_steps(canary=True)
        self.banner(f"提交 {final_step}-step formal")
        canary_output = self.runs / _environment("CANARY_RUN_NAME")
        canary_checkpoint = canary_output / f"checkpoints/step_{canary_step:08d}"
        report = (
            self.release
            / "eval"
            / f"{canary_output.name}_{canary_checkpoint.name}"
            / "report.json"
        )
        if not self.dry_run and _json(report).get("pass") is not True:
            raise PipelineError("没有通过的 canary eval，不允许提交 formal")
        config, output = self._materialize(canary=False)
        final = output / f"checkpoints/step_{final_step:08d}/COMMITTED.json"
        if final.exists():
            print(f"formal 已完成：{final}")
            return
        job_id = self._training_job(config, output, wait=False)
        print(f"formal 已提交，Slurm job={job_id}")
        if not self.dry_run:
            self.submissions.mkdir(parents=True, exist_ok=True)
            receipt = self.submissions / f"formal_job_{job_id.replace(';', '_')}.json"
            atomic_write_json(
                receipt,
                {
                    "schema": "wm3d_v7_native5b_slurm_submission_v1",
                    "job_id": job_id,
                    "config": str(config),
                    "output_root": str(output),
                    "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
                },
                exclusive=True,
            )

    def all(self) -> None:
        self.data()
        self.canary()
        self.train()

    def data(self) -> None:
        self.doctor()
        self._mkdirs()
        self.lock()
        self.download()
        self.prepare()
        self.cache()

    def status(self) -> None:
        canary = self.runs / _environment("CANARY_RUN_NAME")
        formal = self.runs / _environment("FORMAL_RUN_NAME")
        canary_step = self._total_steps(canary=True)
        latest = self._highest_checkpoint(formal)
        report = {
            "raw_lock": self.raw_lock.is_file(),
            "dataset_seal": (self.dataset / "receipts/dataset_seal.json").is_file(),
            "canary_checkpoint": (
                canary / f"checkpoints/step_{canary_step:08d}/COMMITTED.json"
            ).is_file(),
            "canary_eval": (
                self.release
                / "eval"
                / f"{canary.name}_step_{canary_step:08d}/report.json"
            ).is_file(),
            "formal_latest_numbered_checkpoint": str(latest) if latest else None,
        }
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))


def main() -> None:
    args = parse_args()
    pipeline = Pipeline(args)
    command = args.command
    if command == "doctor":
        pipeline.doctor()
    elif command == "data":
        pipeline.data()
    elif command == "lock":
        pipeline.lock()
    elif command == "download":
        pipeline.download()
    elif command == "prepare":
        pipeline.prepare()
    elif command == "cache":
        pipeline.cache()
    elif command == "canary":
        pipeline.canary()
    elif command == "eval":
        if args.checkpoint is None:
            raise PipelineError("eval 需要 --checkpoint")
        print(pipeline.evaluate(args.checkpoint))
    elif command == "train":
        pipeline.train()
    elif command == "all":
        pipeline.all()
    elif command == "status":
        pipeline.status()
    else:
        raise AssertionError(command)


if __name__ == "__main__":
    try:
        main()
    except (PipelineError, subprocess.CalledProcessError) as error:
        print(f"WM3D-V7 pipeline FAIL: {error}", file=sys.stderr)
        raise SystemExit(2) from error
