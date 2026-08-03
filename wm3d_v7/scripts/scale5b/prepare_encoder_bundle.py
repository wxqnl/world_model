#!/usr/bin/env python3
"""Explicitly download and seal the offline VGGT/task-encoder asset bundle."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile

from huggingface_hub import HfApi, snapshot_download


REVISION_RE = re.compile(r"^[0-9a-f]{40,64}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--vggt-source-commit", required=True)
    parser.add_argument("--vggt-model-revision", required=True)
    parser.add_argument("--task-model-revision", default="AUTO")
    return parser.parse_args()


def _token(path: Path) -> str:
    info = os.lstat(path)
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ValueError("HF token file 必须是普通文件")
    safe = path.resolve(strict=True)
    if stat.S_IMODE(safe.stat().st_mode) & 0o077:
        raise PermissionError("HF token file 权限必须不高于 0600")
    value = safe.read_text(encoding="utf-8").strip()
    if not value or any(character.isspace() for character in value):
        raise ValueError("HF token file 内容非法")
    return value


def _git_source(root: Path, commit: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"vggt-{commit}"
    if target.exists():
        actual = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=target, text=True
        ).strip()
        if actual != commit:
            raise ValueError(f"已有 VGGT source commit 漂移：{actual}")
        return target.resolve(strict=True)
    temporary = root / f".vggt-{commit}.incomplete"
    if temporary.exists():
        raise FileExistsError(f"存在未审失败目录：{temporary}")
    subprocess.run(
        [
            "git",
            "clone",
            "--filter=blob:none",
            "https://github.com/facebookresearch/vggt.git",
            str(temporary),
        ],
        check=True,
    )
    subprocess.run(["git", "checkout", "--detach", commit], cwd=temporary, check=True)
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=temporary, text=True
    ).strip()
    if actual != commit:
        raise ValueError("VGGT checkout 未落在指定 commit")
    os.replace(temporary, target)
    return target.resolve(strict=True)


def _snapshot(
    *,
    root: Path,
    repo_id: str,
    revision: str,
    token: str,
) -> Path:
    target = root / repo_id.replace("/", "--") / revision
    marker = target / ".wm3d_v7_snapshot.json"
    expected = {"repo_id": repo_id, "revision": revision}
    if marker.exists():
        if json.loads(marker.read_text(encoding="utf-8")) != expected:
            raise ValueError(f"snapshot marker 漂移：{target}")
        return target.resolve(strict=True)
    if target.exists():
        raise FileExistsError(f"snapshot 目录存在但没有完成 marker：{target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{revision}.incomplete"
    if temporary.exists():
        raise FileExistsError(f"存在未审失败目录：{temporary}")
    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
        token=token,
        local_dir=temporary,
    )
    descriptor, name = tempfile.mkstemp(
        prefix=".snapshot.", suffix=".tmp", dir=temporary
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(expected, handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(name, temporary / marker.name)
    os.replace(temporary, target)
    return target.resolve(strict=True)


def main() -> None:
    args = parse_args()
    revisions = (args.vggt_source_commit, args.vggt_model_revision)
    if any(not REVISION_RE.fullmatch(value) for value in revisions):
        raise ValueError("VGGT revision 必须是不可变 commit")
    if args.output_root.exists():
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("verify_encoder_assets.py")),
                "--asset-root",
                str(args.output_root),
                "--deep",
            ],
            check=True,
        )
        return

    token = _token(args.token_file)
    api = HfApi(token=token)
    vggt_info = api.model_info("facebook/VGGT-1B", revision=args.vggt_model_revision)
    if str(vggt_info.sha) != args.vggt_model_revision:
        raise ValueError("VGGT model revision 解析漂移")
    if args.task_model_revision == "AUTO":
        task_revision = str(api.model_info("google/flan-t5-xl", revision="main").sha)
    else:
        task_revision = args.task_model_revision
        task_info = api.model_info("google/flan-t5-xl", revision=task_revision)
        if str(task_info.sha) != task_revision:
            raise ValueError("task model revision 解析漂移")
    if not REVISION_RE.fullmatch(task_revision):
        raise ValueError("task model revision 不是不可变 commit")

    staging = args.staging_root.resolve()
    staging.mkdir(parents=True, exist_ok=True)
    source = _git_source(staging / "source", args.vggt_source_commit)
    models = staging / "models"
    vggt_snapshot = _snapshot(
        root=models,
        repo_id="facebook/VGGT-1B",
        revision=args.vggt_model_revision,
        token=token,
    )
    task_snapshot = _snapshot(
        root=models,
        repo_id="google/flan-t5-xl",
        revision=task_revision,
        token=token,
    )
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("prepare_encoder_assets.py")),
            "--vggt-source-root",
            str(source),
            "--vggt-source-commit",
            args.vggt_source_commit,
            "--vggt-snapshot",
            str(vggt_snapshot),
            "--vggt-revision",
            args.vggt_model_revision,
            "--task-snapshot",
            str(task_snapshot),
            "--task-revision",
            task_revision,
            "--output-root",
            str(args.output_root),
        ],
        check=True,
    )
    print(
        json.dumps(
            {
                "pass": True,
                "asset_root": str(args.output_root),
                "vggt_revision": args.vggt_model_revision,
                "task_revision": task_revision,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
