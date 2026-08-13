from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

import wm3d_v3.stage1_planner.execution_snapshot as snapshot_contract
from wm3d_v3.stage1_planner.execution_snapshot import (
    ExecutionSnapshotError,
    ExecutionSnapshotPlan,
    PinnedExecutionPath,
    ReadOnlyBindMount,
    scan_regular_tree,
    validate_execution_snapshot,
)
from wm3d_v3.stage1_planner.rollout_audit import TrustedOutputRoot


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def test_execution_snapshot_survives_provenance_swap_during_child(tmp_path: Path) -> None:
    provenance = tmp_path / "provenance"
    provenance.mkdir()
    source = provenance / "input.json"
    original = b'{"sealed":true}\n'
    source.write_bytes(original)
    output_parent = tmp_path / "authority"
    with TrustedOutputRoot(output_parent) as output:
        root = output_parent / "execution/snapshot"
        plan = ExecutionSnapshotPlan(output, root)
        snapshot = root / "inputs/input.json"
        plan.add_file(
            source, snapshot, expected_sha256=_digest(original),
            size=len(original), kind="selected input",
        )
        manifest, _digest_value = plan.seal(
            output_parent / "execution_snapshot_manifest.json"
        )

    replacement = b'{"attacker":true}\n'
    source.write_bytes(replacement)
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            "from pathlib import Path; import sys; "
            "sys.stdout.buffer.write(Path(sys.argv[1]).read_bytes())",
            str(snapshot),
        ],
        check=True,
        capture_output=True,
    )
    assert child.stdout == original
    validate_execution_snapshot(
        manifest, verify_provenance=False, verify_snapshots=True
    )
    with pytest.raises(ExecutionSnapshotError, match="provenance mismatch"):
        validate_execution_snapshot(
            manifest, verify_provenance=True, verify_snapshots=True
        )
    source.write_bytes(original)
    validate_execution_snapshot(
        manifest, verify_provenance=True, verify_snapshots=True
    )


def test_execution_snapshot_fails_closed_on_space_symlink_and_clobber(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"sealed")
    symlink = tmp_path / "link"
    symlink.symlink_to(source)
    with pytest.raises(Exception, match="symlink"):
        scan_regular_tree(tmp_path, label="test tree")

    output_parent = tmp_path / "output"
    with TrustedOutputRoot(output_parent) as output:
        root = output_parent / "snapshot"
        plan = ExecutionSnapshotPlan(output, root)
        target = root / "source"
        plan.add_verified_file(source, target, kind="input")
        monkeypatch.setattr(
            snapshot_contract.os,
            "statvfs",
            lambda _path: SimpleNamespace(f_bavail=0, f_frsize=4096),
        )
        with pytest.raises(ExecutionSnapshotError, match="insufficient space"):
            plan.preflight(safety_margin_bytes=0)

    monkeypatch.undo()
    output_parent_2 = tmp_path / "output2"
    with TrustedOutputRoot(output_parent_2) as output:
        root = output_parent_2 / "snapshot"
        plan = ExecutionSnapshotPlan(output, root)
        target = root / "source"
        plan.add_verified_file(source, target, kind="input")
        output.publish(target, b"different", label="collision seed")
        with pytest.raises(FileExistsError):
            plan.seal(output_parent_2 / "manifest.json")


def test_execution_snapshot_manifest_tamper_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"sealed")
    output_parent = tmp_path / "output"
    with TrustedOutputRoot(output_parent) as output:
        root = output_parent / "snapshot"
        plan = ExecutionSnapshotPlan(output, root)
        plan.add_verified_file(source, root / "source", kind="input")
        manifest, _sha = plan.seal(output_parent / "manifest.json")
    tampered = json.loads(json.dumps(manifest))
    tampered["rows"][0]["snapshot_sha256"] = "0" * 64
    with pytest.raises(ExecutionSnapshotError):
        validate_execution_snapshot(
            tampered, verify_provenance=False, verify_snapshots=True
        )


def test_materialized_symlink_tree_is_regular_and_swap_independent(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside.py"
    outside.write_bytes(b"sealed stdlib referent")
    source = tmp_path / "stdlib"
    source.mkdir()
    (source / "sitecustomize.py").symlink_to(outside)
    rows = scan_regular_tree(
        source, label="stdlib", materialize_file_symlinks=True
    )
    assert rows == [{
        "path": "sitecustomize.py",
        "source_path": str(outside),
        "size": len(b"sealed stdlib referent"),
        "sha256": _digest(b"sealed stdlib referent"),
    }]
    output_parent = tmp_path / "output"
    with TrustedOutputRoot(output_parent) as output:
        root = output_parent / "snapshot"
        plan = ExecutionSnapshotPlan(output, root)
        plan.add_tree(source, root / "stdlib", rows, kind="stdlib")
        manifest, _sha = plan.seal(output_parent / "manifest.json")
    copied = root / "stdlib/sitecustomize.py"
    assert copied.is_file() and not copied.is_symlink()
    outside.write_bytes(b"changed")
    assert copied.read_bytes() == b"sealed stdlib referent"
    validate_execution_snapshot(
        manifest, verify_provenance=False, verify_snapshots=True
    )


def test_pinned_execution_dirfd_survives_directory_swap_restore(
    tmp_path: Path,
) -> None:
    named = tmp_path / "snapshot"
    named.mkdir()
    (named / "input").write_bytes(b"sealed")
    anchor = PinnedExecutionPath(named, directory=True, label="snapshot")
    original = tmp_path / "snapshot.original"
    named.rename(original)
    named.mkdir()
    (named / "input").write_bytes(b"replacement")
    try:
        child = subprocess.run(
            [
                sys.executable, "-c",
                "from pathlib import Path; import sys; "
                "sys.stdout.buffer.write(Path('./input').read_bytes())",
            ],
            cwd=anchor.alias(),
            pass_fds=(anchor.fd,),
            check=True,
            capture_output=True,
        )
        assert child.stdout == b"sealed"
    finally:
        anchor.close()
        (named / "input").unlink()
        named.rmdir()
        original.rename(named)
    assert (named / "input").read_bytes() == b"sealed"


def test_read_only_inputs_survive_root_swap_and_reject_child_replacement(
    tmp_path: Path,
) -> None:
    named = tmp_path / "snapshot"
    inputs = named / "inputs"
    inputs.mkdir(parents=True)
    (inputs / "payload").write_bytes(b"sealed")
    (named / "generated").mkdir()
    anchor = PinnedExecutionPath(named, directory=True, label="snapshot")
    mount = ReadOnlyBindMount(inputs, label="snapshot inputs")
    moved = tmp_path / "snapshot.moved"
    mount.__enter__()
    try:
        named.rename(moved)
        replacement_inputs = named / "inputs"
        replacement_inputs.mkdir(parents=True)
        (replacement_inputs / "payload").write_bytes(b"replacement")
        child = subprocess.run(
            [
                sys.executable,
                "-c",
                "from pathlib import Path; import sys; "
                "sys.stdout.buffer.write(Path('inputs/payload').read_bytes())",
            ],
            cwd=anchor.alias(),
            pass_fds=(anchor.fd,),
            check=True,
            capture_output=True,
        )
        assert child.stdout == b"sealed"
        write = subprocess.run(
            [
                sys.executable,
                "-c",
                "from pathlib import Path; Path('inputs/payload').write_bytes(b'x')",
            ],
            cwd=anchor.alias(),
            pass_fds=(anchor.fd,),
            capture_output=True,
            text=True,
        )
        assert write.returncode != 0
        with pytest.raises(OSError):
            os.rename(
                "inputs",
                "inputs.replaced",
                src_dir_fd=anchor.fd,
                dst_dir_fd=anchor.fd,
            )
        mount.verify(
            target=anchor.alias("inputs"), pass_fds=(anchor.fd,)
        )
    finally:
        mount.close(
            target=anchor.alias("inputs"), pass_fds=(anchor.fd,)
        )
        anchor.close()
        if named.exists():
            for path in sorted(named.rglob("*"), reverse=True):
                path.unlink() if path.is_file() else path.rmdir()
            named.rmdir()
        moved.rename(named)
    assert (named / "inputs/payload").read_bytes() == b"sealed"


def test_private_mount_namespace_reexec_is_distinct() -> None:
    child = subprocess.run(
        [
            "unshare",
            "--mount",
            "--propagation",
            "private",
            "--",
            sys.executable,
            "-c",
            "import os; "
            "assert os.stat('/proc/self/ns/mnt').st_ino != "
            "os.stat(f'/proc/{os.getppid()}/ns/mnt').st_ino",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert child.returncode == 0


def test_private_mount_namespace_rejects_forged_marker() -> None:
    code = (
        "from wm3d_v3.stage1_planner.execution_snapshot import "
        "enter_private_mount_namespace; enter_private_mount_namespace()"
    )
    forged_environment = dict(os.environ)
    forged_environment["WM3D_STAGE1_REPLAY_PRIVATE_MOUNT_NAMESPACE"] = "1"
    forged_environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    forged = subprocess.run(
        [sys.executable, "-c", code],
        env=forged_environment,
        capture_output=True,
        text=True,
    )
    assert forged.returncode != 0
    assert "marker was set outside" in forged.stderr


def test_private_mount_namespace_removes_shared_propagation() -> None:
    environment = dict(os.environ)
    environment["WM3D_STAGE1_REPLAY_PRIVATE_MOUNT_NAMESPACE"] = "1"
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    code = (
        "from wm3d_v3.stage1_planner.execution_snapshot import "
        "enter_private_mount_namespace; enter_private_mount_namespace(); "
        "from pathlib import Path; "
        "assert not any(any(field.startswith(prefix) "
        "for prefix in ('shared:', 'master:', 'propagate_from:')) "
        "for line in Path('/proc/self/mountinfo').read_text().splitlines() "
        "for field in line.split()[6:line.split().index('-')])"
    )
    child = subprocess.run(
        [
            "unshare",
            "--mount",
            "--propagation",
            "shared",
            "--",
            sys.executable,
            "-c",
            code,
        ],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert child.returncode == 0
