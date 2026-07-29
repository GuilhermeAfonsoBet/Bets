"""Atomic write helpers and last-known-good management."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Optional


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(text, encoding=encoding)
    os.replace(tmp, path)


def atomic_write_json(path: Path, obj: Any, *, indent: int = 2) -> None:
    atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=indent, default=str) + "\n")


def update_latest_symlink(latest: Path, target: Path) -> None:
    latest = Path(latest)
    target = Path(target)
    latest.parent.mkdir(parents=True, exist_ok=True)
    tmp = latest.with_name(latest.name + f".tmp.{os.getpid()}")
    if tmp.exists() or tmp.is_symlink():
        tmp.unlink()
    try:
        tmp.symlink_to(target.resolve())
        os.replace(tmp, latest)
    except OSError:
        # Fallback copy for filesystems without symlink replace support
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        try:
            latest.symlink_to(target.resolve())
        except OSError:
            shutil.copy2(target, latest)


def promote_last_known_good(lkg_dir: Path, snapshot_path: Path, report_path: Path) -> None:
    lkg_dir = Path(lkg_dir)
    lkg_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(snapshot_path, lkg_dir / "last_known_good_snapshot.json")
    shutil.copy2(report_path, lkg_dir / "last_known_good_report.md")
    atomic_write_json(
        lkg_dir / "last_known_good_meta.json",
        {"snapshot": str(snapshot_path), "report": str(report_path)},
    )


def read_json(path: Path) -> Optional[dict]:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
