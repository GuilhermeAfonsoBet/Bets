"""Filesystem helpers for accounting (no scraper dependency)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.partial")
    try:
        with tmp.open("wb") as f:
            f.write(data)
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass
        os.replace(tmp, path)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


def atomic_write_json(path: Path, obj: Dict[str, Any]) -> None:
    raw = (json.dumps(obj, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    atomic_write_bytes(path, raw)
