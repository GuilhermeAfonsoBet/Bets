"""Read-only security checksums — confirm operational artifacts unchanged."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional


WATCH_RELATIVE = [
    "logs/wf_policy_current.json",
    "logs/bridge_risk_params.json",
    # executor / accounting / clv / timers are VPS paths; hashed when present
]


def sha256_file(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot_checksums(root: Path, extra: Optional[List[Path]] = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for rel in WATCH_RELATIVE:
        p = root / rel
        out[rel] = {"path": str(p), "sha256": sha256_file(p), "exists": p.is_file()}
    for p in extra or []:
        out[str(p)] = {"path": str(p), "sha256": sha256_file(p), "exists": p.is_file()}
    return out


def compare_checksums(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    changed = []
    for k, b in before.items():
        a = after.get(k) or {}
        if b.get("sha256") != a.get("sha256"):
            changed.append(k)
    return {
        "unchanged": len(changed) == 0,
        "changed": changed,
        "policy_altered": any("wf_policy" in c or "policy" in c.lower() for c in changed),
        "bridge_altered": any("bridge_risk" in c for c in changed),
    }
