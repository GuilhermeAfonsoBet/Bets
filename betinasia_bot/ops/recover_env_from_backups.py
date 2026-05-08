#!/usr/bin/env python3
"""
Recupera .env com segurança a partir de backups locais.

Uso comum:
  python3 ops/recover_env_from_backups.py --env .env
  python3 ops/recover_env_from_backups.py --env .env --apply
  python3 ops/recover_env_from_backups.py --env .env --apply --override-existing
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass
class EnvSnapshot:
    path: Path
    mtime: float
    kv: Dict[str, str]


def _parse_env_file(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return out
    for raw in text.splitlines():
        # Recupera casos comuns de arquivo corrompido com "\n" literal em linha única.
        chunks = raw.split("\\n") if "\\n" in raw else [raw]
        for chunk in chunks:
            line = chunk.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].strip()
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            if not ENV_KEY_RE.match(k):
                continue
            out[k] = v
    return out


def _discover_snapshots(env_path: Path) -> List[EnvSnapshot]:
    parent = env_path.parent
    name = env_path.name
    patterns = [
        f"{name}.bak*",
        f"{name}.backup*",
        f"{name}.old*",
        f"{name}.save*",
        f"{name}.*.bak*",
        f"{name}.before_recover.*",
    ]
    seen: set[str] = set()
    paths: List[Path] = []
    for pat in patterns:
        for m in glob.glob(str(parent / pat)):
            p = Path(m)
            if not p.exists() or p.is_dir():
                continue
            rp = str(p.resolve())
            if rp in seen:
                continue
            seen.add(rp)
            paths.append(p)
    snaps: List[EnvSnapshot] = []
    for p in paths:
        try:
            st = p.stat()
        except Exception:
            continue
        snaps.append(EnvSnapshot(path=p, mtime=float(st.st_mtime), kv=_parse_env_file(p)))
    snaps.sort(key=lambda s: (s.mtime, str(s.path)))
    return snaps


def _latest_values(snaps: Iterable[EnvSnapshot]) -> Dict[str, str]:
    latest: Dict[str, str] = {}
    for s in snaps:
        for k, v in s.kv.items():
            latest[k] = v
    return latest


def _build_history(snaps: Iterable[EnvSnapshot]) -> Dict[str, List[str]]:
    hist: Dict[str, List[str]] = {}
    for s in snaps:
        label = f"{s.path.name} @ {dt.datetime.utcfromtimestamp(s.mtime).isoformat()}Z"
        for k in s.kv:
            hist.setdefault(k, []).append(label)
    return hist


def _rewrite_env(
    env_path: Path,
    *,
    recovered_values: Dict[str, str],
    override_existing: bool,
) -> Tuple[int, int, Path]:
    existing_kv = _parse_env_file(env_path) if env_path.exists() else {}
    now = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    backup_path = env_path.with_name(f"{env_path.name}.before_recover.{now}")
    if env_path.exists():
        backup_path.write_text(env_path.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
    else:
        backup_path.write_text("", encoding="utf-8")

    original_lines = []
    if env_path.exists():
        original_lines = env_path.read_text(encoding="utf-8", errors="replace").splitlines()

    updated = 0
    kept = 0
    out_lines: List[str] = []
    already_written: set[str] = set()

    for raw in original_lines:
        line = raw
        striped = raw.strip()
        if not striped or striped.startswith("#") or "=" not in striped:
            out_lines.append(line)
            continue
        norm = striped
        prefix = ""
        if norm.startswith("export "):
            prefix = "export "
            norm = norm[7:].strip()
        if "=" not in norm:
            out_lines.append(line)
            continue
        key, _old_val = norm.split("=", 1)
        key = key.strip()
        if not ENV_KEY_RE.match(key):
            out_lines.append(line)
            continue
        already_written.add(key)
        if key in recovered_values and override_existing:
            out_lines.append(f"{prefix}{key}={recovered_values[key]}")
            updated += 1
        else:
            out_lines.append(line)
            kept += 1

    appended = 0
    for key in sorted(recovered_values):
        if key in already_written:
            continue
        if (not override_existing) and (key in existing_kv):
            continue
        out_lines.append(f"{key}={recovered_values[key]}")
        appended += 1

    env_path.write_text("\n".join(out_lines).rstrip() + "\n", encoding="utf-8")
    return updated, appended, backup_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Recover .env values from backup files.")
    ap.add_argument("--env", default=".env", help="Path do .env alvo (default: .env)")
    ap.add_argument("--apply", action="store_true", help="Aplica recuperação no arquivo .env")
    ap.add_argument(
        "--override-existing",
        action="store_true",
        help="Também sobrescreve chaves já existentes com último valor encontrado no histórico.",
    )
    ap.add_argument(
        "--critical",
        default="TELEGRAM_BOT_TOKEN,TELEGRAM_CHAT_ID,DATABASE_URL",
        help="Lista CSV de chaves críticas para alerta.",
    )
    args = ap.parse_args()

    env_path = Path(args.env).resolve()
    cur_kv = _parse_env_file(env_path) if env_path.exists() else {}
    snaps = _discover_snapshots(env_path)
    latest = _latest_values(snaps)
    history = _build_history(snaps)

    union_keys = set(latest.keys()) | set(cur_kv.keys())
    missing_from_current = sorted([k for k in union_keys if k not in cur_kv and k in latest])
    critical = [k.strip() for k in str(args.critical or "").split(",") if k.strip()]
    critical_missing = [k for k in critical if k not in cur_kv]
    critical_recoverable = [k for k in critical_missing if k in latest]

    report = {
        "env_path": str(env_path),
        "env_exists": env_path.exists(),
        "current_keys": len(cur_kv),
        "backup_files_found": len(snaps),
        "historical_union_keys": len(union_keys),
        "missing_from_current_count": len(missing_from_current),
        "missing_from_current_sample": missing_from_current[:50],
        "critical_missing": critical_missing,
        "critical_recoverable": critical_recoverable,
        "backup_inventory": [
            {
                "file": s.path.name,
                "mtime_utc": dt.datetime.utcfromtimestamp(s.mtime).isoformat() + "Z",
                "keys": len(s.kv),
            }
            for s in snaps
        ],
        "key_history": {
            k: {"first_seen": v[0], "last_seen": v[-1], "seen_count": len(v)}
            for k, v in sorted(history.items())
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if not args.apply:
        return

    if not latest:
        print("\n[recover] Nenhum backup com chaves válidas foi encontrado; nada para aplicar.")
        return

    updated, appended, backup_path = _rewrite_env(
        env_path,
        recovered_values=latest,
        override_existing=bool(args.override_existing),
    )
    print(
        f"\n[recover] aplicado em {env_path} | updated={updated} appended={appended} "
        f"| backup={backup_path}"
    )


if __name__ == "__main__":
    main()
