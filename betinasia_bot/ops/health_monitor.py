# -*- coding: utf-8 -*-
"""
Monitor operacional (collector + audit + DB + telemetria).

Objetivo:
- detectar "hang silencioso" (serviço ativo, mas sem gravar telemetria/DB)
- reportar WARN/FAIL com critérios objetivos
- opcional: enviar alerta Telegram e/ou reiniciar serviços em FAIL

Uso (exemplo):
  cd /home/betbot/Bets/betinasia_bot
  source venv/bin/activate
  python3 -m ops.health_monitor --since-minutes 30 --telegram
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List
from urllib.parse import urlencode
from urllib.request import Request, urlopen

sys.path.insert(0, ".")

from storage.database import Database
from sqlalchemy import text


@dataclass(frozen=True)
class CheckResult:
    level: str  # PASS/WARN/FAIL
    message: str


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _read_last_jsonl(path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not path.exists():
        return None, "MISSING"
    try:
        # lê só o final (arquivo pode ser grande)
        with path.open("rb") as f:
            f.seek(0, 2)
            end = f.tell()
            # lê últimos 64KB
            size = min(65536, end)
            f.seek(max(0, end - size))
            chunk = f.read().decode("utf-8", errors="ignore")
        lines = [ln.strip() for ln in chunk.splitlines() if ln.strip()]
        if not lines:
            return None, "EMPTY"
        last = lines[-1]
        try:
            return json.loads(last), None
        except Exception:
            return None, "INVALID_JSON"
    except Exception:
        return None, "READ_ERROR"


def _parse_iso_ts(ts: Any) -> Optional[datetime]:
    if not ts:
        return None
    if isinstance(ts, (int, float)):
        # epoch seconds
        try:
            return datetime.fromtimestamp(float(ts), tz=timezone.utc)
        except Exception:
            return None
    if not isinstance(ts, str):
        return None
    s = ts.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _systemctl_show(service: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    try:
        p = subprocess.run(
            ["systemctl", "show", service, "-p", "ActiveState", "-p", "SubState", "-p", "NRestarts"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        for ln in (p.stdout or "").splitlines():
            if "=" in ln:
                k, v = ln.split("=", 1)
                out[k.strip()] = v.strip()
    except Exception:
        pass
    return out


def _systemctl_restart(service: str) -> bool:
    try:
        p = subprocess.run(
            ["sudo", "systemctl", "restart", service],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return p.returncode == 0
    except Exception:
        return False


def _telegram_send(token: str, chat_id: str, text_msg: str) -> bool:
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        data = urlencode({"chat_id": chat_id, "text": text_msg}).encode("utf-8")
        req = Request(url, data=data, method="POST")
        with urlopen(req, timeout=10) as resp:
            return 200 <= int(resp.status) < 300
    except Exception:
        return False


async def _db_metrics(db: Database, since: datetime) -> Dict[str, Any]:
    q = text(
        """
        SELECT
          (SELECT max(scraped_at) FROM best_odds_history) AS last_best_odds_utc,
          (SELECT max(audited_at) FROM betslip_audit_results) AS last_audit_utc,
          (SELECT max(detected_at) FROM h3b_temporal_reversal_events) AS last_h3b_utc,
          (SELECT count(*) FROM best_odds_history WHERE scraped_at >= :since) AS best_odds_n,
          (SELECT count(*) FROM betslip_audit_results WHERE audited_at >= :since) AS audits_n,
          (SELECT count(*) FROM h3b_temporal_reversal_events WHERE detected_at >= :since) AS h3b_n
        """
    )
    async with db.async_session() as session:
        r = await session.execute(q, {"since": since})
        row = r.fetchone()
        if not row:
            return {}
        return dict(row._mapping)


async def run_checks(
    *,
    since_minutes: int,
    telemetry_max_age_sec: int,
    collector_service: str,
    audit_service: str,
    collector_telemetry: Path,
    audit_telemetry: Path,
    restart_on_fail: bool,
) -> Tuple[List[CheckResult], int, Dict[str, Any]]:
    now = _utcnow()
    since = now - timedelta(minutes=int(max(1, since_minutes)))

    results: List[CheckResult] = []
    exit_code = 0

    # 1) systemd
    for svc in [collector_service, audit_service]:
        s = _systemctl_show(svc)
        active = s.get("ActiveState", "unknown")
        sub = s.get("SubState", "unknown")
        restarts = s.get("NRestarts", "unknown")
        if active == "active" and sub == "running":
            results.append(CheckResult("PASS", f"{svc}: ativo (restarts={restarts})"))
        else:
            results.append(CheckResult("FAIL", f"{svc}: fora do esperado ({active}/{sub}, restarts={restarts})"))
            exit_code = max(exit_code, 2)
            if restart_on_fail:
                ok = _systemctl_restart(svc)
                results.append(CheckResult("WARN", f"{svc}: restart acionado={ok}"))
                exit_code = max(exit_code, 1)

    # 2) telemetria freshness
    def _check_telemetry(name: str, path: Path):
        nonlocal exit_code
        payload, err = _read_last_jsonl(path)
        if err:
            results.append(CheckResult("FAIL", f"{name}: telemetria inválida ({err}) em {path}"))
            exit_code = max(exit_code, 2)
            return
        ts = _parse_iso_ts(payload.get("ts_utc") or payload.get("timestamp") or payload.get("ts"))
        if not ts:
            results.append(CheckResult("FAIL", f"{name}: sem timestamp no último JSONL ({path})"))
            exit_code = max(exit_code, 2)
            return
        age = int((now - ts).total_seconds())
        if age <= int(telemetry_max_age_sec):
            results.append(CheckResult("PASS", f"{name}: telemetria atualizada (age={age}s)"))
        else:
            results.append(CheckResult("FAIL", f"{name}: telemetria parada (age={age}s > {telemetry_max_age_sec}s)"))
            exit_code = max(exit_code, 2)

    _check_telemetry("collector", collector_telemetry)
    _check_telemetry("audit-api", audit_telemetry)

    # 3) DB freshness
    db = Database()
    await db.connect()
    try:
        m = await _db_metrics(db, since)
    finally:
        await db.close()

    def _age_sec(dt: Any) -> Optional[int]:
        if isinstance(dt, datetime):
            x = dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
            return int((now - x).total_seconds())
        return None

    last_best = m.get("last_best_odds_utc")
    last_audit = m.get("last_audit_utc")
    a_best = _age_sec(last_best)
    a_audit = _age_sec(last_audit)
    best_n = int(m.get("best_odds_n") or 0)
    audits_n = int(m.get("audits_n") or 0)
    h3b_n = int(m.get("h3b_n") or 0)

    if a_best is not None and a_best <= telemetry_max_age_sec:
        results.append(CheckResult("PASS", f"DB: best_odds_history fresco (age={a_best}s, n={best_n} desde {since_minutes}m)"))
    else:
        results.append(CheckResult("WARN", f"DB: best_odds_history possivelmente atrasado (age={a_best}s, n={best_n} desde {since_minutes}m)"))
        exit_code = max(exit_code, 1)

    if a_audit is not None and a_audit <= telemetry_max_age_sec:
        results.append(CheckResult("PASS", f"DB: betslip_audit_results fresco (age={a_audit}s, n={audits_n} desde {since_minutes}m)"))
    else:
        results.append(CheckResult("WARN", f"DB: betslip_audit_results possivelmente atrasado (age={a_audit}s, n={audits_n} desde {since_minutes}m)"))
        exit_code = max(exit_code, 1)

    results.append(CheckResult("PASS", f"DB: h3b_temporal_reversal_events (n={h3b_n} desde {since_minutes}m)"))

    return results, exit_code, {
        "now_utc": now.isoformat(),
        "since_utc": since.isoformat(),
        "db": {
            "last_best_odds_utc": str(last_best),
            "last_audit_utc": str(last_audit),
            "best_odds_n": best_n,
            "audits_n": audits_n,
            "h3b_n": h3b_n,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since-minutes", type=int, default=30)
    ap.add_argument("--telemetry-max-age-sec", type=int, default=int(os.getenv("OPS_TELEMETRY_MAX_AGE_SEC", "600")))
    ap.add_argument("--collector-service", default=os.getenv("COLLECTOR_SERVICE", "betinasia-collector"))
    ap.add_argument("--audit-service", default=os.getenv("AUDIT_SERVICE", "betinasia-audit-api"))
    ap.add_argument("--collector-telemetry", default=os.getenv("COLLECTOR_TELEMETRY_FILE", "logs/collector_telemetry.jsonl"))
    ap.add_argument("--audit-telemetry", default=os.getenv("AUDIT_TELEMETRY_FILE", "logs/audit_api_telemetry.jsonl"))
    ap.add_argument("--telegram", action="store_true", help="Envia alerta no Telegram em WARN/FAIL")
    ap.add_argument("--restart-on-fail", action="store_true", help="Reinicia serviços quando FAIL (requer sudo sem prompt)")
    args = ap.parse_args()

    results, code, meta = asyncio.run(
        run_checks(
            since_minutes=int(args.since_minutes),
            telemetry_max_age_sec=int(args.telemetry_max_age_sec),
            collector_service=str(args.collector_service),
            audit_service=str(args.audit_service),
            collector_telemetry=Path(str(args.collector_telemetry)),
            audit_telemetry=Path(str(args.audit_telemetry)),
            restart_on_fail=bool(args.restart_on_fail),
        )
    )

    # imprime output legível
    print("=" * 70)
    print(f"OPS HEALTH MONITOR | {meta.get('now_utc')}")
    print("=" * 70)
    for r in results:
        print(f"[{r.level}] {r.message}")
    print("-" * 70)
    print(f"Exit code: {code}")

    if args.telegram and code > 0:
        token = os.getenv("TELEGRAM_BOT_TOKEN") or ""
        chat_id = os.getenv("TELEGRAM_CHAT_ID") or ""
        if token and chat_id:
            lines = [f"OPS HEALTH ({'FAIL' if code>=2 else 'WARN'}) @ {meta.get('now_utc')}"]
            for r in results:
                if r.level in ("WARN", "FAIL"):
                    lines.append(f"- [{r.level}] {r.message}")
            _telegram_send(token, chat_id, "\n".join(lines))
        else:
            print("[WARN] Telegram habilitado, mas TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID não estão setados.")

    return int(code)


if __name__ == "__main__":
    raise SystemExit(main())

