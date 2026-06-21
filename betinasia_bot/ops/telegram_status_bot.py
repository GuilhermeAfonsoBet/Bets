# -*- coding: utf-8 -*-
"""
Bot Telegram (long polling) para status on-demand do operacional.

Objetivo:
- Você manda "/status" no Telegram e ele responde com um diagnóstico completo:
  - systemd (collector + audit)
  - frescor da telemetria
  - frescor do DB (best_odds_history + betslip_audit_results + h3b)
  - sinais de "erro de gravação" no collector (save_errors) e eventos com status != OK

Requisitos (.env):
- TELEGRAM_BOT_TOKEN
- TELEGRAM_CHAT_ID (chat autorizado)
- DATABASE_URL (para métricas de DB)

Opcional (.env):
- OPS_TELEMETRY_MAX_AGE_SEC (default 600)
- OPS_STATUS_SINCE_MINUTES (default 30)
- OPS_STATUS_TAIL_LINES (default 250)
- OPS_TELEGRAM_POLL_TIMEOUT_SEC (default 30)
- OPS_TELEGRAM_OFFSET_FILE (default logs/ops_telegram_offset.json)
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode
from urllib.request import Request, urlopen

sys.path.insert(0, ".")

from storage.database import Database
from sqlalchemy import text


@dataclass(frozen=True)
class CheckLine:
    level: str
    message: str


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_iso_ts(ts: Any) -> Optional[datetime]:
    if not ts:
        return None
    if isinstance(ts, (int, float)):
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


def _read_last_jsonl_n(path: Path, n: int) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    if n <= 0:
        return [], None
    if not path.exists():
        return [], "MISSING"
    try:
        with path.open("rb") as f:
            f.seek(0, 2)
            end = f.tell()
            # tenta ler um chunk razoável; se não der, lê menor
            size = min(256 * 1024, end)  # 256KB
            f.seek(max(0, end - size))
            chunk = f.read().decode("utf-8", errors="ignore")
        lines = [ln.strip() for ln in chunk.splitlines() if ln.strip()]
        if not lines:
            return [], "EMPTY"
        out: List[Dict[str, Any]] = []
        for ln in reversed(lines):
            try:
                out.append(json.loads(ln))
                if len(out) >= n:
                    break
            except Exception:
                continue
        return list(reversed(out)), None
    except Exception:
        return [], "READ_ERROR"


def _systemctl_show(service: str) -> Dict[str, str]:
    # evita dependências extras: chama systemctl show
    import subprocess

    props = [
        "ActiveState",
        "SubState",
        "NRestarts",
        "MainPID",
        "MemoryCurrent",
        "CPUUsageNSec",
    ]
    args = ["systemctl", "show", service]
    for p in props:
        args += ["-p", p]
    p = subprocess.run(args, check=False, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    out: Dict[str, str] = {}
    for ln in (p.stdout or "").splitlines():
        if "=" in ln:
            k, v = ln.split("=", 1)
            out[k.strip()] = v.strip()
    return out


async def _db_metrics(since: datetime) -> Dict[str, Any]:
    db = Database()
    await db.connect()
    try:
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
            return dict(row._mapping) if row else {}
    finally:
        await db.close()


def _age_sec(now: datetime, dt: Any) -> Optional[int]:
    if isinstance(dt, datetime):
        x = dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        return int((now - x).total_seconds())
    return None


def _fmt_bytes(x: str) -> str:
    try:
        b = int(x)
    except Exception:
        return x or "?"
    # systemctl usa bytes; formata simples
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if b < 1024 or unit == "TB":
            return f"{b:.0f}{unit}" if unit == "B" else f"{b:.1f}{unit}"
        b = b / 1024.0
    return str(x)


def _fmt_cpu_ns(x: str) -> str:
    try:
        ns = int(x)
    except Exception:
        return x or "?"
    sec = ns / 1e9
    if sec < 60:
        return f"{sec:.1f}s"
    m = sec / 60
    if m < 60:
        return f"{m:.1f}min"
    h = m / 60
    return f"{h:.1f}h"


def _summarize_collector_telemetry(payloads: List[Dict[str, Any]]) -> List[str]:
    if not payloads:
        return ["collector telemetry: (sem dados)"]
    # pega o último "OK" de ciclo (ou último com save_errors)
    ok = None
    for p in reversed(payloads):
        if p.get("status") == "OK" and "cycle_total_ms" in p:
            ok = p
            break
    if ok is None:
        ok = payloads[-1]

    save_errors = int(ok.get("save_errors") or 0) if isinstance(ok.get("save_errors"), (int, float, str)) else 0
    events_with_odds = ok.get("events_with_odds")
    matches_saved = ok.get("matches_saved")
    save_ms = ok.get("save_ms")
    collect_ms = ok.get("collect_ms")
    cycle = ok.get("cycle")

    # contagem de status problemáticos nos últimos payloads
    bad_status = {"COLLECT_TIMEOUT", "SAVE_TIMEOUT", "BROWSER_TIMEOUT"}
    bad = [p for p in payloads if str(p.get("status")) in bad_status]
    zeros = 0
    for p in payloads:
        if p.get("status") == "OK" and "events_with_odds" in p:
            try:
                if int(p.get("events_with_odds") or 0) <= 0:
                    zeros += 1
            except Exception:
                pass

    lines = []
    lines.append(f"collector: último ciclo={cycle} collect_ms={collect_ms} save_ms={save_ms} saved={matches_saved} with_odds={events_with_odds}")
    if save_errors > 0:
        lines.append(f"collector: ALERTA save_errors={save_errors} no último OK (pode estar 'alive' mas gravando com erro)")
    if bad:
        lines.append(f"collector: timeouts recentes={len(bad)} (últimos {len(payloads)} registros)")
    if zeros > 0:
        lines.append(f"collector: ciclos OK com 0 odds úteis={zeros} (últimos {len(payloads)} registros)")
    return lines


def _summarize_audit_telemetry(payloads: List[Dict[str, Any]]) -> List[str]:
    if not payloads:
        return ["audit telemetry: (sem dados)"]
    # audit é por evento; tenta achar últimos erros
    bad = []
    stale = 0
    ok = 0
    for p in payloads:
        st = str(p.get("status") or "")
        if st == "OK":
            ok += 1
        elif st == "STALE_QUEUE_WAIT":
            stale += 1
        else:
            # API_FAILED ou outros
            if st:
                bad.append(p)
    lines = [f"audit-api: eventos recentes OK={ok} stale={stale} outros={len(bad)} (janela tail={len(payloads)})"]
    if bad:
        last = bad[-1]
        eid = last.get("event_id")
        err = last.get("error") or ""
        lines.append(f"audit-api: último não-OK status={last.get('status')} event_id={eid} err='{str(err)[:120]}'")
    return lines


async def build_status_report() -> Tuple[str, int]:
    now = _utcnow()
    telemetry_max_age_sec = int(os.getenv("OPS_TELEMETRY_MAX_AGE_SEC", "600"))
    since_minutes = int(os.getenv("OPS_STATUS_SINCE_MINUTES", "30"))
    tail_n = int(os.getenv("OPS_STATUS_TAIL_LINES", "250"))

    collector_service = os.getenv("COLLECTOR_SERVICE", "betinasia-collector")
    audit_service = os.getenv("AUDIT_SERVICE", "betinasia-audit-api")
    collector_telemetry = Path(os.getenv("COLLECTOR_TELEMETRY_FILE", "logs/collector_telemetry.jsonl"))
    audit_telemetry = Path(os.getenv("AUDIT_TELEMETRY_FILE", "logs/audit_api_telemetry.jsonl"))

    lines: List[CheckLine] = []
    exit_code = 0

    # systemd status
    for svc in (collector_service, audit_service):
        s = _systemctl_show(svc)
        active = s.get("ActiveState", "unknown")
        sub = s.get("SubState", "unknown")
        restarts = s.get("NRestarts", "?")
        mem = _fmt_bytes(s.get("MemoryCurrent", "") or "?")
        cpu = _fmt_cpu_ns(s.get("CPUUsageNSec", "") or "?")
        if active == "active" and sub == "running":
            lines.append(CheckLine("PASS", f"{svc}: ativo (restarts={restarts}, mem={mem}, cpu={cpu})"))
        else:
            lines.append(CheckLine("FAIL", f"{svc}: fora do esperado ({active}/{sub}, restarts={restarts}, mem={mem}, cpu={cpu})"))
            exit_code = max(exit_code, 2)

    # telemetria freshness (último ts_utc em cada arquivo)
    for name, path in (("collector", collector_telemetry), ("audit-api", audit_telemetry)):
        payloads, err = _read_last_jsonl_n(path, 1)
        if err or not payloads:
            lines.append(CheckLine("FAIL", f"{name}: telemetria inválida ({err}) em {path}"))
            exit_code = max(exit_code, 2)
            continue
        ts = _parse_iso_ts(payloads[-1].get("ts_utc") or payloads[-1].get("timestamp") or payloads[-1].get("ts"))
        if not ts:
            lines.append(CheckLine("FAIL", f"{name}: sem timestamp no último JSONL ({path})"))
            exit_code = max(exit_code, 2)
            continue
        age = int((now - ts).total_seconds())
        if age <= telemetry_max_age_sec:
            lines.append(CheckLine("PASS", f"{name}: telemetria atualizada (age={age}s)"))
        else:
            lines.append(CheckLine("FAIL", f"{name}: telemetria parada (age={age}s > {telemetry_max_age_sec}s)"))
            exit_code = max(exit_code, 2)

    # DB freshness
    since = now - timedelta(minutes=max(1, since_minutes))
    m = await _db_metrics(since)
    a_best = _age_sec(now, m.get("last_best_odds_utc"))
    a_audit = _age_sec(now, m.get("last_audit_utc"))
    best_n = int(m.get("best_odds_n") or 0)
    audits_n = int(m.get("audits_n") or 0)
    h3b_n = int(m.get("h3b_n") or 0)

    if a_best is not None and a_best <= telemetry_max_age_sec:
        lines.append(CheckLine("PASS", f"DB: best_odds_history fresco (age={a_best}s, n={best_n} desde {since_minutes}m)"))
    else:
        lines.append(CheckLine("WARN", f"DB: best_odds_history possivelmente atrasado (age={a_best}s, n={best_n} desde {since_minutes}m)"))
        exit_code = max(exit_code, 1)

    if a_audit is not None and a_audit <= telemetry_max_age_sec:
        lines.append(CheckLine("PASS", f"DB: betslip_audit_results fresco (age={a_audit}s, n={audits_n} desde {since_minutes}m)"))
    else:
        lines.append(CheckLine("WARN", f"DB: betslip_audit_results possivelmente atrasado (age={a_audit}s, n={audits_n} desde {since_minutes}m)"))
        exit_code = max(exit_code, 1)

    lines.append(CheckLine("PASS", f"DB: h3b_temporal_reversal_events (n={h3b_n} desde {since_minutes}m)"))

    # Telemetria "completa": últimos N registros
    coll_tail, _ = _read_last_jsonl_n(collector_telemetry, tail_n)
    audit_tail, _ = _read_last_jsonl_n(audit_telemetry, tail_n)
    extra: List[str] = []
    extra += _summarize_collector_telemetry(coll_tail)
    extra += _summarize_audit_telemetry(audit_tail)

    level = "OK" if exit_code == 0 else "WARN" if exit_code == 1 else "FAIL"
    msg_lines: List[str] = [f"OPS STATUS ({level}) @ {now.isoformat()}"]
    msg_lines.append("-" * 34)
    for ln in lines:
        msg_lines.append(f"[{ln.level}] {ln.message}")
    msg_lines.append("-" * 34)
    msg_lines.append("Resumo telemetria (tail):")
    for ln in extra:
        msg_lines.append(f"- {ln}")
    msg_lines.append("-" * 34)
    msg_lines.append(f"exit_code={exit_code}")
    # Garante ficar < 4096 chars (Telegram). Se estourar, corta.
    out = "\n".join(msg_lines)
    if len(out) > 3800:
        out = out[:3800] + "\n...(truncado)"
    return out, exit_code


def _telegram_call(token: str, method: str, params: Dict[str, Any], *, timeout: int = 30) -> Dict[str, Any]:
    url = f"https://api.telegram.org/bot{token}/{method}"
    data = urlencode({k: v for k, v in params.items() if v is not None}).encode("utf-8")
    req = Request(url, data=data, method="POST")
    with urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="ignore")
    try:
        return json.loads(raw)
    except Exception:
        return {"ok": False, "raw": raw}


def _send_message(token: str, chat_id: str, text_msg: str) -> bool:
    try:
        r = _telegram_call(token, "sendMessage", {"chat_id": chat_id, "text": text_msg}, timeout=15)
        return bool(r.get("ok"))
    except Exception:
        return False


def _load_offset(path: Path) -> int:
    try:
        if not path.exists():
            return 0
        d = json.loads(path.read_text(encoding="utf-8"))
        return int(d.get("offset") or 0)
    except Exception:
        return 0


def _save_offset(path: Path, offset: int) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps({"offset": int(offset)}, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
    except Exception:
        pass


def _is_authorized(chat_id: Any, allowed: str) -> bool:
    try:
        cid = str(int(chat_id))
    except Exception:
        return False
    # allowed pode ser "123" ou "123,456"
    allowed_ids = [x.strip() for x in (allowed or "").split(",") if x.strip()]
    return cid in allowed_ids


async def main_loop() -> int:
    token = os.getenv("TELEGRAM_BOT_TOKEN") or ""
    allowed = os.getenv("TELEGRAM_CHAT_ID") or ""
    if not token or not allowed:
        print("ERRO: defina TELEGRAM_BOT_TOKEN e TELEGRAM_CHAT_ID no ambiente/.env.")
        return 2

    poll_timeout = int(os.getenv("OPS_TELEGRAM_POLL_TIMEOUT_SEC", "30"))
    offset_file = Path(os.getenv("OPS_TELEGRAM_OFFSET_FILE", "logs/ops_telegram_offset.json"))
    offset = _load_offset(offset_file)

    print(f"telegram_status_bot iniciado. allowed_chat_id(s)={allowed} offset={offset}")

    # long polling loop
    while True:
        try:
            resp = _telegram_call(
                token,
                "getUpdates",
                {"offset": offset, "timeout": poll_timeout, "allowed_updates": json.dumps(["message"])},
                timeout=poll_timeout + 10,
            )
            if not resp.get("ok"):
                time.sleep(2)
                continue
            updates = resp.get("result") or []
            if not updates:
                continue

            for u in updates:
                try:
                    uid = int(u.get("update_id"))
                except Exception:
                    continue
                offset = max(offset, uid + 1)
                _save_offset(offset_file, offset)

                msg = u.get("message") or {}
                chat = msg.get("chat") or {}
                chat_id = chat.get("id")
                text_msg = (msg.get("text") or "").strip()
                if not text_msg:
                    continue
                if not _is_authorized(chat_id, allowed):
                    continue

                t = text_msg.lower()
                if t in ("/status", "status", "/ops", "/health"):
                    report, _ = await build_status_report()
                    _send_message(token, str(int(chat_id)), report)
                elif t in ("/help", "help", "/start"):
                    _send_message(
                        token,
                        str(int(chat_id)),
                        "Comandos:\n"
                        "- /status  (status completo)\n"
                        "- /help\n",
                    )
                else:
                    # ignora qualquer outra coisa (para evitar spam)
                    continue
        except Exception:
            time.sleep(2)


def main() -> int:
    return int(asyncio.run(main_loop()))


if __name__ == "__main__":
    raise SystemExit(main())

