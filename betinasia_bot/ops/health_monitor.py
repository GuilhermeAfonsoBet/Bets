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

def _safe_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)

def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _pctl(xs: List[float], p: float) -> Optional[float]:
    if not xs:
        return None
    xs2 = sorted(xs)
    k = (len(xs2) - 1) * (float(p) / 100.0)
    f = int(k)
    c = min(len(xs2) - 1, f + 1)
    if f == c:
        return float(xs2[f])
    return float(xs2[f] + (k - f) * (xs2[c] - xs2[f]))


def _timing_from_exec_jsonl(o: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """
    Extrai métricas timing.* do payload do executor JSONL.
    Mantém compatibilidade com versões anteriores.
    """
    try:
        r = o.get("result") if isinstance(o.get("result"), dict) else {}
        timing = r.get("timing") if isinstance(r.get("timing"), dict) else {}
    except Exception:
        timing = {}
    q = _safe_float(timing.get("queue_delay_ms"))
    c = _safe_float(timing.get("call_to_done_ms"))
    p = _safe_float(timing.get("post_ms"))
    return {"queue_delay_ms": q, "call_to_done_ms": c, "post_ms": p}


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


def _read_tail_jsonl(path: Path, *, max_bytes: int, max_lines: int) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Lê uma janela do final de um JSONL (best-effort) e retorna objetos parseados.
    Evita carregar arquivos enormes.
    """
    if not path.exists():
        return [], "MISSING"
    try:
        mb = int(max(1024, max_bytes))
    except Exception:
        mb = 2_000_000
    try:
        ml = int(max(1, max_lines))
    except Exception:
        ml = 5000
    try:
        with path.open("rb") as f:
            f.seek(0, 2)
            end = f.tell()
            size = min(mb, end)
            f.seek(max(0, end - size))
            chunk = f.read().decode("utf-8", errors="ignore")
        lines = [ln.strip() for ln in chunk.splitlines() if ln.strip()]
        if not lines:
            return [], "EMPTY"
        lines = lines[-ml:]
        out: List[Dict[str, Any]] = []
        for ln in lines:
            try:
                obj = json.loads(ln)
                if isinstance(obj, dict):
                    out.append(obj)
            except Exception:
                continue
        return out, None
    except Exception:
        return [], "READ_ERROR"


def _audit_telemetry_default_for_service(audit_service: str, audit_telemetry_arg: str) -> Path:
    """
    Melhor UX: quando o usuário troca `--audit-service` (ex.: betinasia-audit-api-back),
    mas esquece de trocar `--audit-telemetry`, tentamos inferir um default consistente.
    Regra:
      - se o argumento explícito veio do usuário, respeita;
      - se está no default "logs/audit_api_telemetry.jsonl", e o service contém "-back",
        preferimos "logs/audit_api_back_telemetry.jsonl" (se existir).
    Observação: o `audit_h3b_api.py` atual escreve em `logs/audit_api_telemetry.jsonl` por default,
    mas alguns deployments podem ter sido customizados via symlink/override.
    """
    try:
        svc = str(audit_service or "").strip()
        arg = str(audit_telemetry_arg or "").strip()
        if not arg:
            arg = "logs/audit_api_telemetry.jsonl"
        p = Path(arg)
        # se usuário já passou algo não-default, não mexe
        if str(p) != "logs/audit_api_telemetry.jsonl":
            return p
        if "-back" in svc:
            alt = Path("logs/audit_api_back_telemetry.jsonl")
            if alt.exists():
                return alt
        return p
    except Exception:
        return Path(str(audit_telemetry_arg or "logs/audit_api_telemetry.jsonl"))


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
        # Se estivermos rodando como root (comum em systemd units sem User=),
        # não dependemos de sudo. Se não-root, usamos sudo *não-interativo* (-n)
        # para evitar ficar pendurado esperando senha.
        try:
            is_root = (os.geteuid() == 0)  # type: ignore[attr-defined]
        except Exception:
            is_root = False

        cmd = ["systemctl", "restart", service] if is_root else ["sudo", "-n", "systemctl", "restart", service]
        p = subprocess.run(
            cmd,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return p.returncode == 0
    except Exception:
        return False


def _systemctl_stop(service: str) -> bool:
    try:
        try:
            is_root = (os.geteuid() == 0)  # type: ignore[attr-defined]
        except Exception:
            is_root = False
        cmd = ["systemctl", "stop", service] if is_root else ["sudo", "-n", "systemctl", "stop", service]
        p = subprocess.run(
            cmd,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return p.returncode == 0
    except Exception:
        return False


def _exit_code_from_results(results: List[CheckResult]) -> int:
    code = 0
    for r in results:
        if r.level == "FAIL":
            code = max(code, 2)
        elif r.level == "WARN":
            code = max(code, 1)
    return int(code)


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


async def _db_audit_friction(db: Database, since: datetime) -> Dict[str, Any]:
    """
    Sinais de fricção/bloqueio da API de betslip.
    - 401/auth_error (sessão ausente)
    - NO_ROOT_SESSION_COOKIE (login/sessão não gerou cookie)
    - No betslip_id received (POST/WS api/betslip não retornou id)
    """
    q = text(
        """
        SELECT
          count(*) AS total_n,
          count(*) FILTER (WHERE upper(status)='OK') AS ok_n,
          count(*) FILTER (WHERE upper(status) LIKE 'API_%') AS api_n,
          count(*) FILTER (WHERE upper(status)='STALE_QUEUE_WAIT') AS stale_queue_n,
          count(*) FILTER (
            WHERE (hypothesis_details::text) ILIKE '%HTTP_401%'
               OR (hypothesis_details::text) ILIKE '%auth_error%'
          ) AS auth_401_n,
          count(*) FILTER (
            WHERE (hypothesis_details::text) ILIKE '%NO_ROOT_SESSION_COOKIE%'
          ) AS no_root_session_n,
          count(*) FILTER (
            WHERE (hypothesis_details::text) ILIKE '%No betslip_id received%'
          ) AS no_betslip_id_n
        FROM betslip_audit_results
        WHERE audited_at >= :since
        """
    )
    async with db.async_session() as session:
        r = await session.execute(q, {"since": since})
        row = r.fetchone()
        return dict(row._mapping) if row else {}


async def run_checks(
    *,
    since_minutes: int,
    telemetry_max_age_sec: int,
    collector_service: str,
    audit_service: str,
    executor_service: str,
    bridge_back_service: str,
    bridge_lay_service: str,
    collector_telemetry: Path,
    audit_telemetry: Path,
    executor_jsonl: Path,
    restart_on_fail: bool,
) -> Tuple[List[CheckResult], int, Dict[str, Any]]:
    now = _utcnow()
    since = now - timedelta(minutes=int(max(1, since_minutes)))

    results: List[CheckResult] = []
    exit_code = 0

    # 1) systemd
    services = []
    for svc in [collector_service, audit_service, executor_service, bridge_back_service, bridge_lay_service]:
        s = str(svc or "").strip()
        if not s or s.lower() in ("0", "off", "none", "false"):
            continue
        services.append(s)
    for svc in services:
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
        # Sinais rápidos de fricção/bloqueio (janela menor)
        fric_min = _safe_int(os.getenv("OPS_AUDIT_FRICTION_MINUTES", "10"), 10)
        fric_min = max(1, min(int(since_minutes), int(fric_min)))
        fric_since = now - timedelta(minutes=int(fric_min))
        fr = await _db_audit_friction(db, fric_since)
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

    # 4) Audit API friction / possible block (Telegram alert)
    try:
        total_n = int(fr.get("total_n") or 0)
        ok_n = int(fr.get("ok_n") or 0)
        auth_401_n = int(fr.get("auth_401_n") or 0)
        no_root_n = int(fr.get("no_root_session_n") or 0)
        no_bid_n = int(fr.get("no_betslip_id_n") or 0)
        staleq_n = int(fr.get("stale_queue_n") or 0)
        min_n = _safe_int(os.getenv("OPS_AUDIT_FRICTION_MIN_AUDITS", "10"), 10)
        warn_pct = float(os.getenv("OPS_AUDIT_FRICTION_WARN_PCT", "0.20"))
        fail_pct = float(os.getenv("OPS_AUDIT_FRICTION_FAIL_PCT", "0.50"))
        nobid_warn_pct = float(os.getenv("OPS_AUDIT_NOBETSLIP_WARN_PCT", "0.25"))
        nobid_fail_pct = float(os.getenv("OPS_AUDIT_NOBETSLIP_FAIL_PCT", "0.60"))
        warn_pct = max(0.0, min(1.0, warn_pct))
        fail_pct = max(0.0, min(1.0, fail_pct))
        nobid_warn_pct = max(0.0, min(1.0, nobid_warn_pct))
        nobid_fail_pct = max(0.0, min(1.0, nobid_fail_pct))
        denom = max(1, total_n)
        auth_n = auth_401_n + no_root_n
        auth_rate = auth_n / float(denom)
        ok_rate = ok_n / float(denom)
        no_bid_rate = no_bid_n / float(denom)

        if total_n >= int(min_n):
            hard_fail = (auth_rate >= float(fail_pct)) or (no_bid_rate >= float(nobid_fail_pct))
            hard_warn = (auth_rate >= float(warn_pct)) or (no_bid_rate >= float(nobid_warn_pct))

            if hard_fail:
                results.append(
                    CheckResult(
                        "FAIL",
                        f"audit-api: fricção crítica (possível bloqueio/rate-limit/WS) (janela={fric_min}m) "
                        f"auth401+no_root={auth_n}/{total_n} ({auth_rate:.0%}), ok={ok_n}/{total_n} ({ok_rate:.0%}), "
                        f"no_betslip_id={no_bid_n}/{total_n} ({no_bid_rate:.0%}), staleq={staleq_n}",
                    )
                )
                exit_code = max(exit_code, 2)
            elif hard_warn:
                results.append(
                    CheckResult(
                        "WARN",
                        f"audit-api: fricção elevada (possível bloqueio/rate-limit/WS) (janela={fric_min}m) "
                        f"auth401+no_root={auth_n}/{total_n} ({auth_rate:.0%}), ok={ok_n}/{total_n} ({ok_rate:.0%}), "
                        f"no_betslip_id={no_bid_n}/{total_n} ({no_bid_rate:.0%}), staleq={staleq_n}",
                    )
                )
                exit_code = max(exit_code, 1)
            else:
                results.append(
                    CheckResult(
                        "PASS",
                        f"audit-api: auth OK (janela={fric_min}m) ok={ok_n}/{total_n} ({ok_rate:.0%}), no_betslip_id={no_bid_n} staleq={staleq_n}",
                    )
                )
        else:
            results.append(
                CheckResult(
                    "PASS",
                    f"audit-api: amostra baixa p/ fricção (janela={fric_min}m) n={total_n} < min={min_n}",
                )
            )
    except Exception:
        # Nunca deixa o monitor quebrar por esse check
        pass

    # 5) Executor activity/health (via JSONL)
    try:
        # parâmetros
        tail_lines = _safe_int(os.getenv("OPS_EXECUTOR_TAIL_LINES", "5000"), 5000)
        tail_bytes = _safe_int(os.getenv("OPS_EXECUTOR_TAIL_BYTES", "2000000"), 2_000_000)
        min_events = _safe_int(os.getenv("OPS_EXECUTOR_MIN_EVENTS", "30"), 30)
        max_nonhb_age = _safe_int(os.getenv("OPS_EXECUTOR_NONHEARTBEAT_MAX_AGE_SEC", "900"), 900)
        warn_rate = float(os.getenv("OPS_EXECUTOR_FAIL_RATE_WARN", "0.20"))
        fail_rate = float(os.getenv("OPS_EXECUTOR_FAIL_RATE_FAIL", "0.40"))
        min_audits_for_idle_fail = _safe_int(os.getenv("OPS_EXECUTOR_IDLE_MIN_AUDITS", "10"), 10)
        lat_min_events = _safe_int(os.getenv("OPS_EXECUTOR_LAT_MIN_EVENTS", "30"), 30)
        lat_call_p50_warn = _safe_int(os.getenv("OPS_EXECUTOR_LAT_CALL_P50_WARN_MS", "12000"), 12000)
        lat_call_p50_fail = _safe_int(os.getenv("OPS_EXECUTOR_LAT_CALL_P50_FAIL_MS", "20000"), 20000)
        lat_post_p50_warn = _safe_int(os.getenv("OPS_EXECUTOR_LAT_POST_P50_WARN_MS", "7000"), 7000)
        lat_post_p50_fail = _safe_int(os.getenv("OPS_EXECUTOR_LAT_POST_P50_FAIL_MS", "12000"), 12000)
        lat_queue_p50_warn = _safe_int(os.getenv("OPS_EXECUTOR_LAT_QUEUE_P50_WARN_MS", "1000"), 1000)
        lat_queue_p50_fail = _safe_int(os.getenv("OPS_EXECUTOR_LAT_QUEUE_P50_FAIL_MS", "3000"), 3000)
        # Opcional: p90 (mais robusto contra medianas "ok" com cauda explodindo)
        lat_call_p90_warn = _safe_int(os.getenv("OPS_EXECUTOR_LAT_CALL_P90_WARN_MS", "0"), 0)
        lat_call_p90_fail = _safe_int(os.getenv("OPS_EXECUTOR_LAT_CALL_P90_FAIL_MS", "0"), 0)

        rows, err = _read_tail_jsonl(Path(str(executor_jsonl)), max_bytes=int(tail_bytes), max_lines=int(tail_lines))
        if err:
            results.append(CheckResult("FAIL", f"executor: jsonl inválido ({err}) em {executor_jsonl}"))
            exit_code = max(exit_code, 2)
        else:
            # agregação
            def _get_status(o: Dict[str, Any]) -> str:
                r = o.get("result") if isinstance(o.get("result"), dict) else {}
                return str(r.get("status") or "UNKNOWN")

            def _get_ts(o: Dict[str, Any]) -> Optional[datetime]:
                r = o.get("result") if isinstance(o.get("result"), dict) else {}
                q = o.get("request") if isinstance(o.get("request"), dict) else {}
                return _parse_iso_ts(r.get("created_at") or q.get("created_at"))

            def _get_err(o: Dict[str, Any]) -> str:
                r = o.get("result") if isinstance(o.get("result"), dict) else {}
                return str(r.get("error") or "")

            recent = []
            for o in rows:
                ts = _get_ts(o)
                if ts is None:
                    continue
                if ts >= since:
                    recent.append(o)

            # se não tiver recorte por janela, usa o tail inteiro (melhor do que "nada")
            window = recent if recent else rows
            nonhb = [o for o in window if _get_status(o) != "HEARTBEAT"]
            hb = [o for o in window if _get_status(o) == "HEARTBEAT"]

            last_any = None
            last_nonhb = None
            last_liveok = None
            for o in window:
                ts = _get_ts(o)
                if ts and ((last_any is None) or ts > last_any):
                    last_any = ts
            for o in nonhb:
                ts = _get_ts(o)
                if ts and ((last_nonhb is None) or ts > last_nonhb):
                    last_nonhb = ts
            for o in window:
                if _get_status(o) == "LIVE_OK":
                    ts = _get_ts(o)
                    if ts and ((last_liveok is None) or ts > last_liveok):
                        last_liveok = ts

            age_nonhb = int((now - last_nonhb).total_seconds()) if last_nonhb else None
            age_liveok = int((now - last_liveok).total_seconds()) if last_liveok else None

            # taxa de falha (somente em eventos não-heartbeat)
            fail_statuses = {"API_FAILED", "NO_SESSION", "INTERNAL_ERROR"}
            fail_n = sum(1 for o in nonhb if _get_status(o) in fail_statuses)
            ok_n = sum(1 for o in nonhb if _get_status(o) == "LIVE_OK")
            denom = max(1, len(nonhb))
            fr = float(fail_n) / float(denom)

            # padrões fatais
            fatal_n = 0
            for o in nonhb:
                if _get_status(o) not in fail_statuses:
                    continue
                e = _get_err(o).lower()
                if ("execution context was destroyed" in e) or ("target closed" in e) or ("no_root_session_cookie" in e) or ("auth_error" in e) or ("http_401" in e):
                    fatal_n += 1

            # idle: sem non-heartbeat por tempo alto *e* audit gerando oportunidades
            if age_nonhb is None:
                results.append(CheckResult("WARN", f"executor: sem eventos não-heartbeat no tail (hb={len(hb)})."))
                exit_code = max(exit_code, 1)
            else:
                if age_nonhb > int(max_nonhb_age) and int(audits_n) >= int(min_audits_for_idle_fail):
                    results.append(
                        CheckResult(
                            "FAIL",
                            f"{executor_service or 'betinasia-executor'}: sem execução recente (non_heartbeat_age={age_nonhb}s > {max_nonhb_age}s) "
                            f"com audits_n={audits_n} desde {since_minutes}m (possível bridge/executor travado).",
                        )
                    )
                    exit_code = max(exit_code, 2)
                else:
                    results.append(
                        CheckResult(
                            "PASS",
                            f"executor: atividade ok (nonhb_n={len(nonhb)} hb_n={len(hb)} nonhb_age={age_nonhb}s live_ok_n={ok_n} fail_n={fail_n})",
                        )
                    )

            # fail-rate: só se volume mínimo
            if len(nonhb) >= int(min_events):
                if fr >= float(fail_rate):
                    results.append(
                        CheckResult(
                            "FAIL",
                            f"{executor_service or 'betinasia-executor'}: taxa alta de falhas (fail={fail_n}/{len(nonhb)} {fr:.0%}, fatal={fatal_n}) "
                            f"(janela~{since_minutes}m; min_events={min_events})",
                        )
                    )
                    exit_code = max(exit_code, 2)
                elif fr >= float(warn_rate):
                    results.append(
                        CheckResult(
                            "WARN",
                            f"{executor_service or 'betinasia-executor'}: falhas elevadas (fail={fail_n}/{len(nonhb)} {fr:.0%}, fatal={fatal_n}) "
                            f"(janela~{since_minutes}m)",
                        )
                    )
                    exit_code = max(exit_code, 1)
            else:
                # pouca amostra: apenas informativo
                results.append(
                    CheckResult(
                        "PASS",
                        f"executor: amostra baixa p/ taxa de falhas (nonhb_n={len(nonhb)} < min_events={min_events}) live_ok_age={age_liveok}s",
                    )
                )

            # Latência (p50/p90) em execuções bem-sucedidas
            try:
                ok = [o for o in nonhb if _get_status(o) == "LIVE_OK"]
                # fallback: se quase não tiver LIVE_OK, usa DRY_OK também (shadow)
                if len(ok) < int(lat_min_events):
                    ok = [o for o in nonhb if _get_status(o) in ("LIVE_OK", "DRY_OK")]
                call_ms: List[float] = []
                post_ms: List[float] = []
                queue_ms: List[float] = []
                for o in ok:
                    t = _timing_from_exec_jsonl(o)
                    c = t.get("call_to_done_ms")
                    p = t.get("post_ms")
                    q = t.get("queue_delay_ms")
                    if c is not None and c > 0:
                        call_ms.append(float(c))
                    if p is not None and p > 0:
                        post_ms.append(float(p))
                    if q is not None and q >= 0:
                        queue_ms.append(float(q))

                # só checa se houver volume mínimo de call_to_done_ms (métrica principal)
                if len(call_ms) >= int(lat_min_events):
                    p50_call = _pctl(call_ms, 50) or 0.0
                    p90_call = _pctl(call_ms, 90) or 0.0
                    p50_post = _pctl(post_ms, 50) if post_ms else None
                    p50_queue = _pctl(queue_ms, 50) if queue_ms else None

                    # thresholds p50
                    hard_fail = False
                    hard_warn = False
                    reasons: List[str] = []
                    if p50_call >= float(lat_call_p50_fail):
                        hard_fail = True
                        reasons.append(f"call_p50={int(p50_call)}ms>={int(lat_call_p50_fail)}")
                    elif p50_call >= float(lat_call_p50_warn):
                        hard_warn = True
                        reasons.append(f"call_p50={int(p50_call)}ms>={int(lat_call_p50_warn)}")

                    if p50_post is not None:
                        if float(p50_post) >= float(lat_post_p50_fail):
                            hard_fail = True
                            reasons.append(f"post_p50={int(p50_post)}ms>={int(lat_post_p50_fail)}")
                        elif float(p50_post) >= float(lat_post_p50_warn):
                            hard_warn = True
                            reasons.append(f"post_p50={int(p50_post)}ms>={int(lat_post_p50_warn)}")

                    if p50_queue is not None:
                        if float(p50_queue) >= float(lat_queue_p50_fail):
                            hard_fail = True
                            reasons.append(f"queue_p50={int(p50_queue)}ms>={int(lat_queue_p50_fail)}")
                        elif float(p50_queue) >= float(lat_queue_p50_warn):
                            hard_warn = True
                            reasons.append(f"queue_p50={int(p50_queue)}ms>={int(lat_queue_p50_warn)}")

                    # thresholds p90 (opcional, somente se configurado >0)
                    if int(lat_call_p90_fail) > 0 and p90_call >= float(lat_call_p90_fail):
                        hard_fail = True
                        reasons.append(f"call_p90={int(p90_call)}ms>={int(lat_call_p90_fail)}")
                    elif int(lat_call_p90_warn) > 0 and p90_call >= float(lat_call_p90_warn):
                        hard_warn = True
                        reasons.append(f"call_p90={int(p90_call)}ms>={int(lat_call_p90_warn)}")

                    msg = (
                        f"{executor_service or 'betinasia-executor'}: latência alta "
                        f"(n_ok={len(call_ms)} p50_call={int(p50_call)}ms p90_call={int(p90_call)}ms"
                        + (f" p50_post={int(p50_post)}ms" if p50_post is not None else "")
                        + (f" p50_queue={int(p50_queue)}ms" if p50_queue is not None else "")
                        + f") [{', '.join(reasons) if reasons else 'no_reasons'}]"
                    )
                    if hard_fail:
                        results.append(CheckResult("FAIL", msg))
                        exit_code = max(exit_code, 2)
                    elif hard_warn:
                        results.append(CheckResult("WARN", msg))
                        exit_code = max(exit_code, 1)
                    else:
                        results.append(
                            CheckResult(
                                "PASS",
                                f"{executor_service or 'betinasia-executor'}: latência ok "
                                f"(n_ok={len(call_ms)} p50_call={int(p50_call)}ms p90_call={int(p90_call)}ms"
                                + (f" p50_post={int(p50_post)}ms" if p50_post is not None else "")
                                + (f" p50_queue={int(p50_queue)}ms" if p50_queue is not None else "")
                                + ")",
                            )
                        )
                else:
                    results.append(
                        CheckResult(
                            "PASS",
                            f"{executor_service or 'betinasia-executor'}: amostra baixa p/ latência "
                            f"(n_ok={len(call_ms)} < lat_min_events={lat_min_events})",
                        )
                    )
            except Exception:
                # não quebra monitor por esse check
                pass
    except Exception as e:
        results.append(CheckResult("WARN", f"executor: check falhou (ignored) err={str(e)[:120]}"))
        exit_code = max(exit_code, 1)

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

def _load_state(path: Path) -> Dict[str, Any]:
    try:
        if not path.exists():
            return {"v": 1, "services": {}}
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"v": 1, "services": {}}


def _save_state(path: Path, state: Dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Escrita atômica (reduz risco de state corrompido em timers concorrentes)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)
    except Exception:
        pass


def _rate_limited(
    state: Dict[str, Any],
    service: str,
    *,
    now: datetime,
    max_restarts_per_hour: int,
    cooldown_sec: int,
) -> Tuple[bool, str]:
    """
    Retorna (allowed, reason). Usa state['services'][svc]['restarts'] como lista de ISO timestamps.
    """
    svc = state.setdefault("services", {}).setdefault(service, {})
    restarts: List[str] = list(svc.get("restarts") or [])
    # filtra janela 1h
    cutoff = now - timedelta(hours=1)
    kept: List[str] = []
    last_dt: Optional[datetime] = None
    for ts in restarts:
        dt = _parse_iso_ts(ts)
        if not dt:
            continue
        if dt >= cutoff:
            kept.append(dt.isoformat())
        if (last_dt is None) or (dt > last_dt):
            last_dt = dt
    svc["restarts"] = kept

    if last_dt is not None:
        age = int((now - last_dt).total_seconds())
        if age < int(cooldown_sec):
            return False, f"cooldown ({age}s < {cooldown_sec}s)"
    if len(kept) >= int(max_restarts_per_hour):
        return False, f"rate_limit ({len(kept)}/{max_restarts_per_hour} restarts na última hora)"
    return True, "ok"


def _mark_failure(state: Dict[str, Any], service: str) -> int:
    svc = state.setdefault("services", {}).setdefault(service, {})
    n = _safe_int(svc.get("consecutive_fail"), 0) + 1
    svc["consecutive_fail"] = n
    return n


def _reset_failure(state: Dict[str, Any], service: str) -> None:
    svc = state.setdefault("services", {}).setdefault(service, {})
    svc["consecutive_fail"] = 0


def _append_restart(state: Dict[str, Any], service: str, now: datetime) -> None:
    svc = state.setdefault("services", {}).setdefault(service, {})
    restarts: List[str] = list(svc.get("restarts") or [])
    restarts.append(now.isoformat())
    svc["restarts"] = restarts[-50:]


def _append_action(state: Dict[str, Any], service: str, action: str, now: datetime) -> None:
    svc = state.setdefault("services", {}).setdefault(service, {})
    acts = svc.setdefault("actions", {})
    seq: List[str] = list(acts.get(action) or [])
    seq.append(now.isoformat())
    acts[action] = seq[-50:]


def _rate_limited_action(
    state: Dict[str, Any],
    service: str,
    action: str,
    *,
    now: datetime,
    max_per_hour: int,
    cooldown_sec: int,
) -> Tuple[bool, str]:
    """
    Rate limit genérico por (service, action).
    """
    svc = state.setdefault("services", {}).setdefault(service, {})
    acts = svc.setdefault("actions", {})
    seq: List[str] = list(acts.get(action) or [])

    cutoff = now - timedelta(hours=1)
    kept: List[str] = []
    last_dt: Optional[datetime] = None
    for ts in seq:
        dt = _parse_iso_ts(ts)
        if not dt:
            continue
        if dt >= cutoff:
            kept.append(dt.isoformat())
        if (last_dt is None) or (dt > last_dt):
            last_dt = dt
    acts[action] = kept

    if last_dt is not None:
        age = int((now - last_dt).total_seconds())
        if age < int(cooldown_sec):
            return False, f"cooldown ({age}s < {cooldown_sec}s)"
    if len(kept) >= int(max_per_hour):
        return False, f"rate_limit ({len(kept)}/{max_per_hour} ações na última hora)"
    return True, "ok"


def _set_paused(state: Dict[str, Any], service: str, *, now: datetime, reason: str) -> None:
    svc = state.setdefault("services", {}).setdefault(service, {})
    svc["paused"] = True
    svc["paused_utc"] = now.isoformat()
    svc["paused_reason"] = str(reason)[:240]


def _clear_paused(state: Dict[str, Any], service: str) -> None:
    svc = state.setdefault("services", {}).setdefault(service, {})
    if "paused" in svc:
        svc["paused"] = False


def _is_paused(state: Dict[str, Any], service: str) -> bool:
    try:
        svc = state.get("services", {}).get(service, {}) if isinstance(state, dict) else {}
        return bool(svc.get("paused"))
    except Exception:
        return False


def _should_restart_from_results(results: List[CheckResult], service: str) -> bool:
    """
    Heurística conservadora:
    - se houver FAIL específico do serviço ou do seu subsistema (telemetria/DB)
    """
    key = str(service).strip()
    for r in results:
        if r.level != "FAIL":
            continue
        msg = r.message.lower()
        if key.lower() in msg:
            return True
        # mapeamentos simples
        if "collector" in key.lower() and ("collector" in msg or "best_odds_history" in msg):
            return True
        if "audit" in key.lower() and ("audit" in msg or "betslip_audit_results" in msg):
            return True
    return False


def _has_executor_latency_fail(results: List[CheckResult], executor_service: str) -> bool:
    key = str(executor_service or "").strip().lower()
    for r in results:
        if r.level != "FAIL":
            continue
        msg = str(r.message or "").lower()
        if "latência alta" not in msg:
            continue
        if key and key in msg:
            return True
        if (not key) and ("executor" in msg):
            return True
    return False


def _get_executor_latency_fail_message(results: List[CheckResult], executor_service: str) -> Optional[str]:
    key = str(executor_service or "").strip().lower()
    for r in results:
        if r.level != "FAIL":
            continue
        msg = str(r.message or "")
        low = msg.lower()
        if "latência alta" not in low:
            continue
        if key and key in low:
            return msg
        if (not key) and ("executor" in low):
            return msg
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since-minutes", type=int, default=30)
    ap.add_argument("--telemetry-max-age-sec", type=int, default=int(os.getenv("OPS_TELEMETRY_MAX_AGE_SEC", "600")))
    ap.add_argument("--collector-service", default=os.getenv("COLLECTOR_SERVICE", "betinasia-collector"))
    ap.add_argument("--audit-service", default=os.getenv("AUDIT_SERVICE", "betinasia-audit-api"))
    ap.add_argument("--executor-service", default=os.getenv("EXECUTOR_SERVICE", "betinasia-executor"))
    ap.add_argument("--bridge-back-service", default=os.getenv("BRIDGE_BACK_SERVICE", "betinasia-executor-bridge-back"))
    ap.add_argument("--bridge-lay-service", default=os.getenv("BRIDGE_LAY_SERVICE", "betinasia-executor-bridge-lay"))
    ap.add_argument("--collector-telemetry", default=os.getenv("COLLECTOR_TELEMETRY_FILE", "logs/collector_telemetry.jsonl"))
    ap.add_argument("--audit-telemetry", default=os.getenv("AUDIT_TELEMETRY_FILE", "logs/audit_api_telemetry.jsonl"))
    ap.add_argument("--executor-jsonl", default=os.getenv("EXECUTOR_JSONL", "logs/executor_live.jsonl"))
    ap.add_argument("--telegram", action="store_true", help="Envia alerta no Telegram em WARN/FAIL")
    ap.add_argument(
        "--telegram-recovery",
        action="store_true",
        default=(os.getenv("OPS_TELEGRAM_SEND_RECOVERY", "1").strip() not in ("0", "false", "False", "no", "NO")),
        help="Envia mensagem no Telegram quando sair de WARN/FAIL e voltar para PASS (default=on).",
    )
    ap.add_argument("--restart-on-fail", action="store_true", help="Reinicia serviços quando FAIL (requer sudo sem prompt)")
    ap.add_argument(
        "--autopilot",
        action="store_true",
        help="Modo auto-pilot seguro: só reinicia após N FAILs consecutivos, com cooldown e rate limit.",
    )
    ap.add_argument("--state-file", default=os.getenv("OPS_AUTOPILOT_STATE_FILE", "logs/ops_autopilot_state.json"))
    ap.add_argument("--consecutive-fails-to-restart", type=int, default=int(os.getenv("OPS_FAILS_TO_RESTART", "2")))
    ap.add_argument("--cooldown-sec", type=int, default=int(os.getenv("OPS_RESTART_COOLDOWN_SEC", "1800")))
    ap.add_argument("--max-restarts-per-hour", type=int, default=int(os.getenv("OPS_MAX_RESTARTS_PER_HOUR", "2")))
    args = ap.parse_args()

    # Se o user customizou --audit-service mas não customizou --audit-telemetry,
    # tenta inferir um default consistente.
    try:
        args.audit_telemetry = str(_audit_telemetry_default_for_service(str(args.audit_service), str(args.audit_telemetry)))
    except Exception:
        pass

    results, code, meta = asyncio.run(
        run_checks(
            since_minutes=int(args.since_minutes),
            telemetry_max_age_sec=int(args.telemetry_max_age_sec),
            collector_service=str(args.collector_service),
            audit_service=str(args.audit_service),
            executor_service=str(args.executor_service),
            bridge_back_service=str(args.bridge_back_service),
            bridge_lay_service=str(args.bridge_lay_service),
            collector_telemetry=Path(str(args.collector_telemetry)),
            audit_telemetry=Path(str(args.audit_telemetry)),
            executor_jsonl=Path(str(args.executor_jsonl)),
            restart_on_fail=bool(args.restart_on_fail) and (not bool(args.autopilot)),
        )
    )

    # Estado (para detectar RECOVERY e evitar spam)
    state_path = Path(str(args.state_file))
    state = _load_state(state_path)
    prev_code = _safe_int(state.get("last_overall_code"), 0)
    prev_non_ok = _safe_int(state.get("last_non_ok_code"), 0)
    prev_non_ok_utc = str(state.get("last_non_ok_utc") or "")
    prev_non_ok_lines = list(state.get("last_non_ok_lines") or [])
    recovered = (prev_code > 0 and int(code) == 0)

    # AUTO-PILOT (seguro): restarts com rate limit/cooldown, após FAILs consecutivos
    now = _utcnow()
    autopilot_actions: List[str] = []
    if args.autopilot:
        state["last_run_utc"] = now.isoformat()

        autopilot_svcs = []
        for svc in [str(args.collector_service), str(args.audit_service), str(args.executor_service), str(args.bridge_back_service), str(args.bridge_lay_service)]:
            s = str(svc or "").strip()
            if not s or s.lower() in ("0", "off", "none", "false"):
                continue
            autopilot_svcs.append(s)

        # Se algum serviço estava "paused" e voltou a ficar ativo, liberamos o pause.
        # Isso permite unpause manual via `systemctl start ...` sem mexer em state file.
        try:
            for r in results:
                if r.level != "PASS":
                    continue
                msg = str(r.message or "")
                for svc in autopilot_svcs:
                    if _is_paused(state, svc) and msg.startswith(f"{svc}: ativo"):
                        _clear_paused(state, svc)
                        autopilot_actions.append(f"{svc}: unpaused (serviço ativo)")
        except Exception:
            pass

        # Ação forte: em degradação de latência, pausar bridges (evita operar em Back Pre/In lento).
        # Compat: aceita nomes antigos (OPS_LAT_FAIL_PAUSE_BRIDGES_ENABLE).
        lat_pause_flag = os.getenv("OPS_LATENCY_FAIL_PAUSE_BRIDGES", os.getenv("OPS_LAT_FAIL_PAUSE_BRIDGES_ENABLE", "0"))
        latency_pause_enabled = str(lat_pause_flag or "0").strip().lower() in (
            "1",
            "true",
            "yes",
            "y",
            "on",
        )
        skip_restart_on_latency_fail = str(os.getenv("OPS_LATENCY_FAIL_SKIP_RESTART", "1") or "1").strip().lower() in (
            "1",
            "true",
            "yes",
            "y",
            "on",
        )
        try:
            pause_after_fails = int(float(os.getenv("OPS_LATENCY_FAIL_PAUSE_BRIDGES_FAILS", "1") or 1))
        except Exception:
            pause_after_fails = 1
        pause_after_fails = max(1, int(pause_after_fails))
        try:
            pause_cooldown = int(float(os.getenv("OPS_LATENCY_FAIL_PAUSE_COOLDOWN_SEC", "1800") or 1800))
        except Exception:
            pause_cooldown = 1800
        try:
            pause_max_per_hour = int(float(os.getenv("OPS_LATENCY_FAIL_PAUSE_MAX_PER_HOUR", "1") or 1))
        except Exception:
            pause_max_per_hour = 1
        latency_fail = _has_executor_latency_fail(results, str(args.executor_service))
        # streak de FAIL de latência (por executor_service) para evitar pausar por spikes únicos
        try:
            exsvc = str(args.executor_service or "").strip() or "betinasia-executor"
            ex_state = state.setdefault("services", {}).setdefault(exsvc, {})
            if latency_fail:
                ex_state["latency_fail_streak"] = _safe_int(ex_state.get("latency_fail_streak"), 0) + 1
            else:
                ex_state["latency_fail_streak"] = 0
            latency_fail_streak = _safe_int(ex_state.get("latency_fail_streak"), 0)
        except Exception:
            latency_fail_streak = 0

        paused_now: List[str] = []
        if latency_pause_enabled and latency_fail and int(latency_fail_streak) >= int(pause_after_fails):
            to_pause = []
            for svc in [str(args.bridge_back_service), str(args.bridge_lay_service)]:
                s = str(svc or "").strip()
                if not s or s.lower() in ("0", "off", "none", "false"):
                    continue
                to_pause.append(s)
            for svc in to_pause:
                allowed, reason = _rate_limited_action(
                    state,
                    svc,
                    "pause_on_latency_fail",
                    now=now,
                    max_per_hour=int(pause_max_per_hour),
                    cooldown_sec=int(pause_cooldown),
                )
                if not allowed:
                    autopilot_actions.append(f"{svc}: pause bloqueado ({reason})")
                    continue
                ok = _systemctl_stop(svc)
                _append_action(state, svc, "pause_on_latency_fail", now)
                _set_paused(state, svc, now=now, reason="latency_fail")
                autopilot_actions.append(f"{svc}: paused(stop) por latência FAIL ok={ok}")
                paused_now.append(str(svc))
        elif latency_pause_enabled and latency_fail:
            autopilot_actions.append(
                f"pause_on_latency_fail: aguardando streak (latency_fail_streak={int(latency_fail_streak)}/{int(pause_after_fails)})"
            )

        # marca falhas por serviço
        for svc in autopilot_svcs:
            if _is_paused(state, svc):
                # Se está pausado por decisão operacional, não acumula FAIL (evita loop de restart).
                continue
            if latency_fail and skip_restart_on_latency_fail and (str(svc) == str(args.executor_service)):
                # Evita loop: latência FAIL pode ser externa (site/proxy). Preferimos pausar bridges e investigar.
                continue
            if code >= 2 and _should_restart_from_results(results, svc):
                nfail = _mark_failure(state, svc)
                autopilot_actions.append(f"{svc}: consecutive_fail={nfail}")
            else:
                _reset_failure(state, svc)

        # decide restarts
        for svc in autopilot_svcs:
            if _is_paused(state, svc):
                continue
            if latency_fail and skip_restart_on_latency_fail and (str(svc) == str(args.executor_service)):
                continue
            svc_state = state.get("services", {}).get(svc, {})
            nfail = _safe_int(svc_state.get("consecutive_fail"), 0)
            if nfail < int(args.consecutive_fails_to_restart):
                continue
            allowed, reason = _rate_limited(
                state,
                svc,
                now=now,
                max_restarts_per_hour=int(args.max_restarts_per_hour),
                cooldown_sec=int(args.cooldown_sec),
            )
            if not allowed:
                autopilot_actions.append(f"{svc}: restart bloqueado ({reason})")
                continue
            ok = _systemctl_restart(svc)
            _append_restart(state, svc, now)
            autopilot_actions.append(f"{svc}: restart acionado={ok}")

        _save_state(state_path, state)

    # Atualiza estado de "último status"
    state["last_overall_utc"] = str(meta.get("now_utc") or now.isoformat())
    state["last_overall_code"] = int(code)
    if int(code) > 0:
        state["last_non_ok_utc"] = str(meta.get("now_utc") or now.isoformat())
        state["last_non_ok_code"] = int(code)
        state["last_non_ok_lines"] = [f"[{r.level}] {r.message}" for r in results if r.level in ("WARN", "FAIL")]
    elif recovered:
        state["last_recovery_utc"] = str(meta.get("now_utc") or now.isoformat())
    _save_state(state_path, state)

    # imprime output legível
    print("=" * 70)
    print(f"OPS HEALTH MONITOR | {meta.get('now_utc')}")
    print("=" * 70)
    for r in results:
        print(f"[{r.level}] {r.message}")
    if args.autopilot and autopilot_actions:
        print("-" * 70)
        print("Auto-pilot:")
        for a in autopilot_actions:
            print(f"- {a}")
    print("-" * 70)
    print(f"Exit code: {code}")

    should_send_recovery = bool(args.telegram_recovery) and recovered
    if args.telegram and (code > 0 or (args.autopilot and autopilot_actions) or should_send_recovery):
        token = os.getenv("TELEGRAM_BOT_TOKEN") or ""
        chat_id = os.getenv("TELEGRAM_CHAT_ID") or ""
        if token and chat_id:
            if should_send_recovery:
                lines = [f"OPS HEALTH (RECOVERY/OK) @ {meta.get('now_utc')}"]
                # Inclui o último problema conhecido (para contexto)
                if prev_non_ok > 0:
                    prev_level = "FAIL" if prev_non_ok >= 2 else "WARN"
                    lines.append(f"Anterior: {prev_level} @ {prev_non_ok_utc}")
                    for ln in prev_non_ok_lines[:20]:
                        lines.append(f"- {ln}")
                if args.autopilot and autopilot_actions:
                    lines.append("- Auto-pilot:")
                    for a in autopilot_actions:
                        lines.append(f"  - {a}")
                _telegram_send(token, chat_id, "\n".join(lines))
            else:
                level = "FAIL" if code >= 2 else "WARN" if code > 0 else "OK"
                pause_alert = bool(args.autopilot) and any("paused(stop) por latência FAIL" in str(a) for a in (autopilot_actions or []))
                lines = []
                if pause_alert:
                    lat_msg = _get_executor_latency_fail_message(results, str(args.executor_service)) or ""
                    lines.append("!!! ALERTA CRÍTICO: BRIDGES PAUSADOS POR LATÊNCIA (FAIL) !!!")
                    if lat_msg:
                        lines.append(f"Latência: {lat_msg}")
                lines.append(f"OPS HEALTH ({level}) @ {meta.get('now_utc')}")
                for r in results:
                    if r.level in ("WARN", "FAIL"):
                        lines.append(f"- [{r.level}] {r.message}")
                if args.autopilot and autopilot_actions:
                    lines.append("- Auto-pilot:")
                    for a in autopilot_actions:
                        lines.append(f"  - {a}")
                _telegram_send(token, chat_id, "\n".join(lines))
        else:
            print("[WARN] Telegram habilitado, mas TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID não estão setados.")

    return int(code)


if __name__ == "__main__":
    raise SystemExit(main())

