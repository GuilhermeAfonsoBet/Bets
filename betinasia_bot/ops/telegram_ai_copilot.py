# -*- coding: utf-8 -*-
"""
Copiloto operacional via Telegram (Camada B).

Objetivo:
- Rodar diagnósticos read-only contínuos na VPS.
- Acumular sugestões de ação em fila (pending).
- Executar ações SOMENTE com aprovação explícita via chat.

Comandos:
- /status
- /diag [minutos]
- /pendentes
- /aprovar <ID>
- /rejeitar <ID>
- /historico
- /help
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shlex
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from sqlalchemy import text

from ops.health_monitor import CheckResult, run_checks
from ops.telegram_status_bot import build_status_report
from storage.database import Database


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _log(msg: str) -> None:
    try:
        print(f"{_utcnow().isoformat()} [ops_ai_copilot] {msg}", flush=True)
    except Exception:
        pass


def _safe_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _safe_float(x: Any, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _env_bool(name: str, default: str = "0") -> bool:
    v = str(os.getenv(name, default) or default).strip().lower()
    return v in ("1", "true", "yes", "y", "on")


def _load_env_file(path: Path) -> None:
    try:
        if not path.exists():
            return
        for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw.strip()
            if (not line) or line.startswith("#") or ("=" not in line):
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            if (not k) or (k in os.environ):
                continue
            os.environ[k] = v.strip()
    except Exception:
        return


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


def _state_default() -> Dict[str, Any]:
    return {
        "next_id": 1,
        "last_monitor_utc": "",
        "last_diag_signature": "",
        "proposals": [],
    }


def _load_state(path: Path) -> Dict[str, Any]:
    try:
        if not path.exists():
            return _state_default()
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return _state_default()
        out = _state_default()
        out.update(data)
        if not isinstance(out.get("proposals"), list):
            out["proposals"] = []
        out["next_id"] = max(1, _safe_int(out.get("next_id"), 1))
        return out
    except Exception:
        return _state_default()


def _save_state(path: Path, state: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _allowed_chat_ids() -> List[str]:
    raw = str(
        os.getenv(
            "OPS_AI_TELEGRAM_CHAT_ID",
            os.getenv("TELEGRAM_CHAT_ID", ""),
        )
        or ""
    )
    out = [x.strip() for x in raw.split(",") if x.strip()]
    return out


def _is_authorized(chat_id: Any) -> bool:
    try:
        cid = str(int(chat_id))
    except Exception:
        return False
    return cid in _allowed_chat_ids()


def _primary_chat_id() -> Optional[str]:
    ids = _allowed_chat_ids()
    return ids[0] if ids else None


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
    # Telegram limita 4096 chars; enviamos em blocos.
    chunks: List[str] = []
    txt = str(text_msg or "")
    step = 3800
    for i in range(0, len(txt), step):
        chunks.append(txt[i : i + step])
    if not chunks:
        chunks = [""]
    ok_all = True
    for c in chunks:
        sent = False
        last_err = ""
        for attempt in range(4):
            try:
                r = _telegram_call(token, "sendMessage", {"chat_id": chat_id, "text": c}, timeout=20)
                if bool(r.get("ok")):
                    sent = True
                    break
                last_err = str(r.get("description") or r.get("raw") or "unknown_send_error")[:240]
                params = r.get("parameters") if isinstance(r.get("parameters"), dict) else {}
                retry_after = _safe_int(params.get("retry_after"), 0) if isinstance(params, dict) else 0
                if retry_after > 0:
                    time.sleep(min(30, int(retry_after) + 1))
                else:
                    time.sleep(1.0 + attempt)
            except Exception as e:
                last_err = str(e)[:240]
                time.sleep(1.0 + attempt)
        if not sent:
            ok_all = False
            _log(f"sendMessage failed chat_id={chat_id} err={last_err}")
    return ok_all


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


def _parse_down_services(results: List[CheckResult]) -> List[str]:
    out: List[str] = []
    for r in results:
        if str(r.level) != "FAIL":
            continue
        msg = str(r.message or "")
        if ": fora do esperado (" in msg:
            svc = msg.split(": fora do esperado", 1)[0].strip()
            if svc:
                out.append(svc)
    # ordem estável sem duplicatas
    seen = set()
    uniq: List[str] = []
    for s in out:
        if s in seen:
            continue
        seen.add(s)
        uniq.append(s)
    return uniq


async def _bridge_reason_counts(since_dt: datetime) -> Dict[str, int]:
    db = Database()
    await db.connect()
    try:
        q = text(
            """
            SELECT
              COALESCE(meta->>'reason',
                       CASE WHEN (meta->>'accepted')='true' THEN 'accepted' ELSE 'other' END) AS reason,
              COUNT(*)::bigint AS n
            FROM executor_bridge_seen
            WHERE created_at >= :since_dt
              AND action='live:Back'
            GROUP BY 1
            ORDER BY n DESC
            """
        )
        async with db.async_session() as s:
            r = await s.execute(q, {"since_dt": since_dt})
            rows = [dict(x._mapping) for x in (r.fetchall() or [])]
        out: Dict[str, int] = {}
        for row in rows:
            k = str(row.get("reason") or "other")
            out[k] = int(row.get("n") or 0)
        return out
    finally:
        await db.close()


def _tail_text(path: Path, *, max_bytes: int = 120_000) -> str:
    try:
        if not path.exists():
            return ""
        with path.open("rb") as f:
            f.seek(0, 2)
            end = f.tell()
            size = min(max_bytes, end)
            f.seek(max(0, end - size))
            return f.read().decode("utf-8", errors="ignore")
    except Exception:
        return ""


def _load_policy_meta(path: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "key_by_league": None,
        "key_by_league_scope": None,
        "ah_max_abs_line": None,
        "ah_scope": None,
        "active_keys_n": None,
        "active_keys_base_n": None,
    }
    try:
        if not path.exists():
            return out
        obj = json.loads(path.read_text(encoding="utf-8"))
        wf = obj.get("wf") if isinstance(obj.get("wf"), dict) else {}
        steps = obj.get("steps") if isinstance(obj.get("steps"), list) else []
        last = steps[-1] if steps else {}
        out["key_by_league"] = wf.get("key_by_league")
        out["key_by_league_scope"] = wf.get("key_by_league_scope")
        out["ah_max_abs_line"] = wf.get("ah_max_abs_line")
        out["ah_scope"] = wf.get("ah_scope")
        out["active_keys_n"] = len(last.get("active_keys") or [])
        out["active_keys_base_n"] = len(last.get("active_keys_base") or [])
        return out
    except Exception:
        return out


def _diag_signature(diag: Dict[str, Any]) -> str:
    keys = {
        "overall_code": diag.get("overall_code"),
        "down_services": diag.get("down_services"),
        "reason_counts_top": diag.get("reason_counts_top"),
        "socket_ok": diag.get("socket_ok"),
        "policy": diag.get("policy"),
    }
    raw = json.dumps(keys, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:12]


def _new_proposal_id(state: Dict[str, Any]) -> str:
    n = max(1, _safe_int(state.get("next_id"), 1))
    state["next_id"] = n + 1
    return f"P{n:04d}"


def _proposal_fingerprint(p: Dict[str, Any]) -> str:
    payload = {
        "kind": p.get("kind"),
        "title": p.get("title"),
        "command": p.get("command"),
    }
    return hashlib.sha1(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _append_proposals(state: Dict[str, Any], proposals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    existing = state.get("proposals") if isinstance(state.get("proposals"), list) else []
    pending_fps = {
        str(p.get("fingerprint"))
        for p in existing
        if str(p.get("status")) == "pending" and str(p.get("fingerprint") or "").strip()
    }
    created: List[Dict[str, Any]] = []
    max_pending = max(5, _safe_int(os.getenv("OPS_AI_MAX_PENDING", "30"), 30))
    pending_count = sum(1 for p in existing if str(p.get("status")) == "pending")
    for p in proposals:
        fp = _proposal_fingerprint(p)
        if fp in pending_fps:
            continue
        if pending_count >= max_pending:
            break
        item = dict(p)
        item["id"] = _new_proposal_id(state)
        item["status"] = "pending"
        item["created_utc"] = _utcnow().isoformat()
        item["fingerprint"] = fp
        existing.append(item)
        pending_fps.add(fp)
        created.append(item)
        pending_count += 1
    state["proposals"] = existing[-400:]
    return created


def _pending_list(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    items = state.get("proposals") if isinstance(state.get("proposals"), list) else []
    return [p for p in items if str(p.get("status")) == "pending"]


def _find_proposal(state: Dict[str, Any], proposal_id: str) -> Optional[Dict[str, Any]]:
    pid = str(proposal_id or "").strip().upper()
    items = state.get("proposals") if isinstance(state.get("proposals"), list) else []
    for p in items:
        if str(p.get("id") or "").strip().upper() == pid:
            return p
    return None


def _allowed_action_command(cmd: Any) -> bool:
    if not isinstance(cmd, list) or not cmd:
        return False
    parts = [str(x) for x in cmd]
    # only infrastructure service actions through systemctl are allowed.
    if parts[:3] == ["sudo", "-n", "systemctl"] and len(parts) == 5:
        return parts[3] in ("restart", "start", "stop")
    if parts[:1] == ["systemctl"] and len(parts) == 3:
        return parts[1] in ("restart", "start", "stop")
    return False


def _run_action_command(cmd: List[str], *, timeout_sec: int) -> Tuple[int, str]:
    p = subprocess.run(
        cmd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=max(5, int(timeout_sec)),
    )
    out = str(p.stdout or "").strip()
    return int(p.returncode), out


def _openai_enrich(diag: Dict[str, Any]) -> Dict[str, Any]:
    provider = str(os.getenv("OPS_AI_PROVIDER", "none") or "none").strip().lower()
    if provider != "openai":
        return {}
    api_key = str(os.getenv("OPENAI_API_KEY", "") or "").strip()
    if not api_key:
        return {}
    model = str(os.getenv("OPS_AI_MODEL", "gpt-4o-mini") or "gpt-4o-mini").strip()
    base_url = str(os.getenv("OPS_AI_BASE_URL", "https://api.openai.com/v1/chat/completions") or "").strip()
    timeout = max(8, _safe_int(os.getenv("OPS_AI_TIMEOUT_SEC", "25"), 25))
    prompt = {
        "task": "Analise diagnóstico operacional de bot de apostas e retorne JSON enxuto.",
        "rules": [
            "não sugerir mudanças de .env/policy/código sem aprovação humana",
            "priorizar ações de infraestrutura de baixo risco",
            "responder em português",
        ],
        "diag": diag,
        "output_schema": {
            "summary": "string",
            "confidence": "low|medium|high",
            "recommendations": ["string"],
        },
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Você é um SRE assistente. Responda apenas JSON válido."},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
        "temperature": 0.1,
    }
    try:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = Request(
            base_url,
            data=data,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
        obj = json.loads(raw)
        txt = (
            (((obj.get("choices") or [{}])[0]).get("message") or {}).get("content")
            if isinstance(obj, dict)
            else None
        )
        if not isinstance(txt, str) or (not txt.strip()):
            return {}
        parsed = json.loads(txt)
        if isinstance(parsed, dict):
            return parsed
        return {}
    except Exception:
        return {}


def _heuristic_proposals(diag: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    down = list(diag.get("down_services") or [])
    if down:
        for svc in down[:4]:
            out.append(
                {
                    "kind": "infra_restart",
                    "title": f"Reiniciar serviço {svc}",
                    "rationale": f"Serviço reportado fora do esperado no health-check: {svc}.",
                    "risk": "baixo",
                    "command": ["sudo", "-n", "systemctl", "restart", str(svc)],
                    "source": "heuristic",
                }
            )
    if not bool(diag.get("socket_ok", True)):
        for svc in ("betinasia-executor.service", "betinasia-executor-bridge-back.service"):
            out.append(
                {
                    "kind": "infra_restart",
                    "title": f"Reiniciar {svc} (socket ausente)",
                    "rationale": "Socket do executor ausente; ação recomendada é restart controlado.",
                    "risk": "baixo",
                    "command": ["sudo", "-n", "systemctl", "restart", svc],
                    "source": "heuristic",
                }
            )
    reason_counts = dict(diag.get("reason_counts") or {})
    total = int(diag.get("seen_total") or 0)
    n_not_active = int(reason_counts.get("not_active") or 0)
    n_ah = int(reason_counts.get("wf_ah_max_abs_line") or 0)
    n_acc = int(reason_counts.get("accepted") or 0)
    if total > 0:
        na_rate = n_not_active / float(total)
        ah_rate = n_ah / float(total)
        acc_rate = n_acc / float(total)
        if na_rate >= 0.45 and acc_rate < 0.08:
            out.append(
                {
                    "kind": "manual_review",
                    "title": "Revisar restrição de policy (bloqueio elevado por not_active)",
                    "rationale": (
                        f"Janela atual: not_active={n_not_active}/{total} ({na_rate:.0%}), "
                        f"accepted={n_acc}/{total} ({acc_rate:.0%})."
                    ),
                    "risk": "médio",
                    "command": None,
                    "source": "heuristic",
                }
            )
        if ah_rate >= 0.30:
            out.append(
                {
                    "kind": "manual_review",
                    "title": "Revisar impacto do gate AH (comparabilidade deve ser mantida)",
                    "rationale": f"wf_ah_max_abs_line representa {n_ah}/{total} ({ah_rate:.0%}) dos bloqueios.",
                    "risk": "médio",
                    "command": None,
                    "source": "heuristic",
                }
            )
    return out


async def _build_diag_snapshot(since_minutes: int) -> Dict[str, Any]:
    since_minutes = max(5, int(since_minutes))
    collector_service = os.getenv("COLLECTOR_SERVICE", "betinasia-collector")
    audit_service = os.getenv("AUDIT_SERVICE", "betinasia-audit-api")
    executor_service = os.getenv("EXECUTOR_SERVICE", "betinasia-executor")
    bridge_back_service = os.getenv("BRIDGE_BACK_SERVICE", "betinasia-executor-bridge-back")
    bridge_lay_service = os.getenv("BRIDGE_LAY_SERVICE", "betinasia-executor-bridge-lay")
    collector_telemetry = Path(os.getenv("COLLECTOR_TELEMETRY_FILE", "logs/collector_telemetry.jsonl"))
    audit_telemetry = Path(os.getenv("AUDIT_TELEMETRY_FILE", "logs/audit_api_telemetry.jsonl"))
    executor_jsonl = Path(os.getenv("EXECUTOR_JSONL", "logs/executor_live.jsonl"))
    telemetry_max_age_sec = _safe_int(os.getenv("OPS_TELEMETRY_MAX_AGE_SEC", "600"), 600)

    results, code, _meta = await run_checks(
        since_minutes=since_minutes,
        telemetry_max_age_sec=telemetry_max_age_sec,
        collector_service=str(collector_service),
        audit_service=str(audit_service),
        executor_service=str(executor_service),
        bridge_back_service=str(bridge_back_service),
        bridge_lay_service=str(bridge_lay_service),
        collector_telemetry=collector_telemetry,
        audit_telemetry=audit_telemetry,
        executor_jsonl=executor_jsonl,
        restart_on_fail=False,
    )

    fail_msgs = [str(r.message) for r in results if str(r.level) == "FAIL"]
    warn_msgs = [str(r.message) for r in results if str(r.level) == "WARN"]
    down_services = _parse_down_services(results)

    since_dt = _utcnow() - timedelta(minutes=int(since_minutes))
    reason_counts = await _bridge_reason_counts(since_dt)
    seen_total = int(sum(reason_counts.values()))
    reason_counts_top = sorted(reason_counts.items(), key=lambda kv: kv[1], reverse=True)[:6]

    socket_path = Path(str(os.getenv("EXECUTOR_UNIX_SOCKET", "/tmp/betinasia-exec.sock") or "/tmp/betinasia-exec.sock"))
    socket_ok = socket_path.exists()

    policy_path = Path(str(os.getenv("BRIDGE_POLICY_JSON", "logs/wf_policy_current.json") or "logs/wf_policy_current.json"))
    policy = _load_policy_meta(policy_path)

    bridge_err_tail = _tail_text(Path("logs/executor_bridge_back_error.log"), max_bytes=80_000)
    socket_err_recent = "Cannot connect to unix socket" in bridge_err_tail

    diag: Dict[str, Any] = {
        "ts_utc": _utcnow().isoformat(),
        "since_minutes": int(since_minutes),
        "overall_code": int(code),
        "fail_messages": fail_msgs[:8],
        "warn_messages": warn_msgs[:8],
        "down_services": down_services,
        "reason_counts": reason_counts,
        "reason_counts_top": reason_counts_top,
        "seen_total": seen_total,
        "socket_path": str(socket_path),
        "socket_ok": bool(socket_ok),
        "socket_error_recent": bool(socket_err_recent),
        "policy": policy,
    }
    diag["signature"] = _diag_signature(diag)
    diag["proposals"] = _heuristic_proposals(diag)
    ai = _openai_enrich(diag)
    if ai:
        diag["ai"] = ai
    return diag


def _parse_diag_minutes(text_msg: str, *, default: int = 30) -> int:
    """
    Aceita formatos:
    - /diag
    - /diag 30
    - /diag [30]
    - /diag (30min)
    """
    try:
        txt = str(text_msg or "").strip()
        parts = txt.split(maxsplit=1)
        if len(parts) < 2:
            return max(5, int(default))
        raw = str(parts[1]).strip()
        m = re.search(r"(\d{1,4})", raw)
        if not m:
            return max(5, int(default))
        return max(5, int(m.group(1)))
    except Exception:
        return max(5, int(default))


def _fmt_diag(diag: Dict[str, Any]) -> str:
    code = int(diag.get("overall_code") or 0)
    lvl = "PASS" if code == 0 else "WARN" if code == 1 else "FAIL"
    lines: List[str] = []
    lines.append(f"OPS AI DIAG ({lvl}) @ {diag.get('ts_utc')}")
    lines.append("-" * 34)
    lines.append(f"since_minutes={diag.get('since_minutes')} signature={diag.get('signature')}")
    lines.append(f"socket_ok={diag.get('socket_ok')} (path={diag.get('socket_path')})")
    pol = diag.get("policy") if isinstance(diag.get("policy"), dict) else {}
    lines.append(
        "policy:"
        f" key_by_league={pol.get('key_by_league')}"
        f" ah={pol.get('ah_max_abs_line')}({pol.get('ah_scope')})"
        f" active={pol.get('active_keys_n')}/{pol.get('active_keys_base_n')}"
    )
    down = list(diag.get("down_services") or [])
    if down:
        lines.append(f"serviços down: {', '.join(down)}")
    top = list(diag.get("reason_counts_top") or [])
    if top:
        lines.append("bridge reasons top:")
        for k, n in top[:6]:
            lines.append(f"- {k}: {n}")
    fail_msgs = list(diag.get("fail_messages") or [])
    warn_msgs = list(diag.get("warn_messages") or [])
    if fail_msgs:
        lines.append("fails:")
        for m in fail_msgs[:4]:
            lines.append(f"- {m}")
    if warn_msgs:
        lines.append("warns:")
        for m in warn_msgs[:4]:
            lines.append(f"- {m}")
    ai = diag.get("ai") if isinstance(diag.get("ai"), dict) else {}
    if ai:
        lines.append("-" * 34)
        lines.append(f"IA: {str(ai.get('summary') or '')}")
        if ai.get("confidence"):
            lines.append(f"confiança={ai.get('confidence')}")
    return "\n".join(lines)


def _fmt_pending(state: Dict[str, Any]) -> str:
    pending = _pending_list(state)
    if not pending:
        return "Sem ações pendentes."
    lines = [f"Ações pendentes: {len(pending)}", "-" * 28]
    for p in pending[:20]:
        cmd = p.get("command")
        cmd_txt = " | cmd: " + " ".join(shlex.quote(str(x)) for x in cmd) if isinstance(cmd, list) and cmd else ""
        lines.append(f"{p.get('id')} [{p.get('kind')}] {p.get('title')}{cmd_txt}")
        lines.append(f"  motivo: {p.get('rationale')}")
    if len(pending) > 20:
        lines.append(f"... +{len(pending)-20} pendentes")
    lines.append("Use: /aprovar <ID> ou /rejeitar <ID>")
    return "\n".join(lines)


def _fmt_history(state: Dict[str, Any], limit: int = 12) -> str:
    items = state.get("proposals") if isinstance(state.get("proposals"), list) else []
    hist = [p for p in items if str(p.get("status")) in ("executed", "failed", "rejected")]
    if not hist:
        return "Sem histórico de ações executadas/rejeitadas."
    hist = hist[-limit:]
    lines = [f"Histórico ({len(hist)}):", "-" * 24]
    for p in hist:
        lines.append(f"{p.get('id')} status={p.get('status')} title={p.get('title')}")
        if p.get("executed_utc"):
            lines.append(f"  executed_utc={p.get('executed_utc')} exit={p.get('exit_code')}")
        if p.get("rejected_utc"):
            lines.append(f"  rejected_utc={p.get('rejected_utc')}")
    return "\n".join(lines)


async def _run_diag_and_queue(state: Dict[str, Any], *, since_minutes: int, source: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    diag = await _build_diag_snapshot(since_minutes)
    proposals = list(diag.get("proposals") or [])
    for p in proposals:
        p.setdefault("source", source)
    created = _append_proposals(state, proposals)
    return diag, created


async def _execute_approved_action(state: Dict[str, Any], proposal_id: str, *, actor_chat_id: str) -> str:
    proposal = _find_proposal(state, proposal_id)
    if not proposal:
        return f"ID não encontrado: {proposal_id}"
    if str(proposal.get("status")) != "pending":
        return f"{proposal.get('id')}: status atual={proposal.get('status')} (não está pendente)"
    cmd = proposal.get("command")
    if not cmd:
        proposal["status"] = "approved_noop"
        proposal["approved_utc"] = _utcnow().isoformat()
        proposal["approved_by"] = str(actor_chat_id)
        return f"{proposal.get('id')} aprovado (sem comando executável)."
    if not _allowed_action_command(cmd):
        proposal["status"] = "blocked"
        proposal["approved_utc"] = _utcnow().isoformat()
        proposal["approved_by"] = str(actor_chat_id)
        return f"{proposal.get('id')} bloqueado: comando fora da allowlist."
    timeout = _safe_int(os.getenv("OPS_AI_ACTION_TIMEOUT_SEC", "120"), 120)
    try:
        rc, out = _run_action_command(list(cmd), timeout_sec=timeout)
        proposal["approved_utc"] = _utcnow().isoformat()
        proposal["approved_by"] = str(actor_chat_id)
        proposal["executed_utc"] = _utcnow().isoformat()
        proposal["exit_code"] = int(rc)
        proposal["output_preview"] = str(out or "")[:2000]
        proposal["status"] = "executed" if int(rc) == 0 else "failed"
        msg = f"{proposal.get('id')} executado: exit_code={rc}"
        if out:
            msg += "\n--- output ---\n" + str(out)[:1500]
        return msg
    except subprocess.TimeoutExpired:
        proposal["status"] = "failed"
        proposal["approved_utc"] = _utcnow().isoformat()
        proposal["approved_by"] = str(actor_chat_id)
        proposal["executed_utc"] = _utcnow().isoformat()
        proposal["exit_code"] = 124
        proposal["output_preview"] = "timeout"
        return f"{proposal.get('id')} falhou: timeout."
    except Exception as e:
        proposal["status"] = "failed"
        proposal["approved_utc"] = _utcnow().isoformat()
        proposal["approved_by"] = str(actor_chat_id)
        proposal["executed_utc"] = _utcnow().isoformat()
        proposal["exit_code"] = 1
        proposal["output_preview"] = str(e)[:500]
        return f"{proposal.get('id')} falhou: {str(e)[:200]}"


async def _process_command(text_msg: str, *, chat_id: str, state: Dict[str, Any]) -> str:
    t = str(text_msg or "").strip()
    low = t.lower()

    if low in ("/help", "/start", "help"):
        return (
            "Comandos do Copiloto IA:\n"
            "- /status\n"
            "- /diag [min]\n"
            "- /pendentes\n"
            "- /aprovar <ID>\n"
            "- /rejeitar <ID>\n"
            "- /historico\n"
        )

    if low in ("/status", "status", "/ops", "/health"):
        report, _ = await build_status_report()
        pending = len(_pending_list(state))
        return report + f"\n\n[AI Copilot] pendentes={pending}"

    if low.startswith("/diag"):
        mins = _parse_diag_minutes(t, default=30)
        diag, created = await _run_diag_and_queue(state, since_minutes=mins, source="manual_diag")
        msg = _fmt_diag(diag)
        if created:
            msg += "\n\nNovas sugestões criadas:\n"
            for p in created[:10]:
                msg += f"- {p.get('id')} {p.get('title')}\n"
        return msg

    if low in ("/pendentes", "pendentes", "/acoes"):
        return _fmt_pending(state)

    if low.startswith("/aprovar"):
        parts = t.split()
        if len(parts) < 2:
            return "Uso: /aprovar <ID>"
        return await _execute_approved_action(state, parts[1], actor_chat_id=str(chat_id))

    if low.startswith("/rejeitar"):
        parts = t.split()
        if len(parts) < 2:
            return "Uso: /rejeitar <ID>"
        p = _find_proposal(state, parts[1])
        if not p:
            return f"ID não encontrado: {parts[1]}"
        if str(p.get("status")) != "pending":
            return f"{p.get('id')}: status atual={p.get('status')} (não está pendente)"
        p["status"] = "rejected"
        p["rejected_utc"] = _utcnow().isoformat()
        p["rejected_by"] = str(chat_id)
        return f"{p.get('id')} rejeitado."

    if low in ("/historico", "historico"):
        return _fmt_history(state)

    return "Comando não reconhecido. Use /help."


async def main_loop() -> int:
    _load_env_file(Path(os.getenv("ENV_FILE", ".env")))

    token = str(
        os.getenv(
            "OPS_AI_TELEGRAM_BOT_TOKEN",
            os.getenv("TELEGRAM_BOT_TOKEN", ""),
        )
        or ""
    ).strip()
    if not token or not _allowed_chat_ids():
        print("ERRO: defina OPS_AI_TELEGRAM_BOT_TOKEN e OPS_AI_TELEGRAM_CHAT_ID (ou fallback TELEGRAM_*).")
        return 2

    poll_timeout = _safe_int(os.getenv("OPS_TELEGRAM_POLL_TIMEOUT_SEC", "30"), 30)
    offset_file = Path(os.getenv("OPS_AI_TELEGRAM_OFFSET_FILE", "logs/ops_ai_telegram_offset.json"))
    state_file = Path(os.getenv("OPS_AI_STATE_FILE", "logs/ops_ai_copilot_state.json"))
    monitor_enable = _env_bool("OPS_AI_MONITOR_ENABLE", "1")
    monitor_interval = max(30, _safe_int(os.getenv("OPS_AI_MONITOR_INTERVAL_SEC", "180"), 180))
    monitor_since = max(5, _safe_int(os.getenv("OPS_AI_MONITOR_SINCE_MINUTES", "30"), 30))
    notify_new = _env_bool("OPS_AI_NOTIFY_ON_NEW_PROPOSAL", "1")

    offset = _load_offset(offset_file)
    state = _load_state(state_file)

    print(
        "telegram_ai_copilot iniciado "
        f"allowed_chat_id(s)={','.join(_allowed_chat_ids())} "
        f"offset={offset} monitor_enable={monitor_enable}"
    )

    while True:
        try:
            now = _utcnow()

            if monitor_enable:
                last_run = _parse_iso_ts(state.get("last_monitor_utc"))
                due = (last_run is None) or ((now - last_run).total_seconds() >= monitor_interval)
                if due:
                    diag, created = await _run_diag_and_queue(state, since_minutes=monitor_since, source="auto_monitor")
                    state["last_monitor_utc"] = now.isoformat()
                    state["last_diag_signature"] = str(diag.get("signature") or "")
                    if created and notify_new:
                        cid = _primary_chat_id()
                        if cid:
                            txt = _fmt_diag(diag) + "\n\n" + _fmt_pending(state)
                            ok_send = _send_message(token, cid, txt)
                            if not ok_send:
                                _log("falha ao enviar notificação automática de novas propostas")
                    _save_state(state_file, state)

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
                if not _is_authorized(chat_id):
                    continue
                try:
                    if str(text_msg).strip().lower().startswith("/diag"):
                        _send_message(token, str(int(chat_id)), "Recebido. Processando diagnóstico...")
                    reply = await _process_command(text_msg, chat_id=str(int(chat_id)), state=state)
                except Exception as e:
                    reply = f"Falha no comando: {str(e)[:200]}"
                _save_state(state_file, state)
                ok_send = _send_message(token, str(int(chat_id)), reply)
                if not ok_send:
                    _log(f"falha ao enviar resposta do comando chat_id={int(chat_id)} cmd={text_msg[:60]}")
        except Exception:
            _log("falha no loop principal; retry em 2s")
            time.sleep(2)


def main() -> int:
    return int(asyncio.run(main_loop()))


if __name__ == "__main__":
    raise SystemExit(main())
