# -*- coding: utf-8 -*-
"""
API HTTP do Copiloto operacional (Camada B).

Objetivo:
- Expor diagnóstico e fila de aprovações para integração com ChatGPT Actions/MCP.
- Reaproveitar motor de diagnóstico/propostas do copiloto Telegram.
- Executar ações SOMENTE com aprovação explícita.

Endpoints:
- GET  /health
- GET  /diag?minutes=30
- GET  /pending
- GET  /history?limit=20
- POST /approve
- POST /reject
- POST /query   (comandos shell read-only em allowlist)
- POST /sql     (consulta SQL read-only)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shlex
import subprocess
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

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
    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
        try:
            return json.loads(raw)
        except Exception:
            return {"ok": False, "raw": raw}
    except HTTPError as e:
        raw = ""
        try:
            raw = e.read().decode("utf-8", errors="ignore")
            parsed = json.loads(raw) if raw else {}
            if isinstance(parsed, dict):
                parsed.setdefault("ok", False)
                parsed.setdefault("http_status", int(getattr(e, "code", 0) or 0))
                return parsed
        except Exception:
            pass
        return {
            "ok": False,
            "error": "http_error",
            "http_status": int(getattr(e, "code", 0) or 0),
            "description": str(e)[:240],
            "raw": raw[:500],
        }
    except URLError as e:
        return {"ok": False, "error": "url_error", "description": str(e)[:240]}
    except Exception as e:
        return {"ok": False, "error": "request_error", "description": str(e)[:240]}


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


_READONLY_CMD_BLOCKLIST = {
    "sudo",
    "su",
    "bash",
    "sh",
    "zsh",
    "fish",
    "python",
    "python3",
    "pip",
    "pip3",
    "apt",
    "apt-get",
    "yum",
    "dnf",
    "apk",
    "git",
    "curl",
    "wget",
    "scp",
    "rsync",
    "mv",
    "cp",
    "rm",
    "touch",
    "tee",
    "chmod",
    "chown",
    "kill",
    "pkill",
    "killall",
    "reboot",
    "shutdown",
}

_READONLY_CMD_ALLOWLIST = {
    "journalctl",
    "systemctl",
    "tail",
    "head",
    "ls",
    "pwd",
    "date",
    "uptime",
    "df",
    "free",
    "ps",
    "ss",
    "ip",
    "rg",
    "wc",
    "cut",
    "sed",
    "awk",
}

_SENSITIVE_ARG_SNIPPETS = (
    ".env",
    ".ssh",
    "id_rsa",
    "authorized_keys",
    "/etc/shadow",
)

_SQL_WRITE_RE = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|grant|revoke|vacuum|analyze|copy|call|do|merge|replace)\b",
    flags=re.IGNORECASE,
)


def _normalize_query_argv(body: Dict[str, Any]) -> Tuple[Optional[List[str]], str]:
    argv_raw = body.get("argv")
    cmd_raw = body.get("command")
    argv: List[str] = []
    if isinstance(argv_raw, list) and argv_raw:
        argv = [str(x) for x in argv_raw if str(x).strip()]
    elif isinstance(cmd_raw, str) and cmd_raw.strip():
        try:
            argv = shlex.split(cmd_raw)
        except Exception:
            return None, "command inválido (falha no parser)"
    else:
        return None, "informe 'command' (string) ou 'argv' (lista)"
    if not argv:
        return None, "comando vazio"
    if len(argv) > 64:
        return None, "comando muito longo"
    if any(len(str(x)) > 800 for x in argv):
        return None, "argumento muito longo"
    return argv, ""


def _validate_readonly_command(argv: List[str]) -> Tuple[bool, str]:
    exe = str(argv[0] or "").strip()
    exe_l = exe.lower()
    if (not exe_l) or (exe_l in _READONLY_CMD_BLOCKLIST):
        return False, f"comando bloqueado: {exe}"
    if exe_l not in _READONLY_CMD_ALLOWLIST:
        return False, f"comando fora da allowlist read-only: {exe}"

    # Guardrails específicos por comando.
    rest = [str(x) for x in argv[1:]]
    rest_l = [x.lower() for x in rest]
    if exe_l == "systemctl":
        if not rest:
            return False, "systemctl requer subcomando"
        sub = rest_l[0]
        if sub not in {"status", "is-active", "is-failed", "show", "list-units"}:
            return False, f"systemctl subcomando não permitido: {rest[0]}"
    if exe_l == "journalctl":
        forbidden_prefixes = ("--vacuum", "--rotate")
        for arg in rest_l:
            if arg.startswith(forbidden_prefixes):
                return False, f"journalctl opção não permitida: {arg}"

    # Bloqueia leitura de arquivos sensíveis.
    for arg in rest_l:
        for snip in _SENSITIVE_ARG_SNIPPETS:
            if snip in arg:
                return False, f"argumento sensível bloqueado: {arg}"
    return True, ""


def _run_readonly_query(argv: List[str], *, timeout_sec: int, max_output_chars: int) -> Tuple[int, str]:
    p = subprocess.run(
        argv,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=max(3, int(timeout_sec)),
    )
    out = str(p.stdout or "")
    limit = max(500, int(max_output_chars))
    if len(out) > limit:
        out = out[:limit] + f"\n... [truncado em {limit} chars]"
    return int(p.returncode), out


def _normalize_sql_query(raw_query: Any) -> Tuple[Optional[str], str]:
    if not isinstance(raw_query, str):
        return None, "query deve ser string"
    q = str(raw_query or "").strip()
    if not q:
        return None, "query vazia"
    # Remove ; final para facilitar wrappers.
    while q.endswith(";"):
        q = q[:-1].rstrip()
    if not q:
        return None, "query vazia"
    if ";" in q:
        return None, "múltiplas instruções não permitidas"
    q_l = q.lower()
    if not (q_l.startswith("select ") or q_l.startswith("with ")):
        return None, "somente SELECT/WITH read-only"
    if _SQL_WRITE_RE.search(q):
        return None, "query contém palavra-chave de escrita/bloqueada"
    return q, ""


def _jsonify_value(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, datetime):
        return v.isoformat()
    return str(v)


async def _run_readonly_sql(query: str, *, max_rows: int) -> Tuple[List[Dict[str, Any]], bool]:
    max_rows = max(1, int(max_rows))
    wrapped = f"SELECT * FROM ({query}) __ops_q LIMIT {max_rows + 1}"
    db = Database()
    await db.connect()
    try:
        async with db.async_session() as s:
            r = await s.execute(text(wrapped))
            fetched = r.fetchall() or []
        rows: List[Dict[str, Any]] = []
        truncated = len(fetched) > max_rows
        for row in fetched[:max_rows]:
            item = dict(row._mapping)
            rows.append({str(k): _jsonify_value(v) for k, v in item.items()})
        return rows, truncated
    finally:
        await db.close()


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


def _proposal_to_summary(p: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "id": str(p.get("id") or ""),
        "kind": p.get("kind"),
        "title": p.get("title"),
        "rationale": p.get("rationale"),
        "risk": p.get("risk"),
        "status": p.get("status"),
        "source": p.get("source"),
        "created_utc": p.get("created_utc"),
        "approved_utc": p.get("approved_utc"),
        "approved_by": p.get("approved_by"),
        "executed_utc": p.get("executed_utc"),
        "exit_code": p.get("exit_code"),
    }
    cmd = p.get("command")
    if isinstance(cmd, list) and cmd:
        out["command"] = [str(x) for x in cmd]
    return out


class CopilotApiApp:
    def __init__(self) -> None:
        _load_env_file(Path(os.getenv("ENV_FILE", ".env")))
        self.state_file = Path(os.getenv("OPS_AI_STATE_FILE", "logs/ops_ai_copilot_state.json"))
        self.state_lock = threading.Lock()
        self.state = _load_state(self.state_file)
        self.monitor_enable = _env_bool("OPS_AI_MONITOR_ENABLE", "1")
        self.monitor_interval = max(30, _safe_int(os.getenv("OPS_AI_MONITOR_INTERVAL_SEC", "180"), 180))
        self.monitor_since = max(5, _safe_int(os.getenv("OPS_AI_MONITOR_SINCE_MINUTES", "30"), 30))
        self._stop_event = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None
        self.api_token = str(os.getenv("OPS_AI_API_TOKEN", "") or "").strip()
        self.allow_no_token = _env_bool("OPS_AI_API_ALLOW_NO_TOKEN", "0")

    def _save_state(self) -> None:
        _save_state(self.state_file, self.state)

    def _auth_ok(self, auth_header: str) -> bool:
        if self.allow_no_token:
            return True
        expected = self.api_token
        if not expected:
            return False
        raw = str(auth_header or "").strip()
        if raw.lower().startswith("bearer "):
            raw = raw[7:].strip()
        return bool(raw) and (raw == expected)

    def start_monitor(self) -> None:
        if not self.monitor_enable:
            return
        if self._monitor_thread and self._monitor_thread.is_alive():
            return
        self._monitor_thread = threading.Thread(target=self._monitor_loop, name="ops-ai-monitor", daemon=True)
        self._monitor_thread.start()

    def stop_monitor(self) -> None:
        self._stop_event.set()
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2)

    def _monitor_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                now = _utcnow()
                with self.state_lock:
                    last_run = _parse_iso_ts(self.state.get("last_monitor_utc"))
                due = (last_run is None) or ((now - last_run).total_seconds() >= self.monitor_interval)
                if due:
                    diag, created = asyncio.run(
                        _run_diag_and_queue(self.state, since_minutes=self.monitor_since, source="auto_monitor_api")
                    )
                    with self.state_lock:
                        self.state["last_monitor_utc"] = now.isoformat()
                        self.state["last_diag_signature"] = str(diag.get("signature") or "")
                        self._save_state()
                    if created:
                        _log(f"monitor_api criou {len(created)} proposta(s); signature={diag.get('signature')}")
            except Exception as e:
                _log(f"falha monitor_api: {type(e).__name__}: {str(e)[:200]}")
            self._stop_event.wait(timeout=5)


class CopilotApiHandler(BaseHTTPRequestHandler):
    app: CopilotApiApp

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        _log(f"http {self.address_string()} {fmt % args}")

    def _read_json_body(self) -> Dict[str, Any]:
        try:
            n = int(self.headers.get("Content-Length", "0") or 0)
            if n <= 0:
                return {}
            raw = self.rfile.read(min(n, 1_000_000)).decode("utf-8", errors="ignore")
            obj = json.loads(raw)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}

    def _write_json(self, code: int, payload: Dict[str, Any]) -> None:
        blob = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(int(code))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(blob)))
        self.end_headers()
        self.wfile.write(blob)

    def _require_auth(self) -> bool:
        auth = str(self.headers.get("Authorization", "") or "")
        if self.app._auth_ok(auth):
            return True
        self._write_json(401, {"ok": False, "error": "unauthorized"})
        return False

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = str(parsed.path or "")
        qs = parse_qs(parsed.query or "", keep_blank_values=False)
        if path != "/health" and not self._require_auth():
            return
        if path == "/health":
            self._write_json(
                200,
                {
                    "ok": True,
                    "service": "ops_ai_copilot_api",
                    "ts_utc": _utcnow().isoformat(),
                    "monitor_enable": self.app.monitor_enable,
                },
            )
            return
        if path == "/diag":
            mins = max(5, _safe_int((qs.get("minutes") or ["30"])[0], 30))
            try:
                with self.app.state_lock:
                    diag, created = asyncio.run(_run_diag_and_queue(self.app.state, since_minutes=mins, source="api_diag"))
                    self.app.state["last_diag_signature"] = str(diag.get("signature") or "")
                    self.app._save_state()
                self._write_json(
                    200,
                    {
                        "ok": True,
                        "diag": diag,
                        "created": [_proposal_to_summary(p) for p in created],
                        "pending_count": len(_pending_list(self.app.state)),
                    },
                )
                return
            except Exception as e:
                self._write_json(500, {"ok": False, "error": str(e)[:240]})
                return
        if path == "/pending":
            with self.app.state_lock:
                pending = _pending_list(self.app.state)
            self._write_json(
                200,
                {"ok": True, "pending_count": len(pending), "pending": [_proposal_to_summary(p) for p in pending]},
            )
            return
        if path == "/history":
            limit = max(1, min(200, _safe_int((qs.get("limit") or ["30"])[0], 30)))
            with self.app.state_lock:
                items = self.app.state.get("proposals") if isinstance(self.app.state.get("proposals"), list) else []
                hist = [p for p in items if str(p.get("status")) in ("executed", "failed", "rejected", "approved_noop")]
                hist = hist[-limit:]
            self._write_json(
                200,
                {"ok": True, "history_count": len(hist), "history": [_proposal_to_summary(p) for p in hist]},
            )
            return
        self._write_json(404, {"ok": False, "error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        if not self._require_auth():
            return
        parsed = urlparse(self.path)
        path = str(parsed.path or "")
        body = self._read_json_body()
        actor = str(body.get("actor") or "api")

        if path == "/query":
            if not _env_bool("OPS_AI_QUERY_ENABLE", "1"):
                self._write_json(403, {"ok": False, "error": "query_disabled"})
                return
            argv, err = _normalize_query_argv(body)
            if argv is None:
                self._write_json(400, {"ok": False, "error": err})
                return
            ok_cmd, reason = _validate_readonly_command(argv)
            if not ok_cmd:
                self._write_json(403, {"ok": False, "error": reason, "argv": argv})
                return
            timeout = max(3, min(120, _safe_int(body.get("timeout_sec"), _safe_int(os.getenv("OPS_AI_QUERY_TIMEOUT_SEC", "30"), 30))))
            out_limit = max(500, min(200_000, _safe_int(body.get("max_output_chars"), _safe_int(os.getenv("OPS_AI_QUERY_MAX_OUTPUT_CHARS", "12000"), 12000))))
            try:
                rc, out = _run_readonly_query(argv, timeout_sec=timeout, max_output_chars=out_limit)
                self._write_json(
                    200,
                    {
                        "ok": True,
                        "argv": argv,
                        "exit_code": int(rc),
                        "output": out,
                        "timeout_sec": timeout,
                        "max_output_chars": out_limit,
                    },
                )
                return
            except subprocess.TimeoutExpired:
                self._write_json(408, {"ok": False, "error": "timeout", "argv": argv})
                return
            except Exception as e:
                self._write_json(500, {"ok": False, "error": str(e)[:240], "argv": argv})
                return

        if path == "/sql":
            if not _env_bool("OPS_AI_SQL_ENABLE", "1"):
                self._write_json(403, {"ok": False, "error": "sql_disabled"})
                return
            q, err = _normalize_sql_query(body.get("query"))
            if q is None:
                self._write_json(400, {"ok": False, "error": err})
                return
            max_rows = max(1, min(5000, _safe_int(body.get("max_rows"), _safe_int(os.getenv("OPS_AI_SQL_MAX_ROWS", "500"), 500))))
            try:
                rows, truncated = asyncio.run(_run_readonly_sql(q, max_rows=max_rows))
                self._write_json(
                    200,
                    {
                        "ok": True,
                        "query": q,
                        "row_count": len(rows),
                        "truncated": bool(truncated),
                        "max_rows": max_rows,
                        "rows": rows,
                    },
                )
                return
            except Exception as e:
                self._write_json(500, {"ok": False, "error": str(e)[:240]})
                return

        if path == "/approve":
            pid = str(body.get("id") or "").strip()
            if not pid:
                self._write_json(400, {"ok": False, "error": "missing_id"})
                return
            with self.app.state_lock:
                msg = asyncio.run(_execute_approved_action(self.app.state, pid, actor_chat_id=actor))
                self.app._save_state()
                proposal = _find_proposal(self.app.state, pid)
            self._write_json(
                200,
                {"ok": True, "message": msg, "proposal": _proposal_to_summary(proposal) if proposal else None},
            )
            return

        if path == "/reject":
            pid = str(body.get("id") or "").strip()
            if not pid:
                self._write_json(400, {"ok": False, "error": "missing_id"})
                return
            with self.app.state_lock:
                p = _find_proposal(self.app.state, pid)
                if not p:
                    self._write_json(404, {"ok": False, "error": "not_found"})
                    return
                if str(p.get("status")) != "pending":
                    self._write_json(409, {"ok": False, "error": f"invalid_status:{p.get('status')}"})
                    return
                p["status"] = "rejected"
                p["rejected_utc"] = _utcnow().isoformat()
                p["rejected_by"] = actor
                self.app._save_state()
            self._write_json(200, {"ok": True, "proposal": _proposal_to_summary(p)})
            return

        self._write_json(404, {"ok": False, "error": "not_found"})


def main() -> int:
    app = CopilotApiApp()
    host = str(os.getenv("OPS_AI_API_HOST", "127.0.0.1") or "127.0.0.1").strip()
    port = max(1, min(65535, _safe_int(os.getenv("OPS_AI_API_PORT", "8787"), 8787)))

    if (not app.allow_no_token) and (not app.api_token):
        print("ERRO: defina OPS_AI_API_TOKEN (ou use OPS_AI_API_ALLOW_NO_TOKEN=1 apenas em ambiente local fechado).")
        return 2

    handler = CopilotApiHandler
    handler.app = app
    server = ThreadingHTTPServer((host, port), handler)
    app.start_monitor()
    _log(f"ops_ai_copilot_api escutando em http://{host}:{port} monitor_enable={app.monitor_enable}")
    try:
        server.serve_forever(poll_interval=0.5)
        return 0
    except KeyboardInterrupt:
        return 0
    finally:
        app.stop_monitor()
        server.server_close()


if __name__ == "__main__":
    raise SystemExit(main())
