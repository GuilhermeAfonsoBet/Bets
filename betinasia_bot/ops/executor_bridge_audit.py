from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
from loguru import logger
from sqlalchemy import text

sys.path.insert(0, ".")

from storage.database import Database
from executor.client import submit_execution
from executor.contracts import ExecSide, ExecutionRequest, MarketType


def _safe_float_money(x: Any) -> Optional[float]:
    """
    Similar a _safe_float, mas tolera strings com separadores (ex.: "1,234.56" ou "1.234,56").
    """
    try:
        if x is None:
            return None
        if isinstance(x, (int, float)):
            return float(x)
        if not isinstance(x, str):
            return float(x)  # type: ignore[arg-type]
        s = x.strip()
        if not s:
            return None
        # heurística simples:
        # - se tiver "," e ".", assume "," = milhares e remove ","
        # - senão, troca "," por "."
        if "," in s and "." in s:
            s = s.replace(",", "")
        else:
            s = s.replace(",", ".")
        # remove símbolos comuns
        for ch in ["$", "€", "£", "R$", "USD", "USDT"]:
            s = s.replace(ch, "")
        s = s.strip()
        return float(s)
    except Exception:
        return None


async def _fetch_executor_account(*, unix_socket: Optional[str], http_base: Optional[str]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Consulta o endpoint /account do executor (mesma sessão do browser), retornando JSON.
    """
    try:
        timeout = aiohttp.ClientTimeout(total=12)
        if unix_socket:
            conn = aiohttp.UnixConnector(path=str(unix_socket))
            async with aiohttp.ClientSession(connector=conn, timeout=timeout) as sess:
                async with sess.get("http://localhost/account") as resp:
                    data = await resp.json()
                    if isinstance(data, dict):
                        data["_http_status"] = int(resp.status)
                    return (data if isinstance(data, dict) else {"_http_status": int(resp.status), "data": data}), None
        base = (http_base or "http://127.0.0.1:8089").rstrip("/")
        async with aiohttp.ClientSession(timeout=timeout) as sess:
            async with sess.get(f"{base}/account") as resp:
                data = await resp.json()
                if isinstance(data, dict):
                    data["_http_status"] = int(resp.status)
                return (data if isinstance(data, dict) else {"_http_status": int(resp.status), "data": data}), None
    except Exception as e:
        return None, str(e)[:250]


def _extract_balance_current_usd_from_executor_account(payload: Dict[str, Any]) -> Optional[float]:
    """
    Best-effort: tenta extrair um número de saldo disponível/atual (USD) do JSON retornado por /account.
    O formato varia conforme endpoint descoberto pelo scraper.
    """
    if not isinstance(payload, dict):
        return None

    bal = payload.get("balance")
    # get_balance_any() retorna {"path": ..., "resp": ..., "attempts": [...]}
    if isinstance(bal, dict) and "resp" in bal:
        bal = bal.get("resp")

    candidates: List[Tuple[int, str, float]] = []  # (score, path, value)

    def _score(path: str) -> int:
        p = path.lower()
        sc = 0
        if "available" in p or "free" in p:
            sc += 100
        if "balance" in p:
            sc += 60
        if "amount" in p:
            sc += 20
        # penaliza coisas que não são saldo
        if "limit" in p or "stake" in p or "liabil" in p or "pnl" in p:
            sc -= 40
        return sc

    def _walk(obj: Any, path: str) -> None:
        if obj is None:
            return
        if isinstance(obj, dict):
            for k, v in obj.items():
                kp = f"{path}.{k}" if path else str(k)
                # candidato numérico direto
                fv = _safe_float_money(v)
                if fv is not None and 0 <= float(fv) <= 50_000_000:
                    candidates.append((_score(kp), kp, float(fv)))
                # recursão
                if isinstance(v, (dict, list)):
                    _walk(v, kp)
            return
        if isinstance(obj, list):
            for i, v in enumerate(obj[:200]):
                kp = f"{path}[{i}]"
                fv = _safe_float_money(v)
                if fv is not None and 0 <= float(fv) <= 50_000_000:
                    candidates.append((_score(kp), kp, float(fv)))
                if isinstance(v, (dict, list)):
                    _walk(v, kp)
            return

    _walk(bal, "balance")
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], x[2]), reverse=True)
    best = candidates[0]
    if best[0] <= 0:
        return None
    return float(best[2])


def _telegram_send(token: str, chat_id: str, text_msg: str) -> bool:
    try:
        from urllib.parse import urlencode
        from urllib.request import Request, urlopen

        url = f"https://api.telegram.org/bot{token}/sendMessage"
        data = urlencode({"chat_id": chat_id, "text": text_msg}).encode("utf-8")
        req = Request(url, data=data, method="POST")
        with urlopen(req, timeout=10) as resp:
            return 200 <= int(resp.status) < 300
    except Exception:
        return False


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_iso(s: str) -> Optional[datetime]:
    try:
        if not s:
            return None
        t = str(s).strip()
        if t.endswith("Z"):
            t = t[:-1] + "+00:00"
        dt = datetime.fromisoformat(t)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _norm_line(line: str) -> str:
    return (str(line or "").strip()).replace(",", ".").replace("−", "-")


@dataclass
class BridgeConfig:
    poll_sec: float = 2.0
    lookback_sec: int = 120
    max_per_cycle: int = 3
    mode: str = "shadow"  # shadow|live
    exec_side: ExecSide = ExecSide.BACK
    stake: float = 3.0
    unix_socket: str = "/tmp/betinasia-exec.sock"
    http_url: Optional[str] = None
    only_hypothesis: str = "H3B"
    only_prematch: bool = True
    # Policy OOS (export do report walk-forward)
    policy_json: Optional[str] = None
    policy_reload_sec: float = 5.0
    policy_use_base: bool = False
    # Bankroll (para sizing dinâmico)
    bankroll_ref: Optional[float] = None
    bankroll_json: Optional[str] = None  # ex.: logs/accounting_daily_report.json
    bankroll_reload_sec: float = 30.0
    # Engine de budget por jogo (match_budget do WF)
    use_wf_budget: bool = False
    # Contagem de sinais (para signals_sqrt/linear): janela curta (best-effort)
    signals_lookback_h: float = 36.0
    # Override manual do risk_mode do WF (mantém daily estável).
    # Valores: fixed|signals_sqrt|signals_linear
    wf_risk_mode_override: Optional[str] = None
    # Override manual dos parâmetros numéricos de risco (frações/caps),
    # separados do daily/policy (para go-live controlado).
    # Formato esperado (exemplo):
    # {
    #   "budget_back_frac": 0.01,
    #   "budget_lay_frac": 0.005,
    #   "cap_signal_frac": 0.33,
    #   "cap_event_back_frac": 0.02,
    #   "cap_event_lay_frac": 0.01,
    #   "signals_lookback_h": 36
    # }
    risk_params_json: Optional[str] = None
    risk_params_reload_sec: float = 5.0
    # Guardrail adicional: cap por jogo como fração da banca (0=off). Back em stake; Lay em liability.
    cap_event_back_frac: float = 0.0
    cap_event_lay_frac: float = 0.0
    # Guardrails simples
    min_limit: float = 0.0
    # Alinhamento OOS vs bridge: quando True (default), aplica filtros/limiares do WF (policy_json.wf)
    # no bridge, evitando divergência com o OOS.
    enforce_wf_filters: bool = True


def _wf_float(wf: Dict[str, Any], *keys: str) -> Optional[float]:
    for k in keys:
        if not k:
            continue
        if k in wf and wf.get(k) is not None:
            try:
                return float(wf.get(k))
            except Exception:
                continue
    return None


def _wf_str(wf: Dict[str, Any], *keys: str, default: str = "") -> str:
    for k in keys:
        if not k:
            continue
        if k in wf and wf.get(k) is not None:
            try:
                s = str(wf.get(k)).strip()
                if s:
                    return s
            except Exception:
                continue
    return default


def _wf_effective_min_limit(cfg: BridgeConfig, wf: Dict[str, Any]) -> float:
    wf_min = _wf_float(wf, "liquidity_min_limit", "liquidity_min", "min_limit")
    if cfg.enforce_wf_filters and wf_min is not None:
        return float(max(0.0, float(wf_min)))
    # compat: se não estiver enforcing, preserva guardrail local (pode ser mais restritivo)
    try:
        wf_min2 = float(wf_min) if wf_min is not None else 0.0
    except Exception:
        wf_min2 = 0.0
    return float(max(0.0, float(cfg.min_limit or 0.0), float(wf_min2)))


def _wf_apply_scope(scope: str, *, is_live: bool) -> bool:
    sc = (scope or "").strip().lower() or "pre"
    if sc == "all":
        return True
    if sc == "in":
        return bool(is_live)
    # default/pre
    return not bool(is_live)


def _wf_filters_summary(cfg: BridgeConfig, wf: Dict[str, Any]) -> str:
    try:
        thr = _wf_float(wf, "ah_max_abs_line")
        scope = _wf_str(wf, "ah_scope", default="pre")
        key_by_league = bool(wf.get("key_by_league"))
        key_scope = _wf_str(wf, "key_by_league_scope", default="pre")
        mlim = _wf_effective_min_limit(cfg, wf)
        return f"enforce={int(bool(cfg.enforce_wf_filters))} ah_max_abs_line={thr if thr is not None else '-'} ah_scope={scope} key_by_league={int(key_by_league)} key_scope={key_scope} min_limit_eff={mlim:.4g}"
    except Exception:
        return f"enforce={int(bool(cfg.enforce_wf_filters))}"


DDL_SEEN = """
CREATE TABLE IF NOT EXISTS executor_bridge_seen (
  id BIGSERIAL PRIMARY KEY,
  src_table TEXT NOT NULL,
  src_id BIGINT NOT NULL,
  action TEXT NOT NULL,
  execution_id UUID NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  meta JSONB NULL,
  UNIQUE (src_table, src_id, action)
);
"""

DDL_SEEN_KEYS = """
CREATE TABLE IF NOT EXISTS executor_bridge_seen_keys (
  id BIGSERIAL PRIMARY KEY,
  src_table TEXT NOT NULL,
  src_key TEXT NOT NULL,
  action TEXT NOT NULL,
  execution_id UUID NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  meta JSONB NULL,
  UNIQUE (src_table, src_key, action)
);
"""

DDL_POSITIONS = """
CREATE TABLE IF NOT EXISTS executor_bridge_positions (
  id BIGSERIAL PRIMARY KEY,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  action TEXT NOT NULL,
  mode TEXT NOT NULL,
  exec_side TEXT NOT NULL,
  event_id TEXT NOT NULL,
  match_key TEXT NOT NULL,
  src_id BIGINT NULL,
  src_key TEXT NULL,
  execution_id UUID NULL,
  status TEXT NOT NULL DEFAULT 'SUBMITTED',
  stake_requested DOUBLE PRECISION NULL,
  liability_requested DOUBLE PRECISION NULL,
  bankroll_ref DOUBLE PRECISION NULL,
  budget_match DOUBLE PRECISION NULL,
  cap_signal DOUBLE PRECISION NULL,
  cap_event DOUBLE PRECISION NULL,
  spent_before DOUBLE PRECISION NULL,
  spent_after DOUBLE PRECISION NULL,
  n_signals_est INTEGER NULL,
  risk_mode TEXT NULL,
  meta JSONB NULL
);
"""

DDL_POSITIONS_IDX = [
    "CREATE INDEX IF NOT EXISTS idx_bridge_pos_match ON executor_bridge_positions (match_key, exec_side, created_at);",
    "CREATE INDEX IF NOT EXISTS idx_bridge_pos_event ON executor_bridge_positions (event_id, exec_side, created_at);",
    "CREATE INDEX IF NOT EXISTS idx_bridge_pos_execid ON executor_bridge_positions (execution_id);",
]


async def _ensure_seen_table(db: Database) -> None:
    async with db.engine.begin() as conn:
        await conn.execute(text(DDL_SEEN))
        await conn.execute(text(DDL_SEEN_KEYS))
        await conn.execute(text(DDL_POSITIONS))
        for stmt in DDL_POSITIONS_IDX:
            await conn.execute(text(stmt))


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def _parse_details(row: Dict[str, Any]) -> Dict[str, Any]:
    v = row.get("hypothesis_details")
    if isinstance(v, dict):
        return v
    if isinstance(v, str) and v.strip():
        try:
            obj = json.loads(v)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
    return {}


def _finance_snapshot(details: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    fin = details.get("finance")
    return fin if isinstance(fin, dict) else None


def _limit_from_finance(details: Dict[str, Any], *, exec_side: ExecSide) -> Tuple[Optional[float], Optional[float]]:
    """
    Retorna (available_limit, odd) a partir de hypothesis_details.finance, se existir.
    Para Back: usa finance.back.available_limit e finance.back.odd
    Para Lay: usa finance.lay.available_limit e finance.lay.odd
    """
    fin = _finance_snapshot(details) or {}
    try:
        if exec_side == ExecSide.BACK:
            blk = fin.get("back") if isinstance(fin.get("back"), dict) else {}
        else:
            blk = fin.get("lay") if isinstance(fin.get("lay"), dict) else {}
        lim = _safe_float((blk or {}).get("available_limit"))
        odd = _safe_float((blk or {}).get("odd"))
        return (float(lim) if lim is not None else None, float(odd) if odd is not None else None)
    except Exception:
        return None, None


def _stake_from_limit(*, limit_value: float, stake_pct_of_limit: float, stake_cap_abs: float) -> float:
    st = max(0.0, float(limit_value)) * max(0.0, float(stake_pct_of_limit))
    if stake_cap_abs and float(stake_cap_abs) > 0:
        st = min(float(st), float(stake_cap_abs))
    return float(st)


def _base_exposure_from_limit(
    *,
    exec_side: ExecSide,
    limit_value: Optional[float],
    odd: Optional[float],
    stake_pct_of_limit: float,
    stake_cap_abs: float,
) -> Tuple[Optional[float], Optional[str]]:
    """
    Usa limit_value + stake_pct_of_limit para definir base exposure:
      - Back: exposure = stake
      - Lay: exposure = liability = stake*(odd-1)
    """
    if limit_value is None or float(limit_value) <= 0:
        return None, "no_limit"
    st = _stake_from_limit(limit_value=float(limit_value), stake_pct_of_limit=float(stake_pct_of_limit), stake_cap_abs=float(stake_cap_abs))
    if st <= 0:
        return None, "stake_from_limit<=0"
    if exec_side == ExecSide.BACK:
        return float(st), "limit"
    if odd is None or float(odd) <= 1.0:
        return None, "no_odd_for_lay"
    return float(st) * max(0.0, float(odd) - 1.0), "limit_x_(odd-1)"


def _base_exposure_from_finance(
    *,
    exec_side: ExecSide,
    details: Dict[str, Any],
    odd_hint: Optional[float],
    stake_fallback: float,
) -> Tuple[Optional[float], Optional[str]]:
    """
    Retorna (exposure, why) em unidade:
      - Back: exposure = stake
      - Lay: exposure = liability
    """
    fin = _finance_snapshot(details) or {}
    if exec_side == ExecSide.BACK:
        try:
            s = _safe_float((fin.get("back", {}) or {}).get("suggested_stake"))
            if s is not None and s > 0:
                return float(s), "finance.back.suggested_stake"
        except Exception:
            pass
        return float(max(0.0, float(stake_fallback))), "fallback_stake"

    try:
        ll = _safe_float((fin.get("lay", {}) or {}).get("liability_if_lose"))
        if ll is not None and ll > 0:
            return float(ll), "finance.lay.liability_if_lose"
    except Exception:
        pass
    if odd_hint is not None and float(odd_hint) > 1.0:
        try:
            liab = float(stake_fallback) * max(0.0, float(odd_hint) - 1.0)
            if liab > 0:
                return float(liab), "fallback_stake_x_(odd-1)"
        except Exception:
            pass
    return None, "no_finance_no_odd"


def _load_bankroll_from_json(path: str) -> Optional[float]:
    """
    Lê um JSON (ex.: logs/accounting_daily_report.json) e retorna balance_current.
    """
    try:
        p = Path(path)
        if not p.exists():
            return None
        obj = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            return None
        b = _safe_float(obj.get("balance_current"))
        return float(b) if b is not None and b > 0 else None
    except Exception:
        return None


def _load_risk_params_json(path: str) -> Dict[str, Any]:
    """
    Lê overrides manuais de risco/caps (best-effort).
    """
    try:
        p = Path(path)
        if not p.exists():
            return {}
        obj = json.loads(p.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _rp_float(rp: Dict[str, Any], key: str) -> Optional[float]:
    try:
        if not rp or key not in rp:
            return None
        v = _safe_float(rp.get(key))
        return float(v) if v is not None else None
    except Exception:
        return None


def _rp_str(rp: Dict[str, Any], key: str) -> Optional[str]:
    try:
        if not rp or key not in rp:
            return None
        v = rp.get(key)
        if v is None:
            return None
        s = str(v).strip()
        return s or None
    except Exception:
        return None


def _rp_bool(rp: Dict[str, Any], key: str) -> Optional[bool]:
    try:
        if not rp or key not in rp:
            return None
        v = rp.get(key)
        if v is None:
            return None
        if isinstance(v, bool):
            return bool(v)
        s = str(v).strip().lower()
        if s in ("1", "true", "yes", "y", "on"):
            return True
        if s in ("0", "false", "no", "n", "off"):
            return False
        return None
    except Exception:
        return None


def _event_key(row: Dict[str, Any], cfg: BridgeConfig) -> str:
    event_id = str(row.get("event_id") or "").strip()
    market = str(row.get("market_type") or "AH").strip().upper()
    line = _norm_line(str(row.get("line") or ""))
    side = str(row.get("side") or "").strip().lower()
    is_live = bool(row.get("is_live")) if row.get("is_live") is not None else False
    hyp = str(row.get("hypothesis_type") or "").strip().upper()
    regime = "in" if is_live else "pre"
    return f"{event_id}|{market}|{line}|{side}|{cfg.exec_side.value}|{cfg.mode}|{regime}|{hyp}"


async def _reserve_seen_key(
    db: Database,
    *,
    src_key: str,
    action: str,
    meta: Dict[str, Any],
) -> bool:
    q = """
    INSERT INTO executor_bridge_seen_keys (src_table, src_key, action, execution_id, meta)
    VALUES ('betslip_audit_results', :src_key, :action, NULL, (:meta)::jsonb)
    ON CONFLICT (src_table, src_key, action) DO NOTHING
    RETURNING id
    """
    async with db.async_session() as session:
        r = await session.execute(
            text(q),
            {"src_key": str(src_key), "action": str(action), "meta": json.dumps(meta, ensure_ascii=False)},
        )
        row = r.fetchone()
        await session.commit()
        return bool(row and row[0])


async def _finalize_seen_key(
    db: Database,
    *,
    src_key: str,
    action: str,
    execution_id: Optional[str],
) -> None:
    if not execution_id:
        return
    q = """
    UPDATE executor_bridge_seen_keys
    SET execution_id = :execution_id
    WHERE src_table='betslip_audit_results' AND src_key=:src_key AND action=:action
    """
    async with db.async_session() as session:
        await session.execute(
            text(q),
            {"execution_id": str(execution_id), "src_key": str(src_key), "action": str(action)},
        )
        await session.commit()

async def _unreserve_seen_key(
    db: Database,
    *,
    src_key: str,
    action: str,
) -> None:
    """
    Remove a reserva em executor_bridge_seen_keys para permitir retry em falhas transitórias
    (ex.: executor not_ready / socket down). Safe-noop se não existir.
    """
    q = """
    DELETE FROM executor_bridge_seen_keys
    WHERE src_table='betslip_audit_results' AND src_key=:src_key AND action=:action AND execution_id IS NULL
    """
    async with db.async_session() as session:
        await session.execute(text(q), {"src_key": str(src_key), "action": str(action)})
        await session.commit()


def _load_policy_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        p = Path(path)
        if not p.exists():
            return None
        obj = json.loads(p.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _combo_key_from_row(row: Dict[str, Any], cfg: BridgeConfig, policy: Dict[str, Any]) -> str:
    is_live = bool(row.get("is_live")) if row.get("is_live") is not None else False
    regime = "In" if is_live else "Pre"

    if cfg.exec_side == ExecSide.BACK:
        comb = f"Back_{regime}_Any"
    else:
        hyp = str(row.get("hypothesis_type") or "").strip().upper()
        details = row.get("hypothesis_details")
        if isinstance(details, str):
            try:
                details = json.loads(details)
            except Exception:
                details = None
        had_rev: Optional[bool] = None
        if hyp == "H3B":
            had_rev = True
        elif isinstance(details, dict) and "had_reversal" in details:
            had_rev = bool(details.get("had_reversal"))
        else:
            had_rev = bool(str(row.get("reversal_direction") or "").strip())
        rev = "Yes" if had_rev else "No"
        comb = f"Lay_{regime}_{rev}"

    wf = policy.get("wf") if isinstance(policy.get("wf"), dict) else {}
    key_by_league = bool(wf.get("key_by_league"))
    scope = str(wf.get("key_by_league_scope") or "pre").strip().lower()
    if key_by_league and (scope == "all" or regime == "Pre"):
        league = str(row.get("league") or "").strip()
        if league:
            comb = f"{comb}__{league}"
    return comb


async def _fetch_candidates(
    db: Database,
    *,
    since: datetime,
    cfg: BridgeConfig,
) -> List[Dict[str, Any]]:
    # Nota: betslip_audit_results está em models_hypothesis (tabela criada pela connect()).
    q = """
    SELECT
      r.id,
      r.hypothesis_type,
      r.event_id,
      r.market_type,
      r.line,
      r.side,
      r.is_live,
      r.websocket_odd,
      r.betslip_odd,
      r.betslip_limit,
      r.league,
      r.reversal_direction,
      r.hypothesis_details,
      r.audited_at
    FROM betslip_audit_results r
    LEFT JOIN executor_bridge_seen s
      ON s.src_table='betslip_audit_results' AND s.src_id=r.id AND s.action=:action
    WHERE r.audited_at >= :since
      AND s.id IS NULL
      AND r.is_valid_opportunity = TRUE
      AND (
        r.hypothesis_details IS NULL
        OR COALESCE((r.hypothesis_details::jsonb->>'exec_side_hint'), '') = ''
        OR lower(r.hypothesis_details::jsonb->>'exec_side_hint') = lower(:exec_side_hint)
      )
      AND r.event_id IS NOT NULL AND r.event_id <> ''
      AND upper(r.market_type) = 'AH'
      AND r.hypothesis_type = :hyp
    ORDER BY r.audited_at ASC
    LIMIT :lim
    """
    params = {
        "since": since,
        "lim": int(cfg.max_per_cycle),
        "action": f"{cfg.mode}:{cfg.exec_side.value}",
        "hyp": str(cfg.only_hypothesis),
        "exec_side_hint": str(cfg.exec_side.value),
    }
    if cfg.only_prematch:
        q = q.replace("AND r.hypothesis_type = :hyp", "AND r.hypothesis_type = :hyp AND (r.is_live IS NULL OR r.is_live = FALSE)")
    async with db.async_session() as session:
        r = await session.execute(text(q), params)
        rows = r.fetchall() or []
        return [dict(x._mapping) for x in rows]


async def _mark_seen(
    db: Database,
    *,
    src_id: int,
    action: str,
    execution_id: Optional[str],
    meta: Dict[str, Any],
) -> None:
    q = """
    INSERT INTO executor_bridge_seen (src_table, src_id, action, execution_id, meta)
    VALUES ('betslip_audit_results', :src_id, :action, :execution_id, (:meta)::jsonb)
    ON CONFLICT (src_table, src_id, action) DO NOTHING
    """
    async with db.async_session() as session:
        await session.execute(
            text(q),
            {
                "src_id": int(src_id),
                "action": str(action),
                "execution_id": execution_id,
                "meta": json.dumps(meta, ensure_ascii=False),
            },
        )
        await session.commit()


def _build_request(row: Dict[str, Any], cfg: BridgeConfig) -> ExecutionRequest:
    odd_dec = None
    try:
        odd_dec = float(row.get("websocket_odd") or 0) or None
    except Exception:
        odd_dec = None
    if odd_dec is None:
        try:
            odd_dec = float(row.get("betslip_odd") or 0) or None
        except Exception:
            odd_dec = None

    req = ExecutionRequest(
        created_at=_utcnow(),
        audit_id=int(row.get("id") or 0),
        event_id=str(row.get("event_id")),
        market_type=MarketType.AH,
        side=str(row.get("side") or "").strip(),
        line=_norm_line(str(row.get("line") or "")),
        exec_side=cfg.exec_side,
        is_live=(cfg.mode == "live"),
        odd_at_decision=odd_dec,
    )
    req.policy.policy_version = f"bridge_{cfg.only_hypothesis.lower()}_{cfg.mode}_v0"
    # stake_requested / liability_requested podem ser sobrescritos pelo engine dinâmico (WF budget).
    if cfg.exec_side == ExecSide.BACK:
        req.policy.stake_requested = float(cfg.stake)
    # meta útil para auditoria
    req.meta["bridge"] = {
        "src": "betslip_audit_results",
        "src_id": int(row.get("id") or 0),
        "hypothesis_type": str(row.get("hypothesis_type") or ""),
        "audited_at": str(row.get("audited_at") or ""),
        "betslip_limit": row.get("betslip_limit"),
    }
    return req


async def _signals_count_estimate(
    db: Database,
    *,
    event_id: str,
    hyp: str,
    exec_side: ExecSide,
    lookback_h: float,
    prematch_only: bool,
) -> int:
    """
    Estimativa do total de sinais para este jogo (event_id) numa janela curta.
    Usado para risk_mode signals_sqrt/signals_linear (aproxima OOS).
    """
    t0 = _utcnow() - timedelta(hours=float(max(1e-6, lookback_h)))
    q = """
    SELECT COUNT(*)::bigint AS n
    FROM betslip_audit_results r
    WHERE r.audited_at >= :t0
      AND r.event_id = :event_id
      AND r.hypothesis_type = :hyp
      AND r.is_valid_opportunity = TRUE
      AND (:prematch_only = FALSE OR r.is_live IS NULL OR r.is_live = FALSE)
      AND (
        r.hypothesis_details IS NULL
        OR COALESCE((r.hypothesis_details::jsonb->>'exec_side_hint'), '') = ''
        OR lower(r.hypothesis_details::jsonb->>'exec_side_hint') = lower(:exec_side_hint)
      )
    """
    async with db.async_session() as session:
        r = await session.execute(
            text(q),
            {
                "t0": t0,
                "event_id": str(event_id),
                "hyp": str(hyp),
                "exec_side_hint": str(exec_side.value),
                "prematch_only": bool(prematch_only),
            },
        )
        row = r.fetchone()
        try:
            n = int(row[0]) if row else 0
        except Exception:
            n = 0
        return max(1, n)


async def _spent_for_match(
    db: Database,
    *,
    match_key: str,
    exec_side: ExecSide,
    mode: str,
    action: str,
) -> float:
    """
    Soma de exposição já "consumida" para este match_key.
    Back: soma stake_requested; Lay: soma liability_requested.
    """
    col = "stake_requested" if exec_side == ExecSide.BACK else "liability_requested"
    q = f"""
    SELECT COALESCE(SUM(COALESCE({col}, 0)), 0)::double precision AS spent
    FROM executor_bridge_positions
    WHERE match_key = :match_key
      AND exec_side = :exec_side
      AND mode = :mode
      AND action = :action
      AND status IN ('SUBMITTED','DRY_OK','LIVE_OK')
    """
    async with db.async_session() as session:
        r = await session.execute(
            text(q),
            {
                "match_key": str(match_key),
                "exec_side": str(exec_side.value),
                "mode": str(mode),
                "action": str(action),
            },
        )
        row = r.fetchone()
        try:
            return float(row[0] or 0.0) if row else 0.0
        except Exception:
            return 0.0


async def _insert_position(
    db: Database,
    *,
    action: str,
    mode: str,
    exec_side: ExecSide,
    event_id: str,
    match_key: str,
    src_id: int,
    src_key: str,
    execution_id: Optional[str],
    stake_requested: Optional[float],
    liability_requested: Optional[float],
    ctx: Dict[str, Any],
) -> None:
    q = """
    INSERT INTO executor_bridge_positions (
      action, mode, exec_side, event_id, match_key, src_id, src_key, execution_id,
      status, stake_requested, liability_requested,
      bankroll_ref, budget_match, cap_signal, cap_event, spent_before, spent_after, n_signals_est, risk_mode, meta
    )
    VALUES (
      :action, :mode, :exec_side, :event_id, :match_key, :src_id, :src_key, :execution_id,
      :status, :stake_requested, :liability_requested,
      :bankroll_ref, :budget_match, :cap_signal, :cap_event, :spent_before, :spent_after, :n_signals_est, :risk_mode, (:meta)::jsonb
    )
    """
    meta = dict(ctx or {})
    async with db.async_session() as session:
        await session.execute(
            text(q),
            {
                "action": str(action),
                "mode": str(mode),
                "exec_side": str(exec_side.value),
                "event_id": str(event_id),
                "match_key": str(match_key),
                "src_id": int(src_id),
                "src_key": str(src_key),
                "execution_id": (str(execution_id) if execution_id else None),
                "status": str(meta.pop("status", "SUBMITTED")),
                "stake_requested": (float(stake_requested) if stake_requested is not None else None),
                "liability_requested": (float(liability_requested) if liability_requested is not None else None),
                "bankroll_ref": _safe_float(meta.get("bankroll_ref")),
                "budget_match": _safe_float(meta.get("budget_match")),
                "cap_signal": _safe_float(meta.get("cap_signal")),
                "cap_event": _safe_float(meta.get("cap_event")),
                "spent_before": _safe_float(meta.get("spent_before")),
                "spent_after": _safe_float(meta.get("spent_after")),
                "n_signals_est": _safe_int(meta.get("n_signals_est")),
                "risk_mode": (str(meta.get("risk_mode")) if meta.get("risk_mode") is not None else None),
                "meta": json.dumps(meta, ensure_ascii=False),
            },
        )
        await session.commit()


async def run_bridge(cfg: BridgeConfig) -> int:
    db = Database()
    await db.connect()
    await _ensure_seen_table(db)

    policy: Optional[Dict[str, Any]] = None
    policy_mtime: Optional[float] = None
    policy_last_check = 0.0
    active_keys: Optional[set] = None
    active_keys_base: Optional[set] = None
    bankroll_mtime: Optional[float] = None
    bankroll_last_check = 0.0
    bankroll_ref: Optional[float] = cfg.bankroll_ref
    risk_mtime: Optional[float] = None
    risk_last_check = 0.0
    risk_params: Dict[str, Any] = {}

    # Monitoramento de saldo (guardrail operacional)
    bal_last_check = 0.0
    bal_payload: Optional[Dict[str, Any]] = None
    bal_current: Optional[float] = None
    bal_http: Optional[int] = None
    bal_err: Optional[str] = None
    tg_last_alert_ts = 0.0
    tg_token = str(os.getenv("TELEGRAM_BOT_TOKEN", "") or "").strip()
    tg_chat = str(os.getenv("TELEGRAM_CHAT_ID", "") or "").strip()

    logger.info(
        f"[bridge] started mode={cfg.mode} exec_side={cfg.exec_side.value} "
        f"poll_sec={cfg.poll_sec} lookback_sec={cfg.lookback_sec} max_per_cycle={cfg.max_per_cycle} "
        f"hyp={cfg.only_hypothesis} prematch_only={cfg.only_prematch} "
        f"policy_json={cfg.policy_json or '-'} use_base={cfg.policy_use_base} "
        f"use_wf_budget={cfg.use_wf_budget} bankroll_json={cfg.bankroll_json or '-'} "
        f"bankroll_ref={(bankroll_ref if bankroll_ref is not None else '-')} "
        f"min_limit={cfg.min_limit} enforce_wf_filters={int(bool(cfg.enforce_wf_filters))}"
    )

    while True:
        t0 = time.time()
        # reload policy (se configurado)
        if cfg.policy_json and (time.time() - policy_last_check) >= float(cfg.policy_reload_sec):
            policy_last_check = time.time()
            try:
                p = Path(cfg.policy_json)
                mtime = p.stat().st_mtime if p.exists() else None
                if mtime and (policy_mtime is None or float(mtime) > float(policy_mtime)):
                    pol = _load_policy_json(cfg.policy_json)
                    if pol:
                        policy = pol
                        policy_mtime = float(mtime)
                        steps = pol.get("steps") if isinstance(pol.get("steps"), list) else []
                        last = steps[-1] if steps else None
                        if isinstance(last, dict):
                            active_keys = set(last.get("active_keys") or [])
                            active_keys_base = set(last.get("active_keys_base") or [])
                            logger.info(
                                f"[bridge] policy reloaded mtime={policy_mtime:.0f} "
                                f"active_keys={len(active_keys)} active_base={len(active_keys_base)}"
                            )
                        try:
                            wf = pol.get("wf") if isinstance(pol.get("wf"), dict) else {}
                            if isinstance(wf, dict) and wf:
                                logger.info(f"[bridge] wf filters: {_wf_filters_summary(cfg, wf)}")
                        except Exception:
                            pass
            except Exception as e:
                logger.warning(f"[bridge] policy reload failed: {e}")

        # reload bankroll (se configurado)
        if cfg.bankroll_json and (time.time() - bankroll_last_check) >= float(cfg.bankroll_reload_sec):
            bankroll_last_check = time.time()
            try:
                p = Path(cfg.bankroll_json)
                mtime = p.stat().st_mtime if p.exists() else None
                if mtime and (bankroll_mtime is None or float(mtime) > float(bankroll_mtime)):
                    b = _load_bankroll_from_json(cfg.bankroll_json)
                    if b is not None and b > 0:
                        bankroll_ref = float(b)
                        bankroll_mtime = float(mtime)
                        logger.info(f"[bridge] bankroll reloaded mtime={bankroll_mtime:.0f} bankroll_ref={bankroll_ref:.2f}")
            except Exception as e:
                logger.warning(f"[bridge] bankroll reload failed: {e}")

        # reload risk params (overrides manuais)
        if cfg.risk_params_json and (time.time() - risk_last_check) >= float(cfg.risk_params_reload_sec):
            risk_last_check = time.time()
            try:
                p = Path(cfg.risk_params_json)
                mtime = p.stat().st_mtime if p.exists() else None
                if mtime and (risk_mtime is None or float(mtime) > float(risk_mtime)):
                    rp = _load_risk_params_json(cfg.risk_params_json)
                    if isinstance(rp, dict):
                        risk_params = rp
                        risk_mtime = float(mtime)
                        logger.info(f"[bridge] risk_params reloaded mtime={risk_mtime:.0f} path={cfg.risk_params_json}")
            except Exception as e:
                logger.warning(f"[bridge] risk_params reload failed: {e}")

        since = _utcnow() - timedelta(seconds=int(cfg.lookback_sec))
        rows = await _fetch_candidates(db, since=since, cfg=cfg)
        if not rows:
            await asyncio.sleep(float(cfg.poll_sec))
            continue

        for row in rows:
            src_id = int(row.get("id") or 0)
            action = f"{cfg.mode}:{cfg.exec_side.value}"
            try:
                wf = policy.get("wf") if (policy and isinstance(policy.get("wf"), dict)) else {}
                if not isinstance(wf, dict):
                    wf = {}

                # -----------------------------
                # Filtros do WF/OOS (alinhamento)
                # -----------------------------
                # 1) AH gate por abs(line)
                try:
                    thr = _wf_float(wf, "ah_max_abs_line")
                    scope = _wf_str(wf, "ah_scope", default="pre")
                    if thr is not None and float(thr) > 0:
                        is_live = bool(row.get("is_live")) if row.get("is_live") is not None else False
                        if _wf_apply_scope(scope, is_live=is_live):
                            ln = str(row.get("line") or "").strip()
                            if ln:
                                try:
                                    x = abs(float(_norm_line(ln)))
                                except Exception:
                                    x = None
                                if x is not None and float(x) > float(thr):
                                    await _mark_seen(
                                        db,
                                        src_id=src_id,
                                        action=action,
                                        execution_id=None,
                                        meta={
                                            "skipped": True,
                                            "reason": "wf_ah_max_abs_line",
                                            "line": ln,
                                            "abs_line": float(x),
                                            "thr": float(thr),
                                            "scope": scope,
                                            "is_live": bool(is_live),
                                        },
                                    )
                                    continue
                except Exception:
                    pass

                # guardrail: limit mínimo
                lim = _safe_float(row.get("betslip_limit"))
                min_lim_eff = _wf_effective_min_limit(cfg, wf)
                if min_lim_eff and lim is not None and float(lim) < float(min_lim_eff):
                    await _mark_seen(
                        db,
                        src_id=src_id,
                        action=action,
                        execution_id=None,
                        meta={
                            "skipped": True,
                            "reason": "min_limit",
                            "betslip_limit": lim,
                            "min_limit_eff": float(min_lim_eff),
                            "min_limit_cfg": float(cfg.min_limit or 0.0),
                            "min_limit_wf": _wf_float(wf, "liquidity_min_limit", "liquidity_min", "min_limit"),
                            "enforce_wf_filters": bool(cfg.enforce_wf_filters),
                        },
                    )
                    continue

                # policy OOS: só executa se combinação estiver ativa
                if policy and active_keys is not None:
                    comb = _combo_key_from_row(row, cfg, policy)
                    ok = False
                    # Alinhamento com OOS: quando enforcing, nunca colapsa a key por liga (não usa active_keys_base)
                    use_base = bool(cfg.policy_use_base) and (not bool(cfg.enforce_wf_filters))
                    if use_base and active_keys_base is not None and active_keys_base:
                        ok = str(comb).split("__", 1)[0] in active_keys_base
                    else:
                        ok = comb in active_keys
                    if not ok:
                        await _mark_seen(
                            db,
                            src_id=src_id,
                            action=action,
                            execution_id=None,
                            meta={
                                "skipped": True,
                                "reason": "not_active",
                                "combo": comb,
                                "policy_use_base_cfg": bool(cfg.policy_use_base),
                                "policy_use_base_eff": bool(use_base),
                                "enforce_wf_filters": bool(cfg.enforce_wf_filters),
                            },
                        )
                        continue

                # dedupe por chave operacional
                skey = _event_key(row, cfg)
                reserved = await _reserve_seen_key(
                    db,
                    src_key=skey,
                    action=action,
                    meta={"src_id": src_id, "audited_at": str(row.get("audited_at") or ""), "event_id": row.get("event_id")},
                )
                if not reserved:
                    await _mark_seen(
                        db,
                        src_id=src_id,
                        action=action,
                        execution_id=None,
                        meta={"skipped": True, "reason": "dup_key", "src_key": skey},
                    )
                    continue

                # Kill-switch por lado (controle manual via risk_params_json)
                try:
                    if cfg.exec_side == ExecSide.BACK and bool(_rp_bool(risk_params, "disable_back")):
                        await _mark_seen(
                            db,
                            src_id=src_id,
                            action=action,
                            execution_id=None,
                            meta={"skipped": True, "reason": "disabled_back", "risk_params_json": cfg.risk_params_json},
                        )
                        await _unreserve_seen_key(db, src_key=skey, action=action)
                        continue
                    if cfg.exec_side == ExecSide.LAY and bool(_rp_bool(risk_params, "disable_lay")):
                        await _mark_seen(
                            db,
                            src_id=src_id,
                            action=action,
                            execution_id=None,
                            meta={"skipped": True, "reason": "disabled_lay", "risk_params_json": cfg.risk_params_json},
                        )
                        await _unreserve_seen_key(db, src_key=skey, action=action)
                        continue
                except Exception:
                    pass

                # --------------------------------
                # Guardrail de saldo (LIVE): alerta < 500 (não bloqueia); bloqueia novas apostas quando <= 50
                # --------------------------------
                try:
                    # defaults alinhados ao pedido operacional
                    min_alert = _rp_float(risk_params, "min_balance_alert_usd")
                    if min_alert is None:
                        min_alert = _safe_float_money(os.getenv("BRIDGE_MIN_BALANCE_ALERT_USD", "500"))
                    min_block = _rp_float(risk_params, "min_balance_block_usd")
                    if min_block is None:
                        min_block = _safe_float_money(os.getenv("BRIDGE_MIN_BALANCE_BLOCK_USD", "50"))
                    bal_check_sec = _rp_float(risk_params, "balance_check_sec")
                    if bal_check_sec is None:
                        bal_check_sec = _safe_float_money(os.getenv("BRIDGE_BALANCE_CHECK_SEC", "20"))
                    tg_cooldown = _rp_float(risk_params, "balance_alert_cooldown_sec")
                    if tg_cooldown is None:
                        tg_cooldown = _safe_float_money(os.getenv("BRIDGE_BALANCE_ALERT_COOLDOWN_SEC", "1800"))
                    guard_shadow = _rp_bool(risk_params, "balance_guard_in_shadow")
                    if guard_shadow is None:
                        guard_shadow = os.getenv("BRIDGE_BALANCE_GUARD_IN_SHADOW", "0").strip() in ("1", "true", "True", "yes", "YES")
                    tg_enable = _rp_bool(risk_params, "balance_alert_telegram")
                    if tg_enable is None:
                        tg_enable = os.getenv("BRIDGE_BALANCE_ALERT_TELEGRAM", "1").strip() in ("1", "true", "True", "yes", "YES")

                    guard_active = (str(cfg.mode).lower() == "live") or bool(guard_shadow)
                    if guard_active and (bal_check_sec is not None and float(bal_check_sec) > 0):
                        now = time.time()
                        if (now - float(bal_last_check)) >= float(bal_check_sec):
                            bal_last_check = now
                            bal_payload, bal_err = await _fetch_executor_account(
                                unix_socket=str(cfg.unix_socket or "").strip() or None,
                                http_base=(str(cfg.http_url).strip() if cfg.http_url else None),
                            )
                            bal_http = int(bal_payload.get("_http_status") or 0) if isinstance(bal_payload, dict) else None
                            bal_current = (
                                _extract_balance_current_usd_from_executor_account(bal_payload) if isinstance(bal_payload, dict) else None
                            )
                            if bal_current is None and bal_err:
                                logger.warning(f"[bridge] balance check failed err={bal_err}")
                            elif bal_current is not None:
                                logger.info(f"[bridge] balance_current≈{bal_current:.2f}USD (http={bal_http or 0})")

                        # anexa ao request para auditoria (mesmo em shadow, quando disponível)
                        try:
                            if bal_current is not None:
                                # meta é JSONB: mantém pequeno
                                row.setdefault("_bridge_balance_current", float(bal_current))  # debug local
                        except Exception:
                            pass

                        # bloqueio duro (somente quando temos saldo numérico)
                        if (
                            bal_current is not None
                            and min_block is not None
                            and float(min_block) > 0
                            and float(bal_current) <= float(min_block)
                        ):
                            await _mark_seen(
                                db,
                                src_id=src_id,
                                action=action,
                                execution_id=None,
                                meta={
                                    "skipped": True,
                                    "reason": "low_balance_block",
                                    "balance_current_usd": float(bal_current),
                                    "min_balance_block_usd": float(min_block),
                                    "min_balance_alert_usd": (float(min_alert) if min_alert is not None else None),
                                    "balance_http": int(bal_http or 0),
                                },
                            )
                            await _unreserve_seen_key(db, src_key=skey, action=action)
                            continue

                        # alerta telegram (não bloqueia)
                        if (
                            tg_enable
                            and tg_token
                            and tg_chat
                            and bal_current is not None
                            and min_alert is not None
                            and float(min_alert) > 0
                            and float(bal_current) < float(min_alert)
                            and (tg_cooldown is not None and float(tg_cooldown) >= 0)
                        ):
                            if (now - float(tg_last_alert_ts)) >= float(tg_cooldown):
                                tg_last_alert_ts = now
                                txt = (
                                    f"[betinasia] ALERTA: banca baixa (≈{float(bal_current):.2f} USD) < {float(min_alert):.2f}.\n"
                                    f"Modo={cfg.mode} side={cfg.exec_side.value}. "
                                    f"Bloqueio só em <= {float(min_block or 0):.2f}."
                                )
                                ok = await asyncio.to_thread(_telegram_send, tg_token, tg_chat, txt)
                                logger.info(f"[bridge] telegram low-balance sent={ok}")
                except Exception:
                    pass

                req = _build_request(row, cfg)
                try:
                    # adiciona snapshot simples para auditoria, sem inflar DB
                    if bal_current is not None:
                        req.meta.setdefault("balance", {})
                        req.meta["balance"]["balance_current_usd"] = float(bal_current)
                        req.meta["balance"]["http"] = int(bal_http or 0) if bal_http is not None else None
                        req.meta["balance"]["ts"] = _utcnow().isoformat()
                except Exception:
                    pass

                # -----------------------------
                # Engine dinâmico (WF budget)
                # -----------------------------
                sizing_meta: Dict[str, Any] = {}
                try:
                    use_budget = (
                        bool(cfg.use_wf_budget)
                        and bool(wf.get("match_budget"))
                        and (bankroll_ref is not None and float(bankroll_ref) > 0)
                    )
                    if use_budget:
                        import math

                        # Parâmetros-base vindos do daily/policy (podem ser sobrescritos manualmente por risk_params_json)
                        bud_back_frac = float(wf.get("budget_back_frac") or 0.0)
                        bud_lay_frac = float(wf.get("budget_lay_frac") or 0.0)
                        cap_sig_frac = float(wf.get("budget_cap_signal_frac") or 0.0)
                        risk_mode = str(wf.get("budget_risk_mode") or "fixed").strip() or "fixed"
                        if cfg.wf_risk_mode_override:
                            rm = str(cfg.wf_risk_mode_override or "").strip()
                            if rm in ("fixed", "signals_sqrt", "signals_linear"):
                                risk_mode = rm
                        # overrides manuais (frações/caps)
                        try:
                            v = _rp_float(risk_params, "budget_back_frac")
                            if v is not None:
                                bud_back_frac = float(v)
                            v = _rp_float(risk_params, "budget_lay_frac")
                            if v is not None:
                                bud_lay_frac = float(v)
                            v = _rp_float(risk_params, "cap_signal_frac")
                            if v is not None:
                                cap_sig_frac = float(v)
                        except Exception:
                            pass

                        ev_id = str(row.get("event_id") or "").strip()
                        match_key = ev_id  # proxy robusto: event_id identifica o jogo

                        # se bud_frac do lado for <=0, não aplicar budget (fallback para stake fixo)
                        bud_frac_side = float(bud_back_frac if cfg.exec_side == ExecSide.BACK else bud_lay_frac)
                        if bud_frac_side <= 0:
                            use_budget = False
                            raise RuntimeError("WF_BUDGET_DISABLED_FOR_SIDE (bud_frac_side<=0)")

                        bud_base = float(bankroll_ref) * float(bud_frac_side)
                        n_sig = await _signals_count_estimate(
                            db,
                            event_id=ev_id,
                            hyp=str(cfg.only_hypothesis),
                            exec_side=cfg.exec_side,
                            lookback_h=float(_rp_float(risk_params, "signals_lookback_h") or cfg.signals_lookback_h),
                            prematch_only=bool(cfg.only_prematch),
                        )
                        bud_match = float(bud_base)
                        if risk_mode == "signals_sqrt":
                            bud_match = float(bud_base) / max(1.0, math.sqrt(float(n_sig)))
                        elif risk_mode == "signals_linear":
                            bud_match = float(bud_base) / max(1.0, float(n_sig))

                        cap_signal = float(cap_sig_frac) * float(bud_match) if cap_sig_frac > 0 else float(bud_match)

                        # Caps absolutos por aposta (por regime) — útil para operar em valores fixos (stake médio)
                        cap_abs = None
                        try:
                            is_live = bool(row.get("is_live")) if row.get("is_live") is not None else False
                            if cfg.exec_side == ExecSide.BACK:
                                cap_abs = _rp_float(risk_params, "cap_back_in_abs" if is_live else "cap_back_pre_abs")
                            else:
                                cap_abs = _rp_float(risk_params, "cap_lay_in_abs" if is_live else "cap_lay_pre_abs")
                        except Exception:
                            cap_abs = None
                        if cap_abs is not None:
                            try:
                                cap_abs = float(cap_abs)
                            except Exception:
                                cap_abs = None
                        if cap_abs is not None:
                            if float(cap_abs) <= 0:
                                # interpretamos cap_abs<=0 como bloqueio
                                await _mark_seen(
                                    db,
                                    src_id=src_id,
                                    action=action,
                                    execution_id=None,
                                    meta={
                                        "skipped": True,
                                        "reason": "abs_cap_blocked",
                                        "cap_abs": cap_abs,
                                        "is_live": bool(row.get("is_live")) if row.get("is_live") is not None else False,
                                        "exec_side": cfg.exec_side.value,
                                    },
                                )
                                await _unreserve_seen_key(db, src_key=skey, action=action)
                                continue
                            # cap por sinal (e base) é limitado pelo cap absoluto
                            try:
                                cap_signal = min(float(cap_signal), float(cap_abs))
                            except Exception:
                                pass

                        cap_event = None
                        cap_back = float(_rp_float(risk_params, "cap_event_back_frac") or cfg.cap_event_back_frac)
                        cap_lay = float(_rp_float(risk_params, "cap_event_lay_frac") or cfg.cap_event_lay_frac)
                        if cfg.exec_side == ExecSide.BACK and cap_back > 0:
                            cap_event = float(cap_back) * float(bankroll_ref)
                        if cfg.exec_side == ExecSide.LAY and cap_lay > 0:
                            cap_event = float(cap_lay) * float(bankroll_ref)

                        spent_before = await _spent_for_match(
                            db, match_key=match_key, exec_side=cfg.exec_side, mode=str(cfg.mode), action=action
                        )
                        rem = max(0.0, float(bud_match) - float(spent_before))

                        # base exposure por oportunidade:
                        # 1) preferimos usar `finance.*.available_limit` (do auditor) para o lado correto (Back/Lay)
                        #    e aplicar stake_pct_of_limit manual (risk_params_json).
                        # 2) fallback: betslip_limit (coluna) quando existir.
                        # 3) fallback: finance snapshot (suggested_stake/liability_if_lose) / stake fixo.
                        # 4) fallback final: cap_signal (deixa o budget governar diretamente).
                        details = _parse_details(row)
                        odd_hint = _safe_float(row.get("betslip_odd")) or _safe_float(row.get("websocket_odd"))
                        stake_pct_of_limit = float(_rp_float(risk_params, "stake_pct_of_limit") or 1.0)
                        stake_cap_abs = float(_rp_float(risk_params, "stake_cap_abs") or 0.0)
                        base_exp = None
                        base_why = None
                        # 1) finance.available_limit (lado correto)
                        lim_fin, odd_fin = _limit_from_finance(details, exec_side=cfg.exec_side)
                        base_exp, base_why = _base_exposure_from_limit(
                            exec_side=cfg.exec_side,
                            limit_value=lim_fin,
                            odd=(odd_fin if odd_fin is not None else odd_hint),
                            stake_pct_of_limit=stake_pct_of_limit,
                            stake_cap_abs=stake_cap_abs,
                        )
                        if base_exp is not None and base_why:
                            base_why = f"finance.{base_why}"
                        # 2) betslip_limit (coluna)
                        if base_exp is None or float(base_exp) <= 0:
                            lim_hint = _safe_float(row.get("betslip_limit"))
                            base_exp, base_why2 = _base_exposure_from_limit(
                                exec_side=cfg.exec_side,
                                limit_value=lim_hint,
                                odd=odd_hint,
                                stake_pct_of_limit=stake_pct_of_limit,
                                stake_cap_abs=stake_cap_abs,
                            )
                            if base_exp is not None:
                                base_why = f"betslip_limit.{base_why2}"
                        if base_exp is None or float(base_exp) <= 0:
                            base_exp, base_why = _base_exposure_from_finance(
                                exec_side=cfg.exec_side,
                                details=details,
                                odd_hint=odd_hint,
                                stake_fallback=float(cfg.stake),
                            )
                        # Se estamos em modo budget e não há limit/finance, não queremos ficar travados no stake fixo.
                        # Nesse caso, deixe o budget governar diretamente (cap_signal).
                        if (
                            (base_why == "fallback_stake")
                            and (cfg.exec_side == ExecSide.BACK)
                            and (cap_signal is not None)
                            and float(cap_signal) > 0
                        ):
                            base_exp = float(cap_signal)
                            base_why = "fallback_cap_signal"
                        if base_exp is None or float(base_exp) <= 0:
                            # sem limit e sem finance: deixa o budget governar diretamente
                            base_exp = float(cap_signal)
                            base_why = "fallback_cap_signal"
                        if base_exp is None or float(base_exp) <= 0:
                            await _mark_seen(
                                db,
                                src_id=src_id,
                                action=action,
                                execution_id=None,
                                meta={"skipped": True, "reason": "no_base_exposure", "base_why": base_why, "odd_hint": odd_hint},
                            )
                            await _unreserve_seen_key(db, src_key=skey, action=action)
                            continue

                        # aplica cap absoluto também em base_exp (garante que exp_use não exceda)
                        if cap_abs is not None and float(cap_abs) > 0:
                            try:
                                base_exp = min(float(base_exp), float(cap_abs))
                            except Exception:
                                pass

                        # hard cap total por jogo (cap_event): limita o rem adicionalmente
                        rem_event = None
                        if cap_event is not None:
                            rem_event = max(0.0, float(cap_event) - float(spent_before))
                            rem = min(float(rem), float(rem_event))

                        exp_use = min(float(base_exp), float(rem), float(cap_signal))
                        if exp_use <= 0:
                            await _mark_seen(
                                db,
                                src_id=src_id,
                                action=action,
                                execution_id=None,
                                meta={
                                    "skipped": True,
                                    "reason": "budget_blocked",
                                    "bankroll_ref": bankroll_ref,
                                    "budget_base": bud_base,
                                    "budget_match": bud_match,
                                    "cap_signal": cap_signal,
                                    "cap_event": cap_event,
                                    "spent_before": spent_before,
                                    "rem": rem,
                                    "rem_event": rem_event,
                                    "n_signals_est": n_sig,
                                    "risk_mode": risk_mode,
                                },
                            )
                            continue

                        ratio = float(exp_use) / max(1e-9, float(base_exp))
                        if cfg.exec_side == ExecSide.BACK:
                            req.policy.stake_requested = float(exp_use)
                            req.policy.liability_requested = None
                        else:
                            req.policy.stake_requested = None
                            req.policy.liability_requested = float(exp_use)

                        req.policy.bankroll_ref = float(bankroll_ref)
                        req.policy.bud_back_frac = float(bud_back_frac)
                        req.policy.bud_lay_frac = float(bud_lay_frac)
                        req.policy.cap_signal_frac = float(cap_sig_frac)
                        req.policy.risk_mode = str(risk_mode)
                        req.policy.spent_before = float(spent_before)
                        req.policy.spent_after = float(spent_before) + float(exp_use)
                        req.policy.policy_version = f"bridge_{cfg.only_hypothesis.lower()}_{cfg.mode}_wfBudget_v1"

                        sizing_meta = {
                            "use_wf_budget": True,
                            "bankroll_ref": bankroll_ref,
                            "budget_base": bud_base,
                            "budget_match": bud_match,
                            "cap_signal": cap_signal,
                            "cap_abs": cap_abs,
                            "cap_event": cap_event,
                            "spent_before": spent_before,
                            "spent_after": float(spent_before) + float(exp_use),
                            "n_signals_est": n_sig,
                            "risk_mode": risk_mode,
                            "base_exp": base_exp,
                            "base_why": base_why,
                            "ratio": ratio,
                            "risk_params_json": (cfg.risk_params_json if cfg.risk_params_json else None),
                        }
                        try:
                            req.meta.setdefault("bridge", {})
                            req.meta["bridge"]["sizing"] = dict(sizing_meta)
                        except Exception:
                            pass
                except Exception as e:
                    # não derruba o bridge: fallback para stake fixo
                    logger.warning(f"[bridge] wf_budget engine failed src_id={src_id}: {e}")

                # Gate de slippage (enforced no executor antes do LIVE): Lay + in-match, delta_pct > limiar
                try:
                    if cfg.exec_side == ExecSide.LAY:
                        is_live = bool(row.get("is_live")) if row.get("is_live") is not None else False
                        thr = _rp_float(risk_params, "slippage_gate_lay_in_delta_pct_gt")
                        if is_live and thr is not None and float(thr) > 0:
                            req.meta.setdefault("slippage_gate", {})
                            req.meta["slippage_gate"]["lay_in_max_delta_pct"] = float(thr)
                            # debug: aponta que o gate está ativo para este request
                            req.meta["slippage_gate"]["enabled"] = True
                except Exception:
                    pass

                res = await submit_execution(req=req, unix_socket=cfg.unix_socket, http_base=cfg.http_url)
                eid = str(res.get("execution_id") or "")
                accepted = bool(res.get("accepted"))
                hs = res.get("_http_status")
                err = (res.get("error") or res.get("detail") or res.get("status") or "")
                err_s = str(err).replace("\n", " ").replace("\r", " ").strip()
                if len(err_s) > 160:
                    err_s = err_s[:160].rstrip() + "…"
                logger.info(
                    f"[bridge] submit src_id={src_id} accepted={accepted} execution_id={eid} http={hs} "
                    + (f"err={err_s}" if (not accepted and err_s) else "")
                )

                # Em falhas transitórias, não “consumir” a oportunidade: desfaz a reserva e tenta novamente em ciclos futuros.
                # Isso evita perder horas de operação quando o executor está not_ready ou o socket está instável.
                retry_transient = (os.getenv("BRIDGE_RETRY_TRANSIENT", "1").strip() not in ("0", "false", "False", "no", "NO"))
                transient_http = int(hs) in (503, 429) if hs is not None else False
                transient_err = ("not_ready" in err_s.lower()) or ("queue_full" in err_s.lower())
                if (not accepted) and retry_transient and (transient_http or transient_err):
                    await _unreserve_seen_key(db, src_key=skey, action=action)
                    # pequeno backoff para não martelar o executor
                    await asyncio.sleep(float(os.getenv("BRIDGE_TRANSIENT_BACKOFF_SEC", "1.0")))
                    continue

                if accepted:
                    try:
                        await _insert_position(
                            db,
                            action=action,
                            mode=str(cfg.mode),
                            exec_side=cfg.exec_side,
                            event_id=str(row.get("event_id") or ""),
                            match_key=str(row.get("event_id") or ""),
                            src_id=src_id,
                            src_key=skey,
                            execution_id=(eid or None),
                            stake_requested=_safe_float(getattr(req.policy, "stake_requested", None)),
                            liability_requested=_safe_float(getattr(req.policy, "liability_requested", None)),
                            ctx={"status": "SUBMITTED", **(sizing_meta or {}), "http_submit": hs},
                        )
                    except Exception as e:
                        logger.warning(f"[bridge] insert_position failed src_id={src_id}: {e}")

                await _finalize_seen_key(db, src_key=skey, action=action, execution_id=(eid or None))
                await _mark_seen(db, src_id=src_id, action=action, execution_id=(eid or None), meta={"accepted": accepted, "resp": res})
            except Exception as e:
                msg = str(e)
                logger.exception(f"[bridge] failed src_id={src_id}: {e}")
                # Se falha de transporte/socket, libera a reserva para retry e NÃO marca seen (senão perde oportunidade).
                retry_transient = (os.getenv("BRIDGE_RETRY_TRANSIENT", "1").strip() not in ("0", "false", "False", "no", "NO"))
                is_transient = ("Connection refused" in msg) or ("Connection reset" in msg) or ("Cannot connect to unix socket" in msg)
                if retry_transient and is_transient:
                    try:
                        await _unreserve_seen_key(db, src_key=skey, action=action)
                    except Exception:
                        pass
                    await asyncio.sleep(float(os.getenv("BRIDGE_TRANSIENT_BACKOFF_SEC", "1.0")))
                    continue
                await _mark_seen(db, src_id=src_id, action=action, execution_id=None, meta={"error": str(e)[:500]})

        dt = time.time() - t0
        # evita loop muito agressivo
        await asyncio.sleep(max(0.1, float(cfg.poll_sec) - dt))


def main() -> int:
    ap = argparse.ArgumentParser(description="Bridge: DB audit -> Executor (/execute).")
    ap.add_argument("--mode", default=os.getenv("BRIDGE_MODE", "shadow"), choices=["shadow", "live"])
    ap.add_argument("--exec-side", default=os.getenv("BRIDGE_EXEC_SIDE", "Back"), choices=["Back", "Lay"])
    ap.add_argument("--stake", type=float, default=float(os.getenv("BRIDGE_STAKE", "3.0")))
    ap.add_argument("--poll-sec", type=float, default=float(os.getenv("BRIDGE_POLL_SEC", "2.0")))
    ap.add_argument("--lookback-sec", type=int, default=int(os.getenv("BRIDGE_LOOKBACK_SEC", "120")))
    ap.add_argument("--max-per-cycle", type=int, default=int(os.getenv("BRIDGE_MAX_PER_CYCLE", "3")))
    ap.add_argument("--unix-socket", default=os.getenv("EXECUTOR_UNIX_SOCKET", "/tmp/betinasia-exec.sock"))
    ap.add_argument("--http-url", default=os.getenv("EXECUTOR_HTTP_URL", "").strip() or None)
    ap.add_argument("--hypothesis", default=os.getenv("BRIDGE_HYPOTHESIS", "H3B"))
    ap.add_argument("--prematch-only", action="store_true", default=(os.getenv("BRIDGE_PREMATCH_ONLY", "1").strip() not in ("0", "false", "False", "no", "NO")))
    ap.add_argument("--policy-json", default=os.getenv("BRIDGE_POLICY_JSON", "").strip() or None, help="Path para WF policy exportado (JSON).")
    ap.add_argument("--policy-reload-sec", type=float, default=float(os.getenv("BRIDGE_POLICY_RELOAD_SEC", "5.0")))
    ap.add_argument(
        "--policy-use-base",
        action="store_true",
        default=(os.getenv("BRIDGE_POLICY_USE_BASE", "0").strip() in ("1", "true", "True", "yes", "YES")),
        help="Se true, usa active_keys_base (ignora sufixo de liga).",
    )
    ap.add_argument("--min-limit", type=float, default=float(os.getenv("BRIDGE_MIN_LIMIT", "0.0")), help="Se >0, exige betslip_limit >= este mínimo.")
    ap.add_argument(
        "--enforce-wf-filters",
        action="store_true",
        default=(os.getenv("BRIDGE_ENFORCE_WF_FILTERS", "1").strip() not in ("0", "false", "False", "no", "NO")),
        help="Se true (default), aplica filtros/limiares do WF (policy_json.wf) no bridge para alinhar com OOS.",
    )
    ap.add_argument(
        "--use-wf-budget",
        action="store_true",
        default=(os.getenv("BRIDGE_USE_WF_BUDGET", "0").strip() in ("1", "true", "True", "yes", "YES")),
        help="Se true, aplica match_budget/budgets/risk_mode do wf_policy_current.json para calcular stake/liability dinamicamente.",
    )
    ap.add_argument(
        "--bankroll-ref",
        type=float,
        default=(_safe_float(os.getenv("BRIDGE_BANKROLL_REF", "")) if os.getenv("BRIDGE_BANKROLL_REF") else None),
        help="Banca de referência (override). Se omitido, pode vir de --bankroll-json.",
    )
    ap.add_argument(
        "--bankroll-json",
        default=os.getenv("BRIDGE_BANKROLL_JSON", "").strip() or None,
        help="JSON com balance_current (ex.: logs/accounting_daily_report.json).",
    )
    ap.add_argument("--bankroll-reload-sec", type=float, default=float(os.getenv("BRIDGE_BANKROLL_RELOAD_SEC", "30.0")))
    ap.add_argument("--signals-lookback-h", type=float, default=float(os.getenv("BRIDGE_SIGNALS_LOOKBACK_H", "36.0")))
    ap.add_argument(
        "--wf-risk-mode-override",
        default=os.getenv("BRIDGE_WF_RISK_MODE_OVERRIDE", "").strip() or None,
        help="Override manual do wf.budget_risk_mode: fixed|signals_sqrt|signals_linear",
    )
    ap.add_argument(
        "--risk-params-json",
        default=os.getenv("BRIDGE_RISK_PARAMS_JSON", "").strip() or None,
        help="JSON com overrides manuais de risco/caps (budget_*_frac, cap_signal_frac, cap_event_*_frac).",
    )
    ap.add_argument("--risk-params-reload-sec", type=float, default=float(os.getenv("BRIDGE_RISK_PARAMS_RELOAD_SEC", "5.0")))
    ap.add_argument("--cap-event-back-frac", type=float, default=float(os.getenv("BRIDGE_CAP_EVENT_BACK_FRAC", "0.0")))
    ap.add_argument("--cap-event-lay-frac", type=float, default=float(os.getenv("BRIDGE_CAP_EVENT_LAY_FRAC", "0.0")))
    args = ap.parse_args()

    cfg = BridgeConfig(
        poll_sec=float(args.poll_sec),
        lookback_sec=int(args.lookback_sec),
        max_per_cycle=int(args.max_per_cycle),
        mode=str(args.mode),
        exec_side=ExecSide(str(args.exec_side)),
        stake=float(args.stake),
        unix_socket=str(args.unix_socket),
        http_url=(str(args.http_url) if args.http_url else None),
        only_hypothesis=str(args.hypothesis),
        only_prematch=bool(args.prematch_only),
        policy_json=(str(args.policy_json) if args.policy_json else None),
        policy_reload_sec=float(args.policy_reload_sec),
        policy_use_base=bool(args.policy_use_base),
        min_limit=float(args.min_limit),
        enforce_wf_filters=bool(args.enforce_wf_filters),
        use_wf_budget=bool(args.use_wf_budget),
        bankroll_ref=(float(args.bankroll_ref) if args.bankroll_ref is not None else None),
        bankroll_json=(str(args.bankroll_json) if args.bankroll_json else None),
        bankroll_reload_sec=float(args.bankroll_reload_sec),
        signals_lookback_h=float(args.signals_lookback_h),
        wf_risk_mode_override=(str(args.wf_risk_mode_override) if args.wf_risk_mode_override else None),
        risk_params_json=(str(args.risk_params_json) if args.risk_params_json else None),
        risk_params_reload_sec=float(args.risk_params_reload_sec),
        cap_event_back_frac=float(args.cap_event_back_frac),
        cap_event_lay_frac=float(args.cap_event_lay_frac),
    )

    logger.remove()
    # stdout vai para logs/executor_bridge.log (systemd StandardOutput)
    logger.add(sys.stdout, level=os.getenv("LOG_LEVEL", "INFO"))
    # erros em stderr vão para logs/executor_bridge_error.log (systemd StandardError)
    logger.add(sys.stderr, level="ERROR")
    try:
        asyncio.run(run_bridge(cfg))
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

