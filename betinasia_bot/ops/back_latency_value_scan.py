from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


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
        return int(float(x))
    except Exception:
        return None


def _parse_iso(s: Any) -> Optional[datetime]:
    try:
        txt = str(s or "").strip()
        if not txt:
            return None
        if txt.endswith("Z"):
            txt = txt[:-1] + "+00:00"
        return datetime.fromisoformat(txt)
    except Exception:
        return None


def _pick_col(cols: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    idx = {str(c).strip().lower(): c for c in cols}
    for c in candidates:
        v = idx.get(str(c).strip().lower())
        if v:
            return v
    return None


def _latest_balance_csv(accounting_dir: Path) -> Optional[Path]:
    cands = sorted(glob.glob(str(accounting_dir / "*__balance.csv")))
    if not cands:
        return None
    return Path(cands[-1])


def _load_pnl_by_order_id(balance_csv: Path) -> Dict[str, float]:
    out: Dict[str, float] = {}
    with balance_csv.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        r = csv.DictReader(f)
        cols = list(r.fieldnames or [])
        oid_col = _pick_col(cols, ("order_id", "order id", "order", "bet id", "bet_id", "id"))
        pnl_col = _pick_col(cols, ("amount", "profit_loss", "profit", "p&l", "pnl", "net", "pl"))
        typ_col = _pick_col(cols, ("type",))
        if not oid_col or not pnl_col:
            return out

        def _is_excluded_type(t: str) -> bool:
            tl = str(t or "").strip().lower()
            return any(k in tl for k in ("deposit", "withdraw", "transfer", "top", "payment", "adjust", "bonus"))

        for row in r:
            oid = str(row.get(oid_col) or "").strip()
            if not oid or not oid.isdigit():
                continue
            pnl = _safe_float(row.get(pnl_col))
            if pnl is None:
                continue
            if typ_col and _is_excluded_type(str(row.get(typ_col) or "")):
                continue
            out[oid] = float(out.get(oid) or 0.0) + float(pnl)
    return out


def _extract_order_id(raw: Dict[str, Any]) -> Optional[str]:
    for k in ("order_id", "orderId", "bet_id", "betId", "id"):
        v = raw.get(k)
        if v is None:
            continue
        s = str(v).strip()
        if s and s.isdigit():
            return s
    return None


@dataclass
class ExecRow:
    created_at: str
    regime: str
    pre_submit_ms: Optional[int]
    slippage_pre_pct: Optional[float]
    stake_actual: float
    pnl_actual: float
    roi_order_pct: float
    stake_sim: float
    pnl_sim: float


def _stake_from_slippage(
    *,
    slippage_pre_pct: Optional[float],
    slip_neg_limit_pct: float,
    slip_pos_limit_pct: float,
    stake_neg: float,
    stake_mid: float,
    stake_pos: float,
) -> float:
    if slippage_pre_pct is None:
        return float(stake_mid)
    sp = float(slippage_pre_pct)
    if sp < float(slip_neg_limit_pct):
        return float(stake_neg)
    if sp <= float(slip_pos_limit_pct):
        return float(stake_mid)
    return float(stake_pos)


def _load_rows(
    *,
    executor_jsonl: Path,
    pnl_by_order_id: Dict[str, float],
    start_day: str,
    regime: str,
    slip_neg_limit_pct: float,
    slip_pos_limit_pct: float,
    stake_neg: float,
    stake_mid: float,
    stake_pos: float,
) -> List[ExecRow]:
    out: List[ExecRow] = []
    if not executor_jsonl.exists():
        return out
    with executor_jsonl.open("r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            req = obj.get("request") if isinstance(obj.get("request"), dict) else {}
            res = obj.get("result") if isinstance(obj.get("result"), dict) else {}
            if str(res.get("status") or "") != "LIVE_OK":
                continue
            side = str(res.get("exec_side") or req.get("exec_side") or "").strip().lower()
            if side != "back":
                continue
            dt = _parse_iso(res.get("created_at") or req.get("created_at"))
            if not dt:
                continue
            day = dt.date().isoformat()
            if start_day and day < start_day:
                continue

            raw = res.get("raw") if isinstance(res.get("raw"), dict) else {}
            sent = raw.get("sent") if isinstance(raw.get("sent"), dict) else {}
            vs = raw.get("value_sizing") if isinstance(raw.get("value_sizing"), dict) else {}
            meta = req.get("meta") if isinstance(req.get("meta"), dict) else {}
            market = meta.get("market") if isinstance(meta.get("market"), dict) else {}

            rg = str(vs.get("market_regime") or market.get("regime") or "").strip().lower()
            if rg not in ("pre", "in"):
                rg = "in" if bool(market.get("is_live")) else "pre"
            if regime in ("pre", "in") and rg != regime:
                continue

            oid = _extract_order_id(raw)
            if not oid:
                continue
            pnl = _safe_float(pnl_by_order_id.get(oid))
            if pnl is None:
                continue

            stake_actual = _safe_float(sent.get("stake"))
            if stake_actual is None or float(stake_actual) <= 0:
                continue
            slip = _safe_float(vs.get("slippage_pre_pct"))
            pre_ms = _safe_int(vs.get("pre_submit_ms"))
            roi_order_pct = (float(pnl) / float(stake_actual)) * 100.0

            stake_sim = _stake_from_slippage(
                slippage_pre_pct=slip,
                slip_neg_limit_pct=float(slip_neg_limit_pct),
                slip_pos_limit_pct=float(slip_pos_limit_pct),
                stake_neg=float(stake_neg),
                stake_mid=float(stake_mid),
                stake_pos=float(stake_pos),
            )
            pnl_sim = (roi_order_pct / 100.0) * float(stake_sim)

            out.append(
                ExecRow(
                    created_at=dt.isoformat(),
                    regime=rg,
                    pre_submit_ms=pre_ms,
                    slippage_pre_pct=(float(slip) if slip is not None else None),
                    stake_actual=float(stake_actual),
                    pnl_actual=float(pnl),
                    roi_order_pct=float(roi_order_pct),
                    stake_sim=float(stake_sim),
                    pnl_sim=float(pnl_sim),
                )
            )
    return out


def _aggregate(rows: Iterable[ExecRow], simulated: bool) -> Dict[str, Optional[float]]:
    xs = list(rows)
    n = len(xs)
    if simulated:
        stake = sum(float(r.stake_sim) for r in xs)
        pnl = sum(float(r.pnl_sim) for r in xs)
    else:
        stake = sum(float(r.stake_actual) for r in xs)
        pnl = sum(float(r.pnl_actual) for r in xs)
    roiw = (100.0 * pnl / stake) if stake > 0 else None
    return {
        "n": float(n),
        "stake": float(stake),
        "pnl": float(pnl),
        "roiw": (float(roiw) if roiw is not None else None),
    }


def _parse_thresholds(raw: str) -> List[float]:
    out: List[float] = []
    for tok in str(raw or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        v = _safe_float(tok)
        if v is None or v <= 0:
            continue
        out.append(float(v))
    return sorted(set(out))


def main() -> int:
    ap = argparse.ArgumentParser(description="Scan estatístico de latência para BACK (executor_jsonl + accounting).")
    ap.add_argument("--executor-jsonl", default=os.getenv("EXECUTOR_JSONL", "logs/executor_live.jsonl"))
    ap.add_argument("--accounting-out-dir", default=os.getenv("ACCOUNTING_OUT_DIR", "logs/accounting"))
    ap.add_argument("--start-day", default="", help="Filtro YYYY-MM-DD (inclusive).")
    ap.add_argument("--regime", default="pre", choices=["pre", "in", "all"])
    ap.add_argument("--latency-thresholds-sec", default="2,3,4,5,6,7,8,10,12")
    ap.add_argument("--min-n", type=int, default=25, help="Mínimo de ordens para considerar threshold.")
    ap.add_argument("--min-tranche-n", type=int, default=10, help="Mínimo de ordens na faixa incremental.")
    ap.add_argument("--slip-neg-limit-pct", type=float, default=-2.0)
    ap.add_argument("--slip-pos-limit-pct", type=float, default=2.0)
    ap.add_argument("--stake-neg", type=float, default=40.0)
    ap.add_argument("--stake-mid", type=float, default=20.0)
    ap.add_argument("--stake-pos", type=float, default=20.0)
    ap.add_argument("--json", action="store_true", help="Saída JSON (além da tabela).")
    args = ap.parse_args()

    exec_path = Path(str(args.executor_jsonl)).expanduser()
    acct_dir = Path(str(args.accounting_out_dir)).expanduser()
    balance_csv = _latest_balance_csv(acct_dir)
    if balance_csv is None:
        print(f"ERRO: balance.csv não encontrado em {acct_dir}")
        return 2
    pnl_by_order = _load_pnl_by_order_id(balance_csv)

    regime = str(args.regime).strip().lower()
    if regime == "all":
        regime = ""
    rows = _load_rows(
        executor_jsonl=exec_path,
        pnl_by_order_id=pnl_by_order,
        start_day=str(args.start_day).strip(),
        regime=regime,
        slip_neg_limit_pct=float(args.slip_neg_limit_pct),
        slip_pos_limit_pct=float(args.slip_pos_limit_pct),
        stake_neg=float(args.stake_neg),
        stake_mid=float(args.stake_mid),
        stake_pos=float(args.stake_pos),
    )
    if not rows:
        print("ERRO: sem linhas elegíveis (LIVE_OK BACK com order_id e P&L no ledger).")
        return 3

    thresholds = _parse_thresholds(str(args.latency_thresholds_sec))
    if not thresholds:
        print("ERRO: --latency-thresholds-sec inválido.")
        return 4

    rows_with_latency = [r for r in rows if r.pre_submit_ms is not None]
    base_actual = _aggregate(rows_with_latency, simulated=False)
    base_sim = _aggregate(rows_with_latency, simulated=True)

    table: List[Dict[str, Any]] = []
    prev_set: List[ExecRow] = []
    degrade_sec: Optional[float] = None

    for thr_sec in thresholds:
        thr_ms = int(round(float(thr_sec) * 1000.0))
        cur = [r for r in rows_with_latency if int(r.pre_submit_ms or 0) <= thr_ms]
        cur_actual = _aggregate(cur, simulated=False)
        cur_sim = _aggregate(cur, simulated=True)
        prev_ids = {id(x) for x in prev_set}
        tranche = [x for x in cur if id(x) not in prev_ids]
        tranche_actual = _aggregate(tranche, simulated=False)
        tranche_sim = _aggregate(tranche, simulated=True)

        row = {
            "threshold_sec": float(thr_sec),
            "n": int(cur_actual["n"] or 0),
            "actual_roiw_pct": cur_actual["roiw"],
            "sim_roiw_pct": cur_sim["roiw"],
            "actual_pnl": cur_actual["pnl"],
            "sim_pnl": cur_sim["pnl"],
            "actual_stake": cur_actual["stake"],
            "sim_stake": cur_sim["stake"],
            "tranche_n": int(tranche_actual["n"] or 0),
            "tranche_actual_roiw_pct": tranche_actual["roiw"],
            "tranche_sim_roiw_pct": tranche_sim["roiw"],
        }
        table.append(row)
        prev_set = cur

        if degrade_sec is None:
            tr_n = int(tranche_actual["n"] or 0)
            tr_roiw = tranche_actual["roiw"]
            if tr_n >= int(args.min_tranche_n) and tr_roiw is not None and float(tr_roiw) < 0:
                degrade_sec = float(thr_sec)

    candidates = [r for r in table if int(r["n"]) >= int(args.min_n) and r.get("actual_roiw_pct") is not None]
    best_by_roiw = max(candidates, key=lambda r: float(r["actual_roiw_pct"])) if candidates else None
    best_by_pnl = max(candidates, key=lambda r: float(r["actual_pnl"])) if candidates else None

    print("")
    print("Latência <= Xs | BACK LIVE_OK | regime={} | start_day={}".format((regime or "all"), str(args.start_day or "none")))
    print("Rows com pre_submit_ms: {} | balance_csv={}".format(len(rows_with_latency), str(balance_csv)))
    print("Base actual: n={} stake={:.2f} pnl={:.2f} roiw={:.3f}%".format(int(base_actual["n"] or 0), float(base_actual["stake"] or 0.0), float(base_actual["pnl"] or 0.0), float(base_actual["roiw"] or 0.0)))
    print("Base sim:    n={} stake={:.2f} pnl={:.2f} roiw={:.3f}%".format(int(base_sim["n"] or 0), float(base_sim["stake"] or 0.0), float(base_sim["pnl"] or 0.0), float(base_sim["roiw"] or 0.0)))
    print("")
    print("thr_s | n | ROIw_act% | ROIw_sim% | PnL_act | PnL_sim | tranche_n | tranche_ROIw_act%")
    for r in table:
        print(
            "{:>5.1f} | {:>4d} | {:>9} | {:>9} | {:>8.2f} | {:>8.2f} | {:>9d} | {:>16}".format(
                float(r["threshold_sec"]),
                int(r["n"]),
                ("{:.3f}".format(float(r["actual_roiw_pct"])) if r["actual_roiw_pct"] is not None else "NA"),
                ("{:.3f}".format(float(r["sim_roiw_pct"])) if r["sim_roiw_pct"] is not None else "NA"),
                float(r["actual_pnl"] or 0.0),
                float(r["sim_pnl"] or 0.0),
                int(r["tranche_n"] or 0),
                ("{:.3f}".format(float(r["tranche_actual_roiw_pct"])) if r["tranche_actual_roiw_pct"] is not None else "NA"),
            )
        )
    print("")
    print("degrade_from_sec={}".format(("NA" if degrade_sec is None else "{:.1f}".format(float(degrade_sec)))))
    if best_by_roiw is not None:
        print("best_roiw_threshold_sec={:.1f} (n={} ROIw_act={:.3f}%)".format(float(best_by_roiw["threshold_sec"]), int(best_by_roiw["n"]), float(best_by_roiw["actual_roiw_pct"])))
    if best_by_pnl is not None:
        print("best_pnl_threshold_sec={:.1f} (n={} pnl_act={:.2f})".format(float(best_by_pnl["threshold_sec"]), int(best_by_pnl["n"]), float(best_by_pnl["actual_pnl"])))

    if args.json:
        out = {
            "executor_jsonl": str(exec_path),
            "balance_csv": str(balance_csv),
            "start_day": str(args.start_day or ""),
            "regime": (regime or "all"),
            "rows_with_latency_n": len(rows_with_latency),
            "rule": {
                "slip_neg_limit_pct": float(args.slip_neg_limit_pct),
                "slip_pos_limit_pct": float(args.slip_pos_limit_pct),
                "stake_neg": float(args.stake_neg),
                "stake_mid": float(args.stake_mid),
                "stake_pos": float(args.stake_pos),
            },
            "base_actual": base_actual,
            "base_sim": base_sim,
            "thresholds": table,
            "degrade_from_sec": degrade_sec,
            "best_roiw_threshold_sec": (best_by_roiw["threshold_sec"] if best_by_roiw is not None else None),
            "best_pnl_threshold_sec": (best_by_pnl["threshold_sec"] if best_by_pnl is not None else None),
        }
        print("")
        print(json.dumps(out, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
