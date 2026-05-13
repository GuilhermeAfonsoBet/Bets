from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


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


def _fmt_pct(v: Optional[float], nd: int = 2) -> str:
    if v is None:
        return "NA"
    return f"{float(v):.{nd}f}%"


def _fmt_num(v: Optional[float], nd: int = 2) -> str:
    if v is None:
        return "NA"
    return f"{float(v):.{nd}f}"


def _pct(part: float, total: float) -> Optional[float]:
    if float(total) <= 0:
        return None
    return 100.0 * float(part) / float(total)


def _pick_col(cols: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    idx = {str(c).strip().lower(): c for c in cols}
    for k in candidates:
        v = idx.get(str(k).strip().lower())
        if v:
            return v
    return None


def _run_psql_copy_audit_map(*, database_url: str, days: int, out_csv: Path) -> None:
    q = (
        "\\copy ("
        " SELECT id, COALESCE(NULLIF(TRIM(league),''),'—') AS league"
        " FROM betslip_audit_results"
        f" WHERE audited_at >= now() - interval '{int(days)} days'"
        f") TO '{str(out_csv)}' CSV HEADER"
    )
    env = dict(os.environ)
    env["DATABASE_URL"] = str(database_url)
    cmd = ["psql", str(database_url), "-X", "-v", "ON_ERROR_STOP=1", "-c", q]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"psql copy falhou rc={proc.returncode}: {proc.stderr.strip()[:300]}")


def _load_audit_league_map(audit_map_csv: Path) -> Dict[int, str]:
    out: Dict[int, str] = {}
    with audit_map_csv.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            aid = _safe_int(row.get("id"))
            if aid is None:
                continue
            lg = str(row.get("league") or "—").strip() or "—"
            out[int(aid)] = lg
    return out


def _latest_balance_csv(accounting_dir: Path) -> Optional[Path]:
    cands = sorted(glob.glob(str(accounting_dir / "*__balance.csv")))
    if not cands:
        return None
    return Path(cands[-1])


def _load_pnl_like_by_order_id(balance_csv: Path) -> Dict[str, float]:
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
            if typ_col:
                typ = str(row.get(typ_col) or "").strip().lower()
                if typ and _is_excluded_type(typ):
                    continue
            out[oid] = float(out.get(oid) or 0.0) + float(pnl)
    return out


def _extract_order_id(raw: Dict[str, Any]) -> Optional[str]:
    for k in ("order_id", "orderId", "bet_id", "betId", "id"):
        v = raw.get(k)
        if v is None:
            continue
        s = str(v).strip()
        if s.isdigit():
            return s
    return None


@dataclass
class ExecRow:
    day: str
    league: str
    slip_pre: Optional[float]
    stake: float
    pnl: float


def _load_exec_rows(
    *,
    executor_jsonl: Path,
    league_by_audit_id: Dict[int, str],
    pnl_by_order_id: Dict[str, float],
    regime: str,
) -> List[ExecRow]:
    out: List[ExecRow] = []
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

            raw = res.get("raw") if isinstance(res.get("raw"), dict) else {}
            sent = raw.get("sent") if isinstance(raw.get("sent"), dict) else {}
            vs = raw.get("value_sizing") if isinstance(raw.get("value_sizing"), dict) else {}
            meta = req.get("meta") if isinstance(req.get("meta"), dict) else {}
            market = meta.get("market") if isinstance(meta.get("market"), dict) else {}

            rg = str(market.get("regime") or "").strip().lower()
            if rg not in ("pre", "in"):
                is_live = market.get("is_live")
                rg = "in" if is_live is True else "pre"
            if str(regime).strip().lower() in ("pre", "in") and rg != str(regime).strip().lower():
                continue

            dt = _parse_iso(res.get("created_at") or req.get("created_at"))
            if not dt:
                continue

            stake = _safe_float(sent.get("stake"))
            if stake is None or float(stake) <= 0:
                continue

            oid = _extract_order_id(raw)
            if not oid:
                continue
            if str(oid) not in pnl_by_order_id:
                continue

            aid = _safe_int(req.get("audit_id") or res.get("audit_id"))
            league = league_by_audit_id.get(int(aid), "—") if aid is not None else "—"
            slip = _safe_float(vs.get("slippage_pre_pct"))

            out.append(
                ExecRow(
                    day=dt.date().isoformat(),
                    league=str(league or "—"),
                    slip_pre=(float(slip) if slip is not None else None),
                    stake=float(stake),
                    pnl=float(pnl_by_order_id.get(str(oid)) or 0.0),
                )
            )
    return out


def _aggregate(rows: Sequence[ExecRow]) -> Dict[str, Optional[float]]:
    n = len(rows)
    stake = sum(float(r.stake) for r in rows)
    pnl = sum(float(r.pnl) for r in rows)
    roiw = ((100.0 * pnl / stake) if stake > 0 else None)
    return {"n": float(n), "stake": float(stake), "pnl": float(pnl), "roiw": (float(roiw) if roiw is not None else None)}


def _metrics_by_league(rows: Sequence[ExecRow]) -> Dict[str, Dict[str, Optional[float]]]:
    by: Dict[str, List[ExecRow]] = {}
    for r in rows:
        by.setdefault(str(r.league or "—"), []).append(r)
    out: Dict[str, Dict[str, Optional[float]]] = {}
    for lg, sub in by.items():
        out[lg] = _aggregate(sub)
    return out


def run_ab(
    *,
    rows_all: Sequence[ExecRow],
    train_days: int,
    slip_max: float,
    min_n_current: int,
    min_n_keep: int,
    keep_roi_min: float,
    bad_roi_max: float,
    bad_min_n: int,
    bad_streak_cycles: int,
    min_n_universe: int,
) -> Dict[str, Any]:
    days = sorted({str(r.day) for r in rows_all})
    if len(days) <= int(train_days):
        raise RuntimeError(f"dias insuficientes: {len(days)} <= train_days={train_days}")

    base_rows: List[ExecRow] = []
    a_rows: List[ExecRow] = []
    b_rows: List[ExecRow] = []
    c_rows: List[ExecRow] = []

    active_a_ns: List[int] = []
    active_b_ns: List[int] = []
    active_c_ns: List[int] = []
    active_d_ns: List[int] = []
    bad_streak: Dict[str, int] = {}
    blocked_count: Dict[str, int] = {}
    active_b_prev: Set[str] = set()
    active_c_prev: Set[str] = set()

    for idx in range(int(train_days), len(days)):
        train_set = set(days[idx - int(train_days):idx])
        test_day = days[idx]

        tr_all = [r for r in rows_all if str(r.day) in train_set]
        tr_sl = [r for r in tr_all if (r.slip_pre is not None and float(r.slip_pre) <= float(slip_max))]
        met_all = _metrics_by_league(tr_all)
        met_sl = _metrics_by_league(tr_sl)

        active_a: Set[str] = set()
        for lg, m in met_all.items():
            n = int(m.get("n") or 0)
            roiw = m.get("roiw")
            if n >= int(min_n_current) and roiw is not None and float(roiw) > 0.0:
                active_a.add(str(lg))

        # Cenário 2: estabilidade (A + histerese), sem bloqueio de ruins.
        active_b: Set[str] = set(active_a)
        for lg in list(active_b_prev):
            m = met_all.get(str(lg)) or {}
            n = int(m.get("n") or 0)
            roiw = m.get("roiw")
            if n >= int(min_n_keep) and roiw is not None and float(roiw) >= float(keep_roi_min):
                active_b.add(str(lg))
        active_b_prev = set(active_b)

        blocked: Set[str] = set()
        for lg, m in met_all.items():
            n = int(m.get("n") or 0)
            roiw = m.get("roiw")
            is_bad = bool(n >= int(bad_min_n) and roiw is not None and float(roiw) <= float(bad_roi_max))
            if is_bad:
                bad_streak[str(lg)] = int(bad_streak.get(str(lg), 0)) + 1
            else:
                bad_streak[str(lg)] = 0
            if int(bad_streak.get(str(lg), 0)) >= int(max(1, bad_streak_cycles)):
                blocked.add(str(lg))
                blocked_count[str(lg)] = int(blocked_count.get(str(lg), 0)) + 1

        # Cenário 3: excluir ruins (não exige ROI>0 para entrar).
        universe_c = {
            str(lg) for lg, m in met_all.items()
            if int(m.get("n") or 0) >= int(min_n_universe)
        }
        active_c: Set[str] = set(x for x in universe_c if x not in blocked)
        for lg in list(active_c_prev):
            if lg not in blocked:
                active_c.add(str(lg))
        active_c_prev = set(active_c)

        base_day = [r for r in rows_all if str(r.day) == str(test_day) and (r.slip_pre is not None and float(r.slip_pre) <= float(slip_max))]
        a_day = [r for r in base_day if str(r.league) in active_a]
        b_day = [r for r in base_day if str(r.league) in active_b]
        c_day = [r for r in base_day if str(r.league) in active_c]

        base_rows.extend(base_day)
        a_rows.extend(a_day)
        b_rows.extend(b_day)
        c_rows.extend(c_day)
        active_a_ns.append(len(active_a))
        active_b_ns.append(len(active_b))
        active_c_ns.append(len(active_c))
        active_d_ns.append(len({str(r.league) for r in base_day}))

    base = _aggregate(base_rows)
    a = _aggregate(a_rows)
    b = _aggregate(b_rows)
    c = _aggregate(c_rows)

    return {
        "days_total": len(days),
        "cycles": len(active_a_ns),
        "baseline": base,
        "policy_a": a,
        "policy_b": b,
        "policy_c": c,
        "policy_d": base,
        "active_a_avg": (float(mean(active_a_ns)) if active_a_ns else 0.0),
        "active_b_avg": (float(mean(active_b_ns)) if active_b_ns else 0.0),
        "active_c_avg": (float(mean(active_c_ns)) if active_c_ns else 0.0),
        "active_d_avg": (float(mean(active_d_ns)) if active_d_ns else 0.0),
        "active_a_min": (int(min(active_a_ns)) if active_a_ns else 0),
        "active_a_max": (int(max(active_a_ns)) if active_a_ns else 0),
        "active_b_min": (int(min(active_b_ns)) if active_b_ns else 0),
        "active_b_max": (int(max(active_b_ns)) if active_b_ns else 0),
        "active_c_min": (int(min(active_c_ns)) if active_c_ns else 0),
        "active_c_max": (int(max(active_c_ns)) if active_c_ns else 0),
        "active_d_min": (int(min(active_d_ns)) if active_d_ns else 0),
        "active_d_max": (int(max(active_d_ns)) if active_d_ns else 0),
        "blocked_top": sorted(blocked_count.items(), key=lambda kv: kv[1], reverse=True)[:20],
    }


def _print_report(rep: Dict[str, Any], *, slip_max: float) -> None:
    b = rep.get("baseline") or {}
    a = rep.get("policy_a") or {}
    b2 = rep.get("policy_b") or {}
    c = rep.get("policy_c") or {}
    d = rep.get("policy_d") or {}

    def _n(x: Dict[str, Any], k: str) -> float:
        return float(x.get(k) or 0.0)

    print("\n=== Comparação de cenários por liga ===")
    print(f"dias={int(rep.get('days_total') or 0)} | ciclos={int(rep.get('cycles') or 0)} | avaliação: slippage_pre_pct <= {float(slip_max):.2f}")

    print("\n[Cenário 4 - sem filtro de ligas]")
    print(
        f"n={int(_n(b,'n'))} stake={_fmt_num(_n(b,'stake'))} pnl={_fmt_num(_n(b,'pnl'))} "
        f"ROIw={_fmt_pct(b.get('roiw'))}"
    )
    print(
        f"active_leagues_avg={_fmt_num(rep.get('active_d_avg'))} "
        f"(min={int(rep.get('active_d_min') or 0)}, max={int(rep.get('active_d_max') or 0)})"
    )

    print("\n[Cenário 1 - política atual (A)]")
    print(
        f"n={int(_n(a,'n'))} stake={_fmt_num(_n(a,'stake'))} pnl={_fmt_num(_n(a,'pnl'))} "
        f"ROIw={_fmt_pct(a.get('roiw'))}"
    )
    print(
        f"turnover_n={_fmt_pct(_pct(_n(a,'n'), _n(b,'n')), 1)} | "
        f"turnover_stake={_fmt_pct(_pct(_n(a,'stake'), _n(b,'stake')), 1)}"
    )
    print(
        f"active_leagues_avg={_fmt_num(rep.get('active_a_avg'))} "
        f"(min={int(rep.get('active_a_min') or 0)}, max={int(rep.get('active_a_max') or 0)})"
    )

    print("\n[Cenário 2 - estabilidade (A + histerese)]")
    print(
        f"n={int(_n(b2,'n'))} stake={_fmt_num(_n(b2,'stake'))} pnl={_fmt_num(_n(b2,'pnl'))} "
        f"ROIw={_fmt_pct(b2.get('roiw'))}"
    )
    print(
        f"turnover_n={_fmt_pct(_pct(_n(b2,'n'), _n(b,'n')), 1)} | "
        f"turnover_stake={_fmt_pct(_pct(_n(b2,'stake'), _n(b,'stake')), 1)}"
    )
    print(
        f"active_leagues_avg={_fmt_num(rep.get('active_b_avg'))} "
        f"(min={int(rep.get('active_b_min') or 0)}, max={int(rep.get('active_b_max') or 0)})"
    )

    print("\n[Cenário 3 - excluir ruins (sem seleção de boas)]")
    print(
        f"n={int(_n(c,'n'))} stake={_fmt_num(_n(c,'stake'))} pnl={_fmt_num(_n(c,'pnl'))} "
        f"ROIw={_fmt_pct(c.get('roiw'))}"
    )
    print(
        f"turnover_n={_fmt_pct(_pct(_n(c,'n'), _n(b,'n')), 1)} | "
        f"turnover_stake={_fmt_pct(_pct(_n(c,'stake'), _n(b,'stake')), 1)}"
    )
    print(
        f"active_leagues_avg={_fmt_num(rep.get('active_c_avg'))} "
        f"(min={int(rep.get('active_c_min') or 0)}, max={int(rep.get('active_c_max') or 0)})"
    )

    delta_pnl = _n(c, "pnl") - _n(a, "pnl")
    r_a = a.get("roiw")
    r_c = c.get("roiw")
    delta_roiw = (float(r_c) - float(r_a)) if (r_a is not None and r_c is not None) else None
    delta_turn_stake = None
    if _n(b, "stake") > 0:
        delta_turn_stake = (100.0 * _n(c, "stake") / _n(b, "stake")) - (100.0 * _n(a, "stake") / _n(b, "stake"))
    print("\n[DELTA cenário 3 vs cenário 1]")
    print(
        f"delta_pnl={_fmt_num(delta_pnl)} | "
        f"delta_ROIw_pp={_fmt_num(delta_roiw, 3)} | "
        f"delta_turnover_stake_pp={_fmt_num(delta_turn_stake, 3)}"
    )

    tops = rep.get("blocked_top") or []
    if tops:
        print("\n[TOP ligas bloqueadas por regra de ruim persistente]")
        for lg, k in tops[:12]:
            print(f"- {lg}: {int(k)} ciclos bloqueada")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Comparação de cenários de política por liga: "
            "1) atual, 2) estabilidade, 3) excluir ruins, 4) sem filtro."
        )
    )
    ap.add_argument("--days", type=int, default=int(os.getenv("AB_DAYS", "60")))
    ap.add_argument("--train-days", type=int, default=int(os.getenv("AB_TRAIN_DAYS", "14")))
    ap.add_argument("--slip-max", type=float, default=float(os.getenv("AB_SLIP_MAX", "2.0")))
    ap.add_argument("--min-n-current", type=int, default=int(os.getenv("AB_MIN_N_CURRENT", "20")))
    ap.add_argument("--min-n-keep", type=int, default=int(os.getenv("AB_MIN_N_KEEP", "10")))
    ap.add_argument("--keep-roi-min", type=float, default=float(os.getenv("AB_KEEP_ROI_MIN", "-0.50")))
    ap.add_argument("--bad-roi-max", type=float, default=float(os.getenv("AB_BAD_ROI_MAX", "-3.0")))
    ap.add_argument("--bad-min-n", type=int, default=int(os.getenv("AB_BAD_MIN_N", "20")))
    ap.add_argument("--bad-streak-cycles", type=int, default=int(os.getenv("AB_BAD_STREAK_CYCLES", "2")))
    ap.add_argument("--min-n-universe", type=int, default=int(os.getenv("AB_MIN_N_UNIVERSE", "1")))
    ap.add_argument("--executor-jsonl", default=os.getenv("EXECUTOR_JSONL", "logs/executor_live.jsonl"))
    ap.add_argument("--accounting-dir", default=os.getenv("ACCOUNTING_OUT_DIR", "logs/accounting"))
    ap.add_argument("--database-url", default=os.getenv("DATABASE_URL", ""))
    ap.add_argument("--audit-map-csv", default="", help="Opcional: CSV id->league já pronto. Se vazio, gera com psql.")
    ap.add_argument("--regime", choices=["pre", "in", "all"], default="pre")
    args = ap.parse_args()

    executor_jsonl = Path(str(args.executor_jsonl))
    if not executor_jsonl.exists():
        print(json.dumps({"error": "executor_jsonl_not_found", "path": str(executor_jsonl)}, ensure_ascii=False))
        return 2

    bal = _latest_balance_csv(Path(str(args.accounting_dir)))
    if bal is None or not bal.exists():
        print(json.dumps({"error": "balance_csv_not_found", "dir": str(args.accounting_dir)}, ensure_ascii=False))
        return 2

    with tempfile.TemporaryDirectory(prefix="ab_policy_") as td:
        audit_map_csv = Path(str(args.audit_map_csv)).expanduser() if str(args.audit_map_csv).strip() else (Path(td) / "audit_map.csv")
        if not str(args.audit_map_csv).strip():
            if not str(args.database_url or "").strip():
                print(json.dumps({"error": "database_url_required", "hint": "--database-url ou DATABASE_URL"}, ensure_ascii=False))
                return 2
            _run_psql_copy_audit_map(database_url=str(args.database_url).strip(), days=int(args.days), out_csv=audit_map_csv)

        league_by_aid = _load_audit_league_map(audit_map_csv)
        pnl_by_oid = _load_pnl_like_by_order_id(bal)
        if not pnl_by_oid:
            print(json.dumps({"error": "empty_accounting_order_map", "balance_csv": str(bal)}, ensure_ascii=False))
            return 2

        rg = str(args.regime).strip().lower()
        rows = _load_exec_rows(
            executor_jsonl=executor_jsonl,
            league_by_audit_id=league_by_aid,
            pnl_by_order_id=pnl_by_oid,
            regime=("pre" if rg == "pre" else ("in" if rg == "in" else "")),
        )
        if not rows:
            print(json.dumps({"error": "no_rows_after_join", "regime": rg}, ensure_ascii=False))
            return 2

        rep = run_ab(
            rows_all=rows,
            train_days=int(args.train_days),
            slip_max=float(args.slip_max),
            min_n_current=int(args.min_n_current),
            min_n_keep=int(args.min_n_keep),
            keep_roi_min=float(args.keep_roi_min),
            bad_roi_max=float(args.bad_roi_max),
            bad_min_n=int(args.bad_min_n),
            bad_streak_cycles=int(args.bad_streak_cycles),
            min_n_universe=int(args.min_n_universe),
        )
        _print_report(rep, slip_max=float(args.slip_max))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
