from __future__ import annotations

import argparse
import csv
import json
import os
import random
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _load_env_file(path: Path) -> None:
    try:
        if not path.exists():
            return
        for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            if not k or k in os.environ:
                continue
            os.environ[k] = v.strip()
    except Exception:
        return


def _pick_col(cols: List[str], needles: Tuple[str, ...]) -> Optional[str]:
    try:
        cols = list(cols or [])
        cols_map = {str(c).lower(): str(c) for c in cols if str(c)}
        cols_l = list(cols_map.keys())
        for n0 in (needles or []):
            n = str(n0).lower()
            if not n:
                continue
            if n in cols_map:
                return cols_map[n]
            for cl in cols_l:
                if cl.startswith(n):
                    return cols_map[cl]
            for cl in cols_l:
                if n in cl:
                    return cols_map[cl]
    except Exception:
        return None
    return None


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        s = str(x).strip()
        if not s:
            return None
        s = s.replace(",", ".")
        return float(s)
    except Exception:
        return None


def _approx_eq(a: Any, b: float, *, eps: float = 1e-6) -> bool:
    try:
        if a is None:
            return False
        return abs(float(a) - float(b)) <= float(eps)
    except Exception:
        return False


def _bootstrap_ci_mean(xs: List[float], *, ci: float, n_boot: int, seed: int = 0) -> Optional[Tuple[float, float]]:
    try:
        xs2 = [float(x) for x in (xs or [])]
    except Exception:
        xs2 = []
    if len(xs2) < 5:
        return None
    n = len(xs2)
    nb = int(max(200, int(n_boot)))
    rnd = random.Random(int(seed))
    means: List[float] = []
    for _ in range(nb):
        s = 0.0
        for _j in range(n):
            s += xs2[rnd.randrange(0, n)]
        means.append(s / float(n))
    means.sort()
    alpha = float(max(0.0, min(1.0, 1.0 - float(ci))))
    lo = int(round((alpha / 2.0) * (len(means) - 1)))
    hi = int(round((1.0 - alpha / 2.0) * (len(means) - 1)))
    lo = max(0, min(len(means) - 1, lo))
    hi = max(0, min(len(means) - 1, hi))
    return float(means[lo]), float(means[hi])


def _latest_csv(out_dir: Path, suffix: str) -> Optional[Path]:
    try:
        cands = sorted(out_dir.glob(f"*__{suffix}.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        return cands[0] if cands else None
    except Exception:
        return None


def _open_order_ids_from_open_stakes_csv(path: Path) -> Optional[set[str]]:
    if not path or not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            r = csv.DictReader(f)
            cols = list(r.fieldnames or [])
            if not cols:
                return None
            oid_col = _pick_col(cols, ("order_id", "order id", "bet id", "bet_id", "id", "order"))
            if not oid_col:
                return None
            out: set[str] = set()
            for row in r:
                if not isinstance(row, dict):
                    continue
                oid = str(row.get(oid_col) or "").strip()
                if oid:
                    out.add(oid)
            return out
    except Exception:
        return None


def _ledger_pnl_like_by_order(balance_csv: Path) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not balance_csv.exists():
        return out

    def _excl_type(tl: str) -> bool:
        t = str(tl or "").strip().lower()
        return any(k in t for k in ("deposit", "withdraw", "transfer", "top", "payment", "adjust", "bonus"))

    with balance_csv.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        r = csv.DictReader(f)
        cols = list(r.fieldnames or [])
        if not cols:
            return out
        oid_col = _pick_col(cols, ("order_id", "order id", "order", "bet id", "bet_id", "id"))
        pnl_col = _pick_col(cols, ("amount", "profit_loss", "profit", "p&l", "pnl", "net", "pl"))
        typ_col = _pick_col(cols, ("type",))
        if not oid_col or not pnl_col:
            return out
        for row in r:
            if not isinstance(row, dict):
                continue
            oid = str(row.get(oid_col) or "").strip()
            if not oid or not oid.isdigit():
                continue
            if typ_col and _excl_type(str(row.get(typ_col) or "")):
                continue
            pnl = _safe_float(row.get(pnl_col))
            if pnl is None:
                continue
            out[oid] = float(out.get(oid) or 0.0) + float(pnl)
    return out


def _iter_exec_rows(
    *,
    executor_jsonl: Path,
    start_day: str,
    thr_ms: int,
    hi_min: float,
    hi_max: Optional[float],
) -> List[Dict[str, Any]]:
    """
    Retorna rows com {oid, roi, fast(bool), stake, pre_submit_ms, market_regime}
    Somente Back LIVE_OK, market_regime=pre, stake na faixa HI e created>=start_day.
    """
    rows: List[Dict[str, Any]] = []
    if not executor_jsonl.exists():
        return rows
    for ln in executor_jsonl.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            obj = json.loads(ln)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        req = obj.get("request") if isinstance(obj.get("request"), dict) else {}
        res = obj.get("result") if isinstance(obj.get("result"), dict) else {}
        if str(res.get("status") or "").strip() != "LIVE_OK":
            continue
        if str(res.get("exec_side") or req.get("exec_side") or "").strip().lower() != "back":
            continue
        dt = str(res.get("created_at") or req.get("created_at") or "").strip()
        if not dt or "T" not in dt:
            continue
        day = dt.split("T")[0]
        if start_day and day < start_day:
            continue
        raw = res.get("raw") if isinstance(res.get("raw"), dict) else {}
        oid = str(raw.get("order_id") or "").strip()
        if not oid or not oid.isdigit():
            continue
        sent = raw.get("sent") if isinstance(raw.get("sent"), dict) else {}
        st = _safe_float(sent.get("stake"))
        if st is None and isinstance(res.get("policy"), dict):
            st = _safe_float(res.get("policy", {}).get("stake_requested"))
        if st is None:
            continue
        if float(st) <= float(hi_min):
            continue
        if hi_max is not None and float(st) > float(hi_max):
            continue
        vs = raw.get("value_sizing") if isinstance(raw.get("value_sizing"), dict) else {}
        if str(vs.get("market_regime") or "") != "pre":
            continue
        pre_ms = None
        try:
            pre_ms = int(float(vs.get("pre_submit_ms")))
        except Exception:
            pre_ms = None
        if pre_ms is None:
            continue
        fast = bool(int(pre_ms) <= int(thr_ms))
        rows.append(
            {
                "order_id": oid,
                "day": day,
                "stake": float(st),
                "fast": fast,
                "pre_submit_ms": int(pre_ms),
            }
        )
    return rows


def _fmt_pct(x: Any, nd: int = 2) -> str:
    try:
        if x is None:
            return "—"
        return f"{float(x):.{nd}f}%"
    except Exception:
        return "—"


def _fmt_num(x: Any, nd: int = 2) -> str:
    try:
        if x is None:
            return "—"
        return f"{float(x):,.{nd}f}"
    except Exception:
        return "—"


def main() -> int:
    ap = argparse.ArgumentParser(description="Gera PDF: estatísticas da tese Back Pre fast (pre_submit_ms).")
    ap.add_argument("--out", default=os.getenv("BACKPRE_FAST_STATS_PDF_OUT", "docs/backpre_fast_stats.pdf"))
    ap.add_argument("--env-file", default=os.getenv("ENV_FILE", ".env"))
    ap.add_argument("--executor-jsonl", default=os.getenv("EXECUTOR_JSONL", "logs/executor_live.jsonl"))
    ap.add_argument("--accounting-out-dir", default=os.getenv("ACCOUNTING_OUT_DIR", "logs/accounting"))
    ap.add_argument("--perm-n", type=int, default=int(float(os.getenv("BACKPRE_FAST_PERM_N", "20000") or 20000)))
    ap.add_argument(
        "--manual-text",
        default=None,
        help="Caminho de um .txt com outputs (ex.: colados do terminal) para anexar no PDF.",
    )
    ap.add_argument(
        "--manual-stdin",
        action="store_true",
        help="Lê texto do stdin e anexa no PDF (para colar outputs via pipe).",
    )
    args = ap.parse_args()

    _load_env_file(Path(str(args.env_file)))

    ts = _utcnow().strftime("%Y-%m-%d %H:%M UTC")
    start_day = str(os.getenv("DAILY_BACKPRE_FAST_THESIS_START_DAY", "") or "").strip()
    thr_ms = int(float(os.getenv("EXECUTOR_BACKPRE_FAST_MAX_PRE_SUBMIT_MS", "5000") or 5000))
    stake_hi = float(os.getenv("EXECUTOR_BACKPRE_FAST_STAKE_HI", "20") or 20.0)
    stake_lo = str(os.getenv("EXECUTOR_BACKPRE_FAST_STAKE_LO", "1.50") or "1.50").strip()
    hi_min = float(os.getenv("DAILY_BACKPRE_FAST_HI_MIN", "5.0") or 5.0)
    hi_max_raw = str(os.getenv("DAILY_BACKPRE_FAST_HI_MAX", "") or "").strip()
    hi_max = _safe_float(hi_max_raw) if hi_max_raw else None
    if hi_max is not None and float(hi_max) <= float(hi_min):
        hi_max = None
    hi_label = (
        f"stake > {float(hi_min):.2f}"
        if hi_max is None
        else f"stake em ({float(hi_min):.2f}, {float(hi_max):.2f}]"
    )
    n_boot = int(float(os.getenv("DAILY_BACKPRE_FAST_BOOTSTRAP_N", "2000") or 2000))
    min_n = int(float(os.getenv("DAILY_BACKPRE_FAST_MIN_ORDERS", "25") or 25))

    exec_path = Path(str(args.executor_jsonl)).expanduser()
    acct_dir = Path(str(args.accounting_out_dir)).expanduser()
    bal_csv = _latest_csv(acct_dir, "balance")
    open_csv = _latest_csv(acct_dir, "open_stakes")
    open_oids = _open_order_ids_from_open_stakes_csv(open_csv) if open_csv else None
    ledger = _ledger_pnl_like_by_order(bal_csv) if bal_csv else {}
    exec_rows = _iter_exec_rows(
        executor_jsonl=exec_path,
        start_day=start_day,
        thr_ms=thr_ms,
        hi_min=hi_min,
        hi_max=hi_max,
    )

    # join com ledger -> ROI por ordem
    joined = []
    for r in exec_rows:
        oid = r["order_id"]
        if oid not in ledger:
            continue
        pnl = float(ledger[oid])
        st = float(r["stake"])
        joined.append({**r, "pnl": pnl, "roi": (pnl / st * 100.0)})

    fast = [x for x in joined if x["fast"]]
    slow = [x for x in joined if not x["fast"]]

    def _settled_only(xs):
        if open_oids is None:
            return None
        return [x for x in xs if str(x["order_id"]) not in open_oids]

    fast_set = _settled_only(fast)
    slow_set = _settled_only(slow)

    # permutação (one-sided: fast-slow >= observado)
    perm_p = None
    obs_delta = None
    if fast and slow:
        obs_delta = statistics.fmean([x["roi"] for x in fast]) - statistics.fmean([x["roi"] for x in slow])
        rnd = random.Random(123)
        all_rois = [x["roi"] for x in joined]
        labels = [x["fast"] for x in joined]
        ge = 0
        n = int(max(1000, int(args.perm_n)))
        for _ in range(n):
            perm = labels[:]
            rnd.shuffle(perm)
            mfast = statistics.fmean([all_rois[i] for i, b in enumerate(perm) if b])
            mslow = statistics.fmean([all_rois[i] for i, b in enumerate(perm) if not b])
            if (mfast - mslow) >= float(obs_delta):
                ge += 1
        perm_p = float((ge + 1) / (n + 1))

    def _sum(xs, k):
        return float(sum(float(x.get(k) or 0.0) for x in (xs or [])))

    def _roiw(xs):
        st = _sum(xs, "stake")
        pn = _sum(xs, "pnl")
        return (pn / st * 100.0) if st > 0 else None

    ci_fast90 = _bootstrap_ci_mean([x["roi"] for x in fast], ci=0.90, n_boot=n_boot, seed=1) if len(fast) >= min_n else None
    ci_slow90 = _bootstrap_ci_mean([x["roi"] for x in slow], ci=0.90, n_boot=n_boot, seed=2) if len(slow) >= min_n else None

    alerts: List[str] = []
    if not exec_path.exists():
        alerts.append(f"executor_jsonl não encontrado em `{exec_path}` (ajuste EXECUTOR_JSONL ou rode no VPS).")
    if not acct_dir.exists():
        alerts.append(f"ACCOUNTING_OUT_DIR `{acct_dir}` não existe (ajuste ACCOUNTING_OUT_DIR).")
    if bal_csv is None:
        alerts.append("Nenhum `*__balance.csv` encontrado em ACCOUNTING_OUT_DIR (sem join P&L por order_id).")
    if open_csv is None:
        alerts.append("Nenhum `*__open_stakes.csv` encontrado em ACCOUNTING_OUT_DIR (sem n_liquidadas/ROIw_liquidado).")
    if exec_rows and not joined:
        alerts.append("Há ordens elegíveis no executor_jsonl, mas 0 fizeram join no ledger (order_id ausente no balance.csv ou formatos divergentes).")
    if start_day in ("", "—", None):
        alerts.append("DAILY_BACKPRE_FAST_THESIS_START_DAY não está setado; o recorte pós-início pode estar misturando períodos.")

    manual_blob = ""
    try:
        if args.manual_text:
            p = Path(str(args.manual_text)).expanduser()
            if p.exists():
                manual_blob = p.read_text(encoding="utf-8", errors="ignore").strip()
    except Exception:
        manual_blob = manual_blob or ""
    if args.manual_stdin:
        try:
            stdin_txt = (sys.stdin.read() or "").strip()
            if stdin_txt:
                manual_blob = (manual_blob + "\n\n" + stdin_txt).strip() if manual_blob else stdin_txt
        except Exception:
            pass

    md = f"""# Estatísticas — Tese Back Pre fast (pre_submit_ms)

Gerado em: **{ts}**

## ALERTAS (se aparecerem, o PDF pode ficar “sem análise”)
{(''.join(f'- **{a}**\n' for a in alerts) if alerts else '- (ok) dataset encontrado e join executado.\n')}

## Outputs anexados (manual; opcional)
{('```\n'+manual_blob+'\n```\n' if manual_blob else '_—_\n')}

## Definição da tese (operacional)
- Universo: **Back Pre** (pre-match), apostas efetivas (`LIVE_OK`).
- “Fast”: `pre_submit_ms <= {thr_ms}ms`.
- Sizing operacional (quando habilitado): fast **e** `slippage_pre_pct < EXECUTOR_BACKPRE_FAST_MAX_SLIPPAGE_PCT` ⇒ **stake={stake_hi}**, demais Back ⇒ **stake={stake_lo}**.
- Critério HI no relatório: **{hi_label}**.
- Início operacional (recorte recomendado): `{start_day or '—'}` (UTC).  
  (Use a data em que você ligou `EXECUTOR_BACKPRE_FAST_STAKE_ENABLE=1` em produção.)

## Dataset (VPS) usado neste PDF
- executor_jsonl: `{exec_path}`
- balance_csv (último): `{str(bal_csv) if bal_csv else '—'}`
- open_stakes_csv (último): `{str(open_csv) if open_csv else '—'}`
- n ordens elegíveis (pre, {hi_label}, pós-início): `{len(exec_rows)}`
- n com join no ledger por order_id: `{len(joined)}`

## Resultado principal (ROI por ordem; faixa HI)
| Grupo | n ordens | ROI mean | ROIw | IC90 ROI mean (bootstrap) |
|---|---:|---:|---:|---:|
| Fast (pre_submit_ms<= {thr_ms}ms) | {len(fast)} | {_fmt_pct(statistics.fmean([x['roi'] for x in fast]) if fast else None)} | {_fmt_pct(_roiw(fast))} | {(_fmt_pct(ci_fast90[0])+' .. '+_fmt_pct(ci_fast90[1])) if ci_fast90 else '—'} |
| Slow (pre_submit_ms> {thr_ms}ms) | {len(slow)} | {_fmt_pct(statistics.fmean([x['roi'] for x in slow]) if slow else None)} | {_fmt_pct(_roiw(slow))} | {(_fmt_pct(ci_slow90[0])+' .. '+_fmt_pct(ci_slow90[1])) if ci_slow90 else '—'} |

Delta (Fast − Slow) ROI mean: `{_fmt_pct(obs_delta)}`.

### Teste de permutação (robustez)
- Permutação (one-sided): `p_value = {perm_p if perm_p is not None else '—'}` (n_perm={int(max(1000,int(args.perm_n)))})

## Liquidação (quando open_stakes.csv existe)
_Interpretação: `ROIw_liquidado = (∑P&L_liquidado)/(∑stake_liquidado)`._

| Grupo | n_liquidadas | ROIw_liquidado |
|---|---:|---:|
| Fast | {('—' if fast_set is None else len(fast_set))} | {_fmt_pct(None if fast_set is None else _roiw(fast_set))} |
| Slow | {('—' if slow_set is None else len(slow_set))} | {_fmt_pct(None if slow_set is None else _roiw(slow_set))} |

## Interpretação (como decidir)
- Para afirmar robustez, olhe (i) IC90/IC95 (bootstrap), (ii) teste de permutação, e (iii) estabilidade por dia (no daily report).
- Se `open_stakes.csv` existir e `n_liquidadas` for pequeno, **não** usar ROIw do período como veredito; esperar maturação (settlement).

## Observações
Este PDF é um “sumário metodológico” para acompanhar a tese. Os números do dia devem ser lidos no PDF diário.

## Reprodutibilidade
Este PDF roda os testes diretamente no VPS (sem colar outputs). Para ver detalhes, rode o daily report e compare com as tabelas “Tese: Back Pre fast …”.
"""

    out_pdf = Path(str(args.out)).expanduser()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    tmp_md = out_pdf.with_suffix(".md")
    tmp_md.write_text(md, encoding="utf-8")

    renderer = Path(__file__).resolve().parent.parent / "docs" / "render_markdown_to_pdf.py"
    # IMPORTANT: render_markdown_to_pdf.py precisa rodar no MESMO ambiente (venv) com reportlab instalado.
    # `sys.executable` já aponta para o python atual (ex.: ./venv/bin/python).
    subprocess.run([sys.executable, str(renderer), str(tmp_md), str(out_pdf)], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

