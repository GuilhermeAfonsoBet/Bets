from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from loguru import logger

from .accounting_daily_report import DailyCfg as AcctDailyCfg, run_daily as run_acct_daily
from .accounting_report import compute_pnl_report
from .execution_kpis import compute_kpis_from_lines


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _load_env_file(path: Path) -> None:
    """
    Carrega variáveis de um arquivo .env simples (KEY=VALUE), sem sobrescrever env já definido.
    Ajuda quando rodando manualmente fora do systemd (que usa EnvironmentFile=...).
    """
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


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if not path.exists():
            return None
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


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
        # separador de milhar para leitura operacional (ex.: 1,000,000.00)
        return f"{float(x):,.{nd}f}"
    except Exception:
        return "—"


def _pick_last_day_with_slippage_vs_roi_raw(per_day: list[dict]) -> Optional[dict]:
    """
    O bloco slippage×ROI depende de ROI (placar disponível). Em dias recentes pode estar vazio.
    Pegamos o último dia que tenha pelo menos 1 bucket (Back ou Lay).
    """
    try:
        for it in reversed(per_day or []):
            if not isinstance(it, dict):
                continue
            ex = it.get("execution") if isinstance(it.get("execution"), dict) else {}
            rawblk = ex.get("slippage_vs_roi_raw") if isinstance(ex.get("slippage_vs_roi_raw"), dict) else {}
            b = rawblk.get("back") if isinstance(rawblk.get("back"), dict) else {}
            l = rawblk.get("lay") if isinstance(rawblk.get("lay"), dict) else {}
            bb = b.get("buckets") if isinstance(b.get("buckets"), list) else []
            lb = l.get("buckets") if isinstance(l.get("buckets"), list) else []
            if bb or lb:
                return it
    except Exception:
        return None
    return None


def _slip_raw_3bucket_rows(buckets: list[dict]) -> list[dict]:
    """
    Normaliza para sempre retornar 3 buckets: <=-2%, (-2,2], >2%.
    buckets pode vir incompleto (quando N=0 em algum bucket).
    """
    want = ["<= -2%", "(-2, 2]", "> 2%"]
    by = {}
    for b in buckets or []:
        if isinstance(b, dict) and str(b.get("bucket") or ""):
            by[str(b.get("bucket"))] = b
    out = []
    for lab in want:
        it = by.get(lab) or {}
        out.append(
            {
                "bucket": lab,
                "n": int(it.get("n") or 0),
                "roi_mean": it.get("roi_mean"),
                "roi_se": it.get("roi_se"),
                "roi_ci95": it.get("roi_ci95"),
                "odd_median": it.get("odd_median"),
                "exposure_median": it.get("exposure_median"),
                "exposure_sum": it.get("exposure_sum"),
                "roi_weighted": it.get("roi_weighted"),
            }
        )
    return out


def _fmt_roi_mean_se_ci_pct(row: dict) -> str:
    """
    Formata ROI (mean) com SE e IC95% (quando disponíveis).
    """
    try:
        mean = row.get("roi_mean")
        if mean is None:
            return "—"
        se = row.get("roi_se")
        ci = row.get("roi_ci95") if isinstance(row.get("roi_ci95"), dict) else None
        if se is None and not ci:
            return _fmt_pct(mean)
        se_s = _fmt_pct(se) if se is not None else "—"
        if ci and (ci.get("lb") is not None) and (ci.get("ub") is not None):
            base = f"{_fmt_pct(mean)} (SE {se_s}) [{_fmt_pct(ci.get('lb'))}, {_fmt_pct(ci.get('ub'))}]"
        else:
            base = f"{_fmt_pct(mean)} (SE {se_s})"
        # ROI ponderado por exposição (quando disponível)
        if row.get("roi_weighted") is not None:
            base += f" | ROIw {_fmt_pct(row.get('roi_weighted'))}"
        return base
    except Exception:
        return "—"


def _fmt_ctx_suffix(row: dict) -> str:
    """
    Sufixo opcional com contexto para interpretar ROIs extremos:
    odd_median e exposure_median (stake para Back; liability para Lay).
    """
    try:
        om = row.get("odd_median")
        em = row.get("exposure_median")
        if om is None and em is None:
            return ""
        s = []
        if om is not None:
            s.append(f"odd~{_fmt_num(om,2)}")
        if em is not None:
            s.append(f"exp~{_fmt_num(em,2)}")
        return " (" + ", ".join(s) + ")"
    except Exception:
        return ""


def _append_slippage_vs_roi_raw_section(
    out_lines: list[str],
    *,
    adh_slip: Optional[Dict[str, Any]],
    title: str,
    combo_top_limit: int = 2,
) -> None:
    """
    Renderiza o bloco "Slippage × ROI (raw, com sinal)" preservando as mesmas tabelas:
      - buckets 3-way (Back + Lay)
      - Lay bounded por stake
      - Contrafactual (placar): filtro de slippage
      - Diagnóstico AH (linha)
      - Slippage × ROI por combinação (top N por volume; acumulado)
    """
    try:
        if not isinstance(adh_slip, dict) or not adh_slip:
            return

        # slippage x ROI (3 buckets raw com sinal) — acumulado na janela (não só um dia)
        raw_total: Dict[str, Any] = {}
        try:
            raw_total = (
                adh_slip.get("slippage_vs_roi_raw_total_ctx")
                if isinstance(adh_slip.get("slippage_vs_roi_raw_total_ctx"), dict)
                else (adh_slip.get("slippage_vs_roi_raw_total") if isinstance(adh_slip.get("slippage_vs_roi_raw_total"), dict) else {})
            )
        except Exception:
            raw_total = {}
        if not isinstance(raw_total, dict) or not raw_total:
            return

        try:
            # Para slippage×ROI, respeitamos o range semântico (pós-fix) quando disponível.
            rg = adh_slip.get("slippage_range", None) if isinstance(adh_slip, dict) else None
            if not isinstance(rg, dict) or not rg:
                rg = adh_slip.get("range", {}) if isinstance(adh_slip, dict) else {}
            span = rg.get("span_days") if isinstance(rg, dict) else None
            out_lines.append(f"**{title} (range: `{rg.get('start_day')}` → `{rg.get('end_day')}`; span_days=`{int(span or 0)}`)**\n\n")
        except Exception:
            out_lines.append(f"**{title}**\n\n")

        for side_key, subtitle in (("back", "Back (ROI por stake)"), ("lay", "Lay (ROI por liability)")):
            b = raw_total.get(side_key) if isinstance(raw_total.get(side_key), dict) else {}
            buckets0 = b.get("buckets") if isinstance(b.get("buckets"), list) else []
            buckets = _slip_raw_3bucket_rows(buckets0)
            if not any(int(r.get("n") or 0) > 0 for r in buckets):
                continue
            out_lines.append(f"- **{subtitle}**\n\n")
            out_lines.append("| Bucket slippage_raw_pct | n | ROI mean (SE; IC95) |\n|---|---:|---:|\n")
            for row in buckets:
                out_lines.append(
                    f"| {row.get('bucket')} | {int(row.get('n') or 0)} | {_fmt_roi_mean_se_ci_pct(row)}{_fmt_ctx_suffix(row)} |\n"
                )
            out_lines.append("\n")

        out_lines.append(
            "- Nota: `ROIw` é o **ROI ponderado por exposição** (peso=stake no Back; peso=liability no Lay). "
            "Em prática, dentro de um bucket, `ROIw ≈ (∑P&L)/(∑exposição)`; já o `ROI mean` é a média simples por linha/sinal.\n\n"
        )

        # Lay também em ROI por stake (bounded; sanity-check)
        lay_stake_blk = (
            adh_slip.get("slippage_vs_roi_raw_total_ctx_lay_stake")
            if (isinstance(adh_slip, dict) and isinstance(adh_slip.get("slippage_vs_roi_raw_total_ctx_lay_stake"), dict))
            else {}
        )
        b2 = lay_stake_blk.get("lay") if isinstance(lay_stake_blk.get("lay"), dict) else {}
        buckets02 = b2.get("buckets") if isinstance(b2.get("buckets"), list) else []
        buckets2 = _slip_raw_3bucket_rows(buckets02)
        if any(int(r.get("n") or 0) > 0 for r in buckets2):
            out_lines.append("- **Lay (ROI por stake; bounded)**\n\n")
            out_lines.append("| Bucket slippage_raw_pct | n | ROI mean (SE; IC95) |\n|---|---:|---:|\n")
            for row in buckets2:
                out_lines.append(f"| {row.get('bucket')} | {int(row.get('n') or 0)} | {_fmt_roi_mean_se_ci_pct(row)}{_fmt_ctx_suffix(row)} |\n")
            out_lines.append("\n")

        # Contrafactual: filtro de slippage (placar)
        try:
            cf = adh_slip.get("slippage_filter_counterfactual") if isinstance(adh_slip, dict) else None
            if isinstance(cf, dict) and isinstance(cf.get("rule"), dict):
                b = cf.get("back") if isinstance(cf.get("back"), dict) else {}
                l = cf.get("lay") if isinstance(cf.get("lay"), dict) else {}
                if (int(b.get("n") or 0) + int(l.get("n") or 0)) > 0:
                    out_lines.append("**Contrafactual (placar): aplicar filtro de slippage**\n\n")
                    out_lines.append("- Regra: **Back** pula `slippage_raw_pct <= -2%`; **Lay** pula `slippage_raw_pct > 2%`.\n")
                    out_lines.append("- Observação: usa somente execuções com ROI via placar; não é o P&L do accounting.\n\n")
                    out_lines.append("| Lado | n (base) | P&L (base) | Exposição (base) | n (após filtro) | P&L (após) | Exposição (após) |\n")
                    out_lines.append("|---|---:|---:|---:|---:|---:|---:|\n")
                    out_lines.append(
                        f"| Back | {int(b.get('n') or 0)} | {_fmt_num(b.get('pnl'),2)} | {_fmt_num(b.get('stake'),2)} | {int(b.get('n_filtered') or 0)} | {_fmt_num(b.get('pnl_filtered'),2)} | {_fmt_num(b.get('stake_filtered'),2)} |\n"
                    )
                    out_lines.append(
                        f"| Lay (liab) | {int(l.get('n') or 0)} | {_fmt_num(l.get('pnl'),2)} | {_fmt_num(l.get('liability'),2)} | {int(l.get('n_filtered') or 0)} | {_fmt_num(l.get('pnl_filtered'),2)} | {_fmt_num(l.get('liability_filtered'),2)} |\n"
                    )
                    try:
                        pnl0 = float(b.get("pnl") or 0.0) + float(l.get("pnl") or 0.0)
                        pnl1 = float(b.get("pnl_filtered") or 0.0) + float(l.get("pnl_filtered") or 0.0)
                        out_lines.append(f"| **Total** | — | {_fmt_num(pnl0,2)} | — | — | {_fmt_num(pnl1,2)} | — |\n")
                    except Exception:
                        pass
                    out_lines.append("\n")
        except Exception:
            pass

        # Diagnóstico AH (linha) observado na execução
        try:
            ah = adh_slip.get("observed_ah_line_abs") if isinstance(adh_slip, dict) else None
            if isinstance(ah, dict):
                thr = ah.get("threshold")
                scope = ah.get("scope")
                allx = ah.get("all_exec") if isinstance(ah.get("all_exec"), dict) else {}
                covx = ah.get("cov_placar") if isinstance(ah.get("cov_placar"), dict) else {}
                if int(allx.get("n") or 0) > 0:
                    out_lines.append("**Diagnóstico AH (linha) observado na execução**\n\n")
                    out_lines.append(f"- Policy: `ah_max_abs_line={thr}` | `ah_scope={scope}`\n")
                    out_lines.append(
                        f"- Execuções (todas): `n={int(allx.get('n') or 0)}` | `max|line|={_fmt_num(allx.get('max_abs_line'),2)}` | `n_over={int(allx.get('n_over') or 0)}`\n"
                    )
                    out_lines.append(
                        f"- Execuções com placar/ROI: `n={int(covx.get('n') or 0)}` | `max|line|={_fmt_num(covx.get('max_abs_line'),2)}` | `n_over={int(covx.get('n_over') or 0)}`\n\n"
                    )
        except Exception:
            pass

        # Por combinação (top por volume)
        rows = adh_slip.get("slippage_vs_roi_raw_by_combo_top") if (isinstance(adh_slip, dict) and isinstance(adh_slip.get("slippage_vs_roi_raw_by_combo_top"), list)) else []
        if rows:
            try:
                back_rows = [r for r in rows if isinstance(r, dict) and str(r.get("side")) == "Back"]
                lay_rows = [r for r in rows if isinstance(r, dict) and str(r.get("side")) == "Lay"]

                def _print_combo_block(title2: str, xs: list[dict], limit: int) -> None:
                    if not xs:
                        return
                    out_lines.append(f"**Slippage × ROI por combinação (top {min(limit, len(xs))} por volume; acumulado)**\n\n")
                    out_lines.append(f"- **{title2}**\n\n")
                    out_lines.append("| Combinação | n | ROI<=-2% | n | ROI(-2..2] | n | ROI>2% | n | corr(slip_raw,ROI) |\n")
                    out_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
                    for r in xs[:limit]:
                        comb = str(r.get("comb") or "")
                        n = int(r.get("n") or 0)
                        corr = r.get("corr_raw_pct_vs_roi")
                        bmap = {str(b.get("bucket")): b for b in (r.get("buckets") or []) if isinstance(b, dict)}

                        def _bn(lab: str) -> tuple[int, Any]:
                            bb = bmap.get(lab) or {}
                            return int(bb.get("n") or 0), bb

                        n1, roi1 = _bn("<= -2%")
                        n2, roi2 = _bn("(-2, 2]")
                        n3, roi3 = _bn("> 2%")
                        out_lines.append(
                            f"| {comb} | {n} | {_fmt_roi_mean_se_ci_pct(roi1)} | {n1} | {_fmt_roi_mean_se_ci_pct(roi2)} | {n2} | {_fmt_roi_mean_se_ci_pct(roi3)} | {n3} | {_fmt_num(corr,2)} |\n"
                        )
                    out_lines.append("\n")

                _print_combo_block("Back", back_rows, int(combo_top_limit))
                _print_combo_block("Lay", lay_rows, int(combo_top_limit))
            except Exception:
                pass
    except Exception:
        return


def _demote_h2_to_h3(md: str) -> str:
    # Usado para "embrulhar" o bloco in-sample sem reescrever o conteúdo.
    out = []
    for ln in (md or "").splitlines(True):
        if ln.startswith("## "):
            out.append("### " + ln[3:])
        else:
            out.append(ln)
    return "".join(out)


def _split_base_into_insample_and_oos(md: str) -> tuple[str, str]:
    """
    O relatório robusto pode escrever o bloco OOS no topo-nível como:
      - '## 12) OOS walk-forward ...' (modo "full")
      - '## 1) OOS walk-forward ...'  (modo "oos_first")
    Tudo antes disso é o bloco in-sample.
    """
    txt = md or ""
    keys = ["## 12) OOS walk-forward", "## 1) OOS walk-forward", "## 2) OOS walk-forward"]
    hits = [(txt.find(k), k) for k in keys if txt.find(k) >= 0]
    if not hits:
        # fallback: não encontrou; trata tudo como in-sample
        return txt, ""
    i, _ = sorted(hits, key=lambda x: x[0])[0]
    return txt[:i], txt[i:]


def _extract_md_block(md: str, *, start: str, until_any: list[str]) -> str:
    """
    Extrai um trecho de markdown começando em `start` até antes do primeiro marcador em `until_any`.
    Best-effort: se não achar `start`, retorna "".
    """
    txt = md or ""
    i = txt.find(start)
    if i < 0:
        return ""
    j = None
    for u in until_any:
        k = txt.find(u, i + len(start))
        if k >= 0:
            j = k if j is None else min(j, k)
    return txt[i : (j if j is not None else len(txt))].strip() + "\n"


def _extract_md_table(md: str, *, header_startswith: str) -> tuple[str, list[list[str]]]:
    """
    Extrai uma tabela markdown cujo header começa com `header_startswith` (linha iniciando com '| ...').
    Retorna (table_md, rows) onde rows são as linhas de dados já separadas em colunas (sem pipes).
    """
    txt = md or ""
    lines = txt.splitlines()
    i = None
    for idx, ln in enumerate(lines):
        if ln.strip().startswith(header_startswith):
            i = idx
            break
    if i is None:
        return "", []
    # coletar até a primeira linha vazia após começar
    out_lines = []
    rows: list[list[str]] = []
    for ln in lines[i:]:
        if not ln.strip():
            break
        if not ln.strip().startswith("|"):
            break
        out_lines.append(ln)
        # data row (skip separator)
        if ln.strip().startswith("|---"):
            continue
        cols = [c.strip() for c in ln.strip().strip("|").split("|")]
        # pula header
        if cols and cols[0].lower().startswith("train window"):
            continue
        rows.append(cols)
    return "\n".join(out_lines).strip() + "\n", rows


def _md_table_header_cols(table_md: str) -> list[str]:
    """
    Retorna as colunas do header da tabela (linha 1) sem pipes.
    """
    try:
        for ln in (table_md or "").splitlines():
            s = ln.strip()
            if not s.startswith("|"):
                continue
            if s.startswith("|---"):
                continue
            cols = [c.strip() for c in s.strip().strip("|").split("|")]
            return cols
    except Exception:
        return []


def _parse_md_number(x: Any) -> Optional[float]:
    """
    Parser robusto para números vindos de Markdown (OOS / tabelas no PDF):
    - aceita en-US (1,234.56) e pt-BR (1.234,56)
    - preserva decimais
    - aceita percentuais ("49.54%")
    """
    try:
        t = str(x or "").strip().replace("−", "-")
        if not t:
            return None
        t = t.replace("%", "").strip()
        t = t.replace(" ", "")
        if "." in t and "," in t:
            # decide separador decimal pelo último
            if t.rfind(".") > t.rfind(","):
                t = t.replace(",", "")
            else:
                t = t.replace(".", "").replace(",", ".")
        else:
            if "," in t and "." not in t:
                t = t.replace(",", ".")
        return float(t)
    except Exception:
        return None
    return None

def _tail_lines(path: Path, n: int) -> list[str]:
    try:
        xs = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        return xs[-n:] if n > 0 else xs
    except Exception:
        return []

def _week_start_iso(day_iso: str) -> Optional[str]:
    try:
        from datetime import date as _date, timedelta as _td

        d = _date.fromisoformat(str(day_iso))
        ws = d - _td(days=int(d.weekday()))
        return ws.isoformat()
    except Exception:
        return None


def _month_key(day_iso: str) -> Optional[str]:
    try:
        from datetime import date as _date

        d = _date.fromisoformat(str(day_iso))
        return f"{d.year:04d}-{d.month:02d}"
    except Exception:
        return None


def _agg_by_week(pnls_by_day: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for d, v in pnls_by_day.items():
        ws = _week_start_iso(d)
        if not ws:
            continue
        out[ws] = float(out.get(ws, 0.0)) + float(v or 0.0)
    return dict(sorted(out.items()))


def _agg_by_month(pnls_by_day: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for d, v in pnls_by_day.items():
        mk = _month_key(d)
        if not mk:
            continue
        out[mk] = float(out.get(mk, 0.0)) + float(v or 0.0)
    return dict(sorted(out.items()))


def _max_drawdown(pnls_by_day: Dict[str, float]) -> Dict[str, Any]:
    """
    Max drawdown em unidade monetária, usando curva de equity = cumsum(P&L diário).
    """
    days = sorted([d for d in pnls_by_day.keys() if str(d)])
    eq = 0.0
    peak = 0.0
    mdd = 0.0
    mdd_from = None
    mdd_to = None
    peak_day = None
    for d in days:
        eq += float(pnls_by_day.get(d) or 0.0)
        if eq >= peak:
            peak = eq
            peak_day = d
        dd = peak - eq
        if dd > mdd:
            mdd = dd
            mdd_from = peak_day
            mdd_to = d
    return {"mdd": float(mdd), "from_day": mdd_from, "to_day": mdd_to}


def _sharpe_annualized(pnls_by_day: Dict[str, float], *, bankroll_ref: float) -> Optional[float]:
    """
    Sharpe anualizado (sqrt(252)) usando retornos diários r = pnl / bankroll_ref.
    """
    try:
        br = float(bankroll_ref)
        if br <= 0:
            return None
        rs = [float(v) / br for _, v in sorted(pnls_by_day.items())]
        if len(rs) < 5:
            return None
        import statistics
        import math

        mu = statistics.fmean(rs)
        sd = statistics.pstdev(rs)
        if sd <= 0:
            return None
        return float((mu / sd) * math.sqrt(252.0))
    except Exception:
        return None

def _read_jsonl_last(path: Path, last: int) -> list[str]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if last > 0 and len(lines) > last:
        return lines[-last:]
    return lines


def _parse_iso_dt(s: str) -> Optional[datetime]:
    try:
        t = str(s or "").strip()
        if not t:
            return None
        if t.endswith("Z"):
            t = t[:-1] + "+00:00"
        dt = datetime.fromisoformat(t)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _executor_gaps_summary(lines: list[str]) -> Dict[str, Any]:
    """
    Sumário simples de "downtime" por gaps no JSONL do executor.
    Interpretação: gaps grandes sugerem paradas/restarts ou ausência de tráfego.
    """
    ts = []
    for ln in lines or []:
        try:
            obj = json.loads(ln)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        res = obj.get("result") if isinstance(obj.get("result"), dict) else {}
        req = obj.get("request") if isinstance(obj.get("request"), dict) else {}
        dt = _parse_iso_dt(str(res.get("created_at") or req.get("created_at") or ""))
        if dt:
            ts.append(dt)
    ts.sort()
    if len(ts) < 2:
        return {"n": len(ts), "max_gap_s": None, "gaps_gt_300s": 0, "gaps_gt_900s": 0}
    gaps = [(ts[i] - ts[i - 1]).total_seconds() for i in range(1, len(ts))]
    return {
        "n": int(len(ts)),
        "first_ts": ts[0].isoformat(),
        "last_ts": ts[-1].isoformat(),
        "max_gap_s": float(max(gaps)) if gaps else None,
        "gaps_gt_300s": int(sum(1 for g in gaps if g > 300.0)),
        "gaps_gt_900s": int(sum(1 for g in gaps if g > 900.0)),
    }


def _filter_executor_jsonl_lines_window(lines: list[str], *, since_utc: datetime, until_utc: Optional[datetime] = None) -> list[str]:
    """
    Filtra linhas do executor_jsonl por timestamp (created_at/finished_at) para aproximar "últimas 24h".
    Observação: JSONL não é heartbeat; se o pipeline ficou sem tráfego, a janela pode retornar N baixo.
    """
    until = until_utc or datetime.now(timezone.utc)
    out: list[str] = []
    for ln in lines or []:
        try:
            obj = json.loads(ln)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        res = obj.get("result") if isinstance(obj.get("result"), dict) else {}
        req = obj.get("request") if isinstance(obj.get("request"), dict) else {}
        dt = _parse_iso_dt(str(res.get("created_at") or res.get("finished_at") or req.get("created_at") or ""))
        if not dt:
            continue
        if since_utc <= dt <= until:
            out.append(ln)
    return out


def _executor_gaps_summary_window(lines: list[str], *, since_utc: datetime, until_utc: Optional[datetime] = None) -> Dict[str, Any]:
    """
    Mesmo sumário de gaps, mas focado em uma janela (ex.: últimas 24h).

    Observação: como o JSONL é escrito apenas quando há requisição/resposta, não é um heartbeat.
    Então isso mede "silêncio" do pipeline (executor sem tráfego, audit/bridge parados, ou executor down).
    Para aproximar "tempo em silêncio", somamos (gap - 900s) para gaps>15min (proxy de downtime acima do limiar).
    """
    until = until_utc or datetime.now(timezone.utc)
    ts_all: list[datetime] = []
    for ln in lines or []:
        try:
            obj = json.loads(ln)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        res = obj.get("result") if isinstance(obj.get("result"), dict) else {}
        req = obj.get("request") if isinstance(obj.get("request"), dict) else {}
        dt = _parse_iso_dt(str(res.get("created_at") or req.get("created_at") or ""))
        if dt:
            ts_all.append(dt)
    ts_all.sort()
    if not ts_all:
        return {
            "since_utc": since_utc.isoformat(),
            "until_utc": until.isoformat(),
            "n": 0,
            "first_ts": None,
            "last_ts": None,
            "max_gap_s": None,
            "gaps_gt_300s": 0,
            "gaps_gt_900s": 0,
            "silence_over_15m_s": 0.0,
            "silence_over_15m_pct": None,
        }
    # inclui 1 ponto anterior ao since (se existir) para captar gap cruzando a borda da janela
    prev = None
    for dt in reversed(ts_all):
        if dt < since_utc:
            prev = dt
            break
    tsw = [dt for dt in ts_all if since_utc <= dt <= until]
    if prev:
        tsw = [prev] + tsw
    tsw.sort()
    if len(tsw) < 2:
        return {
            "since_utc": since_utc.isoformat(),
            "until_utc": until.isoformat(),
            "n": int(len(tsw)),
            "first_ts": tsw[0].isoformat() if tsw else None,
            "last_ts": tsw[-1].isoformat() if tsw else None,
            "max_gap_s": None,
            "gaps_gt_300s": 0,
            "gaps_gt_900s": 0,
            "silence_over_15m_s": 0.0,
            "silence_over_15m_pct": None,
        }
    gaps = [(tsw[i] - tsw[i - 1]).total_seconds() for i in range(1, len(tsw))]
    over_15 = [g for g in gaps if g > 900.0]
    silence_over = float(sum((g - 900.0) for g in over_15)) if over_15 else 0.0
    win_s = max(1.0, (until - since_utc).total_seconds())
    return {
        "since_utc": since_utc.isoformat(),
        "until_utc": until.isoformat(),
        "n": int(len(tsw)),
        "first_ts": tsw[0].isoformat(),
        "last_ts": tsw[-1].isoformat(),
        "max_gap_s": float(max(gaps)) if gaps else None,
        "gaps_gt_300s": int(sum(1 for g in gaps if g > 300.0)),
        "gaps_gt_900s": int(sum(1 for g in gaps if g > 900.0)),
        "silence_over_15m_s": float(silence_over),
        "silence_over_15m_pct": float(silence_over / win_s * 100.0) if win_s > 0 else None,
    }


def _mem_available_mib() -> Optional[float]:
    try:
        p = Path("/proc/meminfo")
        if not p.exists():
            return None
        mem_av = None
        for ln in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            if ln.startswith("MemAvailable:"):
                parts = ln.split()
                if len(parts) >= 2:
                    mem_av = float(parts[1])  # kB
                    break
        if mem_av is None:
            return None
        return float(mem_av / 1024.0)
    except Exception:
        return None


def _vcpu_count() -> Optional[int]:
    try:
        n = os.cpu_count()
        if n is None:
            return None
        n2 = int(n)
        return n2 if n2 > 0 else None
    except Exception:
        return None


def _safe_div(a: Any, b: Any) -> Optional[float]:
    try:
        aa = float(a)
        bb = float(b)
        if bb == 0:
            return None
        return float(aa / bb)
    except Exception:
        return None


def _load_wf_policy_last_step(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if not path.exists():
            return None
        d = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(d, dict):
            return None
        steps = d.get("steps") if isinstance(d.get("steps"), list) else []
        last = steps[-1] if steps and isinstance(steps[-1], dict) else None
        return last if isinstance(last, dict) else None
    except Exception:
        return None


def _pick_prev_policy_file(policy_dir: Path, *, cur_day: str) -> Optional[Path]:
    try:
        if not policy_dir.exists():
            return None
        xs = sorted([p for p in policy_dir.glob("wf_policy_*.json") if p.is_file()])
        if not xs:
            return None
        def _day_from_name(p: Path) -> Optional[str]:
            # aceita wf_policy_YYYYMMDD.json ou wf_policy_YYYYMMDD_HHMMSS.json
            s = p.name.replace("wf_policy_", "").replace(".json", "")
            s = s.split("_", 1)[0]
            if len(s) == 8 and s.isdigit():
                return s
            return None

        # preferir o snapshot canônico do dia (sem sufixo), se existir
        by_day: Dict[str, Dict[str, Optional[Path]]] = {}
        for p in xs:
            d = _day_from_name(p)
            if not d:
                continue
            slot = by_day.setdefault(d, {"canonical": None, "fallback": None})
            if p.name == f"wf_policy_{d}.json":
                slot["canonical"] = p
            else:
                # fallback: guarda o "maior" lexicográfico do dia (normalmente o mais recente)
                cur = slot.get("fallback")
                if cur is None or p.name > cur.name:
                    slot["fallback"] = p

        prev_day = None
        for d in sorted(by_day.keys()):
            if str(d) < str(cur_day):
                prev_day = d
        if not prev_day:
            return None
        slot = by_day.get(prev_day) or {}
        return slot.get("canonical") or slot.get("fallback")
    except Exception:
        return None


def _parse_iso_dt_best(s: Any) -> Optional[datetime]:
    try:
        if s is None:
            return None
        t = str(s).strip()
        if not t:
            return None
        if t.endswith("Z"):
            t = t[:-1] + "+00:00"
        dt = datetime.fromisoformat(t)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None

def _env_bool(k: str, default: str = "0") -> bool:
    v = str(os.getenv(k, default) or "").strip()
    return v in ("1", "true", "True", "yes", "YES", "on", "ON")


def _env_float(k: str, default: str) -> float:
    try:
        return float(os.getenv(k, default))
    except Exception:
        return float(default)


def _count_err_substr(audit_rep: Optional[Dict[str, Any]], needle: str) -> int:
    """
    Conta ocorrências em audit_status_kpis.error_rows cujo api_error contém `needle` (case-insensitive).
    """
    try:
        if not isinstance(audit_rep, dict):
            return 0
        xs = audit_rep.get("error_rows") or []
        if not isinstance(xs, list) or not needle:
            return 0
        nd = str(needle).lower()
        tot = 0
        for it in xs:
            if not isinstance(it, dict):
                continue
            err = str(it.get("api_error") or "").lower()
            if nd in err:
                tot += int(it.get("n") or 0)
        return int(tot)
    except Exception:
        return 0


def _sum_status(audit_rep: Optional[Dict[str, Any]], status: str) -> int:
    try:
        if not isinstance(audit_rep, dict):
            return 0
        tot = 0
        for v in audit_rep.get("by_version") or []:
            if not isinstance(v, dict):
                continue
            sc = v.get("status_counts") if isinstance(v.get("status_counts"), dict) else {}
            tot += int(sc.get(status) or 0)
        return int(tot)
    except Exception:
        return 0


def _sum_total(audit_rep: Optional[Dict[str, Any]]) -> int:
    try:
        if not isinstance(audit_rep, dict):
            return 0
        tot = 0
        for v in audit_rep.get("by_version") or []:
            if isinstance(v, dict):
                tot += int(v.get("total") or 0)
        return int(tot)
    except Exception:
        return 0


def _sum_ok_valid(audit_rep: Optional[Dict[str, Any]]) -> int:
    try:
        if not isinstance(audit_rep, dict):
            return 0
        tot = 0
        for v in audit_rep.get("by_version") or []:
            if isinstance(v, dict):
                tot += int(v.get("ok_valid") or 0)
        return int(tot)
    except Exception:
        return 0


def _sum_ok(audit_rep: Optional[Dict[str, Any]]) -> int:
    try:
        if not isinstance(audit_rep, dict):
            return 0
        tot = 0
        for v in audit_rep.get("by_version") or []:
            if isinstance(v, dict):
                sc = v.get("status_counts") if isinstance(v.get("status_counts"), dict) else {}
                tot += int(sc.get("OK") or 0)
        return int(tot)
    except Exception:
        return 0


def _fmt_status(ok: Optional[bool]) -> str:
    if ok is None:
        return "—"
    return "OK" if ok else "FAIL"


def _telegram_send_document(token: str, chat_id: str, *, file_path: Path, caption: str) -> bool:
    url = f"https://api.telegram.org/bot{token}/sendDocument"
    with file_path.open("rb") as f:
        files = {"document": (file_path.name, f, "application/pdf")}
        data = {"chat_id": chat_id, "caption": caption[:900]}
        r = requests.post(url, data=data, files=files, timeout=60)
        return bool(r.ok)


@dataclass
class DailyReportCfg:
    out_dir: Path = Path("logs/daily_reports")
    report_tz: str = "America/Sao_Paulo"
    # Alinhar com o relatório “v38” por default
    versions: str = os.getenv("DAILY_OOS_VERSIONS", "v4.0-api,v5.0-ws-only,v5.1-ws-gate-lay")
    direction: str = os.getenv("DAILY_OOS_DIRECTION", "up")
    # Alinha com o relatório “atual” (ex.: 21d) se o usuário não setar nada.
    lookback_days: str = os.getenv("DAILY_OOS_LOOKBACK_DAYS", "21")
    no_auto_exclude_days: bool = (os.getenv("DAILY_NO_AUTO_EXCLUDE_DAYS", "0").strip() in ("1", "true", "True", "yes", "YES"))
    report_mode: str = os.getenv("DAILY_REPORT_MODE", "oos_first")
    wf_policy_current: Path = Path(os.getenv("DAILY_WF_POLICY_CURRENT", "logs/wf_policy_current.json"))
    wf_policy_history_dir: Path = Path(os.getenv("DAILY_WF_POLICY_HISTORY_DIR", "logs/policy_history"))
    wf_policy_history_jsonl: Path = Path(os.getenv("DAILY_WF_POLICY_HISTORY_JSONL", "logs/wf_policy_history.jsonl"))
    # Walk-forward knobs (para casar com versões como leaguePre / AHgatePre / expanding)
    wf_train_mode: str = os.getenv("DAILY_WF_TRAIN_MODE", "expanding")
    wf_train_days: str = os.getenv("DAILY_WF_TRAIN_DAYS", "2")
    wf_test_days: str = os.getenv("DAILY_WF_TEST_DAYS", "2")
    wf_step_days: str = os.getenv("DAILY_WF_STEP_DAYS", "2")
    wf_key_by_league: bool = (os.getenv("DAILY_WF_KEY_BY_LEAGUE", "1").strip() in ("1", "true", "True", "yes", "YES"))
    wf_key_by_league_scope: str = os.getenv("DAILY_WF_KEY_BY_LEAGUE_SCOPE", "pre")
    # Estatística exploratória no OOS (deve ficar OFF no daily 19h)
    wf_experimental_stats: bool = (os.getenv("DAILY_WF_EXPERIMENTAL_STATS", "0").strip() in ("1", "true", "True", "yes", "YES"))
    wf_ah_max_abs_line: str = os.getenv("DAILY_WF_AH_MAX_ABS_LINE", "2.0")
    wf_ah_scope: str = os.getenv("DAILY_WF_AH_SCOPE", "pre")
    wf_liquidity_mode: str = os.getenv("DAILY_WF_LIQUIDITY_MODE", "none")
    wf_liquidity_scope: str = os.getenv("DAILY_WF_LIQUIDITY_SCOPE", "pre")
    wf_min_matches: str = os.getenv("DAILY_WF_MIN_MATCHES", "0")
    wf_shrinkage: bool = (os.getenv("DAILY_WF_SHRINKAGE", "1").strip() in ("1", "true", "True", "yes", "YES"))
    wf_exclude_exec_buckets_back: str = os.getenv("DAILY_WF_EXCLUDE_EXEC_BUCKETS_BACK", "10-20s")
    wf_exclude_exec_buckets_lay: str = os.getenv("DAILY_WF_EXCLUDE_EXEC_BUCKETS_LAY", "")
    # Sizing no WF (útil para simular in-match governado por budget/caps, sem trocar policy do robô)
    wf_scheme_pre: str = os.getenv("DAILY_WF_SCHEME_PRE", "").strip()
    wf_scheme_in: str = os.getenv("DAILY_WF_SCHEME_IN", "").strip()
    wf_flat_stake_back: str = os.getenv("DAILY_WF_FLAT_STAKE_BACK", "").strip()
    # Importante: o default do analyzer é 1.0; para sensibilidade de banca (Lay in-match FLAT),
    # isso pode "saturar" lucro/turnover. Por default operacional, usamos 50.0 (override via env).
    wf_flat_liab_lay: str = os.getenv("DAILY_WF_FLAT_LIAB_LAY", "50").strip()
    # Budget por match_id no WF (permite rodar manual com EQ 4%/4% cap33% sem mexer no agendado das 19h)
    wf_budget_back_frac: str = os.getenv("DAILY_WF_BUDGET_BACK_FRAC", "").strip()
    wf_budget_lay_frac: str = os.getenv("DAILY_WF_BUDGET_LAY_FRAC", "").strip()
    wf_budget_cap_signal_frac: str = os.getenv("DAILY_WF_BUDGET_CAP_SIGNAL_FRAC", "").strip()
    wf_budget_risk_mode: str = os.getenv("DAILY_WF_BUDGET_RISK_MODE", "").strip()
    # Estudo rápido: sweep de caps absolutos (stake médio) no OOS, para curva lucro×cap (1D + grid 2D).
    wf_sweep_stakes: bool = (os.getenv("DAILY_WF_SWEEP_STAKES", "0").strip() in ("1", "true", "True", "yes", "YES"))
    wf_sweep_back_caps: str = os.getenv("DAILY_WF_SWEEP_BACK_CAPS", "").strip()
    wf_sweep_lay_caps: str = os.getenv("DAILY_WF_SWEEP_LAY_CAPS", "").strip()
    wf_sweep_grid_in: bool = (os.getenv("DAILY_WF_SWEEP_GRID_IN", "1").strip() in ("1", "true", "True", "yes", "YES"))
    # Escala de banca/sizing (manter “10k etc.”)
    kelly_bankroll: str = os.getenv("DAILY_KELLY_BANKROLL", "10000")
    # Grid default para sempre gerar sensibilidade (pequeno o bastante para ser barato).
    wf_bankroll_grid: str = os.getenv("DAILY_WF_BANKROLL_GRID", "10000,50000,100000,500000,1000000,1500000,3000000,5000000").strip()
    executor_jsonl: Path = Path(os.getenv("EXECUTOR_JSONL", "logs/executor_live.jsonl"))
    exec_kpi_last: int = int(os.getenv("DAILY_EXEC_KPI_LAST", "50000"))
    send_telegram: bool = (os.getenv("DAILY_REPORT_TELEGRAM", "1").strip() not in ("0", "false", "False", "no", "NO"))
    skip_accounting: bool = (os.getenv("DAILY_SKIP_ACCOUNTING", "0").strip() in ("1", "true", "True", "yes", "YES"))
    skip_oos: bool = (os.getenv("DAILY_SKIP_OOS", "0").strip() in ("1", "true", "True", "yes", "YES"))

    def __post_init__(self) -> None:
        # Releitura de env em runtime (importante quando rodando manualmente e carregando .env em main()).
        self.versions = os.getenv("DAILY_OOS_VERSIONS", self.versions)
        self.direction = os.getenv("DAILY_OOS_DIRECTION", self.direction)
        self.lookback_days = os.getenv("DAILY_OOS_LOOKBACK_DAYS", self.lookback_days)
        self.report_mode = os.getenv("DAILY_REPORT_MODE", self.report_mode)
        self.wf_policy_current = Path(os.getenv("DAILY_WF_POLICY_CURRENT", str(self.wf_policy_current)))
        self.wf_policy_history_dir = Path(os.getenv("DAILY_WF_POLICY_HISTORY_DIR", str(self.wf_policy_history_dir)))
        self.wf_policy_history_jsonl = Path(os.getenv("DAILY_WF_POLICY_HISTORY_JSONL", str(self.wf_policy_history_jsonl)))
        self.executor_jsonl = Path(os.getenv("EXECUTOR_JSONL", str(self.executor_jsonl)))
        self.wf_exclude_exec_buckets_back = os.getenv("DAILY_WF_EXCLUDE_EXEC_BUCKETS_BACK", self.wf_exclude_exec_buckets_back)
        self.wf_exclude_exec_buckets_lay = os.getenv("DAILY_WF_EXCLUDE_EXEC_BUCKETS_LAY", self.wf_exclude_exec_buckets_lay)
        self.wf_scheme_pre = os.getenv("DAILY_WF_SCHEME_PRE", self.wf_scheme_pre).strip()
        self.wf_scheme_in = os.getenv("DAILY_WF_SCHEME_IN", self.wf_scheme_in).strip()
        self.wf_flat_stake_back = os.getenv("DAILY_WF_FLAT_STAKE_BACK", self.wf_flat_stake_back).strip()
        self.wf_flat_liab_lay = os.getenv("DAILY_WF_FLAT_LIAB_LAY", self.wf_flat_liab_lay).strip()
        self.wf_budget_back_frac = os.getenv("DAILY_WF_BUDGET_BACK_FRAC", self.wf_budget_back_frac).strip()
        self.wf_budget_lay_frac = os.getenv("DAILY_WF_BUDGET_LAY_FRAC", self.wf_budget_lay_frac).strip()
        self.wf_budget_cap_signal_frac = os.getenv("DAILY_WF_BUDGET_CAP_SIGNAL_FRAC", self.wf_budget_cap_signal_frac).strip()
        self.wf_budget_risk_mode = os.getenv("DAILY_WF_BUDGET_RISK_MODE", self.wf_budget_risk_mode).strip()
        self.wf_sweep_stakes = (os.getenv("DAILY_WF_SWEEP_STAKES", "1" if self.wf_sweep_stakes else "0").strip() in ("1", "true", "True", "yes", "YES"))
        self.wf_sweep_back_caps = os.getenv("DAILY_WF_SWEEP_BACK_CAPS", self.wf_sweep_back_caps).strip()
        self.wf_sweep_lay_caps = os.getenv("DAILY_WF_SWEEP_LAY_CAPS", self.wf_sweep_lay_caps).strip()
        self.wf_sweep_grid_in = (os.getenv("DAILY_WF_SWEEP_GRID_IN", "1" if self.wf_sweep_grid_in else "0").strip() in ("1", "true", "True", "yes", "YES"))
        try:
            self.exec_kpi_last = int(os.getenv("DAILY_EXEC_KPI_LAST", str(self.exec_kpi_last)))
        except Exception:
            pass
        self.skip_accounting = (os.getenv("DAILY_SKIP_ACCOUNTING", "1" if self.skip_accounting else "0").strip() in ("1", "true", "True", "yes", "YES"))
        self.skip_oos = (os.getenv("DAILY_SKIP_OOS", "1" if self.skip_oos else "0").strip() in ("1", "true", "True", "yes", "YES"))


async def run_daily_full(cfg: DailyReportCfg) -> Dict[str, Any]:
    ts = _utcnow()
    day = ts.astimezone(timezone.utc).strftime("%Y%m%d")
    day_dir = cfg.out_dir / day
    day_dir.mkdir(parents=True, exist_ok=True)

    # 1) Accounting snapshot + report
    acct_out = day_dir / "accounting_daily_report.json"
    acct: Dict[str, Any] = {}
    if cfg.skip_accounting:
        acct = {"ts": ts.isoformat(), "skipped": True, "error": "ACCOUNTING_SKIPPED (DAILY_SKIP_ACCOUNTING=1)"}
        try:
            acct_out.write_text(json.dumps(acct, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
    else:
        try:
            acct = await run_acct_daily(
                AcctDailyCfg(
                    out_dir=Path(os.getenv("ACCOUNTING_OUT_DIR", "logs/accounting")),
                    jsonl=Path(os.getenv("ACCOUNTING_JSONL", "logs/accounting_snapshots.jsonl")),
                    tz_name=str(os.getenv("REPORT_TZ", cfg.report_tz)),
                    report_out=acct_out,
                    print_json=False,
                )
            )
        except Exception as e:
            # Não aborta o daily: ainda queremos OOS + KPIs + aderência mesmo sem login no accounting.
            acct = {"ts": ts.isoformat(), "error": f"ACCOUNTING_FAILED: {str(e)[:200]}"}
            try:
                acct_out.write_text(json.dumps(acct, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass

    # 2) Execution KPIs (all + success-only)
    exec_lines = []
    if cfg.executor_jsonl.exists():
        exec_lines = _read_jsonl_last(cfg.executor_jsonl, int(cfg.exec_kpi_last))
    kpi_all = compute_kpis_from_lines(exec_lines, path=str(cfg.executor_jsonl))
    kpi_ok = compute_kpis_from_lines(exec_lines, path=str(cfg.executor_jsonl), only_status=["LIVE_OK", "DRY_OK"])
    (day_dir / "execution_kpis_all.json").write_text(json.dumps(kpi_all, ensure_ascii=False, indent=2), encoding="utf-8")
    (day_dir / "execution_kpis_ok.json").write_text(json.dumps(kpi_ok, ensure_ascii=False, indent=2), encoding="utf-8")

    # recorte 24h (para prontidão LIVE): gaps/latência devem ser comparáveis a thresholds (≤8 gaps, p90≤8s, etc.)
    exec_lines_24h: list[str] = []
    try:
        since24 = _utcnow() - timedelta(hours=24.0)
        exec_lines_24h = _filter_executor_jsonl_lines_window(exec_lines, since_utc=since24)
    except Exception:
        exec_lines_24h = []
    kpi_ok_24h = compute_kpis_from_lines(exec_lines_24h, path=str(cfg.executor_jsonl), only_status=["LIVE_OK", "DRY_OK"])
    (day_dir / "execution_kpis_ok_24h.json").write_text(json.dumps(kpi_ok_24h, ensure_ascii=False, indent=2), encoding="utf-8")

    # atividade recente (ajuda a diagnosticar "hoje não teve aposta" sem depender do DB)
    exec_activity: Dict[str, Any] = {"last_live_ok_ts": None, "live_ok_1h": 0, "live_ok_6h": 0, "live_ok_24h": 0}
    try:
        nowu = _utcnow()
        cut1 = nowu - timedelta(hours=1.0)
        cut6 = nowu - timedelta(hours=6.0)
        cut24 = nowu - timedelta(hours=24.0)
        last_live = None
        c1 = c6 = c24 = 0
        for ln in exec_lines:
            try:
                obj = json.loads(ln)
            except Exception:
                continue
            res = obj.get("result") if isinstance(obj, dict) else None
            req = obj.get("request") if isinstance(obj, dict) else None
            if not isinstance(res, dict):
                continue
            st = str(res.get("status") or "")
            if st == "HEARTBEAT":
                continue
            if st != "LIVE_OK":
                continue
            ts0 = _parse_iso_dt_best(str(res.get("finished_at") or res.get("created_at") or (req.get("created_at") if isinstance(req, dict) else "") or ""))
            if not ts0:
                continue
            if last_live is None or ts0 > last_live:
                last_live = ts0
            if ts0 >= cut24:
                c24 += 1
            if ts0 >= cut6:
                c6 += 1
            if ts0 >= cut1:
                c1 += 1
        exec_activity = {
            "last_live_ok_ts": last_live.isoformat() if last_live else None,
            "live_ok_1h": int(c1),
            "live_ok_6h": int(c6),
            "live_ok_24h": int(c24),
        }
    except Exception:
        exec_activity = {"last_live_ok_ts": None, "live_ok_1h": 0, "live_ok_6h": 0, "live_ok_24h": 0}

    # 3) Rodar OOS (walk-forward) e exportar policy
    base_md = day_dir / "report_base.md"
    # Histórico de policy:
    # - `wf_policy_YYYYMMDD.json` é o snapshot canônico do dia (não deve ser sobrescrito em reruns).
    # - reruns escrevem `wf_policy_YYYYMMDD_HHMMSS.json` para evitar “revisar o passado”.
    policy_canon = cfg.wf_policy_history_dir / f"wf_policy_{day}.json"
    policy_hist = policy_canon
    if policy_canon.exists():
        policy_hist = cfg.wf_policy_history_dir / f"wf_policy_{day}_{ts.strftime('%H%M%S')}.json"
    bank_sens_json = day_dir / "wf_bank_sensitivity.json"
    cfg.wf_policy_history_dir.mkdir(parents=True, exist_ok=True)

    args = [
        sys.executable,
        str(Path(__file__).resolve().parent.parent / "analyze_contexto_operacao_b808_robust_report.py"),
        "--direction",
        str(cfg.direction),
        "--versions",
        str(cfg.versions),
        "--out",
        str(base_md),
        "--report-mode",
        str(cfg.report_mode),
        "--walkforward",
        "--wf-export-policy-json",
        str(policy_hist),
        "--wf-export-bank-sensitivity-json",
        str(bank_sens_json),
    ]
    if bool(cfg.no_auto_exclude_days):
        args += ["--no-auto-exclude-days"]
    if str(cfg.lookback_days).strip():
        args += ["--lookback-days", str(cfg.lookback_days).strip()]
    if str(cfg.kelly_bankroll).strip():
        args += ["--kelly-bankroll", str(cfg.kelly_bankroll).strip()]
    if str(cfg.wf_bankroll_grid).strip():
        args += ["--wf-bankroll-grid", str(cfg.wf_bankroll_grid).strip()]
    if str(cfg.wf_train_mode).strip():
        args += ["--wf-train-mode", str(cfg.wf_train_mode).strip()]
    if str(cfg.wf_train_days).strip():
        args += ["--wf-train-days", str(cfg.wf_train_days).strip()]
    if str(cfg.wf_test_days).strip():
        args += ["--wf-test-days", str(cfg.wf_test_days).strip()]
    if str(cfg.wf_step_days).strip():
        args += ["--wf-step-days", str(cfg.wf_step_days).strip()]
    if bool(cfg.wf_key_by_league):
        args += ["--wf-key-by-league"]
        if str(cfg.wf_key_by_league_scope).strip():
            args += ["--wf-key-by-league-scope", str(cfg.wf_key_by_league_scope).strip()]
        if bool(cfg.wf_experimental_stats):
            args += ["--wf-experimental-stats"]
    if str(cfg.wf_ah_max_abs_line).strip():
        args += ["--wf-ah-max-abs-line", str(cfg.wf_ah_max_abs_line).strip()]
        if str(cfg.wf_ah_scope).strip():
            args += ["--wf-ah-scope", str(cfg.wf_ah_scope).strip()]
    if str(cfg.wf_liquidity_mode).strip():
        args += ["--wf-liquidity-mode", str(cfg.wf_liquidity_mode).strip()]
        if str(cfg.wf_liquidity_scope).strip():
            args += ["--wf-liquidity-scope", str(cfg.wf_liquidity_scope).strip()]
    if str(cfg.wf_min_matches).strip():
        args += ["--wf-min-matches", str(cfg.wf_min_matches).strip()]
    if bool(cfg.wf_shrinkage):
        args += ["--wf-shrinkage"]
    if str(cfg.wf_exclude_exec_buckets_back).strip():
        args += ["--wf-exclude-exec-buckets-back", str(cfg.wf_exclude_exec_buckets_back).strip()]
    if str(cfg.wf_exclude_exec_buckets_lay).strip():
        args += ["--wf-exclude-exec-buckets-lay", str(cfg.wf_exclude_exec_buckets_lay).strip()]
    # Restrição por lado (evita que o portfólio OOS selecione Back quando a operação é Lay-only).
    try:
        wf_sides = str(os.getenv("DAILY_WF_SIDES", "") or "").strip().lower()
        if wf_sides:
            args += ["--wf-sides", wf_sides]
    except Exception:
        pass

    # Overrides opcionais (rodagem manual): sizing/budget do WF
    if str(cfg.wf_scheme_pre).strip():
        args += ["--wf-scheme-pre", str(cfg.wf_scheme_pre).strip()]
    if str(cfg.wf_scheme_in).strip():
        args += ["--wf-scheme-in", str(cfg.wf_scheme_in).strip()]
    if str(cfg.wf_flat_stake_back).strip():
        args += ["--wf-flat-stake-back", str(cfg.wf_flat_stake_back).strip()]
    if str(cfg.wf_flat_liab_lay).strip():
        args += ["--wf-flat-liab-lay", str(cfg.wf_flat_liab_lay).strip()]
    if str(cfg.wf_budget_back_frac).strip():
        args += ["--wf-budget-back-frac", str(cfg.wf_budget_back_frac).strip()]
    if str(cfg.wf_budget_lay_frac).strip():
        args += ["--wf-budget-lay-frac", str(cfg.wf_budget_lay_frac).strip()]
    if str(cfg.wf_budget_cap_signal_frac).strip():
        args += ["--wf-budget-cap-signal-frac", str(cfg.wf_budget_cap_signal_frac).strip()]
    if str(cfg.wf_budget_risk_mode).strip():
        args += ["--wf-budget-risk-mode", str(cfg.wf_budget_risk_mode).strip()]

    # Sweep de caps absolutos no OOS (nova seção no PDF)
    if bool(cfg.wf_sweep_stakes):
        args += ["--wf-sweep-stakes"]
        if str(cfg.wf_sweep_back_caps).strip():
            args += ["--wf-sweep-back-caps", str(cfg.wf_sweep_back_caps).strip()]
        if str(cfg.wf_sweep_lay_caps).strip():
            args += ["--wf-sweep-lay-caps", str(cfg.wf_sweep_lay_caps).strip()]
        if bool(cfg.wf_sweep_grid_in):
            args += ["--wf-sweep-grid-in"]

    oos_run = {"skipped": False, "ok": True, "returncode": 0, "error": None, "log": str(day_dir / "oos_run.log")}
    if cfg.skip_oos:
        oos_run = {"skipped": True, "ok": False, "returncode": None, "error": "OOS_SKIPPED (DAILY_SKIP_OOS=1)", "log": None}
    else:
        try:
            log_path = Path(str(oos_run["log"]))
            proc = subprocess.run(args, check=False, cwd=str(Path(__file__).resolve().parent.parent), capture_output=True, text=True)
            oos_run["returncode"] = int(proc.returncode)
            if proc.returncode != 0:
                oos_run["ok"] = False
                oos_run["error"] = f"OOS_FAILED: returncode={proc.returncode}"
            # sempre grava log (stdout+stderr) para debug no VPS
            try:
                log_path.write_text((proc.stdout or "") + "\n\n--- STDERR ---\n\n" + (proc.stderr or ""), encoding="utf-8")
            except Exception:
                pass
        except Exception as e:
            oos_run["ok"] = False
            oos_run["error"] = f"OOS_EXCEPTION: {str(e)[:200]}"

    # Atualiza policy_current (atomic replace) e registra histórico (jsonl) apenas se o OOS rodou com sucesso
    if (not cfg.skip_oos) and bool(oos_run.get("ok")) and policy_hist.exists():
        # Preenche o snapshot canônico do dia (best-effort) apenas se ainda não existir.
        # Isso evita que re-runs manuais sobrescrevam o arquivo `wf_policy_YYYYMMDD.json`.
        try:
            if policy_canon and (not policy_canon.exists()):
                tmpc = policy_canon.with_suffix(".tmp")
                tmpc.write_text(policy_hist.read_text(encoding="utf-8"), encoding="utf-8")
                tmpc.replace(policy_canon)
        except Exception:
            pass
        cfg.wf_policy_current.parent.mkdir(parents=True, exist_ok=True)
        tmp = cfg.wf_policy_current.with_suffix(".tmp")
        tmp.write_text(policy_hist.read_text(encoding="utf-8"), encoding="utf-8")
        tmp.replace(cfg.wf_policy_current)

    active_keys = None
    active_keys_base = None
    policy_wf: Optional[Dict[str, Any]] = None
    policy_last_step: Optional[Dict[str, Any]] = None
    if (not cfg.skip_oos) and bool(oos_run.get("ok")) and policy_hist.exists():
        try:
            pol = json.loads(policy_hist.read_text(encoding="utf-8"))
            steps = pol.get("steps") if isinstance(pol, dict) else []
            last = steps[-1] if isinstance(steps, list) and steps else {}
            if isinstance(last, dict):
                active_keys = last.get("active_keys")
                active_keys_base = last.get("active_keys_base")
                policy_last_step = last
            if isinstance(pol, dict) and isinstance(pol.get("wf"), dict):
                policy_wf = pol.get("wf")
            rec = {
                "ts": ts.isoformat(),
                "policy_path": str(policy_hist),
                "policy_current": str(cfg.wf_policy_current),
                "active_keys": active_keys,
                "active_keys_base": active_keys_base,
            }
            cfg.wf_policy_history_jsonl.parent.mkdir(parents=True, exist_ok=True)
            with cfg.wf_policy_history_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass

    # 4) Relatórios auxiliares para seção 0/1 e apêndices (99.x)
    # Adherence: (a) curto para tabelas diárias; (b) longo/acumulado para slippage/combos/contrafactuais
    adh_short_json = day_dir / "oos_adherence_short.json"
    adh_long_json = day_dir / "oos_adherence_long.json"
    exec_min_json = day_dir / "execution_minimal_by_type_24h.json"
    adh_short: Optional[Dict[str, Any]] = None
    adh_long: Optional[Dict[str, Any]] = None
    exec_min: Optional[Dict[str, Any]] = None
    try:
        slip_cf_start_day = str(os.getenv("OOS_ADHERENCE_SLIP_CF_START_DAY", "") or "").strip() or None
        subprocess.run(
            [
                sys.executable,
                "-m",
                "ops.oos_adherence_report",
                "--policy-json",
                str(cfg.wf_policy_current),
                "--executor-jsonl",
                str(cfg.executor_jsonl),
                "--tz",
                "UTC",
                "--days",
                str(os.getenv("DAILY_ADHERENCE_DAYS_TABLE", os.getenv("DAILY_ADHERENCE_DAYS", "7"))),
                *(
                    ["--slippage-cf-start-day", slip_cf_start_day]
                    if slip_cf_start_day
                    else []
                ),
                "--out",
                str(adh_short_json),
            ],
            check=False,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        adh_short = _read_json(adh_short_json)
    except Exception:
        adh_short = None
    try:
        slip_cf_start_day = str(os.getenv("OOS_ADHERENCE_SLIP_CF_START_DAY", "") or "").strip() or None
        subprocess.run(
            [
                sys.executable,
                "-m",
                "ops.oos_adherence_report",
                "--policy-json",
                str(cfg.wf_policy_current),
                "--executor-jsonl",
                str(cfg.executor_jsonl),
                "--tz",
                "UTC",
                "--days",
                str(os.getenv("DAILY_ADHERENCE_DAYS_SLIPPAGE", "0")),
                "--no-per-day",
                *(
                    ["--slippage-cf-start-day", slip_cf_start_day]
                    if slip_cf_start_day
                    else []
                ),
                "--out",
                str(adh_long_json),
            ],
            check=False,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        adh_long = _read_json(adh_long_json)
    except Exception:
        adh_long = None

    # Execução: métricas mínimas por tipo (Back/Lay × Pre/In) — janela curta (horas)
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "ops.execution_minimal_by_type",
                "--executor-jsonl",
                str(cfg.executor_jsonl),
                "--hours",
                str(os.getenv("DAILY_EXEC_MIN_BY_TYPE_HOURS", "24")),
                "--only-status",
                str(os.getenv("DAILY_EXEC_MIN_BY_TYPE_ONLY_STATUS", "LIVE_OK")),
                "--out",
                str(exec_min_json),
            ],
            check=False,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        exec_min = _read_json(exec_min_json)
    except Exception:
        exec_min = None

    audit_json = day_dir / "audit_status_kpis.json"
    audit_rep: Optional[Dict[str, Any]] = None
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "ops.audit_status_kpis",
                "--hours",
                str(os.getenv("DAILY_AUDIT_KPI_HOURS", "24")),
                "--direction",
                str(cfg.direction),
                "--out",
                str(audit_json),
            ],
            check=False,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        audit_rep = _read_json(audit_json)
    except Exception:
        audit_rep = None

    # 5) Montagem do relatório final:
    # Ordem pedida: 0 (Resumo) -> 1 (Resultados reais) -> 2 (OOS) -> 3 (In-sample) -> 99 (apêndices operacionais)
    base_txt = ""
    if base_md.exists():
        try:
            base_txt = base_md.read_text(encoding="utf-8")
        except Exception:
            base_txt = ""
    insample_txt, oos_txt = _split_base_into_insample_and_oos(base_txt)
    oos_as_annex = (os.getenv("DAILY_OOS_AS_ANNEX", "1").strip() not in ("0", "false", "False", "no", "NO"))
    if oos_txt and (not oos_as_annex):
        oos_txt = (
            oos_txt.replace("## 12) OOS walk-forward", "## 2) OOS walk-forward")
            .replace("## 1) OOS walk-forward", "## 2) OOS walk-forward")
        )

    # Accounting: série por dia/mês a partir do CSV (quando disponível)
    acct_series = None
    try:
        bal_csv = Path(str(acct.get("balance_csv") or "")).expanduser()
        if bal_csv.exists():
            tz = timezone.utc
            try:
                from zoneinfo import ZoneInfo  # type: ignore

                tz = ZoneInfo(str(os.getenv("REPORT_TZ", cfg.report_tz)))
            except Exception:
                tz = timezone.utc
            acct_series = compute_pnl_report(bal_csv, tz=tz)
    except Exception:
        acct_series = None

    # --- Seção 0: Resumo / conclusões (executivo) ---
    s0 = []
    s0.append("## 0) Resumo e conclusões (executivo)\n\n")
    if cfg.skip_oos or (isinstance(oos_run, dict) and not bool(oos_run.get("ok"))):
        s0.append("**Status do OOS (walk-forward)**\n\n")
        if cfg.skip_oos:
            s0.append("- **OOS**: **SKIPPED** (`DAILY_SKIP_OOS=1`).\n\n")
        else:
            s0.append(f"- **OOS**: **FAILED** — `{oos_run.get('error')}`\n")
            if oos_run.get("log"):
                s0.append(f"- Log: `{oos_run.get('log')}`\n\n")

    # performance “real” (accounting) quando houver
    if isinstance(acct, dict) and not acct.get("error"):
        s0.append(f"- **Banca real (saldo atual)**: `{acct.get('balance_current')}`\n")
        s0.append(f"- **P&L (hoje / semana / mês)**: `{acct.get('pnl_today')} / {acct.get('pnl_week')} / {acct.get('pnl_month')}`\n")
    else:
        s0.append("- **Accounting**: indisponível (ver apêndice 99.1)\n")

    # lucro "esperado operacional" aproximado: aplica a regra do gate de slippage no subconjunto com placar (contrafactual)
    try:
        per_day = (adh_short or {}).get("per_day") if isinstance(adh_short, dict) else None
        if isinstance(per_day, list) and per_day:
            base = 0.0
            filt = 0.0
            n = 0
            for it in per_day:
                if not isinstance(it, dict):
                    continue
                cf = it.get("slippage_filter_counterfactual")
                if not isinstance(cf, dict):
                    continue
                b = cf.get("back") if isinstance(cf.get("back"), dict) else {}
                l = cf.get("lay") if isinstance(cf.get("lay"), dict) else {}
                try:
                    base += float(b.get("pnl") or 0.0) + float(l.get("pnl") or 0.0)
                    filt += float(b.get("pnl_filtered") or 0.0) + float(l.get("pnl_filtered") or 0.0)
                    n += int(b.get("n") or 0) + int(l.get("n") or 0)
                except Exception:
                    continue
            if n > 0:
                s0.append(
                    f"- **Lucro esperado (com gate de slippage; exec c/ placar)**: `{_fmt_num(filt,2)}` "
                    f"(base `{_fmt_num(base,2)}`, Δ `{_fmt_num(filt-base,2)}`)\n"
                )
    except Exception:
        pass

    # risco/estabilidade operacional (últimas 24h) via audit_status_kpis
    top_errs = []
    try:
        for it in (audit_rep or {}).get("error_rows") or []:
            if not isinstance(it, dict):
                continue
            n = int(it.get("n") or 0)
            if n <= 0:
                continue
            top_errs.append((n, str(it.get("audit_version") or ""), str(it.get("status") or ""), str(it.get("api_error") or "")))
        top_errs.sort(key=lambda x: x[0], reverse=True)
    except Exception:
        top_errs = []
    if top_errs:
        s0.append("\n**Principais causas de perda de throughput (24h)**\n\n")
        for n, ver, st, err in top_errs[:6]:
            err2 = (err[:160] + "…") if len(err) > 160 else err
            s0.append(f"- `{ver}`: **{st} ×{n}** — `{err2}`\n")

    # conversão (audit) e saúde do executor (gaps)
    try:
        if isinstance(audit_rep, dict) and isinstance(audit_rep.get("by_version"), list):
            tot = ok = valid = 0
            for v in audit_rep.get("by_version") or []:
                if not isinstance(v, dict):
                    continue
                sc = v.get("status_counts") if isinstance(v.get("status_counts"), dict) else {}
                tot += int(v.get("total") or 0)
                ok += int(sc.get("OK") or 0)
                valid += int(v.get("ok_valid") or 0)
            if tot > 0:
                s0.append("\n**Conversão (últimas 24h; auditoria DB)**\n\n")
                s0.append(f"- OK/total: **{ok}/{tot}** ({(ok/tot)*100.0:.1f}%)\n")
                s0.append(f"- OK_valid/total: **{valid}/{tot}** ({(valid/tot)*100.0:.1f}%)\n")
    except Exception:
        pass
    try:
        gaps = _executor_gaps_summary(exec_lines)
        if gaps.get("n") and gaps.get("max_gap_s") is not None:
            s0.append("\n**Saúde do executor (amostra lida do JSONL; não é 24h)**\n\n")
            s0.append(f"- Janela: `{gaps.get('first_ts')}` → `{gaps.get('last_ts')}` (n={gaps.get('n')})\n")
            s0.append(
                f"- Maior gap: `{_fmt_num(gaps.get('max_gap_s'),1)}s` | gaps>5min: `{gaps.get('gaps_gt_300s')}`\n"
            )
    except Exception:
        pass

    # gaps em janela fixa (24h) para prontidão LIVE
    gaps24 = None
    try:
        since24 = _utcnow() - timedelta(hours=24.0)
        gaps24 = _executor_gaps_summary_window(exec_lines, since_utc=since24)
        if isinstance(gaps24, dict) and gaps24.get("n"):
            s0.append("\n**Saúde do executor (últimas 24h; proxy por gaps no JSONL)**\n\n")
            s0.append(f"- Janela: `{gaps24.get('since_utc')}` → `{gaps24.get('until_utc')}` (n={gaps24.get('n')})\n")
            s0.append(
                f"- Maior gap: `{_fmt_num(gaps24.get('max_gap_s'),1)}s` | gaps>15min: `{gaps24.get('gaps_gt_900s')}` | "
                f"silêncio>15min (est.): `{_fmt_num(gaps24.get('silence_over_15m_s'),0)}s` ({_fmt_num(gaps24.get('silence_over_15m_pct'),2)}%)\n"
            )
    except Exception:
        gaps24 = None

    # snapshot simples de memória (ajuda a explicar latência/timeouts)
    try:
        mav = _mem_available_mib()
        if mav is not None:
            s0.append("\n**Recursos da VPS (snapshot)**\n\n")
            s0.append(f"- MemAvailable: `{_fmt_num(mav,0)} MiB`\n")
            try:
                vc = _vcpu_count()
                if vc is not None:
                    s0.append(f"- vCPUs (os.cpu_count): `{int(vc)}`\n")
            except Exception:
                pass
    except Exception:
        pass

    # Se o JSONL está stale, a seção "Execução por dia" vai aparecer zerada mesmo que o bridge/audit estejam rodando.
    try:
        exec_last_ts = _parse_iso_dt_best((gaps or {}).get("last_ts"))
        if exec_last_ts:
            age_h = (datetime.now(timezone.utc) - exec_last_ts).total_seconds() / 3600.0
            thr_h = float(os.getenv("DAILY_EXECUTOR_JSONL_STALE_HOURS", "6.0"))
            if age_h > thr_h:
                s0.append(
                    f"\n**Alerta: executor_jsonl possivelmente desatualizado**\n\n"
                    f"- Último registro no `executor_jsonl`: `{exec_last_ts.isoformat()}` (idade ≈ `{_fmt_num(age_h,1)}h`, limiar `{_fmt_num(thr_h,1)}h`).\n"
                    "- Isso explica dias com `Exec rows=0` mesmo com auditoria DB (funil) mostrando volume.\n\n"
                )
    except Exception:
        pass

    # atividade recente (LIVE_OK): diagnostica rapidamente "hoje não teve aposta"
    try:
        last_live_ok = exec_activity.get("last_live_ok_ts") if isinstance(exec_activity, dict) else None
        s0.append("\n**Atividade recente (executor)**\n\n")
        s0.append(
            f"- Último `LIVE_OK`: `{last_live_ok or '—'}` | "
            f"`LIVE_OK` (1h/6h/24h): `{int(exec_activity.get('live_ok_1h') or 0)}/{int(exec_activity.get('live_ok_6h') or 0)}/{int(exec_activity.get('live_ok_24h') or 0)}`\n"
        )
        if int(exec_activity.get("live_ok_6h") or 0) == 0:
            s0.append("- Se isso persistir com auditoria OK no DB, suspeite de sessão/PMM/timeout ou bridge travado (ver checklist abaixo).\n")
        s0.append("\n")
    except Exception:
        pass

    # ------------------------------------------------------------
    # Prontidão para LIVE (go/no-go) — checklist objetivo
    # ------------------------------------------------------------
    s0.append("\n**Prontidão para LIVE (go/no-go)**\n\n")
    allow_live = _env_bool("EXECUTOR_ALLOW_LIVE", "0")
    # thresholds (configuráveis por env; defaults conservadores)
    thr_ok_valid_pct = _env_float("DAILY_LIVE_MIN_OK_VALID_PCT", "5.0")
    thr_api_failed_pct = _env_float("DAILY_LIVE_MAX_API_FAILED_PCT", "20.0")
    thr_stale_pct = _env_float("DAILY_LIVE_MAX_STALE_QUEUE_PCT", "10.0")
    thr_gaps_15m = int(_env_float("DAILY_LIVE_MAX_GAPS_15MIN", "8"))
    thr_p90_ms = int(_env_float("DAILY_LIVE_MAX_CALL_TO_DONE_P90_MS", "8000"))
    thr_open_betslips = int(_env_float("DAILY_LIVE_MAX_TOO_MANY_OPEN_BETSLIPS", "0"))
    thr_no_pmms = int(_env_float("DAILY_LIVE_MAX_NO_PMMS", "0"))

    tot24 = _sum_total(audit_rep)
    ok24 = _sum_ok(audit_rep)
    okv24 = _sum_ok_valid(audit_rep)
    api_failed24 = _sum_status(audit_rep, "API_FAILED")
    stale24 = _sum_status(audit_rep, "STALE_QUEUE_WAIT")
    err_open = _count_err_substr(audit_rep, "too_many_open_betslips")
    err_pmms = _count_err_substr(audit_rep, "no pmms received")
    pmm_consults_24h = None
    no_pmms_24h = None
    no_pmms_rate_24h = None
    pmm_ws_diag = None
    try:
        blk = (audit_rep or {}).get("pmm") if isinstance((audit_rep or {}).get("pmm"), dict) else {}
        tot = blk.get("total") if isinstance(blk.get("total"), dict) else {}
        pmm_consults_24h = int(tot.get("pmm_consults")) if tot.get("pmm_consults") is not None else None
        no_pmms_24h = int(tot.get("no_pmms")) if tot.get("no_pmms") is not None else None
        no_pmms_rate_24h = _safe_float(tot.get("no_pmms_rate_pct"))
        pmm_ws_diag = blk.get("ws_diag") if isinstance(blk.get("ws_diag"), dict) else None
    except Exception:
        pmm_consults_24h = None
        no_pmms_24h = None
        no_pmms_rate_24h = None
        pmm_ws_diag = None

    ok_valid_pct = (100.0 * okv24 / tot24) if tot24 > 0 else None
    api_failed_pct = (100.0 * api_failed24 / tot24) if tot24 > 0 else None
    stale_pct = (100.0 * stale24 / tot24) if tot24 > 0 else None

    # latência p90 (somente sucessos) — recorte 24h (do JSONL) para ser comparável ao checklist de LIVE
    p90_call = None
    p50_call = None
    p90_call_24h = None
    p50_call_24h = None
    n_succ_24h = None
    p50_queue_24h = None
    p90_queue_24h = None
    p50_queue_all = None
    try:
        blk = ((kpi_ok.get("timing_ms") or {}).get("call_to_done") or {})
        p90_call = int(blk.get("p90") or 0) or None
        p50_call = int(blk.get("p50") or 0) or None
    except Exception:
        p90_call = None
        p50_call = None
    try:
        blkq = ((kpi_ok.get("timing_ms") or {}).get("queue_delay") or {})
        p50_queue_all = int(blkq.get("p50") or 0) or None
    except Exception:
        p50_queue_all = None
    try:
        blk24 = ((kpi_ok_24h.get("timing_ms") or {}).get("call_to_done") or {}) if isinstance(kpi_ok_24h, dict) else {}
        p90_call_24h = int(blk24.get("p90") or 0) or None
        p50_call_24h = int(blk24.get("p50") or 0) or None
        n_succ_24h = int(blk24.get("n") or 0) if blk24.get("n") is not None else None
    except Exception:
        p90_call_24h = None
        p50_call_24h = None
        n_succ_24h = None
    try:
        blkq24 = ((kpi_ok_24h.get("timing_ms") or {}).get("queue_delay") or {}) if isinstance(kpi_ok_24h, dict) else {}
        p50_queue_24h = int(blkq24.get("p50") or 0) or None
        p90_queue_24h = int(blkq24.get("p90") or 0) or None
    except Exception:
        p50_queue_24h = None
        p90_queue_24h = None

    gaps15 = None
    try:
        # força janela 24h; se não houver amostra, deixa None (não compara com threshold).
        src = gaps24 if isinstance(gaps24, dict) else None
        gaps15 = int(src.get("gaps_gt_900s")) if isinstance(src, dict) and src.get("gaps_gt_900s") is not None else None
    except Exception:
        gaps15 = None

    # checks
    chk_allow = bool(allow_live)
    chk_okv = (ok_valid_pct is not None and float(ok_valid_pct) >= float(thr_ok_valid_pct))
    chk_api = (api_failed_pct is not None and float(api_failed_pct) <= float(thr_api_failed_pct))
    chk_stale = (stale_pct is not None and float(stale_pct) <= float(thr_stale_pct))
    chk_gap = (gaps15 is None) or (int(gaps15) <= int(thr_gaps_15m))
    chk_p90 = (p90_call_24h is None) or (int(p90_call_24h) <= int(thr_p90_ms))
    chk_open = int(err_open) <= int(thr_open_betslips)
    chk_pmms = int(err_pmms) <= int(thr_no_pmms)

    s0.append("| Critério | Atual | Alvo | Status |\n|---|---:|---:|---|\n")
    s0.append(f"| Live liberado (`EXECUTOR_ALLOW_LIVE`) | `{allow_live}` | `True` | **{_fmt_status(chk_allow)}** |\n")
    s0.append(f"| OK_valid/total (24h, DB) | {_fmt_num(ok_valid_pct,1)}% | ≥{_fmt_num(thr_ok_valid_pct,1)}% | **{_fmt_status(chk_okv)}** |\n")
    s0.append(f"| API_FAILED/total (24h, DB) | {_fmt_num(api_failed_pct,1)}% | ≤{_fmt_num(thr_api_failed_pct,1)}% | **{_fmt_status(chk_api)}** |\n")
    s0.append(f"| STALE_QUEUE_WAIT/total (24h, DB) | {_fmt_num(stale_pct,1)}% | ≤{_fmt_num(thr_stale_pct,1)}% | **{_fmt_status(chk_stale)}** |\n")
    if pmm_consults_24h is not None and no_pmms_24h is not None and no_pmms_rate_24h is not None:
        s0.append(
            f"| `No PMMs received` (24h, DB) | {int(no_pmms_24h)} / {int(pmm_consults_24h)} ({_fmt_num(no_pmms_rate_24h,2)}%) | ≤{int(thr_no_pmms)} (abs) | **{_fmt_status(chk_pmms)}** |\n"
        )
    else:
        s0.append(f"| `No PMMs received` (24h, DB) | {int(err_pmms)} | ≤{int(thr_no_pmms)} | **{_fmt_status(chk_pmms)}** |\n")
        s0.append("| `No PMMs` / `PMM-consults` (24h, DB) | — | — | — |\n")
    s0.append(f"| `too_many_open_betslips` (24h, DB) | {int(err_open)} | ≤{int(thr_open_betslips)} | **{_fmt_status(chk_open)}** |\n")
    s0.append(
        f"| Latência p90 `call_to_done_ms` (24h; sucessos) | {_fmt_num(p90_call_24h,0)}ms | ≤{int(thr_p90_ms)}ms | **{_fmt_status(chk_p90)}** |\n"
    )
    s0.append(f"| Latência p50 `call_to_done_ms` (24h; sucessos) | {_fmt_num(p50_call_24h,0)}ms | — | — |\n")
    s0.append(f"| n sucessos no JSONL (24h) | {n_succ_24h if n_succ_24h is not None else '—'} | — | — |\n")
    s0.append(f"| Gaps >15min no executor_jsonl (24h; proxy) | {gaps15 if gaps15 is not None else '—'} | ≤{int(thr_gaps_15m)} | **{_fmt_status(chk_gap)}** |\n")
    s0.append("\n")

    # Diagnóstico WS para "No PMMs received": ajuda a distinguir "timeout curto" vs "WS morto".
    try:
        if isinstance(pmm_ws_diag, dict) and int(pmm_ws_diag.get("no_pmms_total") or 0) > 0:
            s0.append("**Diagnóstico de WebSocket (quando ocorre `No PMMs received`)**\n\n")
            thr_ms = int(pmm_ws_diag.get("ws_stale_ms_thr") or 0)
            n0 = int(pmm_ws_diag.get("no_pmms_total") or 0)
            nst = int(pmm_ws_diag.get("no_pmms_ws_stale") or 0)
            p50 = pmm_ws_diag.get("no_pmms_ws_age_ms_median")
            p90 = pmm_ws_diag.get("no_pmms_ws_age_ms_p90")
            mx = pmm_ws_diag.get("no_pmms_ws_age_ms_max")
            s0.append("| Métrica | Valor |\n|---|---:|\n")
            s0.append(f"| `No PMMs` total (24h) | {n0} |\n")
            s0.append(f"| `No PMMs` com WS stale (ws_age_ms≥{thr_ms} ou NULL) | {nst} ({_fmt_num((100.0*nst/n0) if n0 else None,2)}%) |\n")
            s0.append(f"| ws_age_ms p50 / p90 / max | {_fmt_num(p50,0)} / {_fmt_num(p90,0)} / {_fmt_num(mx,0)} |\n")
            s0.append("\n")
    except Exception:
        pass

    # Diagnóstico curto de causa (latência): quando p50 sobe, quase sempre é fila (queue_delay_ms) e/ou timeout de PMM/relógio.
    try:
        if p50_queue_24h is not None and int(p50_queue_24h) > 500:
            s0.append(
                f"**Diagnóstico (latência)**: p50 `queue_delay_ms` (24h) = `{int(p50_queue_24h)}ms` (p90 `{_fmt_num(p90_queue_24h,0)}ms`)"
            )
            if p50_queue_all is not None:
                s0.append(f" vs baseline `{_fmt_num(p50_queue_all,0)}ms`.\n")
            else:
                s0.append(".\n")
            s0.append(
                "- Interpretação: há backlog na fila do executor (workers/concurrency insuficiente ou bursts). Mitigação típica: aumentar `EXECUTOR_WORKERS` e/ou reduzir bursts no bridge.\n\n"
            )
    except Exception:
        pass

    hard_fails = []
    if not chk_allow:
        hard_fails.append("LIVE bloqueado (`EXECUTOR_ALLOW_LIVE=0`)")
    if ok_valid_pct is None or not chk_okv:
        hard_fails.append("conversão `OK_valid/total` baixa")
    if api_failed_pct is None or not chk_api:
        hard_fails.append("taxa de `API_FAILED` alta")
    if stale_pct is None or not chk_stale:
        hard_fails.append("taxa de `STALE_QUEUE_WAIT` alta")
    if not chk_pmms:
        hard_fails.append("erros `No PMMs received` presentes")
    if not chk_open:
        hard_fails.append("erros `too_many_open_betslips` presentes")

    verdict = "APTO (com cautela)" if not hard_fails else "NÃO APTO"
    s0.append(f"**Veredito**: **{verdict}**\n\n")
    if hard_fails:
        s0.append("**Motivos (prioridade)**\n\n")
        for x in hard_fails[:8]:
            s0.append(f"- {x}\n")
        s0.append("\n")
        s0.append(
            "**Próximos passos recomendados (para destravar LIVE)**\n\n"
            "- Atacar `No PMMs received` (timeout/min_wait/idle + estabilidade de sessão) antes de aumentar volume.\n"
            "- Zerar `too_many_open_betslips` (caps/janelas + cleanup agressivo) para evitar bloqueio global.\n"
            "- Reduzir `STALE_QUEUE_WAIT` (fila/concurrency) para não operar atrasado.\n\n"
        )

    # leitura executiva (heurística): onde atacar primeiro
    s0.append(
        "\n**Conclusões operacionais (prioridades)**\n\n"
        "- **Objetivo 1 (conversão)**: reduzir `API_FAILED` (especialmente `No PMMs received`) e `STALE_QUEUE_WAIT` para aumentar taxa de execução sem inflar risco.\n"
        "- **Objetivo 2 (governança de risco)**: consolidar sizing/limites (banca teórica vs banca real) e travas para evitar picos (`too_many_open_betslips`, rate limit, backoff).\n"
        "- **Objetivo 3 (qualidade de entrada)**: acompanhar slippage **com sinal** e seu impacto em ROI por bucket (negativo/flat/positivo) para validar edge e execução.\n\n"
    )

    # ------------------------------------------------------------
    # Carteira (active_keys): delta vs policy anterior + marginais OOS por key
    # ------------------------------------------------------------
    try:
        cur_step = policy_last_step if isinstance(policy_last_step, dict) else None
        cur_keys = set(cur_step.get("active_keys") or []) if cur_step else set()
        prev_pol = _pick_prev_policy_file(cfg.wf_policy_history_dir, cur_day=str(day))
        prev_step = _load_wf_policy_last_step(prev_pol) if prev_pol else None
        prev_keys = set(prev_step.get("active_keys") or []) if prev_step else set()
        if cur_keys and prev_keys:
            entered = sorted(list(cur_keys - prev_keys))
            exited = sorted(list(prev_keys - cur_keys))
            s0.append("\n**Carteira (policy current): keys que entraram/sairam vs policy anterior**\n\n")
            s0.append(f"- Policy anterior: `{prev_pol}`\n")
            s0.append(f"- Δ keys: `{len(prev_keys)}` → `{len(cur_keys)}` (entraram `{len(entered)}`, saíram `{len(exited)}`)\n\n")
            if entered:
                s0.append("- **Entraram**:\n")
                for k in entered[:40]:
                    s0.append(f"  - `{k}`\n")
                if len(entered) > 40:
                    s0.append(f"  - … (+{len(entered)-40})\n")
                s0.append("\n")
            if exited:
                s0.append("- **Saíram**:\n")
                for k in exited[:40]:
                    s0.append(f"  - `{k}`\n")
                if len(exited) > 40:
                    s0.append(f"  - … (+{len(exited)-40})\n")
                s0.append("\n")

            # Marginal por key (OOS): extraído do texto OOS atual (base_md)
            if oos_txt and isinstance(oos_txt, str):
                tbl_md, rows = _extract_md_table(oos_txt, header_startswith="| Combinação (key) | Turnover 30d |")
                hdr = _md_table_header_cols(tbl_md)
                if rows and hdr:
                    idx = {c: i for i, c in enumerate(hdr)}

                    def _f(x: str) -> Optional[float]:
                        """
                        Parse numérico robusto vindo de Markdown (OOS):
                        - Aceita formatos en-US (1,234.56) e pt-BR (1.234,56)
                        - Preserva decimais (não faz replace('.') indiscriminado)
                        - Aceita percentuais ("49.54%")
                        """
                        try:
                            t = str(x or "").strip().replace("−", "-")
                            if not t:
                                return None
                            t = t.replace("%", "").strip()
                            # remove espaços e símbolos comuns
                            t = t.replace(" ", "")
                            # Se contém ambos '.' e ',', inferimos qual é o separador decimal pelo último.
                            if "." in t and "," in t:
                                if t.rfind(".") > t.rfind(","):
                                    # en-US: ',' milhar, '.' decimal
                                    t = t.replace(",", "")
                                else:
                                    # pt-BR: '.' milhar, ',' decimal
                                    t = t.replace(".", "").replace(",", ".")
                            else:
                                # Apenas ',' => assume decimal pt-BR
                                if "," in t and "." not in t:
                                    t = t.replace(",", ".")
                                # Apenas '.' => já é decimal en-US (mantém)
                            return float(t)
                        except Exception:
                            return None

                    mp: Dict[str, Dict[str, Any]] = {}
                    for cols in rows:
                        if not cols or len(cols) < len(hdr):
                            continue
                        k0 = str(cols[idx.get("Combinação (key)", 0)]).strip()
                        if not k0 or k0.lower().startswith("combina"):
                            continue
                        mp[k0] = {
                            "turn_30d": _f(cols[idx.get("Turnover 30d", 1)]) if "Turnover 30d" in idx else None,
                            "share_turn_pct": _f(cols[idx.get("Share turnover", 2)]) if "Share turnover" in idx else None,
                            "profit_30d": _f(cols[idx.get("Lucro 30d (exp.)", 3)]) if "Lucro 30d (exp.)" in idx else None,
                            "share_profit_pct": _f(cols[idx.get("Share lucro", 4)]) if "Share lucro" in idx else None,
                            "roi_turn_pct": _f(cols[idx.get("ROI/turnover 30d", 5)]) if "ROI/turnover 30d" in idx else None,
                        }

                    if mp and (entered or exited):
                        s0.append("**OOS marginal por key (30d exp.) — entraram/sairam**\n\n")
                        s0.append("| Key | Status | Turnover 30d | Share turn | Lucro 30d (exp.) | Share lucro | ROI/turn |\n")
                        s0.append("|---|---|---:|---:|---:|---:|---:|\n")
                        for k in entered:
                            v = mp.get(k) or {}
                            s0.append(
                                f"| `{k}` | entrou | {_fmt_num(v.get('turn_30d'),2)} | {_fmt_num(v.get('share_turn_pct'),2)}% | {_fmt_num(v.get('profit_30d'),2)} | {_fmt_num(v.get('share_profit_pct'),2)}% | {_fmt_num(v.get('roi_turn_pct'),2)}% |\n"
                            )
                        for k in exited:
                            v = mp.get(k) or {}
                            s0.append(
                                f"| `{k}` | saiu | {_fmt_num(v.get('turn_30d'),2)} | {_fmt_num(v.get('share_turn_pct'),2)}% | {_fmt_num(v.get('profit_30d'),2)} | {_fmt_num(v.get('share_profit_pct'),2)}% | {_fmt_num(v.get('roi_turn_pct'),2)}% |\n"
                            )
                        s0.append("\n")

                        # shares top para contexto
                        try:
                            xs = []
                            for k, v in mp.items():
                                st = _f(str(v.get("share_turn_pct") or "")) if isinstance(v, dict) else None
                                sp = _f(str(v.get("share_profit_pct") or "")) if isinstance(v, dict) else None
                                xs.append((k, st, sp))
                            top_turn = sorted([x for x in xs if x[1] is not None], key=lambda x: float(x[1]), reverse=True)[:10]
                            top_prof = sorted([x for x in xs if x[2] is not None], key=lambda x: float(x[2]), reverse=True)[:10]
                            if top_turn:
                                s0.append("- **Top 10 por share de turnover (OOS 30d exp.)**: " + ", ".join([f"`{k}`({float(st):.2f}%)" for k, st, _ in top_turn]) + "\n")
                            if top_prof:
                                s0.append("- **Top 10 por share de lucro (OOS 30d exp.)**: " + ", ".join([f"`{k}`({float(sp):.2f}%)" for k, _, sp in top_prof]) + "\n")
                            s0.append("\n")
                        except Exception:
                            pass

            # Proxy do efeito na sensibilidade de banca: Δ da tabela 12.2b vs report_base do dia anterior (se existir)
            try:
                prev_day = (ts - timedelta(days=1)).astimezone(timezone.utc).strftime("%Y%m%d")
                prev_base = cfg.out_dir / prev_day / "report_base.md"
                if prev_base.exists() and oos_txt:
                    prev_txt = prev_base.read_text(encoding="utf-8", errors="ignore")
                    cur_blk = _extract_md_block(oos_txt, start="### 12.2b Sensibilidade por banca", until_any=["### 12.2c", "### 12.2d", "### 12.3", "## 10)", "## 11)"])
                    prev_blk = _extract_md_block(prev_txt, start="### 12.2b Sensibilidade por banca", until_any=["### 12.2c", "### 12.2d", "### 12.3", "## 10)", "## 11)"])

                    def _parse_sens(bl: str) -> Dict[float, Dict[str, Any]]:
                        out = {}
                        for ln in (bl or "").splitlines():
                            if not ln.startswith("|") or ln.strip().startswith("|---"):
                                continue
                            cols = [c.strip() for c in ln.strip().strip("|").split("|")]
                            if len(cols) < 6 or cols[0].lower().startswith("banca"):
                                continue
                            def _f2(s: str) -> Optional[float]:
                                # reusa o parser robusto acima (mesma semântica)
                                return _f(s)
                            bank = _f2(cols[0])
                            turn = _f2(cols[1])
                            prof = _f2(cols[2])
                            roi = _f2(cols[4])
                            if bank is None:
                                continue
                            out[float(bank)] = {"turn": turn, "profit": prof, "roi": roi}
                        return out

                    curm = _parse_sens(cur_blk)
                    prevm = _parse_sens(prev_blk)
                    inter = sorted(set(curm.keys()) & set(prevm.keys()))
                    if inter:
                        s0.append("**Efeito na sensibilidade de banca (proxy): Δ 12.2b vs dia anterior**\n\n")
                        s0.append(f"- Base anterior: `{prev_base}`\n\n")
                        s0.append("| Banca(ref) | Δ Turnover 30d | Δ Lucro 30d (exp.) | Δ ROI/banca 30d |\n")
                        s0.append("|---:|---:|---:|---:|\n")
                        for b in inter[:12]:
                            s0.append(
                                f"| {int(b)} | {_fmt_num((curm[b].get('turn') or 0.0)-(prevm[b].get('turn') or 0.0),2)} | "
                                f"{_fmt_num((curm[b].get('profit') or 0.0)-(prevm[b].get('profit') or 0.0),2)} | "
                                f"{_fmt_num((curm[b].get('roi') or 0.0)-(prevm[b].get('roi') or 0.0),2)}% |\n"
                            )
                        s0.append("\n")
            except Exception:
                pass
    except Exception:
        pass

    # --- Seção 1: Resultados reais (shadow/live) ---
    s1 = []
    s1.append("## 1) Resultados reais (shadow/live)\n\n")

    # KPIs por recortes (diário/semana/mês) — quando há série
    if acct_series is not None:
        # preferir P&L filtrado (exclui depósitos/saques) quando existir
        pnls = acct_series.pnl_by_day_filtered or acct_series.pnl_by_day
        # semana corrente (por dia)
        try:
            now = _utcnow().astimezone(timezone.utc)
            # usa o maior dia presente como "today" do dataset para evitar mismatch tz
            days_sorted = sorted(pnls.keys())
            today = days_sorted[-1] if days_sorted else now.date().isoformat()
            ws = _week_start_iso(today) or today
            cur_week_days = [d for d in days_sorted if ws <= d <= today]
        except Exception:
            cur_week_days = []
            today = None
            ws = None

        s1.append("**P&L real por dia (semana corrente)**\n\n")
        s1.append("| Dia | P&L |\n|---|---:|\n")
        for d in (cur_week_days or [])[-14:]:
            s1.append(f"| {d} | {_fmt_num(pnls.get(d), 2)} |\n")
        s1.append("\n")

        # Transparência: regras efetivas de seleção e sizing (operacional)
        try:
            s1.append("**Regras efetivas (seleção + sizing) — aplicadas na execução**\n\n")

            # Policy WF (do daily) — define universo/combos e filtros de mercado
            wf = policy_wf if isinstance(policy_wf, dict) else {}
            if wf:
                s1.append("| Policy (WF) | Valor |\n|---|---|\n")
                for k in [
                    "train_mode",
                    "train_days",
                    "test_days",
                    "step_days",
                    "min_matches",
                    "key_by_league",
                    "key_by_league_scope",
                    "ah_max_abs_line",
                    "ah_scope",
                    "liquidity_mode",
                    "liquidity_scope",
                    "liquidity_min_limit",
                ]:
                    if k in wf:
                        s1.append(f"| {k} | `{wf.get(k)}` |\n")
                s1.append("\n")
                if str(wf.get("train_mode") or "").strip().lower() == "expanding":
                    s1.append(
                        "_Nota (WF expanding): `train_days/test_days/step_days` são parâmetros do calendário do walk-forward. "
                        "O **intervalo real** do treino/teste do step vigente é o que aparece logo abaixo em `train=... | test=...`._\n\n"
                    )
            if isinstance(policy_last_step, dict):
                s1.append(f"- Último step (janelas): `train={policy_last_step.get('train')}` | `test={policy_last_step.get('test')}`\n")
                s1.append(f"- Ativas: `keys={len(list(policy_last_step.get('active_keys') or []))}` | `base={len(list(policy_last_step.get('active_keys_base') or []))}`\n\n")

            # Risk params (manual) — governa budget/caps por jogo e sizing base
            rp_path = os.getenv("BRIDGE_RISK_PARAMS_JSON", "").strip()
            rp = _read_json(Path(rp_path)) if rp_path else None
            if isinstance(rp, dict) and rp:
                s1.append("| Risk params (manual) | Valor |\n|---|---|\n")
                for k in [
                    "budget_back_frac",
                    "budget_lay_frac",
                    "cap_signal_frac",
                    "cap_event_back_frac",
                    "cap_event_lay_frac",
                    "stake_pct_of_limit",
                    "stake_cap_abs",
                ]:
                    if k in rp:
                        s1.append(f"| {k} | `{rp.get(k)}` |\n")
                s1.append("\n")

            # Bridge/executor: modo e fontes principais
            s1.append("| Runtime | Valor |\n|---|---|\n")
            for k in [
                "EXECUTOR_ALLOW_LIVE",
                "BRIDGE_USE_WF_BUDGET",
                "BRIDGE_ENFORCE_WF_FILTERS",
                "BRIDGE_WF_RISK_MODE_OVERRIDE",
                "BRIDGE_BANKROLL_REF",
                "BRIDGE_BANKROLL_JSON",
                "BRIDGE_POLICY_JSON",
                "BRIDGE_RISK_PARAMS_JSON",
            ]:
                v = os.getenv(k, "")
                if v:
                    s1.append(f"| {k} | `{v}` |\n")
            s1.append("\n")

            s1.append(
                "_Nota: filtro **AH** é por **|linha|** (ex.: `ah_max_abs_line=2.0` significa |line|≤2.0), não por odds; odds médias >2 podem ocorrer mesmo com AH válido._\n\n"
            )
        except Exception:
            pass

        # drawdown e sharpe (curto, usando a própria janela da série)
        dd = _max_drawdown({k: float(v) for k, v in pnls.items()})
        dd_w = _max_drawdown(_agg_by_week({k: float(v) for k, v in pnls.items()}))
        dd_m = _max_drawdown(_agg_by_month({k: float(v) for k, v in pnls.items()}))
        br_real = None
        try:
            br_real = float(acct.get("balance_current")) if isinstance(acct, dict) and acct.get("balance_current") is not None else None
        except Exception:
            br_real = None
        br_theo = None
        try:
            br_theo = float(cfg.kelly_bankroll) if str(cfg.kelly_bankroll).strip() else None
        except Exception:
            br_theo = None

        s1.append("**Risco/consistência (a partir do P&L diário)**\n\n")
        s1.append("| Métrica | Valor |\n|---|---:|\n")
        s1.append(f"| Max drawdown (diário, monetário) | {_fmt_num(dd.get('mdd'), 2)} |\n")
        s1.append(f"| Max drawdown (semanal, monetário) | {_fmt_num(dd_w.get('mdd'), 2)} |\n")
        s1.append(f"| Max drawdown (mensal, monetário) | {_fmt_num(dd_m.get('mdd'), 2)} |\n")
        if dd.get("from_day") and dd.get("to_day"):
            s1.append(f"| Janela do DD | {dd.get('from_day')} → {dd.get('to_day')} |\n")
        if br_real:
            sh = _sharpe_annualized(pnls, bankroll_ref=float(br_real))
            s1.append(f"| Sharpe anualizado (vs banca real) | {_fmt_num(sh, 2)} |\n")
            # ROI por banca (recortes simples)
            try:
                pnl_week = float(acct.get("pnl_filtered_week") if acct.get("pnl_filtered_week") is not None else acct.get("pnl_week") or 0.0)
                pnl_month = float(acct.get("pnl_filtered_month") if acct.get("pnl_filtered_month") is not None else acct.get("pnl_month") or 0.0)
                s1.append(f"| ROI/banca real (semana) | {_fmt_num((pnl_week/float(br_real))*100.0, 2)}% |\n")
                s1.append(f"| ROI/banca real (mês) | {_fmt_num((pnl_month/float(br_real))*100.0, 2)}% |\n")
            except Exception:
                pass
        if br_theo:
            sh2 = _sharpe_annualized(pnls, bankroll_ref=float(br_theo))
            s1.append(f"| Sharpe anualizado (vs banca teórica) | {_fmt_num(sh2, 2)} |\n")
            try:
                pnl_week = float(acct.get("pnl_filtered_week") if acct.get("pnl_filtered_week") is not None else acct.get("pnl_week") or 0.0)
                pnl_month = float(acct.get("pnl_filtered_month") if acct.get("pnl_filtered_month") is not None else acct.get("pnl_month") or 0.0)
                s1.append(f"| ROI/banca teórica (semana; ref={_fmt_num(br_theo,0)}) | {_fmt_num((pnl_week/float(br_theo))*100.0, 2)}% |\n")
                s1.append(f"| ROI/banca teórica (mês; ref={_fmt_num(br_theo,0)}) | {_fmt_num((pnl_month/float(br_theo))*100.0, 2)}% |\n")
            except Exception:
                pass
        s1.append("\n")

        # semanas fechadas do mês corrente (visão executiva)
        try:
            weeks = _agg_by_week({k: float(v) for k, v in pnls.items()})
            # identifica mês corrente pelo último dia do dataset
            days_sorted = sorted(pnls.keys())
            if days_sorted:
                mk_cur = _month_key(days_sorted[-1])
                # semanas cujo week_start está no mesmo mês e não é a semana corrente
                ws_cur = _week_start_iso(days_sorted[-1])
                rows = [(ws, val) for ws, val in weeks.items() if _month_key(ws) == mk_cur and ws != ws_cur]
                if rows:
                    s1.append("**Semanas anteriores fechadas (mês corrente)**\n\n")
                    s1.append("| Semana (start) | P&L |\n|---|---:|\n")
                    for ws, val in rows[-6:]:
                        s1.append(f"| {ws} | {_fmt_num(val, 2)} |\n")
                    s1.append("\n")
        except Exception:
            pass
    else:
        s1.append("_Sem série de accounting disponível para métricas diárias/Sharpe/DD (ver 99.1)._ \n\n")

    adh_day = adh_short if isinstance(adh_short, dict) else None
    adh_slip = adh_long if isinstance(adh_long, dict) else (adh_day if isinstance(adh_day, dict) else None)

    # execução (contagens + stake médio) via aderência (janela curta)
    if isinstance(exec_min, dict) and isinstance(exec_min.get("by_type"), dict):
        try:
            s1.append("**Execução — métricas mínimas por tipo (Back/Lay × Pre/In; janela curta)**\n\n")
            s1.append(
                "| Tipo | #ordens | #eventos_jsonl | #linhas_api | #jogos | Valor em risco ($) | Ticket médio ($/ordem) | Stake total ($) | #liq | #pend | P&L (liq, $) | ROI% (liq) |\n"
            )
            s1.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
            order = ["Back_Pre", "Back_In", "Lay_Pre", "Lay_In"]
            for k in order:
                r = exec_min.get("by_type", {}).get(k) if isinstance(exec_min.get("by_type"), dict) else None
                if not isinstance(r, dict):
                    continue
                s1.append(
                    f"| {k.replace('_', ' ')} | {int(r.get('n_orders') or 0)} | {int(r.get('n_bets') or 0)} | {int(r.get('n_bet_lines_api') or 0)} | {int(r.get('n_matches') or 0)} | "
                    f"{_fmt_num(r.get('amount_risk_sum'), 2)} | {_fmt_num(r.get('amount_risk_avg_per_order'), 2)} | "
                    f"{_fmt_num(r.get('stake_sum'), 2)} | {int(r.get('n_settled') or 0)} | {int(r.get('n_unsettled') or 0)} | {_fmt_num(r.get('pnl_real_sum_settled') or r.get('pnl_sum_settled'), 2)} | "
                    f"{_fmt_pct(r.get('roi_pct_settled'))} |\n"
                )
            tot = exec_min.get("total") if isinstance(exec_min.get("total"), dict) else {}
            if isinstance(tot, dict) and tot:
                s1.append(
                    f"| **TOTAL** | **{int(tot.get('n_orders') or 0)}** | **{int(tot.get('n_bets') or 0)}** | **{int(tot.get('n_bet_lines_api') or 0)}** | **{int(tot.get('n_matches') or 0)}** | "
                    f"**{_fmt_num(tot.get('amount_risk_sum'), 2)}** | **{_fmt_num(tot.get('amount_risk_avg_per_order'), 2)}** | "
                    f"**{_fmt_num(tot.get('stake_sum'), 2)}** | **{int(tot.get('n_settled') or 0)}** | **{int(tot.get('n_unsettled') or 0)}** | **{_fmt_num(tot.get('pnl_real_sum_settled') or tot.get('pnl_sum_settled'), 2)}** | "
                    f"**{_fmt_pct(tot.get('roi_pct_settled'))}** |\n"
                )
            s1.append("\n")
        except Exception:
            pass
    else:
        # Ajuda a explicar o caso "apostado mas ROI/resultado vazio": aqui usa filtro por status.
        try:
            only_st = str(os.getenv("DAILY_EXEC_MIN_BY_TYPE_ONLY_STATUS", "LIVE_OK"))
            s1.append(
                f"_Execução — métricas mínimas por tipo: sem dados no recorte (provável filtro `DAILY_EXEC_MIN_BY_TYPE_ONLY_STATUS={only_st}`)._\n\n"
            )
        except Exception:
            pass

    if isinstance(adh_day, dict) and isinstance(adh_day.get("per_day"), list) and adh_day.get("per_day"):
        s1.append("**Execução (últimos dias; executor_jsonl + placares quando disponíveis)**\n\n")
        s1.append(
            "| Dia | Exec rows | Sucessos | LIVE_OK | DRY_OK | API_FAILED | N Back | N Lay | Apostado Back ($) | Apostado Lay stake ($) | Apostado Lay liab ($) | P&L total (acct) | P&L (placar) | ROI/$ (placar) | P&L Back | ROI Back | P&L Lay | ROI Lay/liab | ROI Lay/stake |\n"
        )
        s1.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for it in adh_day.get("per_day") or []:
            if not isinstance(it, dict):
                continue
            ex = it.get("execution") if isinstance(it.get("execution"), dict) else {}
            sc = ex.get("status_counts") if isinstance(ex.get("status_counts"), dict) else {}
            back = ex.get("back") if isinstance(ex.get("back"), dict) else {}
            lay = ex.get("lay") if isinstance(ex.get("lay"), dict) else {}
            pnl_back = back.get("pnl_sum")
            pnl_lay = lay.get("pnl_sum")
            pnl_total_placar = (float(pnl_back or 0.0) + float(pnl_lay or 0.0)) if (pnl_back is not None or pnl_lay is not None) else None
            st_back = back.get("stake_sum")
            st_lay = lay.get("stake_sum")
            liab_lay = lay.get("liability_sum")
            # ROI/$ só faz sentido na mesma base do P&L (placar), então usa denominadores "cobertos"
            st_back_cov = back.get("stake_sum_cov")
            st_lay_cov = lay.get("stake_sum_cov")
            st_total_cov = (float(st_back_cov or 0.0) + float(st_lay_cov or 0.0)) if (st_back_cov is not None or st_lay_cov is not None) else None
            roi_dol = (float(pnl_total_placar) / float(st_total_cov) * 100.0) if (pnl_total_placar is not None and st_total_cov and float(st_total_cov) > 0) else None

            # P&L real (accounting) por dia, quando disponível (evita P&L "falso" em dias sem cobertura de ROI)
            dayk = str(it.get("day") or "")
            pnl_acct = None
            try:
                if isinstance(acct, dict):
                    mp = acct.get("pnl_by_day_filtered_recent") if isinstance(acct.get("pnl_by_day_filtered_recent"), dict) else (
                        acct.get("pnl_by_day_recent") if isinstance(acct.get("pnl_by_day_recent"), dict) else {}
                    )
                    if isinstance(mp, dict) and dayk in mp:
                        pnl_acct = float(mp.get(dayk) or 0.0)
            except Exception:
                pnl_acct = None
            s1.append(
                f"| {it.get('day')} | {int(ex.get('n_exec_rows') or 0)} | {int(ex.get('n_exec_success') or 0)} | {int(sc.get('LIVE_OK') or 0)} | {int(sc.get('DRY_OK') or 0)} | "
                f"{int(sc.get('API_FAILED') or 0)} | {int(back.get('n_success') or 0)} | {int(lay.get('n_success') or 0)} | "
                f"{_fmt_num(st_back,2)} | {_fmt_num(st_lay,2)} | {_fmt_num(liab_lay,2)} | "
                f"{_fmt_num(pnl_acct,2)} | {_fmt_num(pnl_total_placar,2)} | {_fmt_pct(roi_dol)} | {_fmt_num(pnl_back,2)} | {_fmt_pct(back.get('roi_pct'))} | "
                f"{_fmt_num(pnl_lay,2)} | {_fmt_pct(lay.get('roi_pct_per_liability'))} | {_fmt_pct(ex.get('lay_roi_pct_per_stake'))} |\n"
            )
        s1.append("\n")

        # Quebra por tipo (Back/Lay × Pre/In) no P&L por placar (cobertura). Ajuda a explicar dias OOS/placar positivos vs accounting negativo.
        try:
            s1.append("**Quebra (placar): Back/Lay × Pre/In (somente cobertos por ROI)**\n\n")
            s1.append("| Dia | P&L Back Pre | ROI Back Pre | P&L Back In | ROI Back In | P&L Lay Pre | ROI Lay Pre/liab | P&L Lay In | ROI Lay In/liab |\n")
            s1.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
            for it in adh_day.get("per_day") or []:
                if not isinstance(it, dict):
                    continue
                ex = it.get("execution") if isinstance(it.get("execution"), dict) else {}
                bt = ex.get("pnl_placar_by_type") if isinstance(ex.get("pnl_placar_by_type"), dict) else {}
                def _p(k: str) -> float:
                    try:
                        return float((bt.get(k) or {}).get("pnl") or 0.0) if isinstance(bt.get(k), dict) else 0.0
                    except Exception:
                        return 0.0
                def _e(k: str) -> float:
                    try:
                        return float((bt.get(k) or {}).get("exposure") or 0.0) if isinstance(bt.get(k), dict) else 0.0
                    except Exception:
                        return 0.0
                pbp, ebp = _p("Back_Pre"), _e("Back_Pre")
                pbi, ebi = _p("Back_In"), _e("Back_In")
                plp, elp = _p("Lay_Pre"), _e("Lay_Pre")
                pli, eli = _p("Lay_In"), _e("Lay_In")
                r_bp = (pbp / ebp * 100.0) if ebp > 0 else None
                r_bi = (pbi / ebi * 100.0) if ebi > 0 else None
                r_lp = (plp / elp * 100.0) if elp > 0 else None
                r_li = (pli / eli * 100.0) if eli > 0 else None
                s1.append(
                    f"| {it.get('day')} | {_fmt_num(pbp,2)} | {_fmt_pct(r_bp)} | {_fmt_num(pbi,2)} | {_fmt_pct(r_bi)} | "
                    f"{_fmt_num(plp,2)} | {_fmt_pct(r_lp)} | {_fmt_num(pli,2)} | {_fmt_pct(r_li)} |\n"
                )
            s1.append("\n")
        except Exception:
            pass

        # slippage x ROI (3 buckets raw com sinal) — acumulado na janela (não só um dia)
        raw_total = {}
        if isinstance(adh_slip, dict):
            raw_total = adh_slip.get("slippage_vs_roi_raw_total_ctx") if isinstance(adh_slip.get("slippage_vs_roi_raw_total_ctx"), dict) else (
                adh_slip.get("slippage_vs_roi_raw_total") if isinstance(adh_slip.get("slippage_vs_roi_raw_total"), dict) else {}
            )
        if isinstance(raw_total, dict) and raw_total:
            try:
                # Para slippage×ROI, respeitamos o range semântico (pós-fix) quando disponível.
                rg = adh_slip.get("slippage_range", None) if isinstance(adh_slip, dict) else None
                if not isinstance(rg, dict) or not rg:
                    rg = adh_slip.get("range", {}) if isinstance(adh_slip, dict) else {}
                span = rg.get("span_days") if isinstance(rg, dict) else None
                s1.append(
                    f"**Slippage × ROI por bucket (raw, com sinal) — acumulado (range: `{rg.get('start_day')}` → `{rg.get('end_day')}`; span_days=`{int(span or 0)}`)**\n\n"
                )
            except Exception:
                s1.append("**Slippage × ROI por bucket (raw, com sinal) — acumulado**\n\n")
            for side_key, title in (("back", "Back (ROI por stake)"), ("lay", "Lay (ROI por liability)")):
                b = raw_total.get(side_key) if isinstance(raw_total.get(side_key), dict) else {}
                buckets0 = b.get("buckets") if isinstance(b.get("buckets"), list) else []
                buckets = _slip_raw_3bucket_rows(buckets0)
                if not any(int(r.get("n") or 0) > 0 for r in buckets):
                    continue
                s1.append(f"- **{title}**\n\n")
                s1.append("| Bucket slippage_raw_pct | n | ROI mean (SE; IC95) |\n|---|---:|---:|\n")
                for row in buckets:
                    s1.append(
                        f"| {row.get('bucket')} | {int(row.get('n') or 0)} | {_fmt_roi_mean_se_ci_pct(row)}{_fmt_ctx_suffix(row)} |\n"
                    )
                s1.append("\n")
            s1.append(
                "- Nota: `ROIw` é o **ROI ponderado por exposição** (peso=stake no Back; peso=liability no Lay). "
                "Em prática, dentro de um bucket, `ROIw ≈ (∑P&L)/(∑exposição)`; já o `ROI mean` é a média simples por linha/sinal.\n\n"
            )
            # Lay também em ROI por stake (bounded; sanity-check)
            lay_stake_blk = adh_slip.get("slippage_vs_roi_raw_total_ctx_lay_stake") if (isinstance(adh_slip, dict) and isinstance(adh_slip.get("slippage_vs_roi_raw_total_ctx_lay_stake"), dict)) else {}
            b2 = lay_stake_blk.get("lay") if isinstance(lay_stake_blk.get("lay"), dict) else {}
            buckets02 = b2.get("buckets") if isinstance(b2.get("buckets"), list) else []
            buckets2 = _slip_raw_3bucket_rows(buckets02)
            if any(int(r.get("n") or 0) > 0 for r in buckets2):
                s1.append("- **Lay (ROI por stake; bounded)**\n\n")
                s1.append("| Bucket slippage_raw_pct | n | ROI mean (SE; IC95) |\n|---|---:|---:|\n")
                for row in buckets2:
                    s1.append(f"| {row.get('bucket')} | {int(row.get('n') or 0)} | {_fmt_roi_mean_se_ci_pct(row)}{_fmt_ctx_suffix(row)} |\n")
                s1.append("\n")

            # Contrafactual: filtro de slippage (placar) — não apostar se Back raw<=-2% e Lay raw>2%
            try:
                cf = adh_slip.get("slippage_filter_counterfactual") if isinstance(adh_slip, dict) else None
                if isinstance(cf, dict) and isinstance(cf.get("rule"), dict):
                    b = cf.get("back") if isinstance(cf.get("back"), dict) else {}
                    l = cf.get("lay") if isinstance(cf.get("lay"), dict) else {}
                    if (int(b.get("n") or 0) + int(l.get("n") or 0)) > 0:
                        s1.append("**Contrafactual (placar): aplicar filtro de slippage**\n\n")
                        s1.append("- Regra: **Back** pula `slippage_raw_pct <= -2%`; **Lay** pula `slippage_raw_pct > 2%`.\n")
                        s1.append("- Observação: usa somente execuções com ROI via placar; não é o P&L do accounting.\n\n")
                        s1.append("| Lado | n (base) | P&L (base) | Exposição (base) | n (após filtro) | P&L (após) | Exposição (após) |\n")
                        s1.append("|---|---:|---:|---:|---:|---:|---:|\n")
                        s1.append(f"| Back | {int(b.get('n') or 0)} | {_fmt_num(b.get('pnl'),2)} | {_fmt_num(b.get('stake'),2)} | {int(b.get('n_filtered') or 0)} | {_fmt_num(b.get('pnl_filtered'),2)} | {_fmt_num(b.get('stake_filtered'),2)} |\n")
                        s1.append(f"| Lay (liab) | {int(l.get('n') or 0)} | {_fmt_num(l.get('pnl'),2)} | {_fmt_num(l.get('liability'),2)} | {int(l.get('n_filtered') or 0)} | {_fmt_num(l.get('pnl_filtered'),2)} | {_fmt_num(l.get('liability_filtered'),2)} |\n")
                        try:
                            pnl0 = float(b.get("pnl") or 0.0) + float(l.get("pnl") or 0.0)
                            pnl1 = float(b.get("pnl_filtered") or 0.0) + float(l.get("pnl_filtered") or 0.0)
                            s1.append(f"| **Total** | — | {_fmt_num(pnl0,2)} | — | — | {_fmt_num(pnl1,2)} | — |\n")
                        except Exception:
                            pass
                        s1.append("\n")
            except Exception:
                pass

            # Diagnóstico AH (linha) observado na execução
            try:
                ah = adh_slip.get("observed_ah_line_abs") if isinstance(adh_slip, dict) else None
                if isinstance(ah, dict):
                    thr = ah.get("threshold")
                    scope = ah.get("scope")
                    allx = ah.get("all_exec") if isinstance(ah.get("all_exec"), dict) else {}
                    covx = ah.get("cov_placar") if isinstance(ah.get("cov_placar"), dict) else {}
                    if int(allx.get("n") or 0) > 0:
                        s1.append("**Diagnóstico AH (linha) observado na execução**\n\n")
                        s1.append(f"- Policy: `ah_max_abs_line={thr}` | `ah_scope={scope}`\n")
                        s1.append(f"- Execuções (todas): `n={int(allx.get('n') or 0)}` | `max|line|={_fmt_num(allx.get('max_abs_line'),2)}` | `n_over={int(allx.get('n_over') or 0)}`\n")
                        s1.append(f"- Execuções com placar/ROI: `n={int(covx.get('n') or 0)}` | `max|line|={_fmt_num(covx.get('max_abs_line'),2)}` | `n_over={int(covx.get('n_over') or 0)}`\n\n")
            except Exception:
                pass
            # Por combinação (top por volume)
            rows = adh_slip.get("slippage_vs_roi_raw_by_combo_top") if (isinstance(adh_slip, dict) and isinstance(adh_slip.get("slippage_vs_roi_raw_by_combo_top"), list)) else []
            if rows:
                try:
                    back_rows = [r for r in rows if isinstance(r, dict) and str(r.get("side")) == "Back"]
                    lay_rows = [r for r in rows if isinstance(r, dict) and str(r.get("side")) == "Lay"]
                    def _print_combo_block(title: str, xs: list[dict], limit: int = 12) -> None:
                        if not xs:
                            return
                        s1.append(f"**Slippage × ROI por combinação (top {min(limit, len(xs))} por volume; acumulado)**\n\n")
                        s1.append(f"- **{title}**\n\n")
                        s1.append("| Combinação | n | ROI<=-2% | n | ROI(-2..2] | n | ROI>2% | n | corr(slip_raw,ROI) |\n")
                        s1.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
                        for r in xs[:limit]:
                            comb = str(r.get("comb") or "")
                            n = int(r.get("n") or 0)
                            corr = r.get("corr_raw_pct_vs_roi")
                            # buckets dict
                            bmap = {str(b.get("bucket")): b for b in (r.get("buckets") or []) if isinstance(b, dict)}
                            def _bn(lab: str) -> tuple[int, Any]:
                                bb = bmap.get(lab) or {}
                                return int(bb.get("n") or 0), bb
                            n1, roi1 = _bn("<= -2%")
                            n2, roi2 = _bn("(-2, 2]")
                            n3, roi3 = _bn("> 2%")
                            s1.append(
                                f"| {comb} | {n} | {_fmt_roi_mean_se_ci_pct(roi1)} | {n1} | {_fmt_roi_mean_se_ci_pct(roi2)} | {n2} | {_fmt_roi_mean_se_ci_pct(roi3)} | {n3} | {_fmt_num(corr,2)} |\n"
                            )
                        s1.append("\n")
                    # já está ordenado por n desc
                    _print_combo_block("Back", back_rows)
                    _print_combo_block("Lay", lay_rows)
                except Exception:
                    pass
        else:
            # fallback: último dia com dados
            last = _pick_last_day_with_slippage_vs_roi_raw(list(adh_day.get("per_day") or [])) if isinstance(adh_day, dict) else None
            if isinstance(last, dict):
                ex = last.get("execution") if isinstance(last.get("execution"), dict) else {}
                rawblk = ex.get("slippage_vs_roi_raw") if isinstance(ex.get("slippage_vs_roi_raw"), dict) else {}
                if rawblk:
                    s1.append(f"**Slippage × ROI por bucket (raw, com sinal) — exemplo do dia `{last.get('day')}`**\n\n")
                    for side_key, title in (("back", "Back (ROI por stake)"), ("lay", "Lay (ROI por liability)")):
                        b = rawblk.get(side_key) if isinstance(rawblk.get(side_key), dict) else {}
                        buckets0 = b.get("buckets") if isinstance(b.get("buckets"), list) else []
                        buckets = _slip_raw_3bucket_rows(buckets0)
                        if not any(int(r.get("n") or 0) > 0 for r in buckets):
                            continue
                        s1.append(f"- **{title}**\n\n")
                        s1.append("| Bucket slippage_raw_pct | n | ROI mean (SE; IC95) |\n|---|---:|---:|\n")
                        for row in buckets:
                            s1.append(f"| {row.get('bucket')} | {int(row.get('n') or 0)} | {_fmt_roi_mean_se_ci_pct(row)} |\n")
                        s1.append("\n")
            else:
                s1.append(
                    "_Slippage × ROI (por bucket) indisponível na janela: precisa de execuções com odd (decision/final) **e** placar (ROI) no DB._\n\n"
                )

    # Funil (24h) por auditoria: total → OK/valid → erros principais
    if isinstance(audit_rep, dict) and isinstance(audit_rep.get("by_version"), list) and audit_rep.get("by_version"):
        s1.append("**Funil de oportunidades (últimas 24h; auditoria DB)**\n\n")
        s1.append("| audit_version | total | OK | OK_valid | GATE_NOT_ELIGIBLE | API_FAILED | STALE_QUEUE_WAIT |\n")
        s1.append("|---|---:|---:|---:|---:|---:|---:|\n")
        for v in audit_rep.get("by_version") or []:
            if not isinstance(v, dict):
                continue
            sc = v.get("status_counts") if isinstance(v.get("status_counts"), dict) else {}
            s1.append(
                f"| {v.get('audit_version')} | {int(v.get('total') or 0)} | {int(sc.get('OK') or 0)} | {int(v.get('ok_valid') or 0)} | "
                f"{int(sc.get('GATE_NOT_ELIGIBLE') or 0)} | {int(sc.get('API_FAILED') or 0)} | {int(sc.get('STALE_QUEUE_WAIT') or 0)} |\n"
            )
        s1.append("\n")

        # motivos top (api_error) agregados
        errs = []
        try:
            for it in (audit_rep.get("error_rows") or []):
                if not isinstance(it, dict):
                    continue
                n = int(it.get("n") or 0)
                if n <= 0:
                    continue
                errs.append((n, str(it.get("audit_version") or ""), str(it.get("status") or ""), str(it.get("api_error") or "")))
            errs.sort(key=lambda x: x[0], reverse=True)
        except Exception:
            errs = []
        if errs:
            s1.append("**Motivos principais de não-execução / falha (top)**\n\n")
            for n, ver, st, err in errs[:8]:
                err2 = (err[:180] + "…") if len(err) > 180 else err
                s1.append(f"- `{ver}`: {st} ×{n} — `{err2}`\n")
            s1.append("\n")

        s1.append("**Oportunidades identificadas / melhorias propostas (curto prazo)**\n\n")
        s1.append(
            "- **PMM/timeout**: se `No PMMs received` dominar, aumentar timeout efetivo e reduzir bursts (workers/queue) tende a elevar conversão sem mexer na estratégia.\n"
            "- **Betslips abertos**: `too_many_open_betslips` é um gargalo de throughput; manter caps/janelas e garantir cleanup rápido evita bloqueio global.\n"
            "- **Fila**: `STALE_QUEUE_WAIT` indica atraso interno; atacar latência/concorrência antes de aumentar volume/seleção.\n\n"
        )

    # Sensibilidade de banca (reusa tabela OOS existente)
    if oos_txt:
        # OOS pode estar numerado como 12.x (full) ou 1.x (oos_first)
        sens = _extract_md_block(
            oos_txt,
            start="### 12.2b Sensibilidade por banca",
            until_any=["### 12.2c Sensibilidade por banca", "### 12.3 ", "### 1.2c Sensibilidade por banca", "### 1.3 "],
        )
        if not sens.strip():
            sens = _extract_md_block(
                oos_txt,
                start="### 1.2b Sensibilidade por banca",
                until_any=["### 1.2c Sensibilidade por banca", "### 1.3 ", "### 12.2c Sensibilidade por banca", "### 12.3 "],
            )
        if sens.strip():
            s1.append("**Estudo de sensibilidade (banca × lucro)**\n\n")
            s1.append(
                "_A tabela abaixo é reaproveitada do bloco OOS (mesmo layout). Ela responde “até onde a operação escala” antes de bater em caps/limites._\n\n"
            )
            s1.append(sens + "\n")

        # Diagnóstico: por que turnover/jogos/lucro OOS podem cair nos steps recentes
        try:
            tbl_md, rows = _extract_md_table(oos_txt, header_startswith="| Train window")
            if tbl_md.strip() and rows:
                s1.append("**OOS recente: escala (turnover/jogos/lucro) por step**\n\n")
                s1.append(
                    "_Leitura: se `#ativas (keys)` e `Jogos OOS` caem, a causa típica é calendário/fragmentação (por liga) + filtros (AH/exec_bucket) + cobertura de placar. "
                    "Se jogos não caem, mas turnover cai, o gargalo tende a ser budget/caps (governança) e sizing._\n\n"
                )
                # mostrar a tabela original e destacar os últimos 4 steps (no topo executivo)
                s1.append(tbl_md + "\n")
                last4 = rows[-4:]
                hdr = _md_table_header_cols(tbl_md)
                hmap = {str(c).strip(): i for i, c in enumerate(hdr)}
                # índices robustos (compatível com tabela antiga e nova)
                ix_games = hmap.get("Jogos OOS")
                ix_turn = hmap.get("Turnover (teste)")
                ix_pnl = hmap.get("Lucro (estratégia, budget)")
                ix_turn_pre = hmap.get("Turnover Pre")
                ix_turn_in = hmap.get("Turnover In")
                def _g(cols, ix):
                    try:
                        if ix is None:
                            return ""
                        return cols[int(ix)]
                    except Exception:
                        return ""
                # heurística: comparar último vs mediana
                try:
                    games = []
                    turns = []
                    turns_pre = []
                    turns_in = []
                    profs = []
                    for r in rows:
                        try:
                            games.append(float(_g(r, ix_games)) if _g(r, ix_games) else 0.0)
                        except Exception:
                            pass
                        try:
                            turns.append(float(str(_g(r, ix_turn)).replace(",", ".")))
                        except Exception:
                            pass
                        try:
                            if _g(r, ix_turn_pre):
                                turns_pre.append(float(str(_g(r, ix_turn_pre)).replace(",", ".")))
                        except Exception:
                            pass
                        try:
                            if _g(r, ix_turn_in):
                                turns_in.append(float(str(_g(r, ix_turn_in)).replace(",", ".")))
                        except Exception:
                            pass
                        try:
                            profs.append(float(str(_g(r, ix_pnl)).replace(",", ".")))
                        except Exception:
                            pass
                    if games and turns:
                        import statistics
                        med_g = statistics.median(games)
                        med_t = statistics.median(turns)
                        s1.append("**Diagnóstico rápido (último step vs mediana histórica do WF)**\n\n")
                        # último row
                        lr = rows[-1]
                        g_last = float(_g(lr, ix_games) or 0.0)
                        t_last = None
                        try:
                            t_last = float(str(_g(lr, ix_turn)).replace(",", "."))
                        except Exception:
                            t_last = None
                        s1.append(f"- Jogos OOS (último): `{_fmt_num(g_last,0)}` vs mediana `{_fmt_num(med_g,0)}`\n")
                        if t_last is not None:
                            s1.append(f"- Turnover teste (último): `{_fmt_num(t_last,2)}` vs mediana `{_fmt_num(med_t,2)}`\n")
                        # Pre/In (se houver)
                        try:
                            if ix_turn_pre is not None and turns_pre:
                                med_tp = statistics.median(turns_pre)
                                tp_last = float(str(_g(lr, ix_turn_pre)).replace(",", ".")) if _g(lr, ix_turn_pre) else None
                                if tp_last is not None:
                                    s1.append(f"- Turnover Pre (último): `{_fmt_num(tp_last,2)}` vs mediana `{_fmt_num(med_tp,2)}`\n")
                            if ix_turn_in is not None and turns_in:
                                med_ti = statistics.median(turns_in)
                                ti_last = float(str(_g(lr, ix_turn_in)).replace(",", ".")) if _g(lr, ix_turn_in) else None
                                if ti_last is not None:
                                    s1.append(f"- Turnover In (último): `{_fmt_num(ti_last,2)}` vs mediana `{_fmt_num(med_ti,2)}`\n")
                        except Exception:
                            pass
                        s1.append(
                            "- Se a queda é em **Jogos OOS**: problema é **volume/cobertura** (placar, calendário, fragmentação por liga, filtros como AH/exec_bucket).\n"
                            "- Se Jogos OOS está ok mas turnover cai: **governança/sizing** (budgets/caps) está limitando escala.\n\n"
                        )
                except Exception:
                    pass
        except Exception:
            pass

    # Histórico recente de policy (parâmetros “passados” do portfólio ativo)
    try:
        hist_lines = _tail_lines(cfg.wf_policy_history_jsonl, 12)
        recs = []
        for ln in hist_lines:
            try:
                recs.append(json.loads(ln))
            except Exception:
                continue
        recs = [r for r in recs if isinstance(r, dict)]
        if recs:
            s1.append("**Portfólio OOS: vigente vs histórico recente**\n\n")
            s1.append("| ts | n_active_keys |\n|---|---:|\n")
            for r in recs[-8:]:
                nkeys = None
                try:
                    ak = r.get("active_keys")
                    nkeys = len(ak) if isinstance(ak, list) else None
                except Exception:
                    nkeys = None
                s1.append(f"| {r.get('ts')} | {nkeys if nkeys is not None else '—'} |\n")
            s1.append("\n")
    except Exception:
        pass

    # parâmetros de negócio e técnicos: manter 99.6 como fonte, mas resumir aqui
    s1.append("**Parâmetros vigentes (visão executiva)**\n\n")
    # decomposição de active_keys (negócio)
    try:
        if isinstance(active_keys, list) and active_keys:
            def _cnt(prefix: str) -> int:
                return sum(1 for k in active_keys if str(k).startswith(prefix))
            by_league = sum(1 for k in active_keys if "__" in str(k))
            s1.append("| Dimensão | Valor |\n|---|---:|\n")
            s1.append(f"| active_keys (total) | {len(active_keys)} |\n")
            s1.append(f"| chaves por liga (suFIXO `__<League>`) | {by_league} |\n")
            s1.append(f"| Back_Pre | {_cnt('Back_Pre_')} |\n")
            s1.append(f"| Back_In | {_cnt('Back_In_')} |\n")
            s1.append(f"| Lay_Pre_Yes | {_cnt('Lay_Pre_Yes')} |\n")
            s1.append(f"| Lay_Pre_No | {_cnt('Lay_Pre_No')} |\n")
            s1.append(f"| Lay_In_Yes | {_cnt('Lay_In_Yes')} |\n")
            s1.append(f"| Lay_In_No | {_cnt('Lay_In_No')} |\n")
            s1.append("\n")
    except Exception:
        pass
    s1.append(
        "- **Combinações ativas (OOS)**: ver `99.3` (active_keys) e o bloco `2) OOS`.\n"
        "- **Stake sizing operacional (real)**: hoje é **FLAT** via `BRIDGE_STAKE` (ver `99.3` e `99.6`).\n"
        "- **Parâmetros técnicos efetivos** (executor/audit/bridge): ver `99.6 Filtros ativos`.\n\n"
    )

    # Critérios (OOS e real) + clareza do filtro de AH
    # Preferir policy_current (fonte da verdade operacional) quando disponível.
    wf_key_by_league = bool(policy_wf.get("key_by_league")) if isinstance(policy_wf, dict) else (
        str(os.getenv("DAILY_WF_KEY_BY_LEAGUE", "1")).strip() not in ("0", "false", "False", "no", "NO")
    )
    wf_key_scope = str(policy_wf.get("key_by_league_scope") or "") if isinstance(policy_wf, dict) else str(os.getenv("DAILY_WF_KEY_BY_LEAGUE_SCOPE", str(cfg.wf_key_by_league_scope)) or "")
    wf_key_scope = wf_key_scope.strip() or "pre"
    try:
        wf_ah = float(policy_wf.get("ah_max_abs_line")) if isinstance(policy_wf, dict) and policy_wf.get("ah_max_abs_line") is not None else float(os.getenv("DAILY_WF_AH_MAX_ABS_LINE", str(cfg.wf_ah_max_abs_line)) or 0.0)
    except Exception:
        wf_ah = 0.0
    wf_ah_scope = str(policy_wf.get("ah_scope") or "") if isinstance(policy_wf, dict) else str(os.getenv("DAILY_WF_AH_SCOPE", str(cfg.wf_ah_scope)) or "")
    wf_ah_scope = wf_ah_scope.strip() or "pre"
    wf_min_matches = str(policy_wf.get("min_matches") or "") if isinstance(policy_wf, dict) else str(os.getenv("DAILY_WF_MIN_MATCHES", str(cfg.wf_min_matches)) or "")
    wf_min_matches = wf_min_matches.strip() or "0"
    s1.append("**Critérios de seleção (OOS) e critérios do real (bridge/executor)**\n\n")
    s1.append(
        "- **OOS (walk-forward)** decide o portfólio `active_keys`.\n"
        f"  - **Chave por liga**: `{wf_key_by_league}` (scope=`{wf_key_scope}`) ⇒ em pre-match a chave pode virar `...__<League>`.\n"
        f"  - **Filtro de AH ativo?**: `{wf_ah > 0}` (max_abs_line=`{_fmt_num(wf_ah,2)}`; scope=`{wf_ah_scope}`) ⇒ remove eventos com `abs(line)` acima do limiar.\n"
        f"  - **Mínimo de jogos no treino**: `wf_min_matches={wf_min_matches}` (0 = desligado).\n"
        "  - **Regra de decisão (por combinação, no treino)**:\n"
        "    - Se `ROI` for **significativamente negativo** (IC90 inteiro < 0): **bloqueia**.\n"
        "    - Se `ROI` for **significativamente positivo** (IC90 inteiro > 0): **ativa**.\n"
        "    - Se `ROI` > 0 mas **não significativo**:\n"
        "      - **Pre-match**: ativa apenas se **CLV > 0** (CLV não precisa ser sig.).\n"
        "      - **In-match**: ativa se **ROI > 0** (CLV não se aplica).\n"
        "  - Operacionalmente, o OOS também pode excluir buckets de execução (ex.: `wf_exclude_exec_buckets_back`).\n"
        "- **Real (shadow/live)**:\n"
        "  - O bridge só envia oportunidades cuja chave esteja em `active_keys` (policy current).\n"
        "  - `DRY_OK` = **shadow** (não apostou); `LIVE_OK` = **efetivo** (apostou).\n\n"
    )
    # Transparência do step vigente (train/test) usado para gerar o policy current.
    if isinstance(policy_last_step, dict):
        try:
            s1.append("**Policy current: janela de treino/teste (do último step exportado)**\n\n")
            s1.append(
                f"- Train window: `{policy_last_step.get('train')}` | Test window: `{policy_last_step.get('test')}` | "
                f"train_days={len(policy_last_step.get('train_days') or [])} | test_days={len(policy_last_step.get('test_days') or [])}\n\n"
            )
        except Exception:
            pass
    # explicitar shadow vs live na janela
    try:
        live_ok = int((kpi_all.get("status_counts") or {}).get("LIVE_OK") or 0)
        dry_ok = int((kpi_all.get("status_counts") or {}).get("DRY_OK") or 0)
        s1.append("**Este período está rodando shadow ou efetivo?**\n\n")
        if live_ok > 0 and dry_ok > 0:
            s1.append(f"- Misturado: `LIVE_OK={live_ok}` e `DRY_OK={dry_ok}`.\n\n")
        elif live_ok > 0:
            s1.append(f"- Predominantemente **efetivo**: `LIVE_OK={live_ok}` (e `DRY_OK={dry_ok}`).\n\n")
        else:
            s1.append(f"- Predominantemente **shadow**: `DRY_OK={dry_ok}` (e `LIVE_OK={live_ok}`).\n\n")
    except Exception:
        pass

    # aspectos técnicos (latência/gaps/restarts proxy)
    try:
        gaps = _executor_gaps_summary(exec_lines)
        s1.append("**Aspectos técnicos (latência/estabilidade)**\n\n")
        s1.append("- Latência detalhada: ver `99.2` (p50/p90/p99 por etapa).\n")
        if gaps.get("max_gap_s") is not None:
            s1.append(
                f"- Gaps no `executor_jsonl` (proxy de downtime/restart/sem tráfego): max `{_fmt_num(gaps.get('max_gap_s'),1)}s`, "
                f"gaps>5min `{gaps.get('gaps_gt_300s')}`, gaps>15min `{gaps.get('gaps_gt_900s')}`.\n\n"
            )
        else:
            s1.append("- Gaps no `executor_jsonl`: amostra insuficiente.\n\n")
    except Exception:
        pass

    # 6) Combinar markdown (base reordenado + blocos operacionais 99.x)
    extra = []
    extra.append("\n\n## 99) Operacional — saldo, P&L e execução\n\n")
    extra.append("### 99.1 Accounting (saldo + P&L)\n\n")
    extra.append(f"- Arquivo: `{acct_out}`\n")
    if acct.get("error"):
        extra.append(f"- **Erro**: **{acct.get('error')}**\n")
    extra.append(f"- Saldo atual: **{acct.get('balance_current')}**\n")
    extra.append(f"- P&L hoje/semana/mês: **{acct.get('pnl_today')} / {acct.get('pnl_week')} / {acct.get('pnl_month')}**\n")
    extra.append("\nMeses fechados:\n\n")
    extra.append("| Mês | P&L |\n|---|---:|\n")
    for k, v in (acct.get("closed_months") or {}).items():
        extra.append(f"| {k} | {v} |\n")

    extra.append("\n### 99.2 Execução (KPIs)\n\n")
    extra.append(f"- Fonte: `{cfg.executor_jsonl}`\n")
    extra.append("- Nota: métricas abaixo vêm do JSONL; se ele estiver **stale** ou incompleto, podem divergir do volume “24h, DB”.\n\n")

    # Status table
    extra.append("**Status (all)**\n\n")
    extra.append("| Status | N |\n|---|---:|\n")
    for k, v in (kpi_all.get("status_counts") or {}).items():
        extra.append(f"| {k} | {int(v)} |\n")
    extra.append("\n")

    def _timing_table(title: str, obj: Dict[str, Any]) -> str:
        def _row(name: str, a: dict) -> str:
            return (
                f"| {name} | {a.get('n')} | {a.get('p50')} | {a.get('p90')} | {a.get('p99')} | {a.get('mean')} |\n"
            )

        s = []
        s.append(f"**{title}**\n\n")
        s.append("| Métrica | n | p50 | p90 | p99 | mean |\n|---|---:|---:|---:|---:|---:|\n")
        for nm in ("queue_delay", "call_to_done", "post"):
            a = ((obj.get(nm) or {}) if isinstance(obj, dict) else {})
            s.append(_row(nm, a if isinstance(a, dict) else {}))
        s.append("\n")
        return "".join(s)

    timing_ok = (kpi_ok.get("timing_ms") or {}) if isinstance(kpi_ok, dict) else {}
    extra.append(_timing_table("Latência (somente LIVE_OK/DRY_OK) — ms", timing_ok))

    # Recorte 24h (consistente com o checklist de prontidão LIVE)
    try:
        timing_ok24 = (kpi_ok_24h.get("timing_ms") or {}) if isinstance(kpi_ok_24h, dict) else {}
        extra.append(_timing_table("Latência (últimas 24h; somente LIVE_OK/DRY_OK) — ms", timing_ok24))
    except Exception:
        pass

    slip_ok = (kpi_ok.get("slippage") or {}) if isinstance(kpi_ok, dict) else {}
    extra.append("**Slippage (somente LIVE_OK/DRY_OK, quando houver odd_at_decision)**\n\n")
    extra.append(
        "- Definição: `slippage = odd_final - odd_at_decision` (em odds decimais) e `slippage_pct = slippage/odd_at_decision`.\n"
        "- Interpretação depende do lado:\n"
        "  - **Back**: slippage_pct **negativo** = piorou (odd caiu); **positivo** = melhorou.\n"
        "  - **Lay**: slippage_pct **positivo** = piorou (odd subiu); **negativo** = melhorou.\n\n"
    )
    extra.append("| Tipo | n | p50 | p90 | p99 | mean |\n|---|---:|---:|---:|---:|---:|\n")
    for nm in ("abs", "pct"):
        a = (slip_ok.get(nm) or {}) if isinstance(slip_ok, dict) else {}
        extra.append(f"| {nm} | {a.get('n')} | {a.get('p50')} | {a.get('p90')} | {a.get('p99')} | {a.get('mean')} |\n")
    extra.append("\n")

    # Slippage por lado (Back vs Lay)
    slip_by_side = (kpi_ok.get("slippage_by_side") or {}) if isinstance(kpi_ok, dict) else {}
    if isinstance(slip_by_side, dict) and slip_by_side:
        extra.append("**Slippage por lado (Back vs Lay)**\n\n")
        extra.append("| Lado | Métrica | n | p50 | p90 | p99 | mean |\n|---|---|---:|---:|---:|---:|---:|\n")
        for side, obj in slip_by_side.items():
            if not isinstance(obj, dict):
                continue
            for nm, label in (
                ("raw_pct", "slippage_pct (raw)"),
                ("cost_pct", "slippage_pct (custo, >=0)"),
            ):
                a = obj.get(nm) if isinstance(obj.get(nm), dict) else {}
                extra.append(
                    f"| {side} | {label} | {a.get('n')} | {a.get('p50')} | {a.get('p90')} | {a.get('p99')} | {a.get('mean')} |\n"
                )
        extra.append("\n")
    extra.append(
        "_Nota: o p90/p99 de `call_to_done_ms` explode quando inclui `NO_SESSION/API_FAILED` (timeouts/relogin). "
        "Por isso reportamos também o recorte apenas de sucessos._\n\n"
    )

    if active_keys:
        extra.append("\n### 99.3 Regras OOS ativas (último step)\n\n")
        extra.append(f"- active_keys: {len(active_keys) if isinstance(active_keys, list) else '—'}\n")
        extra.append("```json\n" + json.dumps(active_keys, ensure_ascii=False, indent=2) + "\n```\n\n")
        extra.append("**Como ler `active_keys` (regra de aprovação)**\n\n")
        extra.append(
            "- `active_keys` é o **portfólio aprovado** pelo walk-forward (OOS) no **último step** exportado.\n"
            "- O bridge (`ops.executor_bridge_audit`) só envia para o executor oportunidades cuja **chave operacional** (combinação) esteja ativa.\n"
            "- Mapeamento de chaves (simplificado):\n"
            "  - **Back**: `Back_Pre_Any` (pre) ou `Back_In_Any` (in). Se o walk-forward estiver com `key_by_league`, a chave pode ter sufixo `__<League>`.\n"
            "  - **Lay**: `Lay_Pre_Yes/No` (pre) ou `Lay_In_Yes/No` (in). Para **H3B**, `Yes` indica que o sinal envolve reversão (por definição da hipótese).\n\n"
        )
        extra.append("**Regras de execução atuais (stake sizing)**\n\n")
        extra.append(
            "- No operacional (shadow/live), o tamanho enviado pelo bridge é **FLAT** via `BRIDGE_STAKE`.\n"
            "- Em **Back**: stake = `BRIDGE_STAKE`.\n"
            "- Em **Lay**: o executor recebe stake, mas o risco relevante é a **liability**, aproximadamente `liability ≈ stake × (odd - 1)`.\n"
            "- Importante: o Kelly/caps que aparece no relatório OOS é **simulação/diagnóstico** do walk-forward; ele não está sendo aplicado no executor/bridge neste momento.\n\n"
        )
        extra.append("**Lay em `ws_gate_lay`: é só pós-reversão?**\n\n")
        extra.append(
            "- Sim: o audit `v5.1-ws-gate-lay` só abre ticket Lay quando passa pelo gate de **queda** (ex.: >2% em 5s): `WS(t+offset) < ratio × WS(t0)`.\n"
            "- Isso significa que, mesmo em shadow, a amostra Lay desse audit representa apenas casos em que houve a movimentação (reversão/queda) definida pela estratégia.\n\n"
        )

    # 99.6 Config efetiva (filtros ativos) — executor/audit/bridge/OOS
    extra.append("\n### 99.6 Filtros ativos (config efetiva)\n\n")
    extra.append(
        "_Nota: esta seção reflete as variáveis carregadas pelo `daily_full_report` (via `.env`). "
        "Services do systemd podem ter overrides (`Environment=`) que não aparecem aqui; use `systemctl show` para confirmar no VPS._\n\n"
    )
    def _env(k: str, default: str = "") -> str:
        v = os.getenv(k, default)
        return str(v) if v is not None else ""

    extra.append("**Executor**\n\n")
    extra.append("| chave | valor |\n|---|---|\n")
    for k in [
        "EXECUTOR_ALLOW_LIVE",
        "EXECUTOR_WORKERS",
        "EXECUTOR_QUEUE_MAX",
        "EXECUTOR_CAP_WINDOW_SEC",
        "EXECUTOR_CAP_MAX",
        "EXECUTOR_FAST_PMM",
        "EXECUTOR_PMM_TIMEOUT_SEC",
        "EXECUTOR_PMM_MIN_WAIT_SEC",
        "EXECUTOR_PMM_IDLE_TIMEOUT_SEC",
        "EXECUTOR_BETSLIP_CACHE_MAX_KEYS",
    ]:
        extra.append(f"| {k} | `{_env(k)}` |\n")
    extra.append("\n")

    extra.append("**Audit H3B**\n\n")
    extra.append("| chave | valor |\n|---|---|\n")
    for k in [
        "AUDIT_MODE",
        "AUDIT_API_SIDES",
        "AUDIT_EXECUTOR_WORKERS",
        "AUDIT_TEMPORAL_WORKERS",
        "AUDIT_MAX_QUEUE_DEPTH",
        "AUDIT_MAX_QUEUE_WAIT_MS",
        "WS_SAMPLE_OFFSETS_SEC",
        "GATE_DROP_OFFSET_SEC",
        "GATE_DROP_RATIO",
        "GATE_RISE_OFFSET_SEC",
        "GATE_RISE_RATIO",
        "GATE_OPEN_WINDOW_SEC",
        "GATE_OPEN_MAX",
        "GATE_MAX_LATE_SEC",
        "GATE_LAY_REFRESH_TIMES_SEC",
    ]:
        extra.append(f"| {k} | `{_env(k)}` |\n")
    extra.append("\n")
    extra.append(
        "_Nota (Back vs Lay): o `AUDIT_MODE` acima costuma refletir o serviço principal (ex.: `ws_gate_lay`). "
        "Em operação real, o **Back** pode vir de um serviço separado (ex.: `betinasia-audit-api-back`, `audit_version=v5.2-api-back`) "
        "ou de uma variante `ws_gate_back` (dependendo do deploy). Para confirmar o que rodou nas últimas 24h, veja `99.5 Auditoria (DB)`._\n\n"
    )

    # Interpretação operacional (audit/entrada) para reduzir ambiguidade
    extra.append("**Interpretação operacional (timing de entrada)**\n\n")
    extra.append("| Item | Regra efetiva |\n|---|---|\n")
    extra.append("| Back (mais cedo possível) | Depende do executor: `EXECUTOR_FAST_PMM`, `EXECUTOR_PMM_MIN_WAIT_SEC`, `EXECUTOR_PMM_TIMEOUT_SEC` (ver tabela Executor). |\n")
    extra.append(
        "| Lay (reversão vs fim) | Depende do `AUDIT_MODE`/audit_version: `ws_gate_lay` abre Lay só quando o gate em `t+GATE_DROP_OFFSET_SEC` passa; "
        "`ws_reversal_lay` tende a entrar no pós-reversal; `ws_only` usa a série WS (offsets até o último ponto, tipicamente 30s). |\n"
    )
    extra.append("\n")

    extra.append("**Bridge**\n\n")
    extra.append("| chave | valor |\n|---|---|\n")
    for k in [
        "BRIDGE_MODE",
        "BRIDGE_EXEC_SIDE",
        "BRIDGE_STAKE",
        "BRIDGE_POLL_SEC",
        "BRIDGE_LOOKBACK_SEC",
        "BRIDGE_MAX_PER_CYCLE",
        "BRIDGE_PREMATCH_ONLY",
        "BRIDGE_POLICY_JSON",
        "BRIDGE_POLICY_RELOAD_SEC",
        "BRIDGE_POLICY_USE_BASE",
        "BRIDGE_MIN_LIMIT",
    ]:
        extra.append(f"| {k} | `{_env(k)}` |\n")
    extra.append("\n")

    extra.append("**OOS / Walk-forward (daily)**\n\n")
    extra.append("| chave | valor |\n|---|---|\n")
    for k in [
        "DAILY_OOS_DIRECTION",
        "DAILY_OOS_VERSIONS",
        "DAILY_OOS_LOOKBACK_DAYS",
        "DAILY_WF_TRAIN_MODE",
        "DAILY_WF_TRAIN_DAYS",
        "DAILY_WF_TEST_DAYS",
        "DAILY_WF_STEP_DAYS",
        "DAILY_WF_KEY_BY_LEAGUE",
        "DAILY_WF_KEY_BY_LEAGUE_SCOPE",
        "DAILY_WF_AH_MAX_ABS_LINE",
        "DAILY_WF_AH_SCOPE",
        "DAILY_WF_EXCLUDE_EXEC_BUCKETS_BACK",
    ]:
        extra.append(f"| {k} | `{_env(k)}` |\n")
    extra.append("\n")

    # 99.4 Aderência OOS (policy por dia × execução)
    try:
        if isinstance(adh_day, dict) and isinstance(adh_day.get("per_day"), list) and adh_day.get("per_day"):
            extra.append("\n### 99.4 Aderência OOS (portfolio por dia × execução)\n\n")
            extra.append(f"- Arquivo (curto): `{adh_short_json}`\n")
            if adh_long_json:
                extra.append(f"- Arquivo (acumulado/slippage): `{adh_long_json}`\n")
            extra.append(f"- Policy current: `{cfg.wf_policy_current}`\n\n")

            extra.append("**Resumo (últimos dias)**\n\n")
            extra.append("| Dia | Ativas (keys) | Bridge rows | Skipped(not_active) | Exec rows | LIVE_OK | DRY_OK | Back bloqueadas (slip<=-2%; cov) | Lay bloqueadas (slip>2%; cov) | ΔP&L cf (placar; cov) | P&L Back | ROI Back | P&L Lay | ROI Lay/liab | P&L total |\n")
            extra.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
            for it in adh_day.get("per_day") or []:
                if not isinstance(it, dict):
                    continue
                pol = it.get("policy") if isinstance(it.get("policy"), dict) else {}
                nkeys = pol.get("n_active_keys")
                bridge_rows = 0
                skipped_na = 0
                for b in (it.get("bridge") or []):
                    if isinstance(b, dict):
                        bridge_rows += int(b.get("n_rows") or 0)
                        skipped_na += int(b.get("n_not_active") or 0)
                ex = it.get("execution") if isinstance(it.get("execution"), dict) else {}
                sc = ex.get("status_counts") if isinstance(ex.get("status_counts"), dict) else {}
                live_ok = int(sc.get("LIVE_OK") or 0)
                dry_ok = int(sc.get("DRY_OK") or 0)
                back = ex.get("back") if isinstance(ex.get("back"), dict) else {}
                lay = ex.get("lay") if isinstance(ex.get("lay"), dict) else {}
                pnl_b = float(back.get("pnl_sum") or 0.0)
                pnl_l = float(lay.get("pnl_sum") or 0.0)
                # contrafactual slippage gate (por dia; somente cobertos por placar+odd)
                cf = it.get("slippage_filter_counterfactual") if isinstance(it.get("slippage_filter_counterfactual"), dict) else {}
                cfb = cf.get("back") if isinstance(cf.get("back"), dict) else {}
                cfl = cf.get("lay") if isinstance(cf.get("lay"), dict) else {}
                nblock_back = None
                nblock_lay = None
                dpnl_cf = None
                try:
                    nblock_back = int(cfb.get("n") or 0) - int(cfb.get("n_filtered") or 0)
                    nblock_lay = int(cfl.get("n") or 0) - int(cfl.get("n_filtered") or 0)
                    pnl_cf = float(cfb.get("pnl_filtered") or 0.0) + float(cfl.get("pnl_filtered") or 0.0)
                    pnl_base = float(cfb.get("pnl") or 0.0) + float(cfl.get("pnl") or 0.0)
                    dpnl_cf = float(pnl_cf - pnl_base)
                except Exception:
                    nblock_back = None
                    nblock_lay = None
                    dpnl_cf = None
                extra.append(
                    f"| {it.get('day')} | {nkeys if nkeys is not None else '—'} | {bridge_rows} | {skipped_na} | "
                    f"{int(ex.get('n_exec_rows') or 0)} | {live_ok} | {dry_ok} | {nblock_back if nblock_back is not None else '—'} | {nblock_lay if nblock_lay is not None else '—'} | {_fmt_num(dpnl_cf,2)} | {_fmt_num(pnl_b,2)} | {_fmt_pct(back.get('roi_pct'))} | "
                    f"{_fmt_num(pnl_l,2)} | {_fmt_pct(lay.get('roi_pct_per_liability'))} | {_fmt_num(pnl_b + pnl_l,2)} |\n"
                )
            extra.append("\n")

            # Potencial (30d) pela banca que maximiza lucro na sensibilidade OOS (se disponível)
            try:
                sens = _extract_md_block(
                    oos_txt,
                    start="### 12.2b Sensibilidade por banca",
                    until_any=["### 12.2c Sensibilidade por banca", "### 12.3 ", "### 1.2c Sensibilidade por banca", "### 1.3 "],
                )
                if not sens.strip():
                    sens = _extract_md_block(
                        oos_txt,
                        start="### 1.2b Sensibilidade por banca",
                        until_any=["### 1.2c Sensibilidade por banca", "### 1.3 ", "### 12.2c Sensibilidade por banca", "### 12.3 "],
                    )
                best = None
                if sens:
                    for ln in sens.splitlines():
                        if not ln.startswith("|") or ln.strip().startswith("|---"):
                            continue
                        cols = [c.strip() for c in ln.strip().strip("|").split("|")]
                        if len(cols) < 6 or cols[0].lower().startswith("banca"):
                            continue
                        bank_ref = _parse_md_number(cols[0])
                        turn_30 = _parse_md_number(cols[1])
                        prof_30 = _parse_md_number(cols[2])
                        bank_eff = _parse_md_number(cols[3])
                        roi_bank = _parse_md_number(cols[4])
                        dd_p95 = _parse_md_number(cols[5])
                        if prof_30 is None:
                            continue
                        if best is None or float(prof_30) > float(best["profit_30d_exp"]):
                            best = {
                                "bank_ref": bank_ref,
                                "turn_30d": turn_30,
                                "profit_30d_exp": prof_30,
                                "bank_eff": bank_eff,
                                "roi_bank_30d": roi_bank,
                                "dd_p95": dd_p95,
                            }
                if best:
                    # share observado (últimos dias) para decompor back/lay como estimativa
                    pnl_b = pnl_l = 0.0
                    for it in adh_day.get("per_day") or []:
                        ex = it.get("execution") if isinstance(it.get("execution"), dict) else {}
                        back = ex.get("back") if isinstance(ex.get("back"), dict) else {}
                        lay = ex.get("lay") if isinstance(ex.get("lay"), dict) else {}
                        pnl_b += float(back.get("pnl_sum") or 0.0)
                        pnl_l += float(lay.get("pnl_sum") or 0.0)
                    tot = pnl_b + pnl_l
                    w_b = (pnl_b / tot) if tot else 0.5
                    w_l = 1.0 - w_b
                    extra.append("**Potencial de lucro (30d) pela banca ótima (sensibilidade OOS)**\n\n")
                    extra.append(f"- Banca ref (grid): `{_fmt_num(best.get('bank_ref'),2)}` | banca rec. (max): `{_fmt_num(best.get('bank_eff'),2)}`\n")
                    extra.append(f"- Lucro 30d (exp.): `{_fmt_num(best.get('profit_30d_exp'),2)}` | ROI/banca 30d (exp.): `{_fmt_num(best.get('roi_bank_30d'),2)}%` | DD p95 (30d): `{_fmt_num(best.get('dd_p95'),2)}`\n")
                    extra.append(
                        f"- Decomposição *estimada* por lado (proporcional ao P&L observado na janela): total `{_fmt_num(best.get('profit_30d_exp'),2)}` → "
                        f"Back `{_fmt_num(float(best.get('profit_30d_exp'))*w_b,2)}` | Lay `{_fmt_num(float(best.get('profit_30d_exp'))*w_l,2)}`\n\n"
                    )
            except Exception:
                pass

            # Slippage × ROI (com sinal): acumulado na janela (para análise estatística)
            raw_total = adh_slip.get("slippage_vs_roi_raw_total") if (isinstance(adh_slip, dict) and isinstance(adh_slip.get("slippage_vs_roi_raw_total"), dict)) else {}
            if isinstance(raw_total, dict) and raw_total:
                try:
                    rg = adh_slip.get("slippage_range", None) if isinstance(adh_slip, dict) else None
                    if not isinstance(rg, dict) or not rg:
                        rg = adh_slip.get("range", {}) if isinstance(adh_slip, dict) else {}
                    span = rg.get("span_days") if isinstance(rg, dict) else None
                    slip_cut = (adh_slip.get("slippage_start_day_local") if isinstance(adh_slip, dict) else None) or None
                    extra.append(
                        f"**Slippage × ROI (raw, com sinal; 3 buckets) — acumulado (range: `{rg.get('start_day')}` → `{rg.get('end_day')}`; span_days=`{int(span or 0)}`; cut=`{slip_cut}`)**\n\n"
                    )
                except Exception:
                    extra.append("**Slippage × ROI (raw, com sinal; 3 buckets) — acumulado**\n\n")
                for side_key, title in (("back", "Back (ROI por stake)"), ("lay", "Lay (ROI por liability)")):
                    blk = raw_total.get(side_key) if isinstance(raw_total.get(side_key), dict) else {}
                    buckets0 = blk.get("buckets") if isinstance(blk.get("buckets"), list) else []
                    buckets = _slip_raw_3bucket_rows(buckets0)
                    if not any(int(r.get("n") or 0) > 0 for r in buckets):
                        continue
                    extra.append(f"- **{title}**\n\n")
                    extra.append("| Bucket slippage_raw_pct | n | ROI mean |\n|---|---:|---:|\n")
                    for b in buckets:
                        extra.append(f"| {b.get('bucket')} | {int(b.get('n') or 0)} | {_fmt_pct(b.get('roi_mean'))} |\n")
                    extra.append("\n")
            else:
                extra.append(
                    "_Slippage × ROI (por bucket) indisponível na janela: precisa de execuções com odd (decision/final) **e** placar (ROI) no DB._\n\n"
                )
    except Exception:
        pass

    # 99.5 Auditoria (DB): motivos de no-OK por versão + qualidade dos OK
    try:
        rep = audit_rep
        if isinstance(rep, dict) and isinstance(rep.get("by_version"), list) and rep.get("by_version"):
            extra.append("\n### 99.5 Auditoria (DB) — motivos de no-OK (por versão)\n\n")
            extra.append(f"- Arquivo: `{audit_json}`\n")
            extra.append(f"- Janela: últimas **{rep.get('hours')}h** (desde `{rep.get('since_utc')}`)\n\n")
            extra.append("**Definições (colunas)**\n\n")
            extra.append(
                "- **OK**: `status='OK'` no `betslip_audit_results` (a auditoria concluiu com sucesso).\n"
                "- **OK com betslip_odd**: subset de OK em que `betslip_odd` está preenchido (houve snapshot do ticket/odds).\n"
                "- **OK valid**: subset de OK em que `is_valid_opportunity=true` (passou o critério operacional de “oportunidade executável”).\n"
                "  - Na prática, o `is_valid_opportunity` tende a cair quando `difference_pct` está fora do range aceito (edge muito pequeno <2% ou mismatch >10%) ou quando campos essenciais do ticket estão ausentes.\n\n"
            )
            extra.append("**Glossário rápido (`audit_version`)**\n\n")
            extra.append("| padrão | significado |\n|---|---|\n")
            extra.append("| `v5.2-api-back` | Back via API (serviço back-only); tende a abrir betslip e medir limites/odds. |\n")
            extra.append("| `v5.1-ws-gate-lay` | Lay via WS gate (queda em 5s); só abre ticket quando o gate passa. |\n")
            extra.append("| `v5.4-ws-reversal-lay` | Lay no pós-reversal; volume baixo pode ser “evento raro” (depende de reversões). |\n")
            extra.append("| `v5.3-ws-gate-back` | Back via WS gate; se `OK` é baixo, costuma indicar gate muito restritivo, parse/click falhando, ou credenciais/sessão instável. |\n")
            extra.append("| `v4.*` / `v1.*` | versões antigas/legadas do pipeline (API/WS), úteis para comparação histórica. |\n")
            extra.append("\n")
            extra.append("| audit_version | total | OK | OK com betslip_odd | OK valid | top no-OK |\n")
            extra.append("|---|---:|---:|---:|---:|---|\n")
            for v in rep.get("by_version") or []:
                if not isinstance(v, dict):
                    continue
                sc = v.get("status_counts") if isinstance(v.get("status_counts"), dict) else {}
                total = int(v.get("total") or 0)
                nok = 0
                try:
                    nok = int(sc.get("OK") or 0)
                except Exception:
                    nok = 0
                # top no-OK
                pairs = []
                for k, cnt in sc.items():
                    if str(k) == "OK":
                        continue
                    try:
                        pairs.append((str(k), int(cnt)))
                    except Exception:
                        continue
                pairs.sort(key=lambda x: x[1], reverse=True)
                top = ", ".join([f"{k}={c}" for k, c in pairs[:4]]) if pairs else "—"
                extra.append(
                    f"| {v.get('audit_version')} | {total} | {nok} | {int(v.get('ok_with_bs') or 0)} | {int(v.get('ok_valid') or 0)} | {top} |\n"
                )
            extra.append("\n")

            # Diagnóstico (OK): por que OK_with_bs >> OK_valid?
            extra.append("**Diagnóstico dos OK (por versão): buckets de |difference_pct|**\n\n")
            extra.append(
                "_Leitura: `OK valid` tende a ser aproximadamente o bucket `2% ≤ |difference_pct| ≤ 10%` (dependendo da regra vigente)._\n\n"
            )
            extra.append("| audit_version | OK diff nulo | OK |diff|<2% | OK 2–10% | OK |diff|>10% |\n")
            extra.append("|---|---:|---:|---:|---:|\n")
            for v in rep.get("by_version") or []:
                if not isinstance(v, dict):
                    continue
                extra.append(
                    f"| {v.get('audit_version')} | {int(v.get('ok_diff_null') or 0)} | {int(v.get('ok_absdiff_lt2') or 0)} | {int(v.get('ok_absdiff_2_10') or 0)} | {int(v.get('ok_absdiff_gt10') or 0)} |\n"
                )
            extra.append("\n")

            # Top api_error (quando houver) para explicar API_FAILED/NO_SESSION/etc.
            tev = rep.get("top_errors_by_version") if isinstance(rep.get("top_errors_by_version"), dict) else {}
            if tev:
                extra.append("**Top erros (api_error) por versão**\n\n")
                for ver, xs in tev.items():
                    if not isinstance(xs, list) or not xs:
                        continue
                    extra.append(f"- `{ver}`:\n")
                    for it in xs[:5]:
                        if not isinstance(it, dict):
                            continue
                        st = str(it.get("status") or "NA")
                        err = str(it.get("api_error") or "").strip()
                        n = int(it.get("n") or 0)
                        if err:
                            err = (err[:180] + "…") if len(err) > 180 else err
                            extra.append(f"  - {st} ×{n}: `{err}`\n")
                    extra.append("\n")
    except Exception:
        pass

    combined_md = day_dir / "report_daily.md"
    insample_wrapped = ""
    if insample_txt.strip():
        insample_wrapped = "## 3) In-sample (detalhe)\n\n" + _demote_h2_to_h3(insample_txt.strip() + "\n")

    oos_annex = ""
    if oos_as_annex and oos_txt.strip():
        oos_annex = "## Anexo A) OOS walk-forward (Seção 12)\n\n" + _demote_h2_to_h3(oos_txt.strip() + "\n")

    # Ajuste adicional (capacidade): sensibilidade por banca com efeito do gate de slippage.
    # Usa (a) curvas base exportadas pelo OOS (wf_bank_sensitivity.json) e (b) contrafactual observado (execuções com placar).
    try:
        sens = _read_json(bank_sens_json) if "bank_sens_json" in locals() else None
        cf_src = None
        if isinstance(adh_long, dict) and isinstance(adh_long.get("slippage_filter_counterfactual"), dict):
            cf_src = adh_long.get("slippage_filter_counterfactual")
        elif isinstance(adh_short, dict) and isinstance(adh_short.get("slippage_filter_counterfactual"), dict):
            cf_src = adh_short.get("slippage_filter_counterfactual")
        if isinstance(cf_src, dict):
            # Sempre imprime um bloco diagnóstico, mesmo se o JSON de sensibilidade estiver ausente (para não “sumir” no PDF).
            sens_ok = bool(isinstance(sens, dict) and isinstance(sens.get("scenarios"), dict) and sens.get("scenarios"))
            block = []
            block.append("\n### Ajuste operacional: Sensibilidade por banca com gate de slippage (contrafactual)\n\n")
            block.append(
                "_Leitura: aplica a regra `Back: pula slippage_raw_pct<=-2%` e `Lay: pula slippage_raw_pct>2%` "
                "como um ajuste de capacidade, usando a evidência contrafactual nas execuções cobertas por placar. "
                "O ajuste é um **proxy**: usa exposição observada (Back=stake, Lay=liability) para estimar redução de N/turnover e mudança de ROI._\n\n"
            )
            try:
                block.append(
                    f"- Fonte OOS (curvas por banca): `{str(bank_sens_json)}` (existe={('sim' if (bank_sens_json.exists() if 'bank_sens_json' in locals() else False) else 'não')}; "
                    f"sens_ok={('sim' if sens_ok else 'não')}).\n\n"
                )
                if isinstance(sens, dict) and isinstance(sens.get("warn"), str) and sens.get("warn"):
                    block.append(f"- Aviso do export: `{sens.get('warn')}`.\n\n")
            except Exception:
                pass

            if not sens_ok:
                block.append(
                    "_Aviso: não foi possível aplicar o ajuste na sensibilidade por banca porque o export `wf_bank_sensitivity.json` está ausente/vazio/ilegível. "
                    "Isso não afeta o OOS em si; apenas impede esta tabela ajustada. "
                    "Se persistir, verifique se o daily está rodando a versão mais recente do `analyze_contexto_operacao_b808_robust_report.py` com "
                    "`--wf-export-bank-sensitivity-json` habilitado._\n\n"
                )
            else:
                sens = sens or {}
                back = cf_src.get("back") if isinstance(cf_src.get("back"), dict) else {}
                lay = cf_src.get("lay") if isinstance(cf_src.get("lay"), dict) else {}
                # "exposição" do contrafactual (apenas cobertura com placar+odd).
                # Back: stake. Lay: preferir liability (sempre existe no contrafactual); se stake existir (versões novas),
                # podemos usar stake, mas não dependemos dele.
                back_exp_base = float(back.get("stake") or 0.0)
                back_exp_filt = float(back.get("stake_filtered") or 0.0)
                lay_exp_base = (
                    float(lay.get("liability") or 0.0)
                    if lay.get("liability") is not None
                    else float(lay.get("stake") or 0.0)
                )
                lay_exp_filt = (
                    float(lay.get("liability_filtered") or 0.0)
                    if lay.get("liability_filtered") is not None
                    else float(lay.get("stake_filtered") or 0.0)
                )
                exp_base = float(back_exp_base) + float(lay_exp_base)
                exp_filt = float(back_exp_filt) + float(lay_exp_filt)
                pnl_base = float(back.get("pnl") or 0.0) + float(lay.get("pnl") or 0.0)
                pnl_filt = float(back.get("pnl_filtered") or 0.0) + float(lay.get("pnl_filtered") or 0.0)
                n_base = int(back.get("n") or 0) + int(lay.get("n") or 0)
                n_filt = int(back.get("n_filtered") or 0) + int(lay.get("n_filtered") or 0)
                exp_factor = _safe_div(exp_filt, exp_base)
                n_factor = _safe_div(n_filt, n_base)
                roi_base = _safe_div(pnl_base, exp_base)
                roi_filt = _safe_div(pnl_filt, exp_filt)
                roi_factor = _safe_div(roi_filt, roi_base) if (roi_base is not None and roi_filt is not None and roi_base != 0) else None
                profit_factor = _safe_div(pnl_filt, pnl_base) if pnl_base != 0 else None

                block.append(
                    f"- Fatores (da janela contrafactual): pass_exposição≈`{_fmt_num((exp_factor*100.0) if exp_factor is not None else None,2)}%`, "
                    f"pass_N≈`{_fmt_num((n_factor*100.0) if n_factor is not None else None,2)}%`, "
                    f"ROI_mult≈`{_fmt_num(roi_factor,3)}` , lucro_mult≈`{_fmt_num(profit_factor,3)}`.\n\n"
                )
                if exp_factor is None or profit_factor is None:
                    block.append(
                        f"_Aviso: ajuste não pôde ser aplicado (exp_base={_fmt_num(exp_base,2)}, exp_filt={_fmt_num(exp_filt,2)}, pnl_base={_fmt_num(pnl_base,2)})._\n\n"
                    )
                else:
                    scen = sens.get("scenarios") if isinstance(sens.get("scenarios"), dict) else {}
                    name_map = {
                        "12.2b_base": "1.2b (base)",
                        "12.2c_eq4_signals_sqrt": "1.2c (EQ 4%/4% cap50%, signals_sqrt)",
                        "12.2e_eq4_fixed": "1.2e (EQ 4%/4% cap50%, fixed)",
                        "12.2d_eq2_signals_sqrt": "1.2d (EQ 2%/2% cap33%, signals_sqrt)",
                    }
                    for name, payload in scen.items():
                        rows = payload.get("rows") if isinstance(payload, dict) else None
                        if not isinstance(rows, list) or not rows:
                            continue
                        ttl = name_map.get(str(name), str(name))
                        block.append(f"**{ttl} — com gate de slippage (ajuste proxy)**\n\n")
                        block.append("| Banca (ref) | Turnover 30d (adj, proxy) | Lucro 30d (adj) | ROI/banca 30d (adj) | n_after_budget (adj) |\n")
                        block.append("|---:|---:|---:|---:|---:|\n")
                        for r in rows:
                            if not isinstance(r, dict):
                                continue
                            br = r.get("bank_ref")
                            t0 = _safe_float(r.get("turn_30d"))
                            p0 = _safe_float(r.get("profit_30d_exp"))
                            beff = _safe_float(r.get("bank_eff"))
                            n0 = None
                            try:
                                h = r.get("limit_hits") if isinstance(r.get("limit_hits"), dict) else {}
                                n0 = int(h.get("n_after_budget")) if h.get("n_after_budget") is not None else None
                            except Exception:
                                n0 = None
                            # turnover: proxy pelo pass_exposição (principalmente para Lay, onde exposição é liability)
                            t1 = (float(t0) * float(exp_factor)) if t0 is not None else None
                            p1 = (float(p0) * float(profit_factor)) if p0 is not None else None
                            roi_bank = (float(p1) / float(beff) * 100.0) if (p1 is not None and beff is not None and beff > 0) else None
                            n1 = int(round(float(n0) * float(n_factor))) if (n0 is not None and n_factor is not None) else None
                            block.append(
                                f"| {_fmt_num(br,2)} | {_fmt_num(t1,2)} | {_fmt_num(p1,2)} | {_fmt_num(roi_bank,2)}% | {n1 if n1 is not None else '—'} |\n"
                            )
                        block.append("\n")

            add_txt = "".join(block)
            if oos_as_annex and oos_annex:
                oos_annex = oos_annex + _demote_h2_to_h3(add_txt)
            else:
                extra.append("\n\n## Anexo B) Ajuste operacional (slippage gate × capacidade)\n\n")
                extra.append(add_txt)
    except Exception:
        pass
    combined_core = "".join(s0) + "".join(s1) + insample_wrapped
    combined_md.write_text(combined_core + "".join(extra) + oos_annex, encoding="utf-8")

    # 5) PDF
    pdf = day_dir / "report_daily.pdf"
    renderer = Path(__file__).resolve().parent.parent / "docs" / "render_markdown_to_pdf.py"
    subprocess.run([sys.executable, str(renderer), str(combined_md), str(pdf)], check=True)

    out = {
        "ts": ts.isoformat(),
        "day_dir": str(day_dir),
        "pdf": str(pdf),
        "policy_current": str(cfg.wf_policy_current),
    }

    # 6) Telegram
    if cfg.send_telegram:
        token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
        if token and chat_id and pdf.exists():
            ok = _telegram_send_document(token, chat_id, file_path=pdf, caption=f"Relatório diário BetinAsia ({day})")
            out["telegram_sent"] = bool(ok)
        else:
            out["telegram_sent"] = False

    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Relatório diário completo: OOS + execution KPIs + accounting + PDF + Telegram.")
    ap.add_argument("--out-dir", default=os.getenv("DAILY_REPORT_OUT_DIR", "logs/daily_reports"))
    args = ap.parse_args()

    logger.remove()
    logger.add(sys.stderr, level=os.getenv("LOG_LEVEL", "INFO"))

    # Se rodando manualmente, garante que .env seja carregado antes do cfg.
    _load_env_file(Path(os.getenv("ENV_FILE", ".env")))

    cfg = DailyReportCfg(out_dir=Path(str(args.out_dir)))
    import asyncio

    out = asyncio.run(run_daily_full(cfg))
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

