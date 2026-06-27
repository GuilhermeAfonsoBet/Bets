#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gera relatorio executivo em Markdown/PDF para:
- 5Ms por segmento pre/pos x com/sem World Cup
- impacto estatistico do slippage do executor no ROI

O script e intencionalmente autocontido para rodar na VPS usando arquivos ja
materializados em /tmp e logs/accounting/executor_live.jsonl.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _pf(x: Any) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip().replace(" ", "")
    if not s:
        return None
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".") if s.rfind(",") > s.rfind(".") else s.replace(",", "")
    elif "," in s:
        s = s.replace(",", ".")
    try:
        v = float(s)
    except Exception:
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def _norm(s: str) -> str:
    return "".join(ch for ch in str(s or "").lower() if ch.isalnum())


def _get_path(obj: Any, path: Sequence[str]) -> Any:
    cur = obj
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _extract_ids(obj: Any, out: set[str]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            kn = _norm(k)
            if kn in {"orderid", "order_id", "ticketid", "betid"}:
                sv = str(v or "").strip()
                if sv:
                    out.add(sv)
            _extract_ids(v, out)
    elif isinstance(obj, list):
        for x in obj:
            _extract_ids(x, out)


def _fmt(x: Optional[float], nd: int = 2, pct: bool = False) -> str:
    if x is None:
        return "NA"
    return f"{float(x):.{nd}f}" + ("%" if pct else "")


def _roi(rows: Sequence[Dict[str, Any]]) -> Tuple[float, float, float]:
    stake = sum(float(r["stake"]) for r in rows)
    pnl = sum(float(r["pnl"]) for r in rows)
    return (100.0 * pnl / stake if stake > 0 else float("nan"), pnl, stake)


def _latest_balance_csv() -> Path:
    cands = sorted(glob.glob("/home/betbot/Bets/betinasia_bot/logs/accounting/*balance*.csv"), key=os.path.getmtime)
    if not cands:
        raise RuntimeError("balance CSV nao encontrado")
    return Path(cands[-1])


def _build_pnl_by_order(balance_csv: Path) -> Dict[str, float]:
    out: Dict[str, float] = defaultdict(float)
    with balance_csv.open(encoding="utf-8", errors="ignore", newline="") as f:
        rd = csv.DictReader(f)
        fields = list(rd.fieldnames or [])
        by = {_norm(x): x for x in fields}

        def pick(cands: Sequence[str]) -> Optional[str]:
            for c in cands:
                hit = by.get(_norm(c))
                if hit:
                    return hit
            return None

        oid_col = pick(["order id", "order_id", "orderid", "bet id", "ticket id", "ticket_id"])
        val_col = pick(["amount", "pnl_real", "pnl", "profit", "pl", "result", "net"])
        if not oid_col or not val_col:
            raise RuntimeError(f"balance sem colunas esperadas: oid={oid_col} val={val_col}")
        for r in rd:
            oid = str(r.get(oid_col, "")).strip()
            v = _pf(r.get(val_col))
            if oid and v is not None:
                out[oid] += float(v)
    return out


def _executor_slippage_rows(jsonl_path: Path, balance_csv: Path, start_day: str, split_day: str) -> List[Dict[str, Any]]:
    pnl_by_order = _build_pnl_by_order(balance_csv)
    rows: List[Dict[str, Any]] = []
    with jsonl_path.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "2026-" not in line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            req = obj.get("request") if isinstance(obj.get("request"), dict) else {}
            res = obj.get("result") if isinstance(obj.get("result"), dict) else {}
            if str(res.get("status") or "").upper() != "LIVE_OK":
                continue
            if str(res.get("exec_side") or req.get("exec_side") or "").lower() != "back":
                continue
            raw = res.get("raw") if isinstance(res.get("raw"), dict) else {}
            vs = raw.get("value_sizing") if isinstance(raw.get("value_sizing"), dict) else {}
            regime = str(vs.get("market_regime") or _get_path(req, ["meta", "market", "regime"]) or "").lower()
            market_live = vs.get("market_is_live")
            if not (regime == "pre" or market_live is False):
                continue
            created = str(res.get("created_at") or req.get("created_at") or "")
            day = created[:10]
            if day < start_day:
                continue
            slip = _pf(vs.get("slippage_pre_pct"))
            if slip is None:
                continue
            order_ids: set[str] = set()
            _extract_ids(obj, order_ids)
            order_id = ""
            for oid in sorted(order_ids):
                if oid in pnl_by_order:
                    order_id = oid
                    break
            if not order_id:
                continue
            stake = _pf(vs.get("stake_chosen"))
            if stake is None:
                stake = _pf(_get_path(res, ["policy", "stake_requested"])) or _pf(_get_path(req, ["policy", "stake_requested"]))
            if stake is None or stake <= 0:
                continue
            pnl = pnl_by_order[order_id]
            rows.append(
                {
                    "day": day,
                    "period": "pre" if day < split_day else "pos",
                    "slip": float(slip),
                    "pnl": float(pnl),
                    "stake": float(stake),
                    "roi": 100.0 * float(pnl) / float(stake),
                    "order_id": order_id,
                }
            )
    return rows


def _bootstrap_diff(neg: Sequence[Dict[str, Any]], non: Sequence[Dict[str, Any]], iters: int, seed: int) -> Tuple[Optional[float], Optional[float]]:
    if len(neg) <= 1 or len(non) <= 1:
        return None, None
    rng = random.Random(seed)
    vals: List[float] = []
    for _ in range(max(1, int(iters))):
        a = [neg[rng.randrange(len(neg))] for _ in range(len(neg))]
        b = [non[rng.randrange(len(non))] for _ in range(len(non))]
        vals.append(_roi(a)[0] - _roi(b)[0])
    vals.sort()
    return vals[int(0.025 * (len(vals) - 1))], vals[int(0.975 * (len(vals) - 1))]


def _perm_p(neg: Sequence[Dict[str, Any]], non: Sequence[Dict[str, Any]], iters: int, seed: int) -> Optional[float]:
    if not neg or not non:
        return None
    rng = random.Random(seed + 11)
    rows = list(neg) + list(non)
    n_neg = len(neg)
    obs = _roi(neg)[0] - _roi(non)[0]
    ge = 0
    for _ in range(max(1, int(iters))):
        rng.shuffle(rows)
        stat = _roi(rows[:n_neg])[0] - _roi(rows[n_neg:])[0]
        if stat >= obs:
            ge += 1
    return (ge + 1) / (max(1, int(iters)) + 1)


def _spearman(rows: Sequence[Dict[str, Any]]) -> Optional[float]:
    if len(rows) <= 2:
        return None
    xs = [float(r["slip"]) for r in rows]
    ys = [float(r["roi"]) for r in rows]

    def ranks(arr: Sequence[float]) -> List[float]:
        idx = sorted(range(len(arr)), key=lambda i: arr[i])
        rk = [0.0] * len(arr)
        i = 0
        while i < len(arr):
            j = i
            while j + 1 < len(arr) and arr[idx[j + 1]] == arr[idx[i]]:
                j += 1
            avg = (i + j + 2) / 2.0
            for k in range(i, j + 1):
                rk[idx[k]] = avg
            i = j + 1
        return rk

    rx, ry = ranks(xs), ranks(ys)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    if vx <= 0 or vy <= 0:
        return None
    return cov / math.sqrt(vx * vy)


def _slippage_section(label: str, rows: Sequence[Dict[str, Any]], iters: int, seed: int) -> Dict[str, Any]:
    neg = [r for r in rows if float(r["slip"]) < 0]
    non = [r for r in rows if float(r["slip"]) >= 0]
    roi_all, pnl_all, stake_all = _roi(rows)
    roi_neg, pnl_neg, stake_neg = _roi(neg)
    roi_non, pnl_non, stake_non = _roi(non)
    ci_lo, ci_hi = _bootstrap_diff(neg, non, iters, seed)
    p_perm = _perm_p(neg, non, iters, seed)
    bins = []
    for lo, hi in [(-999, -2), (-2, -1), (-1, -0.5), (-0.5, 0), (0, 0.5), (0.5, 1), (1, 2), (2, 999)]:
        br = [r for r in rows if lo <= float(r["slip"]) < hi]
        if br:
            rr, pp, ss = _roi(br)
            bins.append({"range": f"[{lo},{hi})", "n": len(br), "roi": rr, "pnl": pp, "stake": ss})
    return {
        "label": label,
        "all": {"n": len(rows), "pnl": pnl_all, "stake": stake_all, "roi": roi_all},
        "neg": {"n": len(neg), "pnl": pnl_neg, "stake": stake_neg, "roi": roi_neg},
        "nonneg": {"n": len(non), "pnl": pnl_non, "stake": stake_non, "roi": roi_non},
        "diff_roi_pp": roi_neg - roi_non,
        "diff_ci95": [ci_lo, ci_hi],
        "perm_p_one_sided": p_perm,
        "spearman": _spearman(rows),
        "bins": bins,
    }


def _five_ms_tables(five_ms_json: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    data = json.loads(five_ms_json.read_text(encoding="utf-8"))
    segs = data["segments"]
    targets = {
        "M1": "p_perm <= 0.10 e CI90_lo > 0",
        "M2": "ROI sem Top-3 > 0 e Top1_abs <= 35%",
        "M3": "pos_ratio >= 55% e mediana semanal > 0",
        "M4": ">=2 blocos positivos e R3 > 0",
        "M5": "EV/aposta > 0 e EV% > 0",
    }
    rows: List[Dict[str, str]] = []
    for s in segs:
        core = s["core"]
        name = s["name"]
        roi = core["roi_pct"]
        rows.extend(
            [
                {
                    "segment": name,
                    "m": "M1",
                    "value": f"ROI={_fmt(roi,2,True)}; p_perm={_fmt(s['M1']['p_perm'],4)}; CI90=[{_fmt(s['M1']['ci90_lo'],2,True)}, {_fmt(s['M1']['ci90_hi'],2,True)}]",
                    "target": targets["M1"],
                    "status": s["M1"]["status"],
                },
                {
                    "segment": name,
                    "m": "M2",
                    "value": f"ROI sem Top-3={_fmt(s['M2']['roi_sem_top3'],2,True)}; Top1_abs={_fmt(s['M2']['top1_abs_pct'],2,True)}",
                    "target": targets["M2"],
                    "status": s["M2"]["status"],
                },
                {
                    "segment": name,
                    "m": "M3",
                    "value": f"pos_ratio={_fmt(s['M3']['pos_ratio_pct'],2,True)}; mediana semanal={_fmt(s['M3']['mediana_semanal_pct'],2,True)}",
                    "target": targets["M3"],
                    "status": s["M3"]["status"],
                },
                {
                    "segment": name,
                    "m": "M4",
                    "value": f"R1={_fmt(s['M4']['r1_pct'],2,True)}; R2={_fmt(s['M4']['r2_pct'],2,True)}; R3={_fmt(s['M4']['r3_pct'],2,True)}",
                    "target": targets["M4"],
                    "status": s["M4"]["status"],
                },
                {
                    "segment": name,
                    "m": "M5",
                    "value": f"EV/aposta={_fmt(s['M5']['ev_por_aposta'],4)}; EV%={_fmt(s['M5']['ev_pct'],2,True)}",
                    "target": targets["M5"],
                    "status": s["M5"]["status"],
                },
            ]
        )
    return segs, rows


def _render_md(out_md: Path, five_ms_json: Path, slip_sections: Sequence[Dict[str, Any]], base_csv: str) -> None:
    segs, m_rows = _five_ms_tables(five_ms_json)
    lines: List[str] = []
    lines.append("# Relatorio executivo - 5Ms e impacto do slippage no ROI\n")
    lines.append("## Sumario executivo\n")
    lines.append("- A regra operacional Back Pre com `slippage_pre_pct < 0` foi validada como protecao prudencial, especialmente no recorte pos-25/05.")
    lines.append("- A evidencia economica pos-25/05 favorece `slippage < 0`, mas o teste ainda nao prova significancia estatistica isolada.")
    lines.append("- O M5 foi corrigido para avaliar EV nominal e EV%, removendo payoff_ratio como criterio de aprovacao.\n")
    lines.append("## Escopo e fontes\n")
    lines.append(f"- Base 5Ms: `{base_csv}`")
    lines.append(f"- JSON 5Ms: `{five_ms_json}`")
    lines.append("- Janela comparavel: a partir de 2026-04-19; split pre/pos em 2026-05-25.")
    lines.append("- Segmentacao World Cup por aliases: World Cup, FIFA World Cup, Club World Cup, FIFA Club World Cup, World Championship.\n")
    lines.append("## Resumo 5Ms\n")
    lines.append("| Segmento | N apostas | N eventos | ROI | Score | Label |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for s in segs:
        c = s["core"]
        lines.append(f"| {s['name']} | {c['n_bets']} | {c['n_events']} | {_fmt(c['roi_pct'],2,True)} | {s['score_5ms']}/5 | {s['label']} |")
    lines.append("\n## 5Ms expandido\n")
    lines.append("| Segmento | M | Valor calculado | Target | Status |")
    lines.append("|---|---|---|---|---|")
    for r in m_rows:
        lines.append(f"| {r['segment']} | {r['m']} | {r['value']} | {r['target']} | {r['status']} |")
    lines.append("\n## Teste estatistico: slippage do executor e ROI\n")
    for sec in slip_sections:
        lines.append(f"### {sec['label']}\n")
        lines.append("| Grupo | N | P&L | Stake | ROI |")
        lines.append("|---|---:|---:|---:|---:|")
        for key, label in [("all", "Todos"), ("neg", "slippage < 0"), ("nonneg", "slippage >= 0")]:
            g = sec[key]
            lines.append(f"| {label} | {g['n']} | {_fmt(g['pnl'],2)} | {_fmt(g['stake'],2)} | {_fmt(g['roi'],3,True)} |")
        ci = sec["diff_ci95"]
        lines.append("")
        lines.append(f"- Diferenca ROI (`<0` - `>=0`): **{_fmt(sec['diff_roi_pp'],3)} p.p.**")
        lines.append(f"- IC95 bootstrap da diferenca: [{_fmt(ci[0],3)}, {_fmt(ci[1],3)}] p.p.")
        lines.append(f"- p_perm one-sided: {_fmt(sec['perm_p_one_sided'],5)}")
        lines.append(f"- Spearman(slippage, ROI): {_fmt(sec['spearman'],4)}")
        lines.append("")
        lines.append("| Faixa slippage | N | ROI | P&L | Stake |")
        lines.append("|---|---:|---:|---:|---:|")
        for b in sec["bins"]:
            lines.append(f"| {b['range']} | {b['n']} | {_fmt(b['roi'],3,True)} | {_fmt(b['pnl'],2)} | {_fmt(b['stake'],2)} |")
        lines.append("")
    lines.append("## Recomendacao operacional\n")
    lines.append("- Manter `slippage executor < 0` como regra de protecao operacional para Back Pre.")
    lines.append("- Registrar tambem os bloqueios (`CAP_BLOCKED`) com telemetria completa, para analise contrafactual futura.")
    lines.append("- Reavaliar thresholds finos apos acumular amostra bloqueada: `[-0.5%,0)`, `<-2%`, interacoes com liga, World Cup, stake e pre_submit_ms.")
    lines.append("- Nao interpretar slippage isolado como alpha estatisticamente provado neste momento; a evidencia e economica/prudencial, ainda em validacao.\n")
    out_md.write_text("\n".join(lines), encoding="utf-8")


def _render_pdf(md_path: Path, pdf_path: Path) -> None:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except Exception as e:
        raise SystemExit(f"reportlab ausente ou indisponivel: {e}")

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=7.5, leading=9))
    styles.add(ParagraphStyle(name="ExecutiveBody", parent=styles["BodyText"], fontSize=9.2, leading=12))
    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4, rightMargin=1.1 * cm, leftMargin=1.1 * cm, topMargin=1.0 * cm, bottomMargin=1.0 * cm)
    story: List[Any] = []

    def para(text: str, style: str = "ExecutiveBody") -> None:
        story.append(Paragraph(text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"), styles[style]))

    def table_from_md(header: List[str], rows: List[List[str]], widths: Optional[List[float]] = None) -> None:
        data = [[Paragraph(str(x), styles["Small"]) for x in header]]
        for row in rows:
            data.append([Paragraph(str(x), styles["Small"]) for x in row])
        tbl = Table(data, colWidths=widths, repeatRows=1)
        tbl.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d1d5db")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f9fafb")]),
                    ("LEFTPADDING", (0, 0), (-1, -1), 3),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                    ("TOPPADDING", (0, 0), (-1, -1), 2),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ]
            )
        )
        story.append(tbl)
        story.append(Spacer(1, 8))

    lines = md_path.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        ln = lines[i].strip()
        if not ln:
            story.append(Spacer(1, 5))
            i += 1
            continue
        if ln.startswith("# "):
            story.append(Paragraph(ln[2:], styles["Title"]))
            story.append(Spacer(1, 8))
            i += 1
            continue
        if ln.startswith("## "):
            story.append(Paragraph(ln[3:], styles["Heading2"]))
            story.append(Spacer(1, 5))
            i += 1
            continue
        if ln.startswith("### "):
            story.append(Paragraph(ln[4:], styles["Heading3"]))
            story.append(Spacer(1, 4))
            i += 1
            continue
        if ln.startswith("|"):
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i].strip())
                i += 1
            parsed = [[c.strip() for c in row.strip("|").split("|")] for row in block]
            if len(parsed) >= 2:
                header = parsed[0]
                rows = [r for r in parsed[2:]]
                widths = None
                if len(header) == 5 and header[1] == "M":
                    widths = [2.55 * cm, 0.8 * cm, 6.1 * cm, 5.0 * cm, 1.4 * cm]
                table_from_md(header, rows, widths)
            continue
        if ln.startswith("- "):
            para("• " + ln[2:])
            i += 1
            continue
        para(ln)
        i += 1
    doc.build(story)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--five-ms-json", required=True)
    ap.add_argument("--base-csv", default="/tmp/base_5ms_real_20260419_20260626_executor_slip.csv")
    ap.add_argument("--executor-jsonl", default="/home/betbot/Bets/betinasia_bot/logs/executor_live.jsonl")
    ap.add_argument("--balance-csv", default="")
    ap.add_argument("--start-day", default="2026-04-19")
    ap.add_argument("--split-day", default="2026-05-25")
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--out-pdf", required=True)
    ap.add_argument("--out-json", default="")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    five_ms_json = Path(args.five_ms_json)
    if not five_ms_json.exists():
        raise SystemExit(f"JSON 5Ms nao encontrado: {five_ms_json}")
    balance_csv = Path(args.balance_csv) if str(args.balance_csv or "").strip() else _latest_balance_csv()
    rows = _executor_slippage_rows(Path(args.executor_jsonl), balance_csv, args.start_day, args.split_day)
    sections = [
        _slippage_section(f"Desde {args.start_day}", rows, args.iters, args.seed),
        _slippage_section(f"Pre ate {args.split_day}", [r for r in rows if r["day"] < args.split_day], args.iters, args.seed + 1),
        _slippage_section(f"Pos desde {args.split_day}", [r for r in rows if r["day"] >= args.split_day], args.iters, args.seed + 2),
    ]
    out_md = Path(args.out_md)
    out_pdf = Path(args.out_pdf)
    _render_md(out_md, five_ms_json, sections, args.base_csv)
    _render_pdf(out_md, out_pdf)
    if str(args.out_json or "").strip():
        out = {"five_ms_json": str(five_ms_json), "balance_csv": str(balance_csv), "slippage_sections": sections}
        Path(args.out_json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] MD: {out_md}")
    print(f"[OK] PDF: {out_pdf}")
    if args.out_json:
        print(f"[OK] JSON: {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
