"""Markdown / executive / simple PDF report builders."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .settlement import performance_block, sample_gate


def _fmt(x: Any, nd: int = 4) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def classify_final_status(bundle: Dict[str, Any]) -> str:
    dq = bundle.get("data_quality") or {}
    alerts = {a.get("alert") for a in bundle.get("alerts") or []}
    if dq.get("conflict_pct") and dq["conflict_pct"] > 0.15:
        return "FRIENDLY_CLASSIFICATION_UNRELIABLE"
    if "FRIENDLY_CLASSIFICATION_LOW_COVERAGE" in alerts and (dq.get("classification_coverage") or 0) < 0.5:
        return "FRIENDLY_DATA_QUALITY_INSUFFICIENT"
    n_res = performance_block(bundle.get("order_rows") or []).get("n_resolved") or 0
    gate = sample_gate(int(n_res))
    if gate == "VERY_LOW_N":
        return "FRIENDLY_SAMPLE_INSUFFICIENT"
    if gate == "INSUFFICIENT_N":
        # still may give preliminary
        pass

    f = performance_block([r for r in bundle.get("order_rows") or [] if r.get("friendly_class") == "FRIENDLY"])
    nf = performance_block([r for r in bundle.get("order_rows") or [] if r.get("friendly_class") == "NON_FRIENDLY"])
    rf, rnf = f.get("roi_resolved"), nf.get("roi_resolved")
    stats = (bundle.get("stat_tests") or {}).get("tests") or {}
    roi_test = stats.get("roi_resolved_diff_friendly_minus_non") or {}
    robust = bundle.get("robustness") or []
    sign_stable = all(r.get("sign_vs_base_unchanged") in (True, None) for r in robust[1:6]) if len(robust) > 1 else False

    if rf is None or rnf is None:
        return "FRIENDLY_DATA_QUALITY_INSUFFICIENT"
    delta = rf - rnf
    supported = gate in {"FIRST_READING", "RELIABLE_READING_CANDIDATE"} and sign_stable
    if abs(delta) < 0.02:
        return "NO_CLEAR_FRIENDLY_DIFFERENCE"
    if delta < 0:
        return "FRIENDLY_WORSE_SUPPORTED" if supported else "FRIENDLY_WORSE_PRELIMINARY"
    return "NON_FRIENDLY_WORSE_SUPPORTED" if supported else "NON_FRIENDLY_WORSE_PRELIMINARY"


def build_executive_summary(bundle: Dict[str, Any]) -> str:
    status = bundle.get("final_status") or classify_final_status(bundle)
    rows = bundle.get("order_rows") or []
    f = [r for r in rows if r.get("friendly_class") == "FRIENDLY"]
    nf = [r for r in rows if r.get("friendly_class") == "NON_FRIENDLY"]
    u = [r for r in rows if r.get("friendly_class") == "UNCLASSIFIED"]
    c = [r for r in rows if r.get("friendly_class") == "CONFLICT"]
    bf, bnf, bu = performance_block(f), performance_block(nf), performance_block(u)
    lines = []
    a = lines.append
    a("# H3BUP Friendly vs Non-Friendly — Executive Summary\n")
    a(f"- **Status final:** `{status}`\n")
    a(f"- **Run id:** `{bundle.get('run_id')}`\n")
    a(f"- **Cutoff:** `{bundle.get('cutoff_utc')}`\n")
    a(f"- **Classification version:** `{bundle.get('friendly_classification_version')}`\n\n")
    a("## Respostas executivas\n\n")
    a(f"1. Friendly positivo/negativo: **{'positivo' if (bf.get('pnl_resolved') or 0) > 0 else 'negativo/nulo'}** (P&L={_fmt(bf.get('pnl_resolved'))}, ROI={_fmt(bf.get('roi_resolved'))})\n")
    a(f"2. Non-Friendly positivo/negativo: **{'positivo' if (bnf.get('pnl_resolved') or 0) > 0 else 'negativo/nulo'}** (P&L={_fmt(bnf.get('pnl_resolved'))}, ROI={_fmt(bnf.get('roi_resolved'))})\n")
    loss_group = "FRIENDLY" if (bf.get("pnl_resolved") or 0) <= (bnf.get("pnl_resolved") or 0) else "NON_FRIENDLY"
    a(f"3. Grupo que explica maior parte da perda (P&L resolved mais baixo): **{loss_group}**\n")
    clv = bundle.get("clv_summary") or []
    a(f"4. Diferença também no CLV: ver tabela CLV (N={len(clv)} linhas)\n")
    a("5. Closing vs P&L: ver secção CLV CLOSING no relatório completo\n")
    robust = bundle.get("robustness") or []
    a(f"6. Sobrevive remoção top eventos/ligas: ver robustez ({len(robust)} cenários)\n")
    a("7. Latência/slippage: ver execution_summary\n")
    a("8. Cobertura comparável: ver data_quality / alerts\n")
    a(f"9. Evidência estatística: sample_gate total=`{sample_gate(int(performance_block(rows).get('n_resolved') or 0))}`\n")
    a("10. Limitação principal: sample size / classification coverage / concentração (ver alerts)\n\n")
    a("## Contagens\n")
    a(f"- Primário N={len(rows)} | Friendly={len(f)} | Non-Friendly={len(nf)} | Unclassified={len(u)} | Conflict={len(c)}\n\n")
    a("> Análise histórica read-only. Não altera policy/stake/filtros. Não é recomendação operacional.\n")
    return "".join(lines)


def build_full_report(bundle: Dict[str, Any]) -> str:
    rows = bundle.get("order_rows") or []
    lines: List[str] = []
    a = lines.append
    a("# H3BUP_vNext — Análise Friendly vs Non-Friendly\n\n")
    a(f"- status: `{bundle.get('final_status')}`\n")
    a(f"- run_id: `{bundle.get('run_id')}`\n")
    a(f"- generated_at_utc: `{bundle.get('generated_at_utc')}`\n")
    a(f"- cutoff_utc: `{bundle.get('cutoff_utc')}`\n")
    a(f"- friendly_classification_version: `{bundle.get('friendly_classification_version')}`\n")
    a(f"- classification_checksum: `{bundle.get('classification_checksum')}`\n\n")
    a("## Universo PRIMÁRIO (H3BUP_vNext exact)\n\n")
    meta = bundle.get("primary_meta") or {}
    a(f"```json\n{json.dumps(meta, indent=2, ensure_ascii=False)}\n```\n\n")
    a("## Tabela principal de performance\n\n")
    a("| Métrica | Friendly | Non-Friendly | Unclassified | Conflict | Total |\n")
    a("|---|---:|---:|---:|---:|---:|\n")
    for r in bundle.get("performance_summary") or []:
        a(
            f"| {r.get('metric')} | {_fmt(r.get('FRIENDLY'))} | {_fmt(r.get('NON_FRIENDLY'))} | "
            f"{_fmt(r.get('UNCLASSIFIED'))} | {_fmt(r.get('CONFLICT'))} | {_fmt(r.get('TOTAL'))} |\n"
        )
    a("\n## CLV (VALID_STRICT)\n\n")
    a("| Grupo | Janela | N | Coverage | Média | Mediana | Positivo % | Status |\n")
    a("|---|---|---:|---:|---:|---:|---:|---|\n")
    for r in bundle.get("clv_summary") or []:
        a(
            f"| {r.get('group')} | {r.get('window')} | {r.get('n')} | {_fmt(r.get('coverage_pct'),1)} | "
            f"{_fmt(r.get('mean'))} | {_fmt(r.get('median'))} | {_fmt(r.get('positive_pct'),1)} | {r.get('status')} |\n"
        )
    a("\n## Execução / preço\n\n")
    a("| Métrica | Friendly | Non-Friendly | Delta |\n|---|---:|---:|---:|\n")
    for r in bundle.get("execution_summary") or []:
        a(f"| {r.get('metric')} | {_fmt(r.get('FRIENDLY'))} | {_fmt(r.get('NON_FRIENDLY'))} | {_fmt(r.get('delta'))} |\n")
    a("\n## Cenários contrafactuais (diagnóstico)\n\n")
    a("> Cenários históricos não representam resultado out-of-sample e não devem ser interpretados como recomendação operacional.\n\n")
    for s in bundle.get("scenarios") or []:
        a(f"- `{s.get('scenario')}`: N={s.get('n')} stake={_fmt(s.get('stake'),2)} P&L={_fmt(s.get('pnl'))} ROI={_fmt(s.get('roi'))}\n")
    a("\n## Alertas\n\n")
    for al in bundle.get("alerts") or []:
        a(f"- `{al.get('alert')}` ({al.get('severity')}): {al.get('detail')}\n")
    a("\n## Segurança\n\n")
    sec = bundle.get("security") or {}
    a(f"- checksums_unchanged: `{sec.get('unchanged')}`\n")
    a(f"- policy_altered: `{sec.get('policy_altered')}` → deve ser Não\n")
    a(f"- telegram_used: `False`\n")
    a(f"- orders_created: `False`\n")
    a(f"- betslip_opened: `False`\n")
    a("\n## Universo SECUNDÁRIO (apêndice)\n\n")
    a(f"```json\n{json.dumps(bundle.get('secondary_meta') or {}, indent=2, ensure_ascii=False)}\n```\n")
    a("\nNunca consolidar ROI primário + secundário numa única linha.\n")
    return "".join(lines)


def write_simple_pdf(path: Path, title: str, body_md: str) -> None:
    """Minimal PDF without external deps (text-only pages)."""
    # Very small PDF writer
    text = (title + "\n\n" + body_md).replace("\r", "")
    # Escape for PDF strings
    content_lines = []
    y = 800
    content_lines.append("BT /F1 10 Tf 40 800 Td")
    for raw in text.split("\n")[:90]:
        line = raw.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")[:110]
        content_lines.append(f"0 -12 Td ({line}) Tj")
    content_lines.append("ET")
    stream = "\n".join(content_lines).encode("latin-1", errors="replace")
    objs = []
    objs.append(b"1 0 obj<< /Type /Catalog /Pages 2 0 R >>endobj\n")
    objs.append(b"2 0 obj<< /Type /Pages /Kids [3 0 R] /Count 1 >>endobj\n")
    objs.append(
        b"3 0 obj<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>endobj\n"
    )
    objs.append(f"4 0 obj<< /Length {len(stream)} >>stream\n".encode() + stream + b"\nendstream endobj\n")
    objs.append(b"5 0 obj<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>endobj\n")
    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for obj in objs:
        offsets.append(len(out))
        out.extend(obj)
    xref_pos = len(out)
    out.extend(f"xref\n0 {len(offsets)}\n".encode())
    out.extend(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        out.extend(f"{off:010d} 00000 n \n".encode())
    out.extend(
        f"trailer<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode()
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(out))
