"""Markdown renderer — derives ONLY from canonical JSON (no recalculation)."""

from __future__ import annotations

from typing import Any, Dict, List


def _m(metric: Any) -> str:
    if not isinstance(metric, dict):
        return "—"
    st = metric.get("status")
    val = metric.get("value")
    if st in {"MISSING", "STALE", "FAILED", "NOT_IMPLEMENTED", "NOT_DUE", "NOT_APPLICABLE", "UNAVAILABLE_STALE", "INSUFFICIENT_N"} and val is None:
        return f"`{st}`"
    if val is None:
        return f"`{st or 'MISSING'}`"
    unit = metric.get("unit") or ""
    if unit == "percent":
        try:
            return f"{float(val):.2f}% [{st}]"
        except Exception:
            return f"{val} [{st}]"
    if unit == "fraction":
        try:
            return f"{float(val)*100:.2f}% [{st}]"
        except Exception:
            return f"{val} [{st}]"
    return f"{val} [{st}]"


def render_markdown(snapshot: Dict[str, Any]) -> str:
    lines: List[str] = []
    a = lines.append
    # Mandatory preview identity (Telegram/PDF validation depends on this exact banner).
    a("# DAILY V2 — PREVIEW / NÃO OFICIAL\n\n")
    a("> Este relatório está em validação shadow. Não substitui o Daily V1 oficial das 22:00 UTC.\n\n")
    a(f"# H3BUP Daily V2 — {snapshot.get('report_date_utc')}\n")
    a("## 0) Manifesto\n")
    a("- status: `SHADOW / PREVIEW / NÃO OFICIAL`\n")
    a(f"- report_type: `{snapshot.get('report_type')}`\n")
    a(f"- report_date_utc: `{snapshot.get('report_date_utc')}`\n")
    a(f"- window: `{snapshot.get('window_start_utc')}` → `{snapshot.get('window_end_utc')}`\n")
    a(f"- report_cutoff_utc: `{snapshot.get('report_cutoff_utc')}`\n")
    parity = snapshot.get("parity") or {}
    if parity:
        a(f"- v1_report_cutoff_utc: `{parity.get('v1_report_cutoff_utc')}`\n")
        a(f"- v2_comparison_cutoff_utc: `{parity.get('v2_comparison_cutoff_utc')}`\n")
        a(f"- parity_status: `{parity.get('parity_status')}`\n")
    a(f"- generated_at_utc: `{snapshot.get('generated_at_utc')}`\n")
    a(f"- schema_version: `{snapshot.get('schema_version')}`\n")
    a(f"- run_id: `{snapshot.get('run_id')}`\n")
    a(f"- git_commit: `{snapshot.get('git_commit')}`\n")
    a(f"- policy: `{snapshot.get('policy_id')}` / `{snapshot.get('policy_version')}`\n")
    a(f"- report_health: `{(snapshot.get('report_health') or {}).get('status')}`\n")
    a(f"- data_quality: `{snapshot.get('data_quality')}`\n")
    a(f"- statistical_readiness: `{snapshot.get('statistical_readiness')}`\n\n")

    a("## 1) Resumo executivo\n")
    fun = snapshot.get("execution_funnel") or {}
    sett = snapshot.get("settlement") or {}
    perf = snapshot.get("performance") or {}
    a(f"- LIVE_OK: {_m(fun.get('live_ok'))}\n")
    a(f"- maturity: `{sett.get('maturity_status')}`\n")
    a(f"- open/settled/void/missing: `{sett.get('n_open')}` / `{sett.get('n_settled')}` / `{sett.get('n_void_push')}` / `{sett.get('n_missing_accounting')}`\n")
    a(f"- ROI settled (principal): {_m(perf.get('roi_settled'))}\n")
    a(f"- ROIw Total v1 (complementar): {_m(perf.get('roiw_total_v1'))}\n")
    a(f"- ROIw Total v2: {_m(perf.get('roiw_total_v2'))}\n")
    a("- Nenhuma conclusão de edge sem evidência estatística suficiente.\n\n")

    a("## 2) Policy e configuração efectiva\n")
    a(f"- policy_id: `{snapshot.get('policy_id')}`\n")
    a(f"- policy_version: `{snapshot.get('policy_version')}`\n")
    a("- stake alvo H3BUP: `10` USD\n")
    a("- odd band: `1.85–2.15`; capacity `dry.limit_final > 100`; slippage_pre_pct `< 0`\n\n")

    a("## 3) Data health\n")
    a("| Fonte | Status | Cutoff | Age(s) |\n|---|---|---|---:|\n")
    for name, meta in (snapshot.get("source_manifest") or {}).items():
        a(
            f"| {name} | `{meta.get('status')}` | `{meta.get('source_cutoff_utc') or meta.get('mtime_utc')}` | {meta.get('age_seconds')} |\n"
        )
    a("\n")

    a("## 4) Funil operacional\n")
    a(f"- LIVE_OK (coorte created_at UTC): {_m(fun.get('live_ok'))}\n")
    fb = fun.get("fast_buckets") or {}
    a(f"- DAILY_FAST_LE_6S: `{fb.get('DAILY_FAST_LE_6S')}`\n")
    a(f"- STUDY_FAST_LT_4S (exploratório): `{fb.get('STUDY_FAST_LT_4S')}`\n")
    a(f"- PRE_SUBMIT_MS_NA: `{fb.get('PRE_SUBMIT_MS_NA')}`\n\n")

    a("## 5) Settlement e performance\n")
    a(f"- stake placed: `{sett.get('stake_placed_sum')}`\n")
    a(f"- stake settled: `{sett.get('stake_settled_sum')}`\n")
    a(f"- pnl settled: `{sett.get('pnl_settled_sum')}`\n")
    a(f"- ROI settled: {_m(perf.get('roi_settled'))}\n")
    a(f"- ROIw Total v1: {_m(perf.get('roiw_total_v1'))}\n")
    a(f"- ROIw Total v2: {_m(perf.get('roiw_total_v2'))}\n")
    a(f"- principal_metric: `{perf.get('principal_metric')}`\n\n")

    a("## 6) Qualidade de preço / CLV\n")
    clv = snapshot.get("clv") or {}
    a(f"- collection: `{clv.get('collection_status')}` started `{clv.get('collection_started_at_utc')}`\n")
    a(f"- POST_5M strict: {_m(clv.get('post_5m_valid_strict'))}\n")
    a(f"- POST_15M strict: {_m(clv.get('post_15m_valid_strict'))}\n")
    a(f"- CLOSING strict: {_m(clv.get('closing_valid_strict'))}\n")
    a(f"- fair edge: {_m(clv.get('fair_edge'))}\n")
    a(f"- funnel: `{clv.get('funnel')}`\n\n")

    a("## 7) Latência E2E\n")
    lat = snapshot.get("latency") or {}
    a(f"- DAILY_FAST_LE_6S: {_m(lat.get('daily_fast_le_6s'))}\n")
    a(f"- STUDY_FAST_LT_4S: {_m(lat.get('study_fast_lt_4s'))}\n")
    a(f"- E2E WS→LIVE_OK: {_m(lat.get('e2e_ws_to_live_ok'))}\n")
    a(f"- detect→audit overhead: {_m(lat.get('detect_to_audit_overhead'))}\n")
    a(f"- e2e_source_status: `{lat.get('e2e_source_status')}`\n\n")

    a("## 8) Concentração\n")
    a(f"- `{snapshot.get('concentration')}`\n\n")

    a("## 9) Excepções e alertas\n")
    ex = snapshot.get("exceptions") or []
    if not ex:
        a("- nenhum\n\n")
    else:
        for e in ex:
            a(f"- [{e.get('severity')}] `{e.get('alert_id')}` evidence={e.get('evidence')}\n")
        a("\n")

    a("## 10) Mudanças vs relatório anterior\n")
    a("- _comparação incremental gerida pelo runner shadow / compare_v1_\n\n")

    a("## 11) Metodologia e linhagem\n")
    for k, v in (snapshot.get("methodology") or {}).items():
        a(f"- **{k}**: {v}\n")
    a("\n")
    a("---\n\n")
    a("**DAILY V2 — PREVIEW / NÃO OFICIAL** — Uso: validação técnica e metodológica. "
      "Não utilizar este preview como substituto do relatório oficial.\n")
    return "".join(lines)
