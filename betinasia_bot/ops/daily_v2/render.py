"""Markdown renderer — derives ONLY from canonical JSON (no recalculation). P0 enriched."""

from __future__ import annotations

from typing import Any, Dict, List

from .formatters import fmt_age, fmt_int, fmt_money, fmt_ms, fmt_pct, fmt_ts, metric_cell


def _m(metric: Any, *, as_pct: bool = False, as_money: bool = False, as_ms: bool = False) -> str:
    return metric_cell(metric, as_pct=as_pct, as_money=as_money, as_ms=as_ms)


def _status_box(x: Any) -> str:
    if isinstance(x, dict):
        return f"`{x.get('status') or '—'}`"
    return f"`{x or '—'}`"


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
    a(f"- window: `{fmt_ts(snapshot.get('window_start_utc'))}` → `{fmt_ts(snapshot.get('window_end_utc'))}`\n")
    a(f"- report_cutoff_utc: `{fmt_ts(snapshot.get('report_cutoff_utc'))}`\n")
    parity = snapshot.get("parity") or {}
    cut = snapshot.get("cutoffs") or {}
    if parity or cut:
        a(f"- v1_report_cutoff_utc: `{fmt_ts(parity.get('v1_report_cutoff_utc') or cut.get('v1_report_cutoff_utc'))}`\n")
        a(f"- v2_comparison_cutoff_utc: `{fmt_ts(parity.get('v2_comparison_cutoff_utc') or cut.get('v2_comparison_cutoff_utc'))}`\n")
        a(f"- parity_status: `{parity.get('parity_status') or '—'}`\n")
    a(f"- generated_at_utc: `{fmt_ts(snapshot.get('generated_at_utc'))}`\n")
    a(f"- schema_version: `{snapshot.get('schema_version')}`\n")
    a(f"- run_id: `{snapshot.get('run_id')}`\n")
    a(f"- git_commit: `{snapshot.get('git_commit')}`\n")
    a(f"- policy: `{snapshot.get('policy_id')}` / `{snapshot.get('policy_version')}`\n")
    a(f"- REPORT_HEALTH: {_status_box(snapshot.get('report_health'))}\n")
    a(f"- OPERATIONS_HEALTH: {_status_box(snapshot.get('operations_health'))}\n")
    a(f"- DATA_QUALITY: {_status_box(snapshot.get('data_quality'))}\n")
    a(f"- STATISTICAL_READINESS: {_status_box(snapshot.get('statistical_readiness'))}\n\n")

    fun = snapshot.get("execution_funnel") or {}
    sett = snapshot.get("settlement") or {}
    perf = snapshot.get("performance") or {}

    a("## 1) Resumo executivo\n")
    a(f"- LIVE_OK: {_m(fun.get('live_ok'))}\n")
    a(f"- maturity: `{sett.get('maturity_status')}`\n")
    a(
        f"- open / settled decided / void / missing: "
        f"`{sett.get('n_open')}` / `{sett.get('n_settled')}` / `{sett.get('n_void_push')}` / `{sett.get('n_missing_accounting')}`\n"
    )
    a(f"- stake placed: `{fmt_money(sett.get('stake_placed') or sett.get('stake_placed_sum'))}`\n")
    a(f"- stake resolved total: `{fmt_money(sett.get('stake_resolved_total') or sett.get('stake_settled_sum'))}`\n")
    a(f"- stake void: `{fmt_money(sett.get('stake_void'))}`\n")
    a(f"- P&L resolved: `{fmt_money(sett.get('pnl_resolved') or sett.get('pnl_settled_sum'))}`\n")
    roi_principal = perf.get("roi_resolved") or perf.get("roi_settled")
    a(f"- ROI principal (`{perf.get('principal_metric') or 'roi_resolved'}`): {_m(roi_principal, as_pct=True)}\n")
    a(f"- ROI decided ex-void: {_m(perf.get('roi_decided_ex_void'), as_pct=True)}\n")
    a(f"- ROIw Total v2 (complementar): {_m(perf.get('roiw_total_v2'))}\n")
    a("- ROIw Total v1: ver apêndice de paridade V1×V2 (fora do resumo principal).\n")
    a("- Nenhuma conclusão de edge sem evidência estatística suficiente.\n\n")

    a("## 2) Health (4 dimensões)\n")
    a("| Dimensão | Status |\n|---|---|\n")
    a(f"| REPORT_HEALTH | {_status_box(snapshot.get('report_health'))} |\n")
    a(f"| OPERATIONS_HEALTH | {_status_box(snapshot.get('operations_health'))} |\n")
    a(f"| DATA_QUALITY | {_status_box(snapshot.get('data_quality'))} |\n")
    a(f"| STATISTICAL_READINESS | {_status_box(snapshot.get('statistical_readiness'))} |\n\n")

    a("## 3) Configuração (fingerprint / drift)\n")
    a("| Config | File status | Runtime status | Fingerprint | Drift |\n|---|---|---|---|---|\n")
    cfg = snapshot.get("config") or {}
    for name in ("policy", "risk_params"):
        ce = cfg.get(name) or {}
        a(
            f"| {name} | `{ce.get('file_status')}` | `{ce.get('runtime_status')}` | "
            f"`{ce.get('fingerprint')}` | `{ce.get('drift')}` |\n"
        )
    a("\n> `CURRENT_UNCHANGED` não é warning. `CONFIG_DRIFT` é CRITICAL.\n\n")

    a("## 4) Data health (fontes)\n")
    a("| Fonte | Status | Cutoff | Age |\n|---|---|---|---|\n")
    for name, meta in (snapshot.get("source_manifest") or {}).items():
        a(
            f"| {name} | `{meta.get('status')}` | `{fmt_ts(meta.get('source_cutoff_utc') or meta.get('mtime_utc'))}` | "
            f"{fmt_age(meta.get('age_seconds'))} |\n"
        )
    a("\n")

    a("## 5) Funil operacional\n")
    a(f"- Universo temporal: `{fun.get('window_label') or 'coorte created_at UTC'}`\n\n")
    a("| Etapa | N | % etapa anterior | % inicial | Status |\n|---|---:|---:|---:|---|\n")
    stages = fun.get("stages") or []
    if stages:
        for row in stages:
            pp = row.get("pct_prev")
            pi = row.get("pct_initial")
            a(
                f"| {row.get('step')} | {fmt_int(row.get('n'))} | "
                f"{('—' if pp is None else f'{pp:.1f}%')} | "
                f"{('—' if pi is None else f'{pi:.1f}%')} | `{row.get('status')}` |\n"
            )
    else:
        a(f"| LIVE_OK (coorte) | {_m(fun.get('live_ok'))} | — | — | AVAILABLE |\n")
    a("\n### Reasons / bloqueios\n")
    a("| Reason/status | N | % requests |\n|---|---:|---:|\n")
    brs = fun.get("block_reasons") or []
    if brs:
        for r in brs:
            pct = r.get("pct_requests")
            a(f"| {r.get('reason')} | {fmt_int(r.get('n'))} | {('—' if pct is None else f'{pct:.1f}%')} |\n")
    else:
        a("| — | 0 | — |\n")
    a("\n### Buckets de velocidade\n")
    fb = fun.get("fast_buckets") or {}
    daily = fb.get("DAILY_FAST_LE_6S") or {}
    study = fb.get("STUDY_FAST_LT_4S") or {}
    na = fb.get("PRE_SUBMIT_MS_NA") or {}
    n_daily = daily.get("n")
    n_study = study.get("n")
    n_na = na.get("n")
    a(f"- DAILY_FAST_LE_6S: `{n_daily}` de operações com pre_submit ≤ 6s\n")
    a(f"- STUDY_FAST_LT_4S (exploratório): `{n_study}` com pre_submit < 4s\n")
    a(f"- PRE_SUBMIT_MS_NA: N=`{n_na}` · coverage missing=`{_m(snapshot.get('latency', {}).get('pre_submit_ms_na'))}`\n\n")

    a("## 6) Settlement e performance\n")
    a("| Métrica | Valor |\n|---|---|\n")
    a(f"| stake_placed | {fmt_money(sett.get('stake_placed') or sett.get('stake_placed_sum'))} |\n")
    a(f"| stake_resolved_total | {fmt_money(sett.get('stake_resolved_total') or sett.get('stake_settled_sum'))} |\n")
    a(f"| stake_decided_ex_void | {fmt_money(sett.get('stake_decided_ex_void'))} |\n")
    a(f"| stake_void | {fmt_money(sett.get('stake_void'))} |\n")
    a(f"| stake_open | {fmt_money(sett.get('stake_open'))} |\n")
    a(f"| pnl_resolved | {fmt_money(sett.get('pnl_resolved') or sett.get('pnl_settled_sum'))} |\n")
    a(f"| pnl_decided_ex_void | {fmt_money(sett.get('pnl_decided_ex_void'))} |\n")
    a(f"| roi_resolved (principal) | {_m(perf.get('roi_resolved') or perf.get('roi_settled'), as_pct=True)} |\n")
    a(f"| roi_decided_ex_void | {_m(perf.get('roi_decided_ex_void'), as_pct=True)} |\n")
    a(f"| ROIw Total v2 | {_m(perf.get('roiw_total_v2'))} |\n")
    a(f"| maturity | `{sett.get('maturity_status')}` |\n\n")
    a("> Fórmulas: `roi_resolved = pnl_resolved / stake_resolved_total` (void no denominador); "
      "`roi_decided_ex_void = pnl_decided_ex_void / stake_decided_ex_void`.\n\n")

    # Friendly vs Non-Friendly (shadow diagnostic)
    from .friendly_section import render_friendly_markdown

    friendly = snapshot.get("friendly_breakdown") or {}
    if friendly:
        a(render_friendly_markdown(friendly))

    a("## 7) Qualidade de preço / CLV forward\n")
    clv = snapshot.get("clv") or {}
    a(f"- collection: `{clv.get('collection_status')}` started `{fmt_ts(clv.get('collection_started_at_utc'))}`\n")
    a(f"- source priority: `{clv.get('source_priority')}`\n")
    a(f"- collector: `{clv.get('collector_status')}`\n")
    a(f"- fair edge: {_m(clv.get('fair_edge'))}\n\n")
    a("### Funil CLV (diagnóstico)\n")
    funnel = clv.get("funnel") or {}
    for k, v in funnel.items():
        a(f"- {k}: `{v}`\n")
    a("\n### Cobertura por janela (VALID_STRICT)\n")
    a("| Janela | Expected | Due | Attempted | Strict valid | Coverage |\n|---|---:|---:|---:|---:|---:|\n")
    for row in clv.get("windows") or []:
        cov = row.get("coverage_pct")
        a(
            f"| {row.get('window')} | {fmt_int(row.get('expected'))} | {fmt_int(row.get('due'))} | "
            f"{fmt_int(row.get('attempted'))} | {fmt_int(row.get('strict_valid'))} | "
            f"{('—' if cov is None else f'{cov:.1f}%')} |\n"
        )
    a("\n### Performance CLV (VALID_STRICT)\n")
    a("| Janela | N | CLV médio | Mediana | Positivo % | Status |\n|---|---:|---:|---:|---:|---|\n")
    for row in clv.get("performance_rows") or list((clv.get("performance") or {}).values()):
        if not isinstance(row, dict):
            continue
        a(
            f"| {row.get('window')} | {fmt_int(row.get('n'))} | "
            f"{fmt_pct(row.get('clv_mean_pct'), already_percent=True)} | "
            f"{fmt_pct(row.get('clv_median_pct'), already_percent=True)} | "
            f"{fmt_pct(row.get('positive_pct'), already_percent=True)} | `{row.get('status')}` |\n"
        )
    a("\n")

    a("## 8) Latência E2E\n")
    e2e = snapshot.get("e2e") or {}
    lat = snapshot.get("latency") or {}
    a(f"- traces totais: `{fmt_int(e2e.get('n_traces') or lat.get('n_traces'))}`\n")
    a(f"- traces LIVE_OK: `{fmt_int(e2e.get('n_live_ok') or lat.get('n_live_ok_traces'))}`\n")
    a(f"- full-trace coverage: `{fmt_pct((e2e.get('full_trace_coverage_pct') or 0)/100.0) if e2e.get('full_trace_coverage_pct') is not None else '—'}`\n")
    a(f"- etapa dominante: `{e2e.get('dominant_stage') or lat.get('dominant_stage') or '—'}`\n")
    a(f"- ordering violations: `{fmt_int(e2e.get('ordering_violations') or lat.get('ordering_violations'))}`\n")
    a(f"- clock skew: `{fmt_int(e2e.get('clock_skew') or lat.get('clock_skew'))}`\n")
    a(f"- detect→audit overhead: {_m(lat.get('detect_to_audit_overhead') or e2e.get('detect_to_audit_overhead'), as_ms=True)}\n\n")
    a("| Métrica | N | Coverage | Mediana | p95 | Status |\n|---|---:|---:|---:|---:|---|\n")
    segs = e2e.get("segments") or lat.get("segments") or {}
    labels = [
        ("ws_to_detect", "WS→detect"),
        ("detect_to_audit", "detect→audit"),
        ("audit_to_bridge", "audit→bridge"),
        ("bridge_to_request", "bridge→request"),
        ("request_to_executor", "request→executor"),
        ("executor_to_dryrun", "executor→dry-run"),
        ("dryrun_duration", "dry-run duration"),
        ("dryrun_to_gate", "dry-run→gate"),
        ("gate_to_place", "gate→place"),
        ("place_duration", "place duration"),
        ("ws_to_live_ok", "WS→LIVE_OK"),
    ]
    for key, label in labels:
        m = segs.get(key) or {}
        if not isinstance(m, dict):
            a(f"| {label} | — | — | — | — | `MISSING` |\n")
            continue
        cov = m.get("coverage_pct")
        a(
            f"| {label} | {fmt_int(m.get('n'))} | "
            f"{('—' if cov is None else f'{float(cov):.1f}%')} | "
            f"{fmt_ms(m.get('value'))} | {fmt_ms(m.get('p95'))} | `{m.get('status')}` |\n"
        )
    a("\n")

    a("## 9) Excepções e alertas\n")
    ex = snapshot.get("exceptions") or []
    if not ex:
        a("- nenhum\n\n")
    else:
        a("| alert_id | severity | status | message |\n|---|---|---|---|\n")
        for e in ex:
            a(
                f"| `{e.get('alert_id')}` | `{e.get('severity')}` | `{e.get('status')}` | "
                f"{e.get('message') or e.get('evidence')} |\n"
            )
        a("\n")

    a("## 10) Mudanças versus V2 anterior\n")
    prev = snapshot.get("previous_diff") or {}
    if not prev or not prev.get("rows"):
        a("- sem snapshot V2 anterior comparável para esta coorte.\n\n")
    else:
        a(f"- previous_run_id: `{prev.get('previous_run_id')}` → current `{prev.get('current_run_id')}`\n\n")
        a("| Métrica | Anterior | Atual | Delta |\n|---|---:|---:|---:|\n")
        for r in prev.get("rows") or []:
            a(f"| {r.get('metric')} | {r.get('anterior')} | {r.get('atual')} | {r.get('delta')} |\n")
        a(f"\n- novos alertas: `{prev.get('new_alerts')}`\n")
        a(f"- alertas resolvidos: `{prev.get('resolved_alerts')}`\n\n")

    a("## 11) Paridade V1 × V2\n")
    a("| Campo | V1 | V2 | Status |\n|---|---|---|---|\n")
    a(f"| report_date | `{parity.get('v1_report_date') or snapshot.get('report_date_utc')}` | `{snapshot.get('report_date_utc')}` | "
      f"`{parity.get('parity_status') or '—'}` |\n")
    a(f"| cohort start | `{fmt_ts(parity.get('v1_cohort_start') or snapshot.get('window_start_utc'))}` | "
      f"`{fmt_ts(snapshot.get('window_start_utc'))}` | — |\n")
    a(f"| cohort end | `{fmt_ts(parity.get('v1_cohort_end') or snapshot.get('window_end_utc'))}` | "
      f"`{fmt_ts(snapshot.get('window_end_utc'))}` | — |\n")
    a(f"| parity cutoff | `{fmt_ts(parity.get('v1_report_cutoff_utc'))}` | "
      f"`{fmt_ts(parity.get('v2_comparison_cutoff_utc'))}` | `{parity.get('parity_status')}` |\n")
    a(f"| policy | `{parity.get('v1_policy') or 'H3BUP_vNext'}` | `{snapshot.get('policy_version')}` | — |\n")
    a(f"| LIVE_OK universe | `{parity.get('v1_live_ok') or '—'}` | {_m(fun.get('live_ok'))} | — |\n\n")
    a("### Apêndice métricas legado (paridade)\n")
    a("| Métrica | Fórmula | Universo | Inclui open? | Uso |\n|---|---|---|---|---|\n")
    a("| ROI principal | pnl_resolved/stake_resolved_total | settled+void | não | oficial V2 |\n")
    a("| ROIw v2 | settled-aware % | settled+void | não | complementar |\n")
    a(f"| ROIw v1 | {_m(perf.get('roiw_total_v1'))} | ledger join | potencialmente | paridade legado |\n\n")

    a("## 12) Metodologia e linhagem\n")
    for k, v in (snapshot.get("methodology") or {}).items():
        a(f"- **{k}**: {v}\n")
    a("\n")
    a("---\n\n")
    a("**DAILY V2 — PREVIEW / NÃO OFICIAL** — Uso: validação técnica e metodológica. "
      "Não utilizar este preview como substituto do relatório oficial.\n")
    return "".join(lines)
