"""Preview / official labelling for Daily V2 Telegram + PDF."""

from __future__ import annotations

PREVIEW_BANNER = "DAILY V2 — PREVIEW / NÃO OFICIAL"
PREVIEW_BANNER_EMOJI = "🧪 DAILY V2 — PREVIEW / NÃO OFICIAL"
PREVIEW_FOOTER = "DAILY V2 — PREVIEW / NÃO OFICIAL"
PREVIEW_FILENAME_PREFIX = "H3BUP_DAILY_V2_PREVIEW_"


def preview_pdf_filename(day_yyyymmdd: str, run_id: str) -> str:
    return f"{PREVIEW_FILENAME_PREFIX}{day_yyyymmdd}_{run_id}.pdf"


def build_telegram_caption(
    *,
    report_date_utc: str,
    report_cutoff_utc: str,
    generated_at_utc: str,
    schema_version,
    run_id: str,
    parity_status: str,
) -> str:
    lines = [
        PREVIEW_BANNER_EMOJI,
        "",
        "Este relatório está em validação shadow.",
        "Não substitui o Daily V1 oficial das 22:00 UTC.",
        "",
        f"Data da coorte: {report_date_utc} UTC",
        f"Cutoff lógico: {report_cutoff_utc}",
        f"Gerado em: {generated_at_utc}",
        f"Schema: Daily V2 / versão {schema_version}",
        f"Run ID: {run_id}",
        "Status: SHADOW",
        f"Comparação V1 × V2: {parity_status}",
        "",
        "Uso: validação técnica e metodológica.",
        "Não utilizar este preview como substituto do relatório oficial.",
    ]
    return "\n".join(lines)


def markdown_has_preview_label(md: str) -> bool:
    return PREVIEW_BANNER in (md or "")


def ensure_preview_markdown_banner(md: str) -> str:
    if markdown_has_preview_label(md):
        return md
    banner = (
        f"# {PREVIEW_BANNER}\n\n"
        "> Este relatório está em validação shadow. "
        "Não substitui o Daily V1 oficial das 22:00 UTC.\n\n"
    )
    return banner + (md or "")


def validate_preview_artifacts(*, md: str, pdf_name: str) -> tuple[bool, str]:
    if not markdown_has_preview_label(md):
        return False, "PREVIEW_LABEL_VALIDATION_FAILED: markdown missing PREVIEW banner"
    if PREVIEW_FILENAME_PREFIX not in pdf_name:
        return False, "PREVIEW_LABEL_VALIDATION_FAILED: pdf filename missing V2_PREVIEW"
    return True, "OK"
