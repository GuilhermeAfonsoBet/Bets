"""Telegram PREVIEW sender for Daily V2 (fail-open; never touches V1)."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import requests

from .io_atomic import atomic_write_json
from .preview_labels import (
    PREVIEW_BANNER,
    build_telegram_caption,
    markdown_has_preview_label,
    validate_preview_artifacts,
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _redact_secrets(text: str) -> str:
    if not text:
        return ""
    # never leak bot tokens in logs
    return re.sub(r"bot\d+:[A-Za-z0-9_-]+", "bot***:REDACTED", str(text))


def send_document_with_message_id(
    *,
    token: str,
    chat_id: str,
    file_path: Path,
    caption: str,
) -> Tuple[bool, Optional[int], Optional[int], str]:
    """Returns (ok, http_status, message_id, error_redacted)."""
    url = f"https://api.telegram.org/bot{token}/sendDocument"
    try:
        with file_path.open("rb") as f:
            files = {"document": (file_path.name, f, "application/pdf")}
            data = {"chat_id": chat_id, "caption": caption[:1024]}
            r = requests.post(url, data=data, files=files, timeout=90)
        msg_id = None
        if r.ok:
            try:
                payload = r.json()
                msg_id = int((((payload.get("result") or {}) if isinstance(payload, dict) else {}) or {}).get("message_id"))
            except Exception:
                msg_id = None
            return True, int(r.status_code), msg_id, ""
        return False, int(r.status_code), None, _redact_secrets(str(r.text or "")[:500])
    except Exception as e:
        return False, None, None, _redact_secrets(str(e)[:240])


def maybe_send_telegram_preview(
    *,
    root: Path,
    out_dir: Path,
    day_s: str,
    run_id: str,
    snap: Dict[str, Any],
    md_text: str,
    pdf_path: Path,
    parity_status: str,
    telegram_preview_enabled: bool,
    official: bool,
) -> Dict[str, Any]:
    """Send V2 PDF as PREVIEW if enabled. Never marks official. Fail-open."""
    evidence: Dict[str, Any] = {
        "run_id": run_id,
        "report_type": "DAILY_V2_PREVIEW",
        "generated_at_utc": snap.get("generated_at_utc"),
        "report_cutoff_utc": (snap.get("parity") or {}).get("v2_comparison_cutoff_utc")
        or snap.get("report_cutoff_utc"),
        "telegram_preview_enabled": bool(telegram_preview_enabled),
        "official_flag": bool(official),
        "telegram_destination": None,
        "telegram_send_started_at_utc": None,
        "telegram_send_finished_at_utc": None,
        "telegram_message_id": None,
        "telegram_document_name": pdf_path.name if pdf_path else None,
        "telegram_status": "SKIPPED",
        "error": None,
    }

    evid_path = out_dir / f"h3bup_daily_v2_telegram_preview_{day_s}_{run_id}.json"

    if official:
        evidence["telegram_status"] = "SKIPPED"
        evidence["error"] = "OFFICIAL=1 not allowed in this intervention; refuse send as official"
        atomic_write_json(evid_path, evidence)
        return evidence

    if not telegram_preview_enabled:
        evidence["telegram_status"] = "SKIPPED"
        evidence["error"] = "H3BUP_DAILY_V2_TELEGRAM_PREVIEW=0"
        atomic_write_json(evid_path, evidence)
        return evidence

    ok_label, label_reason = validate_preview_artifacts(md=md_text, pdf_name=pdf_path.name)
    if not ok_label or not markdown_has_preview_label(md_text):
        evidence["telegram_status"] = "PREVIEW_LABEL_VALIDATION_FAILED"
        evidence["error"] = label_reason
        atomic_write_json(evid_path, evidence)
        return evidence

    if not pdf_path.exists() or pdf_path.stat().st_size < 64:
        evidence["telegram_status"] = "FAILED"
        evidence["error"] = "pdf_missing_or_partial"
        atomic_write_json(evid_path, evidence)
        return evidence

    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    # optional override destination for preview
    chat_id = os.getenv("H3BUP_DAILY_V2_TELEGRAM_CHAT_ID", chat_id).strip()
    evidence["telegram_destination"] = f"chat_id={chat_id[:4]}***" if chat_id else None

    if not token or not chat_id:
        evidence["telegram_status"] = "FAILED"
        evidence["error"] = "missing_token_or_chat"
        atomic_write_json(evid_path, evidence)
        return evidence

    caption = build_telegram_caption(
        report_date_utc=str(snap.get("report_date_utc")),
        report_cutoff_utc=str(
            (snap.get("parity") or {}).get("v2_comparison_cutoff_utc") or snap.get("report_cutoff_utc")
        ),
        generated_at_utc=str(snap.get("generated_at_utc")),
        schema_version=snap.get("schema_version"),
        run_id=run_id,
        parity_status=parity_status,
    )
    if PREVIEW_BANNER not in caption and "PREVIEW / NÃO OFICIAL" not in caption:
        evidence["telegram_status"] = "PREVIEW_LABEL_VALIDATION_FAILED"
        evidence["error"] = "caption_missing_preview_label"
        atomic_write_json(evid_path, evidence)
        return evidence

    evidence["telegram_send_started_at_utc"] = _utcnow().isoformat()
    retries = max(1, int(float(os.getenv("DAILY_TELEGRAM_RETRIES", "2") or 2)))
    last_err = ""
    ok = False
    msg_id = None
    http_st = None
    for _ in range(retries):
        ok, http_st, msg_id, last_err = send_document_with_message_id(
            token=token, chat_id=chat_id, file_path=pdf_path, caption=caption
        )
        if ok:
            break
    evidence["telegram_send_finished_at_utc"] = _utcnow().isoformat()
    evidence["telegram_message_id"] = msg_id
    evidence["http_status"] = http_st
    if ok:
        evidence["telegram_status"] = "SENT"
        evidence["error"] = None
    else:
        evidence["telegram_status"] = "FAILED"
        evidence["error"] = last_err or "telegram_send_failed"

    # Ensure token never persisted
    blob = json.dumps(evidence)
    if token and token in blob:
        evidence["error"] = "redacted_token_leak_prevented"
    atomic_write_json(evid_path, evidence)
    evidence["_evidence_path"] = str(evid_path)
    return evidence
