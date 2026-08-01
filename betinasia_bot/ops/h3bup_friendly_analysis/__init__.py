"""H3BUP_vNext Friendly vs Non-Friendly historical analysis (read-only).

Reporting-only package. Does not alter policy, stake, filters, execution,
accounting, CLV workers, timers, or Telegram publication.
"""

from __future__ import annotations

FRIENDLY_CLASSIFICATION_VERSION = "FRIENDLY_CLASS_V1_20260731"
POLICY_ID = "H3BUP_vNext"
POLICY_VERSION_EXACT = "H3BUP_vNext_20260629"
POLICY_START_UTC = "2026-06-29T00:00:00+00:00"

CLASSES = ("FRIENDLY", "NON_FRIENDLY", "UNCLASSIFIED", "CONFLICT")

__all__ = [
    "FRIENDLY_CLASSIFICATION_VERSION",
    "POLICY_ID",
    "POLICY_VERSION_EXACT",
    "POLICY_START_UTC",
    "CLASSES",
]
