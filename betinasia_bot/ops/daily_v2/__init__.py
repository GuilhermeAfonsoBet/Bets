"""H3BUP Daily V2 — reporting only (fail-open, shadow-capable).

Does not alter policy, stake, execution, accounting, E2E collectors, or CLV workers.
"""

SCHEMA_VERSION = 2
DAILY_FAST_LE_6S_MS = 6000
STUDY_FAST_LT_4S_MS = 4000

__all__ = [
    "SCHEMA_VERSION",
    "DAILY_FAST_LE_6S_MS",
    "STUDY_FAST_LT_4S_MS",
]
