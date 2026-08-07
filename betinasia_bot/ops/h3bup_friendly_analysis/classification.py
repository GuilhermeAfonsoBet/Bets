"""Versioned Friendly classification — freeze BEFORE joining P&L/settlement/CLV.

friendly_classification_version = FRIENDLY_CLASS_V1_20260731

Source priority:
  1. structured competition flag
  2. competition_type / league_type
  3. normalized league/competition name
  4. normalized tournament name
  5. event name (last fallback)
  6. insufficient evidence → UNCLASSIFIED

CONFLICT when structured flag disagrees with name evidence.
UNCLASSIFIED must never be coerced to NON_FRIENDLY.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from . import FRIENDLY_CLASSIFICATION_VERSION

RULE_VERSION = FRIENDLY_CLASSIFICATION_VERSION

# Explicit Friendly name patterns (word-boundary). Rule ids are stable.
FRIENDLY_NAME_RULES: Tuple[Tuple[str, re.Pattern[str]], ...] = (
    ("R_CLUB_FRIENDLY", re.compile(r"\bclub\s+friendl(y|ies)\b", re.I)),
    ("R_INTL_FRIENDLY", re.compile(r"\binternational\s+friendl(y|ies)\b", re.I)),
    ("R_WOMEN_INTL_FRIENDLY", re.compile(r"\bwomen(?:'s)?\s+international\s+friendl(y|ies)\b", re.I)),
    ("R_YOUTH_INTL_FRIENDLY", re.compile(r"\byouth\s+international\s+friendl(y|ies)\b", re.I)),
    ("R_U19_FRIENDLY", re.compile(r"\bu-?19\s+friendl(y|ies)\b", re.I)),
    ("R_U21_FRIENDLY", re.compile(r"\bu-?21\s+friendl(y|ies)\b", re.I)),
    ("R_OLYMPIC_FRIENDLY", re.compile(r"\bolympic\s+friendl(y|ies)\b", re.I)),
    ("R_FRIENDLY_MATCH", re.compile(r"\bfriendly\s+match(es)?\b", re.I)),
    ("R_FRIENDLIES", re.compile(r"\bfriendlies\b", re.I)),
    ("R_FRIENDLY", re.compile(r"\bfriendly\b", re.I)),
    ("R_AMISTOSO", re.compile(r"\bamistos[oa]s?\b", re.I)),
)

# Structured type tokens that mean Friendly / Official
_STRUCT_FRIENDLY = frozenset(
    {
        "friendly",
        "friendlies",
        "club_friendly",
        "club_friendlies",
        "international_friendly",
        "international_friendlies",
        "amistoso",
        "amistosos",
        "friendly_match",
    }
)
_STRUCT_OFFICIAL = frozenset(
    {
        "official",
        "league",
        "cup",
        "championship",
        "tournament",
        "competitive",
        "domestic",
        "international_competition",
        "premier",
        "division",
    }
)


@dataclass(frozen=True)
class ClassificationResult:
    friendly_class: str
    friendly_source: str
    friendly_rule_id: str
    friendly_rule_version: str
    friendly_confidence: str
    friendly_raw_value: str
    friendly_normalized_value: str
    friendly_conflict_reason: str


def normalize_text(value: Any) -> str:
    """lowercase, trim, strip accents, normalize hyphen/spaces; empty if none."""
    if value is None:
        return ""
    s = str(value).strip()
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _name_match(norm: str) -> Optional[Tuple[str, str]]:
    if not norm:
        return None
    for rule_id, pat in FRIENDLY_NAME_RULES:
        if pat.search(norm):
            return rule_id, norm
    return None


def _struct_token(raw: Any) -> str:
    return normalize_text(raw).replace(" ", "_")


def _structured_verdict(raw: Any) -> Optional[str]:
    """Return FRIENDLY / NON_FRIENDLY / None from structured flag or type."""
    tok = _struct_token(raw)
    if not tok:
        return None
    if tok in _STRUCT_FRIENDLY or tok.endswith("_friendly") or "friendly" in tok.split("_"):
        # Avoid treating e.g. random tokens; require explicit friendly token presence
        if "friendly" in tok or "amistoso" in tok or tok in _STRUCT_FRIENDLY:
            return "FRIENDLY"
    if tok in _STRUCT_OFFICIAL:
        return "NON_FRIENDLY"
    # boolean-ish
    if tok in {"1", "true", "yes", "y"}:
        return "FRIENDLY"
    if tok in {"0", "false", "no", "n"}:
        return "NON_FRIENDLY"
    return None


def classify_entity(
    *,
    structured_flag: Any = None,
    competition_type: Any = None,
    league_type: Any = None,
    league_name: Any = None,
    competition_name: Any = None,
    tournament_name: Any = None,
    event_name: Any = None,
) -> ClassificationResult:
    """Classify one entity. Never consults P&L/settlement/CLV/ROI/odds."""
    version = RULE_VERSION
    raw_preserve = {
        "structured_flag": None if structured_flag is None else str(structured_flag),
        "competition_type": None if competition_type is None else str(competition_type),
        "league_type": None if league_type is None else str(league_type),
        "league_name": None if league_name is None else str(league_name),
        "competition_name": None if competition_name is None else str(competition_name),
        "tournament_name": None if tournament_name is None else str(tournament_name),
        "event_name": None if event_name is None else str(event_name),
    }

    # 1) structured flag
    struct_sources: List[Tuple[str, Any]] = [
        ("structured_flag", structured_flag),
        ("competition_type", competition_type),
        ("league_type", league_type),
    ]
    struct_verdict: Optional[str] = None
    struct_source = ""
    struct_raw = ""
    for src, val in struct_sources:
        v = _structured_verdict(val)
        if v is not None:
            struct_verdict = v
            struct_source = src
            struct_raw = str(val)
            break

    # 3–5) name evidence
    name_layers: List[Tuple[str, Any]] = [
        ("league_name", league_name),
        ("competition_name", competition_name),
        ("tournament_name", tournament_name),
        ("event_name", event_name),
    ]
    name_hit: Optional[Tuple[str, str, str]] = None  # source, rule_id, norm
    for src, val in name_layers:
        norm = normalize_text(val)
        hit = _name_match(norm)
        if hit:
            name_hit = (src, hit[0], hit[1])
            break

    # CONFLICT: structured official vs friendly name (or vice versa)
    if struct_verdict == "NON_FRIENDLY" and name_hit is not None:
        return ClassificationResult(
            friendly_class="CONFLICT",
            friendly_source=f"{struct_source}+{name_hit[0]}",
            friendly_rule_id="R_CONFLICT_STRUCT_VS_NAME",
            friendly_rule_version=version,
            friendly_confidence="conflict",
            friendly_raw_value=struct_raw,
            friendly_normalized_value=name_hit[2],
            friendly_conflict_reason=(
                f"structured:{struct_source}={struct_raw} implies NON_FRIENDLY; "
                f"name:{name_hit[0]} matched {name_hit[1]}"
            ),
        )
    if struct_verdict == "FRIENDLY" and name_hit is None:
        # structured friendly alone is enough
        return ClassificationResult(
            friendly_class="FRIENDLY",
            friendly_source=struct_source,
            friendly_rule_id="R_STRUCT_FRIENDLY",
            friendly_rule_version=version,
            friendly_confidence="high",
            friendly_raw_value=struct_raw,
            friendly_normalized_value=normalize_text(struct_raw),
            friendly_conflict_reason="",
        )
    if struct_verdict == "FRIENDLY" and name_hit is not None:
        return ClassificationResult(
            friendly_class="FRIENDLY",
            friendly_source=struct_source,
            friendly_rule_id="R_STRUCT_FRIENDLY",
            friendly_rule_version=version,
            friendly_confidence="high",
            friendly_raw_value=struct_raw,
            friendly_normalized_value=normalize_text(struct_raw),
            friendly_conflict_reason="",
        )
    if struct_verdict == "NON_FRIENDLY" and name_hit is None:
        return ClassificationResult(
            friendly_class="NON_FRIENDLY",
            friendly_source=struct_source,
            friendly_rule_id="R_STRUCT_OFFICIAL",
            friendly_rule_version=version,
            friendly_confidence="high",
            friendly_raw_value=struct_raw,
            friendly_normalized_value=normalize_text(struct_raw),
            friendly_conflict_reason="",
        )

    # No structured verdict — use name hierarchy
    if name_hit is not None:
        src, rule_id, norm = name_hit
        conf = "medium" if src != "event_name" else "low"
        return ClassificationResult(
            friendly_class="FRIENDLY",
            friendly_source=src,
            friendly_rule_id=rule_id,
            friendly_rule_version=version,
            friendly_confidence=conf,
            friendly_raw_value=str(raw_preserve.get(src) or ""),
            friendly_normalized_value=norm,
            friendly_conflict_reason="",
        )

    # Name present but not Friendly → NON_FRIENDLY only when league/competition name exists
    for src in ("league_name", "competition_name"):
        raw = raw_preserve.get(src)
        norm = normalize_text(raw)
        if norm:
            return ClassificationResult(
                friendly_class="NON_FRIENDLY",
                friendly_source=src,
                friendly_rule_id="R_NAME_OFFICIAL_DEFAULT",
                friendly_rule_version=version,
                friendly_confidence="medium",
                friendly_raw_value=str(raw),
                friendly_normalized_value=norm,
                friendly_conflict_reason="",
            )

    # Tournament-only non-friendly name without friendly match
    t_raw = raw_preserve.get("tournament_name")
    t_norm = normalize_text(t_raw)
    if t_norm:
        return ClassificationResult(
            friendly_class="NON_FRIENDLY",
            friendly_source="tournament_name",
            friendly_rule_id="R_TOURNAMENT_OFFICIAL_DEFAULT",
            friendly_rule_version=version,
            friendly_confidence="low",
            friendly_raw_value=str(t_raw),
            friendly_normalized_value=t_norm,
            friendly_conflict_reason="",
        )

    # Event name alone without Friendly pattern is NOT enough to call non-Friendly
    return ClassificationResult(
        friendly_class="UNCLASSIFIED",
        friendly_source="none",
        friendly_rule_id="R_UNCLASSIFIED",
        friendly_rule_version=version,
        friendly_confidence="none",
        friendly_raw_value="",
        friendly_normalized_value="",
        friendly_conflict_reason="",
    )


def extract_classification_inputs(order: Dict[str, Any]) -> Dict[str, Any]:
    """Pull classification fields from an order-level dict without performance fields.

    Intentionally ignores keys: pnl, settlement*, clv*, roi*, odd_final (post), accounting*.
    """
    blocked = {
        "pnl",
        "settlement_status",
        "settlement_ts",
        "roi",
        "clv_post_5m",
        "clv_post_15m",
        "clv_closing",
        "accounting_amount",
        "accounting_status",
        "odd_final",  # post-decision price — not used for Friendly class
    }

    def dig(*keys: str) -> Any:
        for k in keys:
            if k in blocked:
                continue
            if order.get(k) not in (None, ""):
                return order.get(k)
        # nested meta/request/result/raw
        for nest in ("meta", "request", "result", "raw", "shadow", "policy"):
            node = order.get(nest)
            if isinstance(node, dict):
                for k in keys:
                    if k in blocked:
                        continue
                    if node.get(k) not in (None, ""):
                        return node.get(k)
        return None

    return {
        "structured_flag": dig(
            "is_friendly",
            "friendly_flag",
            "competition_is_friendly",
            "league_is_friendly",
        ),
        "competition_type": dig("competition_type", "comp_type"),
        "league_type": dig("league_type"),
        "league_name": dig("league_name", "league", "competition_league"),
        "competition_name": dig("competition_name", "competition"),
        "tournament_name": dig("tournament_name", "tournament"),
        "event_name": dig("event_name", "match_name"),
    }


def classify_order_identity(order: Dict[str, Any]) -> ClassificationResult:
    """Classify using only identity/competition fields (no performance join)."""
    inp = extract_classification_inputs(order)
    return classify_entity(**inp)


def build_classification_mapping(
    orders: Sequence[Dict[str, Any]],
    *,
    id_key: str = "order_id",
) -> List[Dict[str, Any]]:
    """Build frozen mapping rows. Must be called before performance join."""
    rows: List[Dict[str, Any]] = []
    for o in orders:
        oid = str(o.get(id_key) or "").strip()
        if not oid:
            # fallback identity already resolved upstream; skip empty
            continue
        res = classify_order_identity(o)
        rows.append(
            {
                "order_id": oid,
                "event_id": str(o.get("event_id") or ""),
                "league_id": str(o.get("league_id") or ""),
                "league_name": o.get("league_name") or o.get("league") or "",
                "competition_id": str(o.get("competition_id") or ""),
                "competition_name": o.get("competition_name") or o.get("competition") or "",
                "tournament_name": o.get("tournament_name") or "",
                "event_name": o.get("event_name") or "",
                "friendly_classification_version": RULE_VERSION,
                **asdict(res),
            }
        )
    rows.sort(key=lambda r: r["order_id"])
    return rows


def mapping_checksum(rows: Sequence[Dict[str, Any]]) -> str:
    """SHA256 over canonical JSON of mapping (class fields only)."""
    canon = [
        {
            "order_id": r.get("order_id"),
            "friendly_class": r.get("friendly_class"),
            "friendly_source": r.get("friendly_source"),
            "friendly_rule_id": r.get("friendly_rule_id"),
            "friendly_rule_version": r.get("friendly_rule_version"),
            "friendly_raw_value": r.get("friendly_raw_value"),
            "friendly_normalized_value": r.get("friendly_normalized_value"),
            "friendly_conflict_reason": r.get("friendly_conflict_reason"),
        }
        for r in rows
    ]
    blob = json.dumps(canon, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def rules_document() -> Dict[str, Any]:
    return {
        "friendly_classification_version": RULE_VERSION,
        "classes": ["FRIENDLY", "NON_FRIENDLY", "UNCLASSIFIED", "CONFLICT"],
        "source_priority": [
            "structured_flag",
            "competition_type/league_type",
            "league_name/competition_name",
            "tournament_name",
            "event_name",
            "UNCLASSIFIED",
        ],
        "name_rules": [{"rule_id": rid, "pattern": pat.pattern} for rid, pat in FRIENDLY_NAME_RULES],
        "notes": [
            "Classification must be frozen (checksum) before joining P&L/settlement/CLV.",
            "UNCLASSIFIED is never treated as NON_FRIENDLY.",
            "Word-boundary regex; original text preserved in friendly_raw_value.",
            "Event name alone without Friendly pattern does not imply NON_FRIENDLY.",
        ],
        "freeze_contract": {
            "forbidden_inputs": ["pnl", "settlement", "clv", "roi", "odd_final", "result"],
        },
    }


def write_freeze_artifacts(
    out_dir: Path,
    rows: Sequence[Dict[str, Any]],
    *,
    run_id: str,
) -> Dict[str, Path]:
    """Write mapping CSV, rules JSON, checksum TXT. Call before performance join."""
    out_dir.mkdir(parents=True, exist_ok=True)
    mapping_path = out_dir / f"h3bup_friendly_classification_mapping_{run_id}.csv"
    rules_path = out_dir / f"h3bup_friendly_classification_rules_{run_id}.json"
    checksum_path = out_dir / f"h3bup_friendly_classification_checksum_{run_id}.txt"

    fields = [
        "order_id",
        "event_id",
        "league_id",
        "league_name",
        "competition_id",
        "competition_name",
        "tournament_name",
        "event_name",
        "friendly_classification_version",
        "friendly_class",
        "friendly_source",
        "friendly_rule_id",
        "friendly_rule_version",
        "friendly_confidence",
        "friendly_raw_value",
        "friendly_normalized_value",
        "friendly_conflict_reason",
    ]
    import csv

    with mapping_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})

    rules_path.write_text(json.dumps(rules_document(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    cs = mapping_checksum(rows)
    checksum_path.write_text(
        f"friendly_classification_version={RULE_VERSION}\n"
        f"n_rows={len(rows)}\n"
        f"sha256={cs}\n",
        encoding="utf-8",
    )
    return {"mapping": mapping_path, "rules": rules_path, "checksum": checksum_path, "sha256": cs}  # type: ignore[dict-item]
