from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from sqlalchemy import text

sys.path.insert(0, ".")

from storage.database import Database


def _normalize_league(name: str) -> str:
    s = str(name or "").strip().lower()
    if not s:
        return ""
    s = (
        unicodedata.normalize("NFKD", s)
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _load_policy(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"policy_json nao encontrado: {path}")
    obj = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"policy_json invalido: {path}")
    return obj


def _extract_approved_leagues(policy: Dict[str, Any], key_prefix: str) -> Tuple[List[str], List[str]]:
    steps = policy.get("steps") if isinstance(policy.get("steps"), list) else []
    last = steps[-1] if steps else {}
    active_keys = last.get("active_keys") if isinstance(last, dict) else []
    out: List[str] = []
    all_keys: List[str] = []
    for k in (active_keys or []):
        key = str(k or "").strip()
        if not key:
            continue
        all_keys.append(key)
        if key_prefix and (not key.startswith(key_prefix)):
            continue
        if "__" not in key:
            continue
        league = key.split("__", 1)[1].strip()
        if league:
            out.append(league)
    return sorted(set(out)), all_keys


async def _fetch_signals_by_league(
    db: Database,
    *,
    hours: float,
    hypothesis_type: str,
    reversal_direction: str,
    audit_version: str,
    status: str,
    exec_side: str,
    prematch_only: bool,
) -> List[Tuple[str, int]]:
    where: List[str] = [
        "r.audited_at >= now() - (:hours * interval '1 hour')",
        "r.hypothesis_type = :hypothesis_type",
        "r.reversal_direction = :reversal_direction",
        "r.status = :status",
        "r.is_valid_opportunity = TRUE",
        "lower(COALESCE(r.hypothesis_details->>'exec_side_hint','back')) = :exec_side",
    ]
    params: Dict[str, Any] = {
        "hours": float(hours),
        "hypothesis_type": str(hypothesis_type),
        "reversal_direction": str(reversal_direction),
        "status": str(status),
        "exec_side": str(exec_side).lower(),
    }

    if audit_version.strip():
        where.append("COALESCE(r.audit_version, r.hypothesis_details->>'audit_version') = :audit_version")
        params["audit_version"] = audit_version.strip()

    if prematch_only:
        where.append("(r.is_live IS NULL OR r.is_live = FALSE)")

    q = f"""
    SELECT
      COALESCE(NULLIF(TRIM(r.league), ''), '?') AS league,
      COUNT(*)::bigint AS n
    FROM betslip_audit_results r
    WHERE {' AND '.join(where)}
    GROUP BY 1
    ORDER BY 2 DESC, 1 ASC
    """
    async with db.async_session() as session:
        res = await session.execute(text(q), params)
        rows = res.fetchall()
    return [(str(r[0]), int(r[1])) for r in rows]


def _print_top(title: str, rows: Sequence[Tuple[str, int]], limit: int) -> None:
    print(f"\n{title}: {len(rows)}")
    for league, n in list(rows)[: max(0, int(limit))]:
        print(f"  - {league}: {n}")


async def _run(args: argparse.Namespace) -> int:
    policy = _load_policy(args.policy_json)
    approved, active_keys = _extract_approved_leagues(policy, key_prefix=args.key_prefix)

    db = Database()
    await db.connect()
    try:
        signals_rows = await _fetch_signals_by_league(
            db,
            hours=float(args.hours),
            hypothesis_type=str(args.hypothesis_type),
            reversal_direction=str(args.reversal_direction),
            audit_version=str(args.audit_version),
            status=str(args.status),
            exec_side=str(args.exec_side),
            prematch_only=bool(args.prematch_only),
        )
    finally:
        await db.close()

    approved_set = set(approved)
    approved_norm: Dict[str, List[str]] = {}
    for lg in approved:
        approved_norm.setdefault(_normalize_league(lg), []).append(lg)

    total_signals = sum(n for _, n in signals_rows)
    exact_rows: List[Tuple[str, int]] = []
    norm_only_rows: List[Tuple[str, int]] = []
    outside_rows: List[Tuple[str, int]] = []
    mismatch_rows: List[Tuple[str, int, List[str]]] = []

    for league, n in signals_rows:
        if league in approved_set:
            exact_rows.append((league, n))
            continue
        norm = _normalize_league(league)
        if norm and norm in approved_norm:
            norm_only_rows.append((league, n))
            mismatch_rows.append((league, n, approved_norm[norm]))
        else:
            outside_rows.append((league, n))

    n_exact = sum(n for _, n in exact_rows)
    n_norm_only = sum(n for _, n in norm_only_rows)
    n_outside = sum(n for _, n in outside_rows)
    coverage_exact = (100.0 * n_exact / total_signals) if total_signals else 0.0
    coverage_norm = (100.0 * (n_exact + n_norm_only) / total_signals) if total_signals else 0.0

    print("=== AUDITORIA POLICY x SINAIS (LEAGUE MISMATCH) ===")
    print(f"policy_json: {args.policy_json}")
    print(f"active_keys_total: {len(active_keys)}")
    print(f"approved_leagues(policy via prefix '{args.key_prefix}'): {len(approved)}")
    print(f"signals_total: {total_signals}")
    print(f"signals_exact_match: {n_exact} ({coverage_exact:.2f}%)")
    print(f"signals_norm_match_only: {n_norm_only} ({(coverage_norm - coverage_exact):.2f}%)")
    print(f"signals_outside_policy: {n_outside} ({(100.0 - coverage_norm):.2f}%)")

    _print_top("APROVADAS COM SINAL (match exato)", exact_rows, args.top)
    _print_top("SINAL FORA DA POLICY (match exato e normalizado)", outside_rows, args.top)
    _print_top("SUSPEITA DE MISMATCH (nao exato, mas normalizado bate)", [(x[0], x[1]) for x in mismatch_rows], args.top)

    approved_without_signal = sorted([lg for lg in approved if lg not in {x for x, _ in exact_rows}])
    print(f"\nAPROVADA SEM SINAL (match exato): {len(approved_without_signal)}")
    for lg in approved_without_signal[: max(0, int(args.top))]:
        print(f"  - {lg}")

    if args.json_out:
        payload = {
            "policy_json": args.policy_json,
            "key_prefix": args.key_prefix,
            "active_keys_total": len(active_keys),
            "approved_leagues_total": len(approved),
            "signals_total": total_signals,
            "signals_exact_match": n_exact,
            "signals_norm_match_only": n_norm_only,
            "signals_outside_policy": n_outside,
            "coverage_exact_pct": coverage_exact,
            "coverage_exact_plus_norm_pct": coverage_norm,
            "approved_with_signal_exact": exact_rows,
            "outside_policy": outside_rows,
            "norm_mismatch_suspects": [
                {"signal_league": lg, "n": n, "approved_candidates": cands}
                for lg, n, cands in mismatch_rows
            ],
            "approved_without_signal_exact": approved_without_signal,
        }
        Path(args.json_out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nJSON salvo em: {args.json_out}")

    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Audita mismatch entre ligas de sinais e active_keys da policy atual."
    )
    ap.add_argument("--policy-json", default="logs/wf_policy_current.json")
    ap.add_argument("--key-prefix", default="Back_Pre_Any__")
    ap.add_argument("--hours", type=float, default=36.0)
    ap.add_argument("--hypothesis-type", default="H3B")
    ap.add_argument("--reversal-direction", default="up")
    ap.add_argument("--audit-version", default="v5.3-ws-gate-back")
    ap.add_argument("--status", default="OK")
    ap.add_argument("--exec-side", default="back")
    ap.add_argument("--prematch-only", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
