#!/usr/bin/env python3
"""
Avaliacao estatistica semanal da sensibilidade do ROI a latencia (7-10s vs 10-15s).

Foco principal:
- Estrategia back + pre + slippage_pre_pct < 0
- Inferencia semanal robusta por:
  1) Teste de permutacao estratificado (controle de confusores observaveis)
  2) Bootstrap por cluster (IC de delta de ROI)
  3) Modelo global ajustado com FE de semana (estimativa pooled ajustada)

Entradas:
- CSV de P&L/stake (ex.: /tmp/projecao_por_aposta.csv)
- CSV de E2E canonico por audit_id (ex.: /tmp/e2e_canonico_por_aposta.csv)

Saidas:
- CSV semanal com metricas e testes
- Markdown executivo com resultados e alertas de robustez
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import numpy as np  # type: ignore
except ModuleNotFoundError:
    np = None  # type: ignore


AUDIT_ID_COLS = ("audit_id", "id", "bet_id", "opportunity_id")
TS_COLS = (
    "ts_utc",
    "audited_utc",
    "audited_at",
    "audited_at_utc",
    "timestamp_utc",
    "timestamp",
    "ts",
    "event_ts_utc",
    "created_at",
)
E2E_COLS = (
    "e2e_total_ms",
    "e2e_ms",
    "lag_total_ms",
    "lag_e2e_ms",
    "detect_to_submit_ms",
    "call_to_done_ms",
)
STAKE_COLS = ("stake_exec", "stake_real", "stake", "turnover_unit", "turnover", "turn", "amount")
PNL_COLS = ("pnl_real", "pnl_ledger", "pnl", "daily_unit_pnl", "profit_real", "profit", "result_real")
SIDE_COLS = ("side", "side_lc", "bet_side", "selection_side")
REGIME_COLS = ("market_regime", "market_period", "market_period_lc", "regime")
SLIP_COLS = ("slippage_pre_pct", "slip_pre_pct", "slippage_pct_pre")

LEAGUE_COLS = ("league", "league_name", "competition", "league_lc")
LINE_COLS = ("raw_line", "line", "ah_line")
ODD_COLS = ("odd_exec", "bs_odd", "betslip_odd", "odd", "price")
CLUSTER_COLS = ("match_id", "event_id", "fixture_id", "game_id")

TRUE_VALUES = {"1", "true", "t", "yes", "y", "sim"}
FALSE_VALUES = {"0", "false", "f", "no", "n", "nao", "não"}


@dataclass
class Obs:
    audit_id: int
    ts: datetime
    week: str
    bucket: str  # "7-10s" ou "10-15s"
    treat: int   # 1 se 10-15s, 0 se 7-10s
    stake: float
    pnl: float
    roi_event_pct: float
    league: str
    day: str
    line_bin: str
    cluster_id: str
    odd_exec: Optional[float]
    abs_line: Optional[float]


def _first_nonempty(row: Dict[str, str], candidates: Sequence[str]) -> Optional[str]:
    for key in candidates:
        if key in row:
            raw = row.get(key)
            if raw is None:
                continue
            txt = str(raw).strip()
            if txt != "":
                return txt
    return None


def _to_float(raw: Optional[str]) -> Optional[float]:
    if raw is None:
        return None
    txt = str(raw).strip().replace(",", ".").replace("−", "-")
    if txt == "":
        return None
    try:
        return float(txt)
    except ValueError:
        return None


def _to_int(raw: Optional[str]) -> Optional[int]:
    if raw is None:
        return None
    txt = str(raw).strip()
    if txt == "":
        return None
    try:
        return int(txt)
    except ValueError:
        try:
            return int(float(txt.replace(",", ".")))
        except ValueError:
            return None


def _to_dt(raw: Optional[str]) -> Optional[datetime]:
    if raw is None:
        return None
    txt = str(raw).strip()
    if txt == "":
        return None
    if txt.endswith("Z"):
        txt = txt[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(txt)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _to_bool(raw: Optional[str]) -> Optional[bool]:
    if raw is None:
        return None
    txt = str(raw).strip().lower()
    if txt in TRUE_VALUES:
        return True
    if txt in FALSE_VALUES:
        return False
    return None


def _nonempty_count(rows: Sequence[Dict[str, str]], col: str) -> int:
    n = 0
    for row in rows:
        raw = row.get(col)
        if raw is None:
            continue
        if str(raw).strip() != "":
            n += 1
    return n


def _best_col_report(rows: Sequence[Dict[str, str]], candidates: Sequence[str], label: str) -> str:
    hits: List[Tuple[int, str]] = []
    for c in candidates:
        k = _nonempty_count(rows, c)
        if k > 0:
            hits.append((k, c))
    if not hits:
        return f"{label}: nenhum candidato com dados"
    hits.sort(reverse=True)
    top = ", ".join(f"{c}={k}" for k, c in hits[:6])
    return f"{label}: {top}"


def _iso_week(dt: datetime) -> str:
    iso = dt.isocalendar()
    return f"{int(iso.year)}-W{int(iso.week):02d}"


def _line_bin(v: Optional[float]) -> str:
    if v is None:
        return "line:na"
    x = abs(float(v))
    if x < 0.5:
        return "line:<0.5"
    if x < 1.5:
        return "line:0.5-1.5"
    return "line:>=1.5"


def _bucket_7_15(e2e_ms: float) -> Optional[str]:
    if 7000.0 <= e2e_ms < 10000.0:
        return "7-10s"
    if 10000.0 <= e2e_ms < 15000.0:
        return "10-15s"
    return None


def _quantile(xs: Sequence[float], q: float) -> Optional[float]:
    if not xs:
        return None
    arr = sorted(float(v) for v in xs if math.isfinite(float(v)))
    if not arr:
        return None
    if len(arr) == 1:
        return arr[0]
    qq = min(1.0, max(0.0, float(q)))
    pos = (len(arr) - 1) * qq
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return arr[lo]
    frac = pos - lo
    return arr[lo] * (1.0 - frac) + arr[hi] * frac


def _median(xs: Sequence[float]) -> Optional[float]:
    return _quantile(xs, 0.5)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        rd = csv.DictReader(fh)
        return [dict(r) for r in rd]


def _load_e2e_map(e2e_csv: Path) -> Dict[int, float]:
    rows = _read_csv(e2e_csv)
    out: Dict[int, float] = {}
    for row in rows:
        aid = _to_int(_first_nonempty(row, AUDIT_ID_COLS))
        e2e = _to_float(_first_nonempty(row, E2E_COLS))
        if aid is None or e2e is None or e2e <= 0:
            continue
        out[aid] = e2e
    return out


def _roi_pct_from_rows(rows: Sequence[Obs]) -> Optional[float]:
    turn = sum(r.stake for r in rows)
    if turn <= 0:
        return None
    pnl = sum(r.pnl for r in rows)
    return 100.0 * pnl / turn


def _delta_roi_pct(rows: Sequence[Obs]) -> Optional[float]:
    g0 = [r for r in rows if r.treat == 0]
    g1 = [r for r in rows if r.treat == 1]
    if not g0 or not g1:
        return None
    r0 = _roi_pct_from_rows(g0)
    r1 = _roi_pct_from_rows(g1)
    if r0 is None or r1 is None:
        return None
    return r1 - r0


def _stratum_key(obs: Obs, mode: str) -> str:
    if mode == "none":
        return "all"
    if mode == "day":
        return obs.day
    if mode == "day_league":
        return f"{obs.day}|{obs.league}"
    return f"{obs.day}|{obs.league}|{obs.line_bin}"


def _permutation_pvalue_week(
    rows: Sequence[Obs],
    *,
    stratify_mode: str,
    n_perm: int,
    seed: int,
) -> Tuple[Optional[float], int, int]:
    """
    Teste de permutacao estratificado.
    Estatistica: delta ROI (10-15s minus 7-10s), ponderado por stake.
    """
    obs_delta = _delta_roi_pct(rows)
    if obs_delta is None:
        return None, 0, 0

    strata: Dict[str, List[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        strata[_stratum_key(r, stratify_mode)].append(i)

    # so contam estratos "mistos" (com os 2 buckets) para randomizacao efetiva
    mixed = []
    for _, idxs in strata.items():
        t = [rows[i].treat for i in idxs]
        if 0 in t and 1 in t:
            mixed.append(idxs)
    if not mixed:
        return None, len(strata), 0

    total_stake = sum(r.stake for r in rows)
    total_pnl = sum(r.pnl for r in rows)
    if total_stake <= 0:
        return None, len(strata), len(mixed)

    rng = random.Random(int(seed))
    more_extreme = 0
    valid = 0

    all_indices = list(range(len(rows)))
    # Estratos nao mistos ficam fixos
    fixed_treat = [rows[i].treat for i in all_indices]

    for _ in range(int(n_perm)):
        assign = fixed_treat[:]
        for idxs in mixed:
            n1 = sum(rows[i].treat for i in idxs)
            chosen = set(rng.sample(idxs, n1))
            for i in idxs:
                assign[i] = 1 if i in chosen else 0

        stake1 = pnl1 = 0.0
        for i, r in enumerate(rows):
            if assign[i] == 1:
                stake1 += r.stake
                pnl1 += r.pnl
        stake0 = total_stake - stake1
        pnl0 = total_pnl - pnl1
        if stake0 <= 0 or stake1 <= 0:
            continue
        delta = 100.0 * (pnl1 / stake1 - pnl0 / stake0)
        valid += 1
        if abs(delta) >= abs(obs_delta):
            more_extreme += 1

    if valid == 0:
        return None, len(strata), len(mixed)
    p_two = (more_extreme + 1.0) / (valid + 1.0)
    return float(p_two), len(strata), len(mixed)


def _permutation_with_fallback(
    rows: Sequence[Obs],
    *,
    preferred_mode: str,
    n_perm: int,
    seed: int,
) -> Tuple[Optional[float], str, int, int]:
    if preferred_mode == "day_league_line":
        modes = ["day_league_line", "day_league", "day", "none"]
    elif preferred_mode == "day_league":
        modes = ["day_league", "day", "none"]
    elif preferred_mode == "day":
        modes = ["day", "none"]
    else:
        modes = ["none"]

    last_info = (None, modes[-1], 0, 0)
    for i, mode in enumerate(modes):
        p, n_strata, n_mixed = _permutation_pvalue_week(
            rows, stratify_mode=mode, n_perm=n_perm, seed=seed + i * 100003
        )
        last_info = (p, mode, n_strata, n_mixed)
        if p is not None:
            return p, mode, n_strata, n_mixed
    return last_info


def _bootstrap_ci_week(
    rows: Sequence[Obs],
    *,
    n_boot: int,
    seed: int,
) -> Tuple[Optional[float], Optional[float]]:
    """
    IC90 bootstrap por cluster (cluster_id).
    """
    by_cluster: Dict[str, List[Obs]] = defaultdict(list)
    for r in rows:
        by_cluster[r.cluster_id].append(r)
    clusters = list(by_cluster.keys())
    if len(clusters) < 2:
        return None, None

    rng = random.Random(int(seed))
    vals: List[float] = []
    for _ in range(int(n_boot)):
        sample_rows: List[Obs] = []
        for _k in range(len(clusters)):
            cid = rng.choice(clusters)
            sample_rows.extend(by_cluster[cid])
        d = _delta_roi_pct(sample_rows)
        if d is not None and math.isfinite(d):
            vals.append(float(d))
    if len(vals) < 30:
        return None, None
    lo = _quantile(vals, 0.05)
    hi = _quantile(vals, 0.95)
    if lo is None or hi is None:
        return None, None
    return lo, hi


def _build_obs(
    pnl_rows: Sequence[Dict[str, str]],
    e2e_map: Dict[int, float],
    *,
    start_dt: Optional[datetime],
    end_dt: Optional[datetime],
    require_segment: bool,
    strict_segment: bool,
) -> Tuple[List[Obs], Counter[str]]:
    out: List[Obs] = []
    dropped: Counter[str] = Counter()

    for row in pnl_rows:
        aid = _to_int(_first_nonempty(row, AUDIT_ID_COLS))
        if aid is None:
            dropped["missing_audit_id"] += 1
            continue
        ts = _to_dt(_first_nonempty(row, TS_COLS))
        if ts is None:
            dropped["missing_timestamp"] += 1
            continue
        if start_dt is not None and ts < start_dt:
            dropped["outside_time_window"] += 1
            continue
        if end_dt is not None and ts >= end_dt:
            dropped["outside_time_window"] += 1
            continue

        stake = _to_float(_first_nonempty(row, STAKE_COLS))
        pnl = _to_float(_first_nonempty(row, PNL_COLS))
        if stake is None:
            dropped["missing_stake"] += 1
            continue
        if pnl is None:
            dropped["missing_pnl"] += 1
            continue
        if stake <= 0:
            dropped["invalid_stake_nonpositive"] += 1
            continue

        e2e = e2e_map.get(aid)
        if e2e is None:
            dropped["missing_e2e_join"] += 1
            continue
        if e2e <= 0:
            dropped["invalid_e2e_nonpositive"] += 1
            continue
        bucket = _bucket_7_15(e2e)
        if bucket is None:
            dropped["outside_7_15_window"] += 1
            continue

        side = (_first_nonempty(row, SIDE_COLS) or "").strip().lower()
        regime_raw = (_first_nonempty(row, REGIME_COLS) or "").strip().lower()
        is_live = _to_bool(row.get("is_live"))
        slip = _to_float(_first_nonempty(row, SLIP_COLS))

        is_back = None
        if side == "back":
            is_back = True
        elif side == "lay":
            is_back = False

        is_pre = None
        if regime_raw in {"pre", "prematch", "pre_match", "pre-match"}:
            is_pre = True
        elif regime_raw in {"live", "inplay", "in_play", "in-match", "in_match"}:
            is_pre = False
        elif is_live is not None:
            is_pre = not is_live

        is_slipneg = None if slip is None else (slip < 0)

        if require_segment:
            if strict_segment and (is_back is None or is_pre is None or is_slipneg is None):
                dropped["segment_missing_fields"] += 1
                continue
            if is_back is False or is_pre is False or is_slipneg is False:
                dropped["segment_filtered_out"] += 1
                continue

        league = (_first_nonempty(row, LEAGUE_COLS) or "<na>").strip().lower()
        line_v = _to_float(_first_nonempty(row, LINE_COLS))
        line_bin = _line_bin(line_v)
        odd = _to_float(_first_nonempty(row, ODD_COLS))

        cluster_val = _first_nonempty(row, CLUSTER_COLS)
        if cluster_val is None:
            cluster_val = f"audit:{aid}"
        day = ts.date().isoformat()
        wk = _iso_week(ts)
        treat = 1 if bucket == "10-15s" else 0
        roi_event_pct = 100.0 * pnl / stake

        out.append(
            Obs(
                audit_id=aid,
                ts=ts,
                week=wk,
                bucket=bucket,
                treat=treat,
                stake=stake,
                pnl=pnl,
                roi_event_pct=roi_event_pct,
                league=league,
                day=day,
                line_bin=line_bin,
                cluster_id=str(cluster_val),
                odd_exec=odd,
                abs_line=None if line_v is None else abs(line_v),
            )
        )
    return out, dropped


def _wls_fe_effect(
    rows: Sequence[Obs],
    *,
    max_league_dummies: int,
) -> Tuple[Optional[float], Optional[float], Optional[float], int, List[str]]:
    """
    Efeito ajustado pooled de 10-15s vs 7-10s:
      y = ROI_event_pct
      X = [intercept, treat, FE_semana, controles]
      peso = stake
    Retorna beta_treat, se_robusta, p_aprox, n, regressors.
    """
    n = len(rows)
    if n < 20:
        return None, None, None, n, []
    if np is None:
        return None, None, None, n, ["pooled_fe_skipped_numpy_unavailable"]

    weeks = sorted({r.week for r in rows})
    week_ref = weeks[0]
    week_dummy_names = [f"week:{w}" for w in weeks if w != week_ref]

    leagues = sorted({r.league for r in rows})
    use_league = len(leagues) <= int(max_league_dummies)
    league_ref = leagues[0] if leagues else "<na>"
    league_dummy_names = [f"league:{lg}" for lg in leagues if lg != league_ref] if use_league else []

    odd_vals = [float(r.odd_exec) for r in rows if r.odd_exec is not None]
    line_vals = [float(r.abs_line) for r in rows if r.abs_line is not None]
    odd_med = _median(odd_vals)
    line_med = _median(line_vals)

    reg_names = ["intercept", "treat_10_15", "log_stake"] + week_dummy_names
    if odd_med is not None:
        reg_names.append("odd_exec")
    if line_med is not None:
        reg_names.append("abs_line")
    reg_names.extend(league_dummy_names)

    k = len(reg_names)
    X = np.zeros((n, k), dtype=float)
    y = np.zeros((n,), dtype=float)
    w = np.zeros((n,), dtype=float)
    cluster = []

    week_col = {name: i for i, name in enumerate(reg_names)}
    for i, r in enumerate(rows):
        y[i] = r.roi_event_pct
        w[i] = max(r.stake, 1e-9)
        cluster.append(r.cluster_id)
        X[i, week_col["intercept"]] = 1.0
        X[i, week_col["treat_10_15"]] = float(r.treat)
        X[i, week_col["log_stake"]] = math.log(max(r.stake, 1e-9))
        if r.week != week_ref:
            X[i, week_col[f"week:{r.week}"]] = 1.0
        if odd_med is not None:
            X[i, week_col["odd_exec"]] = r.odd_exec if r.odd_exec is not None else odd_med
        if line_med is not None:
            X[i, week_col["abs_line"]] = r.abs_line if r.abs_line is not None else line_med
        if use_league and r.league != league_ref:
            X[i, week_col[f"league:{r.league}"]] = 1.0

    sw = np.sqrt(w)
    Xw = X * sw[:, None]
    yw = y * sw

    try:
        beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    except np.linalg.LinAlgError:
        return None, None, None, n, reg_names
    resid = yw - Xw @ beta

    # Cov robusta cluster-robust (CR1) por cluster_id quando possivel;
    # fallback para HC1.
    XtX_inv = np.linalg.pinv(Xw.T @ Xw)
    clusters = sorted(set(cluster))
    if len(clusters) >= 15:
        meat = np.zeros((k, k), dtype=float)
        idx_by_c: Dict[str, List[int]] = defaultdict(list)
        for i, c in enumerate(cluster):
            idx_by_c[c].append(i)
        for c in clusters:
            idx = idx_by_c[c]
            Xg = Xw[idx, :]
            ug = resid[idx].reshape(-1, 1)
            meat += Xg.T @ ug @ ug.T @ Xg
        cov = XtX_inv @ meat @ XtX_inv
        g = len(clusters)
        if g > 1 and n > k:
            cov *= (g / (g - 1.0)) * ((n - 1.0) / (n - k))
    else:
        meat = np.zeros((k, k), dtype=float)
        for i in range(n):
            xi = Xw[i, :].reshape(-1, 1)
            meat += (resid[i] ** 2) * (xi @ xi.T)
        cov = XtX_inv @ meat @ XtX_inv
        if n > k:
            cov *= n / (n - k)

    j = reg_names.index("treat_10_15")
    b = float(beta[j])
    var = float(cov[j, j]) if np.isfinite(cov[j, j]) else float("nan")
    if not np.isfinite(var) or var <= 0:
        return b, None, None, n, reg_names
    se = math.sqrt(var)
    z = b / se if se > 0 else float("nan")
    # aproximacao normal
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0)))) if np.isfinite(z) else None
    return b, se, p, n, reg_names


def _fmt(v: Optional[float], nd: int = 2) -> str:
    if v is None:
        return "—"
    return f"{v:.{nd}f}"


def _write_weekly_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader()
        for r in rows:
            wr.writerow(r)


def _write_summary_md(
    path: Path,
    *,
    weekly_rows: List[Dict[str, object]],
    n_input: int,
    n_kept: int,
    drop_reasons: Counter[str],
    stratify_mode: str,
    perm_n: int,
    boot_n: int,
    pooled_beta: Optional[float],
    pooled_se: Optional[float],
    pooled_p: Optional[float],
    reg_names: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append("# Avaliacao estatistica semanal: latencia 7-10s vs 10-15s\n\n")
    lines.append("## 1) Setup\n")
    lines.append(f"- linhas_entrada={n_input}\n")
    lines.append(f"- linhas_analise={n_kept}\n")
    lines.append(f"- estratificacao_permutacao={stratify_mode}\n")
    lines.append(f"- permutacoes={perm_n}\n")
    lines.append(f"- bootstrap_cluster={boot_n}\n")
    lines.append(f"- principais_descartes={dict(drop_reasons.most_common(8))}\n\n")

    lines.append("## 2) Resultado semanal (estimando delta ROI: 10-15s - 7-10s)\n")
    lines.append(
        "| Semana | n_7_10 | n_10_15 | ROI 7-10 (%) | ROI 10-15 (%) | Delta p.p. | p_perm (2s) | IC90 bootstrap |\n"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|\n")
    for r in weekly_rows:
        ic = "—"
        if r["boot_ci90_lo"] is not None and r["boot_ci90_hi"] is not None:
            ic = f"[{_fmt(float(r['boot_ci90_lo']),2)}, {_fmt(float(r['boot_ci90_hi']),2)}]"
        lines.append(
            f"| {r['week']} | {r['n_7_10']} | {r['n_10_15']} | "
            f"{_fmt(float(r['roi_7_10']) if r['roi_7_10'] is not None else None,2)} | "
            f"{_fmt(float(r['roi_10_15']) if r['roi_10_15'] is not None else None,2)} | "
            f"{_fmt(float(r['delta_pp']) if r['delta_pp'] is not None else None,2)} | "
            f"{_fmt(float(r['perm_p_two_sided']) if r['perm_p_two_sided'] is not None else None,4)} | {ic} |\n"
        )

    lines.append("\n## 3) Efeito pooled ajustado (WLS + FE de semana)\n")
    lines.append(
        "- Modelo: ROI_evento(%) ~ I(10-15s) + FE semana + log(stake) + controles disponiveis.\n"
    )
    if pooled_beta is None and "pooled_fe_skipped_numpy_unavailable" in set(reg_names):
        lines.append("- Numpy nao disponivel neste ambiente: bloco pooled FE foi pulado.\n")
        lines.append("- As inferencias semanais (permutacao + bootstrap) seguem validas e foram executadas.\n\n")
    else:
        lines.append(
            f"- Beta ajustado I(10-15s): {_fmt(pooled_beta,3)} p.p.; SE={_fmt(pooled_se,3)}; p~{_fmt(pooled_p,4)}.\n"
        )
        lines.append(f"- Regressors: {list(reg_names)}\n\n")

    lines.append("## 4) Leitura recomendada\n")
    lines.append(
        "- Delta negativo indica deterioracao do ROI quando a latencia sobe de 7-10s para 10-15s.\n"
    )
    lines.append(
        "- Dê mais peso a semanas com n razoavel em ambos os buckets (ex.: n>=10 por bucket).\n"
    )
    lines.append(
        "- p_perm baixo + IC90 totalmente abaixo de zero reforca evidencia de deterioracao na semana.\n"
    )

    path.write_text("".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Avaliacao estatistica semanal de sensibilidade ROI x latencia")
    parser.add_argument("--pnl-csv", required=True)
    parser.add_argument("--e2e-csv", required=True)
    parser.add_argument("--out-weekly-csv", required=True)
    parser.add_argument("--out-summary-md", required=True)
    parser.add_argument("--start-utc", default="")
    parser.add_argument("--end-utc", default="")
    parser.add_argument("--require-back-pre-slipneg", type=int, default=1)
    parser.add_argument("--strict-segment-fields", type=int, default=0)
    parser.add_argument(
        "--stratify-mode",
        choices=["none", "day", "day_league", "day_league_line"],
        default="day_league_line",
    )
    parser.add_argument("--perm-n", type=int, default=5000)
    parser.add_argument("--boot-n", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max-league-dummies", type=int, default=25)
    args = parser.parse_args()

    pnl_path = Path(args.pnl_csv)
    e2e_path = Path(args.e2e_csv)
    if not pnl_path.exists():
        raise SystemExit(f"arquivo inexistente: {pnl_path}")
    if not e2e_path.exists():
        raise SystemExit(f"arquivo inexistente: {e2e_path}")

    pnl_rows = _read_csv(pnl_path)
    e2e_map = _load_e2e_map(e2e_path)
    start_dt = _to_dt(args.start_utc) if str(args.start_utc).strip() else None
    end_dt = _to_dt(args.end_utc) if str(args.end_utc).strip() else None

    obs, dropped = _build_obs(
        pnl_rows,
        e2e_map,
        start_dt=start_dt,
        end_dt=end_dt,
        require_segment=bool(int(args.require_back_pre_slipneg)),
        strict_segment=bool(int(args.strict_segment_fields)),
    )

    if not obs:
        header_cols = sorted({k for r in pnl_rows for k in r.keys()})
        aid_in = set(_to_int(_first_nonempty(r, AUDIT_ID_COLS)) for r in pnl_rows)
        aid_in = {a for a in aid_in if a is not None}
        e2e_cov = sum(1 for a in aid_in if a in e2e_map)
        diag = [
            "nenhuma linha apos filtros para comparacao 7-10s vs 10-15s",
            f"rows_in={len(pnl_rows)} e2e_map_valid={len(e2e_map)} e2e_cov={e2e_cov}/{len(aid_in)}",
            f"drop_reasons={dict(dropped.most_common(12))}",
            _best_col_report(pnl_rows, AUDIT_ID_COLS, "audit_id_candidates"),
            _best_col_report(pnl_rows, TS_COLS, "timestamp_candidates"),
            _best_col_report(pnl_rows, STAKE_COLS, "stake_candidates"),
            _best_col_report(pnl_rows, PNL_COLS, "pnl_candidates"),
            _best_col_report(pnl_rows, SIDE_COLS, "side_candidates"),
            _best_col_report(pnl_rows, REGIME_COLS, "regime_candidates"),
            _best_col_report(pnl_rows, SLIP_COLS, "slippage_candidates"),
            f"headers_detectados={header_cols}",
        ]
        raise SystemExit("\n".join(diag))

    by_week: Dict[str, List[Obs]] = defaultdict(list)
    for r in obs:
        by_week[r.week].append(r)

    weekly_rows: List[Dict[str, object]] = []
    for wk in sorted(by_week.keys()):
        sub = by_week[wk]
        g0 = [r for r in sub if r.treat == 0]
        g1 = [r for r in sub if r.treat == 1]
        roi0 = _roi_pct_from_rows(g0)
        roi1 = _roi_pct_from_rows(g1)
        delta = None if (roi0 is None or roi1 is None) else (roi1 - roi0)
        p_perm, perm_mode, n_strata, n_mixed = _permutation_with_fallback(
            sub,
            preferred_mode=args.stratify_mode,
            n_perm=int(args.perm_n),
            seed=int(args.seed) + hash(wk) % 100000,
        )
        ci_lo, ci_hi = _bootstrap_ci_week(
            sub, n_boot=int(args.boot_n), seed=int(args.seed) + 2 * (hash(wk) % 100000)
        )
        weekly_rows.append(
            {
                "week": wk,
                "n_total": len(sub),
                "n_7_10": len(g0),
                "n_10_15": len(g1),
                "turn_7_10": sum(r.stake for r in g0),
                "turn_10_15": sum(r.stake for r in g1),
                "pnl_7_10": sum(r.pnl for r in g0),
                "pnl_10_15": sum(r.pnl for r in g1),
                "roi_7_10": roi0,
                "roi_10_15": roi1,
                "delta_pp": delta,
                "perm_p_two_sided": p_perm,
                "perm_mode_used": perm_mode,
                "boot_ci90_lo": ci_lo,
                "boot_ci90_hi": ci_hi,
                "n_strata": n_strata,
                "n_mixed_strata": n_mixed,
            }
        )

    pooled_beta, pooled_se, pooled_p, n_reg, reg_names = _wls_fe_effect(
        obs, max_league_dummies=int(args.max_league_dummies)
    )

    out_week = Path(args.out_weekly_csv)
    out_md = Path(args.out_summary_md)
    _write_weekly_csv(out_week, weekly_rows)
    _write_summary_md(
        out_md,
        weekly_rows=weekly_rows,
        n_input=len(pnl_rows),
        n_kept=len(obs),
        drop_reasons=dropped,
        stratify_mode=args.stratify_mode,
        perm_n=int(args.perm_n),
        boot_n=int(args.boot_n),
        pooled_beta=pooled_beta,
        pooled_se=pooled_se,
        pooled_p=pooled_p,
        reg_names=reg_names,
    )

    print("OK")
    print(f"rows_in={len(pnl_rows)} rows_used={len(obs)}")
    print(f"weeks={len(by_week)}")
    print(f"drop_reasons={dict(dropped.most_common(8))}")
    print(f"pooled_beta_10_15_vs_7_10_pp={pooled_beta}")
    print(f"pooled_se={pooled_se} pooled_p={pooled_p} reg_n={n_reg}")
    print(f"out_weekly_csv={out_week}")
    print(f"out_summary_md={out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
