#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Análise robusta de estabilidade (semanal) para o portfólio proba_raw:
- Seg/Ter/Qua: modelos separados
- Qui: modelo SegQui

Inputs:
  - /workspace/analysis_proba_raw/scored_dedup_proba_raw.csv
  - /workspace/analysis_proba_raw/portfolio_proba_raw_reoptimized.json

Outputs (em /workspace/analysis_proba_raw/robust_weekly/):
  - weekly_series.csv / weekly_series_train.csv / weekly_series_oos.csv
  - weekly_bootstrap_summary.csv
  - weekly_bootstrap_summary_winsor.csv
  - weekly_bootstrap_drawdown_summary.csv
  - robust_stability_weekly.md
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


BASE_SCORED = Path("/workspace/analysis_proba_raw/scored_dedup_proba_raw.csv")
CFG_PATH = Path("/workspace/analysis_proba_raw/portfolio_proba_raw_reoptimized.json")
OUT_DIR = Path("/workspace/analysis_proba_raw/robust_weekly")

TRAIN_START = pd.Timestamp("2025-10-01")
TRAIN_END = pd.Timestamp("2025-12-31 23:59:59")
OOS_START = pd.Timestamp("2026-01-01")

RNG_SEED = 7
N_BOOT = 50_000
N_BOOT_DD = 10_000


def is_ft(tipo: str) -> bool:
    return "first half" not in str(tipo).lower()


def safe_house_cap(x) -> float:
    try:
        v = float(x)
    except Exception:
        return float("inf")
    if not np.isfinite(v) or v <= 0:
        return float("inf")
    return v


def load_cfg() -> Dict:
    return json.loads(CFG_PATH.read_text(encoding="utf-8"))


def apply_portfolio(df: pd.DataFrame, cfg: Dict, roi_col: str = "ROI Real") -> pd.DataFrame:
    """
    Aplica regras por dia e calcula stake/profit por aposta.
    Stake: min(stake_frac * bankroll, house_cap).
    """
    bankroll = float(cfg["bankroll"])
    per = cfg["per_day"]

    rows = []
    for day, params in per.items():
        sc = params["score_col"]
        cut = float(params["cutoff"])
        frac = float(params["stake_frac"])

        x = df[df["dow_pt"] == day].copy()
        x = x[x["Tipo Aposta"].apply(is_ft)]

        roi = pd.to_numeric(x[roi_col], errors="coerce")
        score = pd.to_numeric(x[sc], errors="coerce")
        mask = np.isfinite(roi.to_numpy(dtype=float)) & np.isfinite(score.to_numpy(dtype=float))
        x = x[mask].copy()
        roi = roi[mask]
        score = score[mask]
        x = x[score.to_numpy(dtype=float) >= cut]
        if x.empty:
            continue

        stake0 = bankroll * frac
        cap = x["house_cap"].astype(float).to_numpy()
        stake = np.minimum(stake0, cap)
        x["stake_usd"] = stake
        x["profit_usd"] = stake * pd.to_numeric(x[roi_col], errors="coerce").to_numpy(dtype=float)
        x["rule_day"] = day
        x["score_col"] = sc
        rows.append(x)

    if not rows:
        return pd.DataFrame(columns=list(df.columns) + ["stake_usd", "profit_usd", "rule_day", "score_col"])

    out = pd.concat(rows, axis=0, ignore_index=True)
    out = out.sort_values("BIA_ApostaUTC").reset_index(drop=True)
    return out


def weekly_aggregate(df_bets: pd.DataFrame) -> pd.DataFrame:
    x = df_bets.copy()
    x["week"] = pd.to_datetime(x["BIA_ApostaUTC"]).dt.to_period("W-SUN").astype(str)
    g = x.groupby("week", as_index=False).agg(
        n_bets=("profit_usd", "size"),
        stake_usd=("stake_usd", "sum"),
        profit_usd=("profit_usd", "sum"),
    )
    g["roi_on_stake"] = np.where(g["stake_usd"] > 0, g["profit_usd"] / g["stake_usd"], np.nan)
    return g.sort_values("week").reset_index(drop=True)


def summarize_dist(name: str, x: np.ndarray) -> Dict[str, float]:
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"name": name, "n": 0}
    q = np.quantile(a, [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    return {
        "name": name,
        "n": int(a.size),
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)) if a.size >= 2 else 0.0,
        "p_neg": float((a < 0).mean()),
        "q01": float(q[0]),
        "q05": float(q[1]),
        "q10": float(q[2]),
        "q25": float(q[3]),
        "q50": float(q[4]),
        "q75": float(q[5]),
        "q90": float(q[6]),
        "q95": float(q[7]),
        "q99": float(q[8]),
    }


def block_bootstrap_weekly(
    weekly_profit: np.ndarray,
    horizons: Tuple[int, ...] = (4, 13, 26, 52),
    n_boot: int = N_BOOT,
    seed: int = RNG_SEED,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    w = np.asarray(weekly_profit, dtype=float)
    w = w[np.isfinite(w)]
    if w.size == 0:
        return pd.DataFrame()

    rows = []
    for H in horizons:
        idx = rng.integers(0, w.size, size=(n_boot, H))
        sums = w[idx].sum(axis=1)
        s = summarize_dist(f"{H}w_profit_usd", sums)
        var05 = float(np.quantile(sums, 0.05))
        cvar05 = float(np.mean(sums[sums <= var05]))
        s["var05"] = var05
        s["cvar05"] = cvar05
        rows.append(s)
    return pd.DataFrame(rows)


def bootstrap_drawdown_weekly(
    weekly_profit: np.ndarray,
    bankroll0: float,
    horizon_weeks: int = 52,
    n_boot: int = N_BOOT_DD,
    seed: int = RNG_SEED,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 11)
    w = np.asarray(weekly_profit, dtype=float)
    w = w[np.isfinite(w)]
    if w.size == 0:
        return pd.DataFrame()

    idx = rng.integers(0, w.size, size=(n_boot, horizon_weeks))
    pnl = w[idx]
    bank = bankroll0 + np.cumsum(pnl, axis=1)
    peak = np.maximum.accumulate(bank, axis=1)
    dd = peak - bank
    max_dd = dd.max(axis=1)
    peak_max = peak.max(axis=1)
    max_dd_pct = np.where(peak_max > 0, max_dd / peak_max, np.nan)
    end_bank = bank[:, -1]
    ruin = (end_bank <= 0).astype(int)

    return pd.DataFrame(
        {
            "max_dd_usd": max_dd,
            "max_dd_pct": max_dd_pct,
            "end_bank": end_bank,
            "ruin_end_le_0": ruin,
        }
    )


def winsorize_positive_tail(s: pd.Series, q: float = 0.995) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    hi = float(np.nanquantile(x.to_numpy(dtype=float), q))
    return np.minimum(x.to_numpy(dtype=float), hi)


def cap_roi(s: pd.Series, cap: float) -> pd.Series:
    """
    Stress test: limita a cauda positiva do ROI (mantém perdas).
    """
    x = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
    return np.minimum(x, float(cap))

def dd_summary(dd: pd.DataFrame, label: str) -> Dict[str, float]:
    if dd.empty:
        return {"label": label}
    x = dd["max_dd_usd"].to_numpy(dtype=float)
    y = dd["max_dd_pct"].to_numpy(dtype=float)
    endb = dd["end_bank"].to_numpy(dtype=float)
    ruin = dd["ruin_end_le_0"].to_numpy(dtype=int)
    return {
        "label": label,
        "dd_usd_p50": float(np.quantile(x, 0.50)),
        "dd_usd_p90": float(np.quantile(x, 0.90)),
        "dd_usd_p95": float(np.quantile(x, 0.95)),
        "dd_usd_p99": float(np.quantile(x, 0.99)),
        "dd_pct_p50": float(np.nanquantile(y, 0.50)),
        "dd_pct_p90": float(np.nanquantile(y, 0.90)),
        "dd_pct_p95": float(np.nanquantile(y, 0.95)),
        "dd_pct_p99": float(np.nanquantile(y, 0.99)),
        "end_bank_p05": float(np.quantile(endb, 0.05)),
        "end_bank_p50": float(np.quantile(endb, 0.50)),
        "end_bank_p95": float(np.quantile(endb, 0.95)),
        "p_ruin_end_le_0": float(np.mean(ruin)),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = load_cfg()
    bankroll0 = float(cfg["bankroll"])

    df = pd.read_csv(BASE_SCORED, parse_dates=["BIA_ApostaUTC"])
    df["house_cap"] = df["house_cap"].apply(safe_house_cap)

    scenarios = []

    # Cenário base (raw)
    scenarios.append(("raw", df.copy(), "ROI Real"))

    # Stress: winsor p99.5 (leve)
    df_w = df.copy()
    df_w["ROI_winsor_p995"] = winsorize_positive_tail(df_w["ROI Real"], q=0.995)
    scenarios.append(("winsor_p995", df_w, "ROI_winsor_p995"))

    # Stress: cap ROI em 2.0 (remove caudas >2)
    df_c2 = df.copy()
    df_c2["ROI_cap2"] = cap_roi(df_c2["ROI Real"], cap=2.0)
    scenarios.append(("cap2", df_c2, "ROI_cap2"))

    # Stress: cap ROI em 1.0 (AH típico: -1,-0.5,0,0.5,1)
    df_c1 = df.copy()
    df_c1["ROI_cap1"] = cap_roi(df_c1["ROI Real"], cap=1.0)
    scenarios.append(("cap1", df_c1, "ROI_cap1"))

    # Rodar tudo por cenário
    dd_rows = []
    md_blocks = []

    for i, (label, dfx, roi_col) in enumerate(scenarios):
        bets = apply_portfolio(dfx, cfg, roi_col=roi_col)
        weekly = weekly_aggregate(bets)
        bets_train = bets[(bets["BIA_ApostaUTC"] >= TRAIN_START) & (bets["BIA_ApostaUTC"] <= TRAIN_END)].copy()
        bets_oos = bets[bets["BIA_ApostaUTC"] >= OOS_START].copy()
        weekly_train = weekly_aggregate(bets_train)
        weekly_oos = weekly_aggregate(bets_oos)

        bs = block_bootstrap_weekly(weekly_train["profit_usd"].to_numpy(dtype=float), seed=RNG_SEED + i * 100)
        dd = bootstrap_drawdown_weekly(weekly_train["profit_usd"].to_numpy(dtype=float), bankroll0=bankroll0, seed=RNG_SEED + i * 100)
        dd_rows.append(dd_summary(dd, label))

        # persistência por cenário
        weekly.to_csv(OUT_DIR / f"weekly_series_{label}.csv", index=False)
        weekly_train.to_csv(OUT_DIR / f"weekly_series_train_{label}.csv", index=False)
        weekly_oos.to_csv(OUT_DIR / f"weekly_series_oos_{label}.csv", index=False)
        bs.to_csv(OUT_DIR / f"weekly_bootstrap_summary_{label}.csv", index=False)

        # bloco md
        wp = weekly_train["profit_usd"].to_numpy(dtype=float)
        md_blocks.append(f"\n### Cenário: **{label}**\n")
        md_blocks.append(f"- Semanas no treino: **{weekly_train.shape[0]}**; PnL semanal médio: **USD {np.mean(wp):.0f}**; std: **USD {np.std(wp, ddof=1):.0f}**; P(semana<0): **{(wp<0).mean()*100:.1f}%**\n")
        if not bs.empty:
            r52 = bs[bs["name"] == "52w_profit_usd"].iloc[0]
            md_blocks.append(
                f"- **52 semanas (bootstrap)**: média {r52['mean']:.0f}, p05 {r52['q05']:.0f}, VaR5% {r52['var05']:.0f}, CVaR5% {r52['cvar05']:.0f}, P(<0) {r52['p_neg']*100:.1f}%\n"
            )

    dd_sum = pd.DataFrame(dd_rows)
    dd_sum.to_csv(OUT_DIR / "weekly_bootstrap_drawdown_summary.csv", index=False)

    # Relatório MD
    lines = []
    lines.append("## Estabilidade robusta (semanal) — portfólio proba_raw\\n")
    lines.append(f"- Banca usada p/ sizing: **USD {bankroll0:,.0f}**; max 7% já embutido na regra\\n")
    lines.append(f"- Treino: **{TRAIN_START.date()}..{TRAIN_END.date()}**; OOS: **>= {OOS_START.date()}**\\n")
    lines.append(f"- Bootstrap: **{N_BOOT:,}** amostras (semana com reposição); drawdown paths: **{N_BOOT_DD:,}**\\n")

    lines.append("\n### Resultados por cenário (treino)\n")
    lines.extend(md_blocks)

    lines.append("\n### Drawdown (paths semanais, 52 semanas) — resumo\n")
    for _, r in dd_sum.iterrows():
        lines.append(
            f"- **{r['label']}**: MaxDD USD p50={r['dd_usd_p50']:.0f}, p95={r['dd_usd_p95']:.0f}, p99={r['dd_usd_p99']:.0f}; MaxDD% p95={r['dd_pct_p95']*100:.1f}%; P(ruína end<=0)={r['p_ruin_end_le_0']*100:.2f}%\n"
        )

    lines.append("\\n### Nota sobre o lucro alto vs análises anteriores\\n")
    lines.append(
        "- Se o cenário **raw** mostrar lucro irreal, os cenários **cap2/cap1** te dizem o quanto isso depende de outliers/erros de payout. "
        "Para uma mesa profissional, eu recomendaria usar **cap1/cap2** como stress test obrigatório e só aceitar o edge se ele sobreviver a isso.\n"
    )

    (OUT_DIR / "robust_stability_weekly.md").write_text("".join(lines), encoding="utf-8")
    print(str(OUT_DIR / "robust_stability_weekly.md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

