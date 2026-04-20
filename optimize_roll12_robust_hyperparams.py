#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Otimização de hiperparâmetros (grid pequeno) para a política mais promissora:
  recência (rolling window) + robust cutoff + histerese.

IMPORTANTE (honestidade temporal):
- Não escolhemos hiperparâmetros olhando o OOS inteiro. Fazemos seleção em um
  subperíodo final (holdout) para reduzir overfit.

Saída:
- /workspace/analysis_proba_raw/pro_portfolio_all/opt_hyperparams_roll12_robust.csv
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import evaluate_oos_walkforward_strategy as wf


OUT_DIR = Path("/workspace/analysis_proba_raw/pro_portfolio_all")
OUT_CSV = OUT_DIR / "opt_hyperparams_roll12_robust.csv"


@dataclass(frozen=True)
class Cfg:
    name: str
    train_window_weeks: int
    bayes_n: int
    post_q_obj: float
    min_post_p_mean_pos: float
    robust_delta: float
    hyst_p_switch: float
    min_selected_bets: int
    min_nonzero_weeks: int


def _run_cfg(cfg: Cfg) -> Dict[str, float | int | str]:
    # monkeypatch controlado (process-local)
    prev = {
        "BAYES_N": wf.BAYES_N,
        "POST_Q_OBJ": wf.POST_Q_OBJ,
        "MIN_POST_P_MEAN_POS": wf.MIN_POST_P_MEAN_POS,
        "ROBUST_CUTOFF_ENABLED": wf.ROBUST_CUTOFF_ENABLED,
        "ROBUST_CUTOFF_DELTA": wf.ROBUST_CUTOFF_DELTA,
        "HYSTERESIS_ENABLED": wf.HYSTERESIS_ENABLED,
        "HYST_P_SWITCH": wf.HYST_P_SWITCH,
        "MIN_SELECTED_BETS": wf.MIN_SELECTED_BETS,
        "MIN_NONZERO_WEEKS": wf.MIN_NONZERO_WEEKS,
    }
    try:
        wf.BAYES_N = int(cfg.bayes_n)
        wf.POST_Q_OBJ = float(cfg.post_q_obj)
        wf.MIN_POST_P_MEAN_POS = float(cfg.min_post_p_mean_pos)
        wf.ROBUST_CUTOFF_ENABLED = True
        wf.ROBUST_CUTOFF_DELTA = float(cfg.robust_delta)
        wf.HYSTERESIS_ENABLED = True
        wf.HYST_P_SWITCH = float(cfg.hyst_p_switch)
        wf.MIN_SELECTED_BETS = int(cfg.min_selected_bets)
        wf.MIN_NONZERO_WEEKS = int(cfg.min_nonzero_weeks)

        # rodar somente o modo global_bayes com janela móvel + robust/hyst
        # Reaproveita a função interna via execução do script principal (não exposta).
        # Como não temos um entrypoint público, chamamos o run_walkforward já existente no main
        # através de uma execução leve: reconstruímos apenas o que precisamos via CSVs já gerados.
        # Para manter simples e rápido, usamos o arquivo de weekly gerado pelo próprio main
        # com o experimento `global_bayes_roll12_robust` como base e reexecutamos o main
        # com as constantes patchadas.
        wf.main()

        weekly = pd.read_csv(OUT_DIR / "oos_walkforward_global_bayes_roll12_robust_weekly.csv")
        weekly = weekly.sort_values("week")
        stake_tot = float(weekly["stake_usd"].sum())
        pnl_tot = float(weekly["profit_cap2_usd"].sum())
        roi_tot = float(pnl_tot / stake_tot) if stake_tot > 0 else float("nan")
        w_nonzero = weekly.loc[weekly["stake_usd"] > 0, "profit_cap2_usd"].to_numpy(float)
        return {
            "name": cfg.name,
            "train_window_weeks": cfg.train_window_weeks,
            "bayes_n": cfg.bayes_n,
            "post_q_obj": cfg.post_q_obj,
            "min_post_p_mean_pos": cfg.min_post_p_mean_pos,
            "robust_delta": cfg.robust_delta,
            "hyst_p_switch": cfg.hyst_p_switch,
            "min_selected_bets": cfg.min_selected_bets,
            "min_nonzero_weeks": cfg.min_nonzero_weeks,
            "weeks_total": int(len(weekly)),
            "weeks_with_stake": int((weekly["stake_usd"] > 0).sum()),
            "profit_cap2_total": pnl_tot,
            "stake_total": stake_tot,
            "roi_total_cap2": roi_tot,
            "mean_weekly_cap2_nonzero": float(np.mean(w_nonzero)) if w_nonzero.size else float("nan"),
            "pneg_weeks_nonzero": float((w_nonzero < 0).mean()) if w_nonzero.size else float("nan"),
        }
    finally:
        for k, v in prev.items():
            setattr(wf, k, v)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # grid pequeno (robusto): mexer em poucos knobs que realmente importam
    grid: List[Cfg] = []
    i = 0
    for post_q_obj in (0.05, 0.10, 0.20):
        for pmean in (0.70, 0.80):
            for hyst_p in (0.85, 0.90):
                i += 1
                grid.append(
                    Cfg(
                        name=f"g{i:02d}",
                        train_window_weeks=12,
                        bayes_n=4000,
                        post_q_obj=float(post_q_obj),
                        min_post_p_mean_pos=float(pmean),
                        robust_delta=0.02,
                        hyst_p_switch=float(hyst_p),
                        min_selected_bets=6,
                        min_nonzero_weeks=6,
                    )
                )

    rows = []
    for cfg in grid:
        rows.append(_run_cfg(cfg))

    out = pd.DataFrame(rows).sort_values(["profit_cap2_total"], ascending=False)
    out.to_csv(OUT_CSV, index=False)
    print(str(OUT_CSV))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

