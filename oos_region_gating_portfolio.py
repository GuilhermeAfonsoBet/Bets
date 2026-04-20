#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experimento: usar "RegiaoEvento" na otimização do portfólio (não no score), via gating OOS.

Ideia:
- Mantém as regras (cutoff/stake_frac/score_col) do modo escolhido (p10_p70).
- Para cada semana de teste, estima no TREINO (últimas 12 semanas) a performance por região
  dentro de cada segmento (DoW x FT/FH) e acima do cutoff.
- Desliga regiões com mean(ROI_cap2) <= 0 (e com n>=min_n) para aquela semana/segmento.
- Aplica na semana teste: mesma regra, mas filtrando por regiões permitidas.

Isso é OOS (usa apenas passado para decidir as regiões) e mantém comparabilidade com o score atual.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

import evaluate_oos_walkforward_strategy as wf


OUT_DIR = Path("/workspace/analysis_proba_raw/pro_portfolio_all")
MODE = "global_bayes_roll12_robust_p10_p70"
SCORED = Path("/workspace/analysis_proba_raw/scored_dedup_proba_raw_all.csv")
REGION_PRED = OUT_DIR / "region_exante_pred.csv"
WF_RULES = OUT_DIR / f"oos_walkforward_{MODE}_selected_rules.csv"
WF_WEEKLY = OUT_DIR / f"oos_walkforward_{MODE}_weekly.csv"

TRAIN_WINDOW_WEEKS = 12
MIN_N_PER_REGION = 20
REGION_PRED_MIN_PMAX = 0.70  # abaixo disso, tratamos como 'desconhecida' (evita gating por ruído)
# Gating conservador:
# Em vez de "permitir apenas regiões com mean_roi>0" (que pode zerar semanas inteiras),
# usamos uma lógica de BLOQUEIO: só exclui regiões com evidência de ROI ruim.
BLOCK_BAD_REGIONS_ONLY = True
BAD_MEAN_ROI_TH = -0.02  # bloqueia apenas se mean(roi_cap2) <= este valor (com n>=MIN_N_PER_REGION)


def infer_region(text: str) -> str:
    if not text:
        return "desconhecida"
    s = str(text).lower()

    if any(k in s for k in ["australia", "austrália", "new zealand", "nova zelandia", "nova zelândia"]):
        return "oceania"
    if any(k in s for k in ["saudi", "arábia", "arabia", "qatar", "catar", "uae", "emirates", "emirados", "iran", "iraq", "israel", "turkey", "turquia"]):
        return "oriente_medio"
    if any(k in s for k in ["japan", "japão", "korea", "coreia", "china", "índia", "india", "thailand", "tailand", "viet", "malaysia", "singapore", "indonesia"]):
        return "asia"

    west = [
        "england","inglaterra","spain","espanha","france","frança","italy","itália","germany","alemanha",
        "netherlands","holanda","belgium","bélgica","portugal","switzerland","suíça","austria","áustria",
        "scotland","escócia","ireland","irlanda","norway","noruega","sweden","suécia","denmark","dinamarca",
    ]
    east = [
        "poland","polônia","czech","tcheca","slovakia","eslováquia","hungary","hungria","romania","romênia",
        "bulgaria","bulgária","serbia","sérvia","croatia","croácia","ukraine","ucrânia","russia","rússia",
        "greece","grécia",
    ]
    if any(k in s for k in west):
        return "europa_ocidental"
    if any(k in s for k in east):
        return "europa_oriental"

    if any(k in s for k in ["mexico", "méxico", "costa rica", "honduras", "guatemala", "panama", "panamá", "el salvador", "nicaragua", "nicarágua", "jamaica", "haiti", "haití", "dominican", "república dominicana"]):
        return "america_central"
    if any(k in s for k in ["usa", "united states", "estados unidos", "canada", "canadá"]):
        return "america_norte"
    if any(k in s for k in ["brazil","brasil","argentina","chile","colombia","colômbia","peru","uruguay","uruguai","paraguay","paraguai","ecuador","venezuela","bolivia","bolívia"]):
        return "america_sul"

    return "desconhecida"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not WF_RULES.exists() or not WF_WEEKLY.exists():
        raise FileNotFoundError("Arquivos do walk-forward base não encontrados.")

    df = pd.read_csv(SCORED, parse_dates=["BIA_ApostaUTC"])
    df["week"] = wf.week_key(df["BIA_ApostaUTC"]).astype(str)
    df["roi_raw"] = pd.to_numeric(df["roi_calc"], errors="coerce").astype(float)
    df["roi_cap2"] = np.minimum(df["roi_raw"].to_numpy(dtype=float), 2.0)
    df["house_cap"] = pd.to_numeric(df["house_cap"], errors="coerce").astype(float)

    # garantir colunas calibradas do score atual
    if "proba_cal_segqui" not in df.columns:
        df["proba_cal_segqui"] = np.nan
    if "proba_cal_sexdom" not in df.columns:
        df["proba_cal_sexdom"] = np.nan
    if "proba_raw_segqui" in df.columns:
        df["proba_cal_segqui"] = wf._apply_isotonic_vec(
            pd.to_numeric(df["proba_raw_segqui"], errors="coerce").to_numpy(float),
            x=np.asarray(__import__("json").loads(Path(wf.CALIB_SEGQUI).read_text(encoding="utf-8"))["isotonic"]["x"], float),
            y=np.asarray(__import__("json").loads(Path(wf.CALIB_SEGQUI).read_text(encoding="utf-8"))["isotonic"]["y"], float),
            floor=wf.CALIB_FLOOR,
        )
    if "proba_raw_sexdom" in df.columns:
        df["proba_cal_sexdom"] = wf._apply_isotonic_vec(
            pd.to_numeric(df["proba_raw_sexdom"], errors="coerce").to_numpy(float),
            x=np.asarray(__import__("json").loads(Path(wf.CALIB_SEXDOM).read_text(encoding="utf-8"))["isotonic"]["x"], float),
            y=np.asarray(__import__("json").loads(Path(wf.CALIB_SEXDOM).read_text(encoding="utf-8"))["isotonic"]["y"], float),
            floor=wf.CALIB_FLOOR,
        )

    # Região do evento:
    # - Preferência: usar predição "ex-ante" (treinada offline) para ter cobertura ~100% sem depender do BetinAsia.
    # - Fallback: heurística simples (pode ter muita 'desconhecida' se Evento não contiver país/competição).
    if REGION_PRED.exists():
        r = pd.read_csv(REGION_PRED, usecols=["ID Aposta", "region_pred", "region_pred_pmax"])
        r = r.rename(columns={"region_pred": "region_evt"})
        df = df.merge(r, how="left", on="ID Aposta")
        df["region_pred_pmax"] = pd.to_numeric(df.get("region_pred_pmax"), errors="coerce").astype(float)
        df["region_evt"] = df["region_evt"].astype("string").fillna("desconhecida").astype(str)
        # baixa confiança => não usar como split/gating (vira desconhecida)
        low = ~np.isfinite(df["region_pred_pmax"].to_numpy(float)) | (df["region_pred_pmax"].to_numpy(float) < float(REGION_PRED_MIN_PMAX))
        df.loc[low, "region_evt"] = "desconhecida"
    else:
        ev = df.get("Evento", pd.Series("", index=df.index)).astype(str)
        df["region_evt"] = ev.fillna("").astype(str).apply(infer_region).astype(str)

    rules = pd.read_csv(WF_RULES)
    weekly_base = pd.read_csv(WF_WEEKLY)
    weeks = weekly_base["week"].astype(str).tolist()

    out_week_rows = []
    out_gate_rows = []

    for wk in weeks:
        g = rules[rules["test_week"].astype(str) == wk].copy()
        if g.empty:
            continue

        # treino: últimas 12 semanas antes do wk, baseado no índice de weeks
        i = weeks.index(wk)
        train_weeks = weeks[max(0, i - TRAIN_WINDOW_WEEKS) : i]
        df_train = df[df["week"].isin(train_weeks)].copy()
        df_test = df[df["week"] == wk].copy()

        if df_test.empty:
            continue

        # monta gating por segmento (rule_key) -> conjunto de regiões a bloquear
        block_by_rule: Dict[str, Set[str]] = {}
        for _, rr in g.iterrows():
            if str(rr.get("status", "")) != "ok":
                continue
            if float(rr.get("stake_frac", 0.0)) <= 0:
                continue
            rk = str(rr["rule_key"])
            bt = str(rr["bet_type"])
            dow = str(rr["dow_pt"])
            sc = str(rr["score_col"])
            cutoff = float(rr["cutoff"])

            x = df_train[(df_train["bet_type"] == bt) & (df_train["dow_pt"] == dow)].copy()
            if x.empty or sc not in x.columns:
                continue
            score = pd.to_numeric(x[sc], errors="coerce").to_numpy(float)
            roi2 = x["roi_cap2"].to_numpy(float)
            reg = x["region_evt"].astype(str).to_numpy()
            m = np.isfinite(score) & (score >= cutoff) & np.isfinite(roi2)
            if not np.any(m):
                continue
            xt = pd.DataFrame({"region": reg[m], "roi": roi2[m]})
            by = xt.groupby("region", as_index=False).agg(n=("roi", "size"), mean_roi=("roi", "mean"))
            if BLOCK_BAD_REGIONS_ONLY:
                block = set(by.loc[(by["n"] >= MIN_N_PER_REGION) & (by["mean_roi"] <= BAD_MEAN_ROI_TH), "region"].astype(str).tolist())
                if block:
                    block_by_rule[rk] = block
                    out_gate_rows.append({"week": wk, "rule_key": rk, "blocked_regions": ",".join(sorted(block)), "n_regions": int(len(block))})
            else:
                # modo antigo (mais agressivo): permitir apenas regiões com mean_roi>0
                allow = set(by.loc[(by["n"] >= MIN_N_PER_REGION) & (by["mean_roi"] > 0), "region"].astype(str).tolist())
                if allow:
                    # armazenamos como "blocked" vazio e tratamos na aplicação via allow-list abaixo
                    # (mantém compatibilidade com estrutura de saída)
                    out_gate_rows.append({"week": wk, "rule_key": rk, "blocked_regions": "", "n_regions": int(len(allow))})
                    block_by_rule[rk] = set()  # sentinel

        # aplica regras + gating
        bets = []
        alpha = float(g["alpha_effective"].iloc[0]) if "alpha_effective" in g.columns and np.isfinite(float(g["alpha_effective"].iloc[0])) else float(g["alpha_global"].iloc[0])
        for _, rr in g.iterrows():
            if str(rr.get("status", "")) != "ok":
                continue
            stake_frac = float(rr.get("stake_frac", 0.0))
            if stake_frac <= 0:
                continue
            rk = str(rr["rule_key"])
            bt = str(rr["bet_type"])
            dow = str(rr["dow_pt"])
            sc = str(rr["score_col"])
            cutoff = float(rr["cutoff"])
            x = df_test[(df_test["bet_type"] == bt) & (df_test["dow_pt"] == dow)].copy()
            if x.empty or sc not in x.columns:
                continue
            score = pd.to_numeric(x[sc], errors="coerce").to_numpy(float)
            roi2 = x["roi_cap2"].to_numpy(float)
            cap = x["house_cap"].to_numpy(float)
            reg = x["region_evt"].astype(str).to_numpy()

            m = np.isfinite(score) & (score >= cutoff) & np.isfinite(roi2) & np.isfinite(cap) & (cap > 0)
            if rk in block_by_rule and BLOCK_BAD_REGIONS_ONLY:
                block = block_by_rule[rk]
                if block:
                    m = m & (~np.isin(reg, list(block)))
            if not np.any(m):
                continue
            stake0 = wf.BANKROLL * stake_frac * float(alpha)
            stake_eff = np.minimum(stake0, cap[m])
            profit = stake_eff * roi2[m]
            bets.append(pd.DataFrame({"stake_eff": stake_eff, "profit_cap2": profit}))

        if bets:
            b = pd.concat(bets, axis=0, ignore_index=True)
            stake_sum = float(b["stake_eff"].sum())
            pnl_sum = float(b["profit_cap2"].sum())
            n = int(len(b))
        else:
            stake_sum = 0.0
            pnl_sum = 0.0
            n = 0

        out_week_rows.append(
            {
                "week": wk,
                "alpha_effective": float(alpha),
                "n_bets": n,
                "stake_usd": stake_sum,
                "profit_cap2_usd": pnl_sum,
                "roi_on_stake_cap2": float(pnl_sum / stake_sum) if stake_sum > 0 else float("nan"),
            }
        )

    out_week = pd.DataFrame(out_week_rows)
    out_gate = pd.DataFrame(out_gate_rows)

    suffix = "exantepred" if REGION_PRED.exists() else "heuristic"
    if BLOCK_BAD_REGIONS_ONLY:
        suffix = f"{suffix}_blockbad"
    out_week_path = OUT_DIR / f"oos_walkforward_region_gating_{suffix}_weekly.csv"
    out_gate_path = OUT_DIR / f"oos_walkforward_region_gating_{suffix}_blocked_regions.csv"
    out_week.to_csv(out_week_path, index=False)
    out_gate.to_csv(out_gate_path, index=False)

    # summary simples vs baseline
    base = weekly_base.copy()
    base["week"] = base["week"].astype(str)
    base = base[base["week"].isin(out_week["week"].astype(str))].copy()
    stake_b = float(base["stake_usd"].sum())
    pnl_b = float(base["profit_cap2_usd"].sum())
    stake_g = float(out_week["stake_usd"].sum())
    pnl_g = float(out_week["profit_cap2_usd"].sum())
    summ = pd.DataFrame(
        [
            {"name": "baseline", "profit_cap2_total": pnl_b, "stake_total": stake_b, "roi_total_cap2": pnl_b / stake_b if stake_b > 0 else np.nan, "weeks": int(len(base)), "weeks_with_stake": int((base["stake_usd"] > 0).sum())},
            {"name": "region_gating", "profit_cap2_total": pnl_g, "stake_total": stake_g, "roi_total_cap2": pnl_g / stake_g if stake_g > 0 else np.nan, "weeks": int(len(out_week)), "weeks_with_stake": int((out_week["stake_usd"] > 0).sum())},
        ]
    )
    summ_path = OUT_DIR / f"oos_walkforward_region_gating_{suffix}_summary.csv"
    summ.to_csv(summ_path, index=False)

    print(str(summ_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

