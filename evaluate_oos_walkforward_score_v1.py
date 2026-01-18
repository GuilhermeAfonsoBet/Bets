#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Walk-forward OOS comparando score atual vs score v1 (modelo financeiro) de forma HONESTA no tempo.

Crítico: score v1 é treinado a cada fold usando somente semanas anteriores e aplicado na semana teste.

Saídas (em /workspace/analysis_proba_raw/pro_portfolio_all):
- oos_walkforward_score_current_p10_p70_*.csv
- oos_walkforward_scorev1_logit_p10_p70_*.csv
- oos_walkforward_scorev1_logit_p10_p70_summary.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import json

import evaluate_oos_walkforward_strategy as wf


SCORED = Path("/workspace/analysis_proba_raw/scored_dedup_proba_raw_all.csv")
OUT_DIR = Path("/workspace/analysis_proba_raw/pro_portfolio_all")

TRAIN_WINDOW_WEEKS = 12
EVAL_LAST_WEEKS = 8  # para rodar rápido (últimas N semanas OOS)


def _safe_get(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col] if col in df.columns else pd.Series(np.nan, index=df.index)


def _build_score_features(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["odd_rb"] = pd.to_numeric(_safe_get(df, "Odd Indicada no RB"), errors="coerce")
    X["odd_exec"] = pd.to_numeric(_safe_get(df, "Odd Aposta Realizada"), errors="coerce")
    X["odd_rb2"] = pd.to_numeric(_safe_get(df, "RebelBetting.Odds"), errors="coerce")
    X["odd_got"] = pd.to_numeric(_safe_get(df, "BetinAsia.got price"), errors="coerce")
    X["dif_odds_rb_bia"] = pd.to_numeric(_safe_get(df, "Dif Odds RB & BIA"), errors="coerce")
    X["n_books"] = pd.to_numeric(_safe_get(df, "ApostaLive.Número de casas disponíveis no momento da aposta"), errors="coerce")
    X["stake_max_house"] = pd.to_numeric(_safe_get(df, "ApostaLive.Stake máximo da casa da aposta (USD)"), errors="coerce")
    X["dif_top2"] = pd.to_numeric(_safe_get(df, "ApostaLive.Dif % maior odd e segunda maior"), errors="coerce")
    X["dif_med"] = pd.to_numeric(_safe_get(df, "ApostaLive.Dif % maior odd e odd mediana"), errors="coerce")
    X["aux1_maior_odd"] = pd.to_numeric(_safe_get(df, "ApostaLive.Aux1 - maior odd"), errors="coerce")
    X["rb_percentage"] = pd.to_numeric(_safe_get(df, "RebelBetting.Percentage"), errors="coerce")
    X["mins_to_start"] = pd.to_numeric(_safe_get(df, "RebelBetting.MinutesToMatchStart"), errors="coerce")
    X["bot_total"] = pd.to_numeric(_safe_get(df, "TempoApostas.Tempo total bot"), errors="coerce")
    X["tipo_aposta"] = _safe_get(df, "Tipo Aposta").astype(str)
    X["subtipo"] = _safe_get(df, "Subtipo da Aposta").astype(str)
    X["jogo_int_ou_intervalo"] = _safe_get(df, "Jogo inteiro / intervalo").astype(str)
    X["esporte"] = _safe_get(df, "Esporte").astype(str)
    X["bet_type"] = _safe_get(df, "bet_type").astype(str)
    X["dow_pt"] = _safe_get(df, "dow_pt").astype(str)
    X["book"] = _safe_get(df, "ApostaLive.Casa aposta vencedora").astype(str)
    dt = pd.to_datetime(df["BIA_ApostaUTC"], errors="coerce")
    X["hour_utc"] = dt.dt.hour
    X["weekday_utc"] = dt.dt.weekday
    return X


def _fit_predict_logit(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    Xtr = _build_score_features(train_df)
    Xte = _build_score_features(test_df)

    roi_tr = pd.to_numeric(train_df["roi_calc"], errors="coerce").to_numpy(float)
    ok = np.isfinite(roi_tr)
    ytr = (roi_tr[ok] > 0).astype(int)
    Xtr = Xtr.loc[ok].copy()
    if Xtr.shape[0] < 200 or len(np.unique(ytr)) < 2:
        return np.full(len(test_df), np.nan, dtype=float)

    num_cols = [c for c in Xtr.columns if pd.api.types.is_numeric_dtype(Xtr[c])]
    cat_cols = [c for c in Xtr.columns if c not in num_cols]
    pre = ColumnTransformer(
        [
            ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num_cols),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols),
        ]
    )
    clf = LogisticRegression(max_iter=2000, solver="lbfgs")
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(Xtr, ytr)
    return pipe.predict_proba(Xte)[:, 1].astype(float)


def _fit_hgb(train_df: pd.DataFrame):
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OrdinalEncoder

    Xtr = _build_score_features(train_df)

    roi_tr = pd.to_numeric(train_df["roi_calc"], errors="coerce").to_numpy(float)
    ok = np.isfinite(roi_tr)
    ytr = (roi_tr[ok] > 0).astype(int)
    Xtr = Xtr.loc[ok].copy()
    if Xtr.shape[0] < 200 or len(np.unique(ytr)) < 2:
        return None

    num_cols = [c for c in Xtr.columns if pd.api.types.is_numeric_dtype(Xtr[c])]
    cat_cols = [c for c in Xtr.columns if c not in num_cols]
    pre = ColumnTransformer(
        [
            ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num_cols),
            # OrdinalEncoder é muito mais leve que OneHot e costuma ser suficiente para HGB em protótipos.
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]), cat_cols),
        ]
    )
    clf = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=40, random_state=7)
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(Xtr, ytr)
    return pipe


def _predict_proba_hgb(pipe, df_any: pd.DataFrame) -> np.ndarray:
    if pipe is None:
        return np.full(len(df_any), np.nan, dtype=float)
    X = _build_score_features(df_any)
    return pipe.predict_proba(X)[:, 1].astype(float)


def _oof_scores_logit_by_week(df_train: pd.DataFrame) -> np.ndarray:
    """
    Gera scores out-of-fold dentro do treino (sem olhar o futuro):
    para cada semana w_k (a partir da 5ª), treina com semanas anteriores e prevê w_k.
    """
    out = np.full(len(df_train), np.nan, dtype=float)
    weeks = sorted(df_train["week"].astype(str).unique().tolist())
    if len(weeks) < 6:
        return out
    # OOF começa depois de algumas semanas para ter amostra mínima
    for i in range(4, len(weeks)):
        w_te = weeks[i]
        w_tr = weeks[:i]
        tr = df_train[df_train["week"].astype(str).isin(w_tr)].copy()
        te_idx = df_train.index[df_train["week"].astype(str) == w_te]
        te = df_train.loc[te_idx].copy()
        if te.empty:
            continue
        p = _fit_predict_logit(tr, te)
        out[df_train.index.get_indexer(te_idx)] = p
    return out


def _oof_scores_hgb_by_week(df_train: pd.DataFrame) -> np.ndarray:
    out = np.full(len(df_train), np.nan, dtype=float)
    weeks = sorted(df_train["week"].astype(str).unique().tolist())
    if len(weeks) < 6:
        return out
    for i in range(4, len(weeks)):
        w_te = weeks[i]
        w_tr = weeks[:i]
        tr = df_train[df_train["week"].astype(str).isin(w_tr)].copy()
        te_idx = df_train.index[df_train["week"].astype(str) == w_te]
        te = df_train.loc[te_idx].copy()
        if te.empty:
            continue
        p = _fit_predict_hgb(tr, te)
        out[df_train.index.get_indexer(te_idx)] = p
    return out

def _run_wf(df: pd.DataFrame, score_col_for_dow) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    weeks = sorted(df["week"].unique().tolist())
    start_i = max(wf.MIN_GLOBAL_TRAIN_WEEKS, len(weeks) - EVAL_LAST_WEEKS)
    weekly_rows = []
    daily_rows = []
    all_rules_rows = []
    prev_rules: Dict[str, wf.Rule] = {}

    for i in range(start_i, len(weeks)):
        w_test = weeks[i]
        train_weeks = weeks[max(0, i - TRAIN_WINDOW_WEEKS) : i]
        df_train = df[df["week"].isin(train_weeks)].copy()
        df_test = df[df["week"] == w_test].copy()

        rules: Dict[str, wf.Rule] = {}
        for bet_type in ("FT", "FH"):
            for dow in wf.WEEKDAY_PT:
                sc = str(score_col_for_dow(dow))
                x = df_train[(df_train["dow_pt"] == dow) & (df_train["bet_type"] == bet_type)].copy()
                if x.empty:
                    rule = wf.Rule(bet_type=bet_type, dow=dow, score_col=sc, cutoff=1.0, stake_frac=0.0, status="no_data")
                else:
                    prev = prev_rules.get(f"{bet_type}|{dow}")
                    rule = wf.optimize_segment_train(x, sc, bayes_select=True, prev_rule=prev, roi_bias_adj=0.0)
                rules[f"{bet_type}|{dow}"] = rule

        alpha, _, _ = wf.find_alpha_global(df_train, rules)
        bets = wf.apply_rules_on_df(df_test, rules, alpha=float(alpha))

        stake_sum = float(bets["stake_eff"].sum()) if len(bets) else 0.0
        pnl_sum = float(bets["profit_cap2"].sum()) if len(bets) else 0.0
        n_bets = int(len(bets))
        weekly_rows.append(
            {
                "week": w_test,
                "train_weeks": int(len(train_weeks)),
                "alpha_global": float(alpha),
                "n_bets": n_bets,
                "stake_usd": stake_sum,
                "profit_cap2_usd": pnl_sum,
                "roi_on_stake_cap2": float(pnl_sum / stake_sum) if stake_sum > 0 else float("nan"),
            }
        )

        for key, r in rules.items():
            all_rules_rows.append(
                {
                    "test_week": w_test,
                    "train_weeks": int(len(train_weeks)),
                    "bet_type": r.bet_type,
                    "dow_pt": r.dow,
                    "score_col": r.score_col,
                    "cutoff": float(r.cutoff),
                    "stake_frac": float(r.stake_frac),
                    "alpha_global": float(alpha),
                    "status": r.status,
                    "rule_key": key,
                }
            )

        if len(bets):
            dd = bets.groupby("date", as_index=False).agg(stake_usd=("stake_eff", "sum"), profit_cap2_usd=("profit_cap2", "sum"))
            dd["week"] = w_test
            dd["alpha_global"] = float(alpha)
            daily_rows.append(dd)

        prev_rules = rules.copy()

    weekly_df = pd.DataFrame(weekly_rows)
    rules_df = pd.DataFrame(all_rules_rows)
    daily_df = pd.concat(daily_rows, axis=0, ignore_index=True) if daily_rows else pd.DataFrame(columns=["date", "stake_usd", "profit_cap2_usd", "week", "alpha_global"])
    return weekly_df, rules_df, daily_df


def _run_wf_scorev1_logit(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    weeks = sorted(df["week"].unique().tolist())
    weekly_rows = []
    daily_rows = []
    all_rules_rows = []
    prev_rules: Dict[str, wf.Rule] = {}

    for i in range(wf.MIN_GLOBAL_TRAIN_WEEKS, len(weeks)):
        w_test = weeks[i]
        train_weeks = weeks[max(0, i - TRAIN_WINDOW_WEEKS) : i]
        df_train = df[df["week"].isin(train_weeks)].copy()
        df_test = df[df["week"] == w_test].copy()

        # score no treino: OOF por semana (para não otimizar cutoffs em previsões in-sample)
        df_train["score_v1_logit"] = np.nan
        df_test["score_v1_logit"] = np.nan
        for _, dows in [
            ("segunda", {"segunda-feira"}),
            ("terca", {"terça-feira"}),
            ("quarta", {"quarta-feira"}),
            ("quinta", {"quinta-feira"}),
            ("sexdom", {"sexta-feira", "sábado", "domingo"}),
        ]:
            m_tr = df_train["dow_pt"].isin(dows)
            m_te = df_test["dow_pt"].isin(dows)
            if not m_tr.any():
                continue
            # OOF no treino
            df_train.loc[m_tr, "score_v1_logit"] = _oof_scores_logit_by_week(df_train.loc[m_tr].copy())
            # score no teste (modelo treinado em todo treino)
            if not m_te.any():
                continue
            p = _fit_predict_logit(df_train.loc[m_tr].copy(), df_test.loc[m_te].copy())
            df_test.loc[m_te, "score_v1_logit"] = p

        def score_col_for_dow(_dow: str) -> str:
            return "score_v1_logit"

        rules: Dict[str, wf.Rule] = {}
        for bet_type in ("FT", "FH"):
            for dow in wf.WEEKDAY_PT:
                sc = "score_v1_logit"
                x = df_train[(df_train["dow_pt"] == dow) & (df_train["bet_type"] == bet_type)].copy()
                if x.empty:
                    rule = wf.Rule(bet_type=bet_type, dow=dow, score_col=sc, cutoff=1.0, stake_frac=0.0, status="no_data")
                else:
                    prev = prev_rules.get(f"{bet_type}|{dow}")
                    rule = wf.optimize_segment_train(x, sc, bayes_select=True, prev_rule=prev, roi_bias_adj=0.0)
                rules[f"{bet_type}|{dow}"] = rule

        alpha, _, _ = wf.find_alpha_global(df_train, rules)
        bets = wf.apply_rules_on_df(df_test, rules, alpha=float(alpha))

        stake_sum = float(bets["stake_eff"].sum()) if len(bets) else 0.0
        pnl_sum = float(bets["profit_cap2"].sum()) if len(bets) else 0.0
        n_bets = int(len(bets))
        weekly_rows.append(
            {
                "week": w_test,
                "train_weeks": int(len(train_weeks)),
                "alpha_global": float(alpha),
                "n_bets": n_bets,
                "stake_usd": stake_sum,
                "profit_cap2_usd": pnl_sum,
                "roi_on_stake_cap2": float(pnl_sum / stake_sum) if stake_sum > 0 else float("nan"),
            }
        )

        for key, r in rules.items():
            all_rules_rows.append(
                {
                    "test_week": w_test,
                    "train_weeks": int(len(train_weeks)),
                    "bet_type": r.bet_type,
                    "dow_pt": r.dow,
                    "score_col": r.score_col,
                    "cutoff": float(r.cutoff),
                    "stake_frac": float(r.stake_frac),
                    "alpha_global": float(alpha),
                    "status": r.status,
                    "rule_key": key,
                }
            )

        if len(bets):
            dd = bets.groupby("date", as_index=False).agg(stake_usd=("stake_eff", "sum"), profit_cap2_usd=("profit_cap2", "sum"))
            dd["week"] = w_test
            dd["alpha_global"] = float(alpha)
            daily_rows.append(dd)

        prev_rules = rules.copy()

    weekly_df = pd.DataFrame(weekly_rows)
    rules_df = pd.DataFrame(all_rules_rows)
    daily_df = pd.concat(daily_rows, axis=0, ignore_index=True) if daily_rows else pd.DataFrame(columns=["date", "stake_usd", "profit_cap2_usd", "week", "alpha_global"])
    return weekly_df, rules_df, daily_df


def _run_wf_scorev1_hgb(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    weeks = sorted(df["week"].unique().tolist())
    start_i = max(wf.MIN_GLOBAL_TRAIN_WEEKS, len(weeks) - EVAL_LAST_WEEKS)
    weekly_rows = []
    daily_rows = []
    all_rules_rows = []
    prev_rules: Dict[str, wf.Rule] = {}

    for i in range(start_i, len(weeks)):
        w_test = weeks[i]
        train_weeks = weeks[max(0, i - TRAIN_WINDOW_WEEKS) : i]
        df_train = df[df["week"].isin(train_weeks)].copy()
        df_test = df[df["week"] == w_test].copy()

        df_train["score_v1_hgb"] = np.nan
        df_test["score_v1_hgb"] = np.nan
        for _, dows in [
            ("segunda", {"segunda-feira"}),
            ("terca", {"terça-feira"}),
            ("quarta", {"quarta-feira"}),
            ("quinta", {"quinta-feira"}),
            ("sexdom", {"sexta-feira", "sábado", "domingo"}),
        ]:
            m_tr = df_train["dow_pt"].isin(dows)
            m_te = df_test["dow_pt"].isin(dows)
            if not m_tr.any():
                continue
            pipe = _fit_hgb(df_train.loc[m_tr].copy())
            # score in-sample no treino (rápido)
            df_train.loc[m_tr, "score_v1_hgb"] = _predict_proba_hgb(pipe, df_train.loc[m_tr].copy())
            # score no teste
            if m_te.any():
                df_test.loc[m_te, "score_v1_hgb"] = _predict_proba_hgb(pipe, df_test.loc[m_te].copy())

        rules: Dict[str, wf.Rule] = {}
        for bet_type in ("FT", "FH"):
            for dow in wf.WEEKDAY_PT:
                sc = "score_v1_hgb"
                x = df_train[(df_train["dow_pt"] == dow) & (df_train["bet_type"] == bet_type)].copy()
                if x.empty:
                    rule = wf.Rule(bet_type=bet_type, dow=dow, score_col=sc, cutoff=1.0, stake_frac=0.0, status="no_data")
                else:
                    prev = prev_rules.get(f"{bet_type}|{dow}")
                    rule = wf.optimize_segment_train(x, sc, bayes_select=True, prev_rule=prev, roi_bias_adj=0.0)
                rules[f"{bet_type}|{dow}"] = rule

        alpha, _, _ = wf.find_alpha_global(df_train, rules)
        bets = wf.apply_rules_on_df(df_test, rules, alpha=float(alpha))

        stake_sum = float(bets["stake_eff"].sum()) if len(bets) else 0.0
        pnl_sum = float(bets["profit_cap2"].sum()) if len(bets) else 0.0
        n_bets = int(len(bets))
        weekly_rows.append(
            {
                "week": w_test,
                "train_weeks": int(len(train_weeks)),
                "alpha_global": float(alpha),
                "n_bets": n_bets,
                "stake_usd": stake_sum,
                "profit_cap2_usd": pnl_sum,
                "roi_on_stake_cap2": float(pnl_sum / stake_sum) if stake_sum > 0 else float("nan"),
            }
        )

        for key, r in rules.items():
            all_rules_rows.append(
                {
                    "test_week": w_test,
                    "train_weeks": int(len(train_weeks)),
                    "bet_type": r.bet_type,
                    "dow_pt": r.dow,
                    "score_col": r.score_col,
                    "cutoff": float(r.cutoff),
                    "stake_frac": float(r.stake_frac),
                    "alpha_global": float(alpha),
                    "status": r.status,
                    "rule_key": key,
                }
            )

        if len(bets):
            dd = bets.groupby("date", as_index=False).agg(stake_usd=("stake_eff", "sum"), profit_cap2_usd=("profit_cap2", "sum"))
            dd["week"] = w_test
            dd["alpha_global"] = float(alpha)
            daily_rows.append(dd)

        prev_rules = rules.copy()

    weekly_df = pd.DataFrame(weekly_rows)
    rules_df = pd.DataFrame(all_rules_rows)
    daily_df = pd.concat(daily_rows, axis=0, ignore_index=True) if daily_rows else pd.DataFrame(columns=["date", "stake_usd", "profit_cap2_usd", "week", "alpha_global"])
    return weekly_df, rules_df, daily_df


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    wf.POST_Q_OBJ = 0.10
    wf.MIN_POST_P_MEAN_POS = 0.70
    wf.ROBUST_CUTOFF_ENABLED = True
    wf.ROBUST_CUTOFF_DELTA = 0.02
    wf.HYSTERESIS_ENABLED = True
    wf.HYST_P_SWITCH = 0.90

    df = pd.read_csv(SCORED, parse_dates=["BIA_ApostaUTC"])
    df["house_cap"] = df["house_cap"].apply(wf.safe_cap)
    df["week"] = wf.week_key(df["BIA_ApostaUTC"])
    df["date"] = wf.date_key(df["BIA_ApostaUTC"])
    df["roi_raw"] = pd.to_numeric(df["roi_calc"], errors="coerce").astype(float)
    df["roi_cap2"] = np.minimum(df["roi_raw"].to_numpy(dtype=float), 2.0)
    df["roi_cap1"] = np.minimum(df["roi_raw"].to_numpy(dtype=float), 1.0)

    # garantir scores calibrados para quinta e sexdom (score atual)
    if "proba_cal_segqui" not in df.columns:
        df["proba_cal_segqui"] = np.nan
    if "proba_cal_sexdom" not in df.columns:
        df["proba_cal_sexdom"] = np.nan
    if "proba_raw_segqui" in df.columns and wf.CALIB_SEGQUI.exists():
        calib = json.loads(wf.CALIB_SEGQUI.read_text(encoding="utf-8"))
        x = np.asarray(calib.get("isotonic", {}).get("x", []), dtype=float)
        y = np.asarray(calib.get("isotonic", {}).get("y", []), dtype=float)
        p_raw = pd.to_numeric(df["proba_raw_segqui"], errors="coerce").to_numpy(dtype=float)
        df["proba_cal_segqui"] = wf._apply_isotonic_vec(p_raw, x=x, y=y, floor=wf.CALIB_FLOOR)
    if "proba_raw_sexdom" in df.columns and wf.CALIB_SEXDOM.exists():
        calib = json.loads(wf.CALIB_SEXDOM.read_text(encoding="utf-8"))
        x = np.asarray(calib.get("isotonic", {}).get("x", []), dtype=float)
        y = np.asarray(calib.get("isotonic", {}).get("y", []), dtype=float)
        p_raw = pd.to_numeric(df["proba_raw_sexdom"], errors="coerce").to_numpy(dtype=float)
        df["proba_cal_sexdom"] = wf._apply_isotonic_vec(p_raw, x=x, y=y, floor=wf.CALIB_FLOOR)

    wk_cur, rl_cur, dy_cur = _run_wf(df, wf.segment_score_col)
    wk_v1h, rl_v1h, dy_v1h = _run_wf_scorev1_hgb(df)

    def summarize(name: str, wk: pd.DataFrame) -> Dict[str, float | int | str]:
        stake = float(wk["stake_usd"].sum())
        pnl = float(wk["profit_cap2_usd"].sum())
        roi = float(pnl / stake) if stake > 0 else float("nan")
        w_nonzero = wk.loc[wk["stake_usd"] > 0, "profit_cap2_usd"].to_numpy(float)
        return {
            "name": name,
            "profit_cap2_total": pnl,
            "stake_total": stake,
            "roi_total_cap2": roi,
            "weeks_total": int(len(wk)),
            "weeks_with_stake": int((wk["stake_usd"] > 0).sum()),
            "mean_weekly_cap2_nonzero": float(np.mean(w_nonzero)) if w_nonzero.size else float("nan"),
            "pneg_weeks_nonzero": float((w_nonzero < 0).mean()) if w_nonzero.size else float("nan"),
        }

    summary = pd.DataFrame([summarize("score_current", wk_cur), summarize("score_v1_hgb_foldwise", wk_v1h)])
    summary_path = OUT_DIR / "oos_walkforward_scorev1_hgb_p10_p70_summary.csv"
    summary.to_csv(summary_path, index=False)

    wk_cur.to_csv(OUT_DIR / "oos_walkforward_score_current_p10_p70_weekly.csv", index=False)
    rl_cur.to_csv(OUT_DIR / "oos_walkforward_score_current_p10_p70_selected_rules.csv", index=False)
    dy_cur.to_csv(OUT_DIR / "oos_walkforward_score_current_p10_p70_daily.csv", index=False)

    wk_v1h.to_csv(OUT_DIR / "oos_walkforward_scorev1_hgb_p10_p70_weekly.csv", index=False)
    rl_v1h.to_csv(OUT_DIR / "oos_walkforward_scorev1_hgb_p10_p70_selected_rules.csv", index=False)
    dy_v1h.to_csv(OUT_DIR / "oos_walkforward_scorev1_hgb_p10_p70_daily.csv", index=False)

    print(str(summary_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

