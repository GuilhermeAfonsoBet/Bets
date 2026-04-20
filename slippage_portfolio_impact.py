#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Estima impacto histórico de slippage (odd execução vs odd proxy ex-ante) NO PORTFÓLIO OOS escolhido.

Definições:
- odd_aux  := ApostaLive.Aux1 - maior odd / 1000 (heurística: se Aux1 > 10, assume milésimos)
- odd_got  := BetinAsia.got price (ex-post)
- slippage := odd_got - odd_aux

Impacto em PnL:
- Usa stake_eff gerado pela regra (stake_frac × alpha_effective × bankroll, capado por house_cap).
- Converte resultado + odds em ROI (mesma convenção do projeto) e calcula:
    profit_cap2 = stake_eff * min(ROI, 2.0)
  para odd_got e odd_aux e toma delta.

Saídas:
- analysis_proba_raw/pro_portfolio_all/slippage_portfolio_impact_summary.csv
- analysis_proba_raw/pro_portfolio_all/slippage_portfolio_impact_weekly.csv
- analysis_proba_raw/pro_portfolio_all/slippage_portfolio_impact_detail.csv
"""

from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd


OUT_DIR = Path("/workspace/analysis_proba_raw/pro_portfolio_all")
MODE = "global_bayes_roll12_robust_p10_p70"

SCORED = Path("/workspace/analysis_proba_raw/scored_dedup_proba_raw_all.csv")
RULES = OUT_DIR / f"oos_walkforward_{MODE}_selected_rules.csv"

BANKROLL = 2300.0
CALIB_SEGQUI = Path("/workspace/clv_calib_SegQui.json")
CALIB_SEXDOM = Path("/workspace/clv_calib_SexDom.json")
CALIB_FLOOR = 0.005


def _roi_from_result_and_odds(result: object, odds: float) -> float:
    if not np.isfinite(odds) or odds <= 1e-12:
        return float("nan")
    s = str(result).strip().lower()
    if s == "win":
        return float(odds - 1.0)
    if s in {"lose", "loss"}:
        return -1.0
    if s in {"halfwin", "half win"}:
        return float((odds - 1.0) / 2.0)
    if s in {"halfloss", "halflose", "half loss", "half lose"}:
        return -0.5
    if s in {"push", "void", "refund", "cancelled", "canceled"}:
        return 0.0
    return float("nan")


def _aux_odds_from_aux1(aux1: np.ndarray) -> np.ndarray:
    a = np.asarray(aux1, dtype=float)
    # heurística de escala: >10 => milésimos
    out = np.where(np.isfinite(a) & (a > 10.0), a / 1000.0, a)
    return out.astype(float)


def _apply_isotonic_vec(p_raw: np.ndarray, calib_path: Path, floor: float) -> np.ndarray:
    """
    Aplica calibração isotônica salva em JSON (isotonic.x / isotonic.y), com extrapolação constante e floor.
    Se calibrador ausente/inválido, retorna p_raw com floor.
    """
    p = np.asarray(p_raw, dtype=float)
    out = p.copy()
    try:
        if calib_path.exists():
            obj = json.loads(calib_path.read_text(encoding="utf-8"))
            x = np.asarray(obj.get("isotonic", {}).get("x", []), dtype=float)
            y = np.asarray(obj.get("isotonic", {}).get("y", []), dtype=float)
            if x.size and y.size and x.size == y.size:
                out = np.interp(out, x, y, left=float(y[0]), right=float(y[-1]))
    except Exception:
        out = p.copy()
    out = np.maximum(out, float(floor))
    return np.clip(out, 0.0, 1.0)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not SCORED.exists():
        raise SystemExit(f"Arquivo não encontrado: {SCORED}")
    if not RULES.exists():
        raise SystemExit(f"Arquivo não encontrado: {RULES}")

    df = pd.read_csv(SCORED, parse_dates=["BIA_ApostaUTC"])
    rules = pd.read_csv(RULES)
    rules["test_week"] = rules["test_week"].astype(str)

    # colunas necessárias para slippage
    need_cols = {
        "ID Aposta",
        "bet_type",
        "dow_pt",
        "house_cap",
        "RebelBetting.Result",
        "BetinAsia.got price",
        "ApostaLive.Aux1 - maior odd",
    }
    miss = [c for c in need_cols if c not in df.columns]
    if miss:
        raise SystemExit("Colunas ausentes no SCORED (regerar pipeline?): " + ", ".join(miss))

    # Garantir colunas calibradas (SegQui/ SexDom), pois regras podem referenciar proba_cal_*.
    # Replica a lógica do WF (calibração isotônica + floor).
    if "proba_cal_segqui" not in df.columns:
        df["proba_cal_segqui"] = np.nan
    if "proba_raw_segqui" in df.columns and CALIB_SEGQUI.exists():
        p_raw = pd.to_numeric(df["proba_raw_segqui"], errors="coerce").to_numpy(dtype=float)
        df["proba_cal_segqui"] = _apply_isotonic_vec(p_raw, calib_path=CALIB_SEGQUI, floor=CALIB_FLOOR)
    if "proba_cal_sexdom" not in df.columns:
        df["proba_cal_sexdom"] = np.nan
    if "proba_raw_sexdom" in df.columns and CALIB_SEXDOM.exists():
        p_raw = pd.to_numeric(df["proba_raw_sexdom"], errors="coerce").to_numpy(dtype=float)
        df["proba_cal_sexdom"] = _apply_isotonic_vec(p_raw, calib_path=CALIB_SEXDOM, floor=CALIB_FLOOR)

    # Para seleção, precisamos de todas as possíveis score_cols referenciadas nas regras
    score_cols = sorted(set(rules["score_col"].astype(str).unique().tolist()))
    for c in score_cols:
        if c not in df.columns:
            raise SystemExit(f"Coluna de score ausente no SCORED: {c}")

    # adicionar coluna week no df para filtrar rápido
    df["week"] = pd.to_datetime(df["BIA_ApostaUTC"]).dt.to_period("W-SUN").astype(str)

    all_rows = []
    weekly_rows = []

    for w_test, rw in rules.groupby("test_week", as_index=False):
        w_test = str(w_test)
        rw = rw.copy()
        # alpha_effective é o que de fato multiplicou os stakes naquela semana (inclui overlay de regime, se houver)
        if "alpha_effective" in rw.columns and np.isfinite(pd.to_numeric(rw["alpha_effective"], errors="coerce").iloc[0]):
            alpha = float(pd.to_numeric(rw["alpha_effective"], errors="coerce").iloc[0])
        else:
            alpha = float(pd.to_numeric(rw.get("alpha_global", 1.0), errors="coerce").iloc[0]) if "alpha_global" in rw.columns else 1.0

        dfw = df[df["week"].astype(str) == w_test].copy()
        if dfw.empty:
            continue

        selected = []
        for _, r in rw.iterrows():
            if str(r.get("status")) != "ok":
                continue
            stake_frac = float(r.get("stake_frac", 0.0))
            if stake_frac <= 0:
                continue
            bt = str(r["bet_type"])
            dow = str(r["dow_pt"])
            score_col = str(r["score_col"])
            cutoff = float(r["cutoff"])

            x = dfw[(dfw["bet_type"] == bt) & (dfw["dow_pt"] == dow)].copy()
            if x.empty:
                continue
            s = pd.to_numeric(x[score_col], errors="coerce").to_numpy(dtype=float)
            m = np.isfinite(s) & (s >= cutoff)
            if not np.any(m):
                continue
            x = x.iloc[np.where(m)[0]].copy()

            cap = pd.to_numeric(x["house_cap"], errors="coerce").to_numpy(dtype=float)
            stake0 = float(BANKROLL) * float(stake_frac) * float(alpha)
            stake_eff = np.minimum(stake0, cap)
            x["stake_eff"] = stake_eff
            x["rule_key"] = str(r.get("rule_key", f"{bt}|{dow}"))
            x["alpha_effective"] = float(alpha)
            selected.append(
                x[
                    [
                        "ID Aposta",
                        "BIA_ApostaUTC",
                        "week",
                        "bet_type",
                        "dow_pt",
                        "rule_key",
                        "alpha_effective",
                        "stake_eff",
                        "house_cap",
                        "RebelBetting.Result",
                        "BetinAsia.got price",
                        "ApostaLive.Aux1 - maior odd",
                    ]
                ]
            )

        sel = pd.concat(selected, axis=0, ignore_index=True) if selected else pd.DataFrame()
        if sel.empty:
            weekly_rows.append(
                {
                    "week": w_test,
                    "n_bets": 0,
                    "stake_total": 0.0,
                    "stake_covered": 0.0,
                    "delta_profit_cap2": 0.0,
                    "delta_profit_uncapped": 0.0,
                    "delta_roi_on_stake_cap2_all": float("nan"),
                    "delta_roi_on_stake_cap2_covered": float("nan"),
                    "delta_roi_on_stake_uncapped_all": float("nan"),
                    "delta_roi_on_stake_uncapped_covered": float("nan"),
                }
            )
            continue

        # odds
        got = pd.to_numeric(sel["BetinAsia.got price"], errors="coerce").to_numpy(dtype=float)
        aux1 = pd.to_numeric(sel["ApostaLive.Aux1 - maior odd"], errors="coerce").to_numpy(dtype=float)
        aux = _aux_odds_from_aux1(aux1)

        stake_eff = pd.to_numeric(sel["stake_eff"], errors="coerce").to_numpy(dtype=float)
        res = sel["RebelBetting.Result"].to_numpy()

        roi_got = np.array([_roi_from_result_and_odds(r, o) for r, o in zip(res, got)], dtype=float)
        roi_aux = np.array([_roi_from_result_and_odds(r, o) for r, o in zip(res, aux)], dtype=float)

        roi_got_cap2 = np.minimum(roi_got, 2.0)
        roi_aux_cap2 = np.minimum(roi_aux, 2.0)

        # cobertura: onde conseguimos comparar got vs aux
        ok = (
            np.isfinite(got)
            & np.isfinite(aux)
            & (got > 1.0)
            & (aux > 1.0)
            & np.isfinite(stake_eff)
            & (stake_eff > 0)
            & np.isfinite(roi_got)
            & np.isfinite(roi_aux)
        )
        stake_total = float(np.nansum(stake_eff[np.isfinite(stake_eff) & (stake_eff > 0)]))
        stake_covered = float(np.nansum(stake_eff[ok]))

        delta_cap2 = stake_eff * (roi_got_cap2 - roi_aux_cap2)
        delta_uncapped = stake_eff * (roi_got - roi_aux)

        # por convenção, delta fora de cobertura vira 0 (impacto não mensurável => neutro na estimativa)
        delta_cap2_all = float(np.nansum(np.where(ok, delta_cap2, 0.0)))
        delta_unc_all = float(np.nansum(np.where(ok, delta_uncapped, 0.0)))

        delta_cap2_cov = float(np.nansum(delta_cap2[ok])) if int(np.sum(ok)) else 0.0
        delta_unc_cov = float(np.nansum(delta_uncapped[ok])) if int(np.sum(ok)) else 0.0

        sel["odd_got"] = got
        sel["odd_aux"] = aux
        sel["slippage_odds"] = got - aux
        sel["roi_got"] = roi_got
        sel["roi_aux"] = roi_aux
        sel["roi_got_cap2"] = roi_got_cap2
        sel["roi_aux_cap2"] = roi_aux_cap2
        sel["delta_profit_cap2"] = delta_cap2
        sel["delta_profit_uncapped"] = delta_uncapped
        sel["slippage_covered"] = ok.astype(int)

        all_rows.append(sel)

        weekly_rows.append(
            {
                "week": w_test,
                "n_bets": int(len(sel)),
                "stake_total": stake_total,
                "stake_covered": stake_covered,
                "delta_profit_cap2": delta_cap2_all,
                "delta_profit_uncapped": delta_unc_all,
                "delta_roi_on_stake_cap2_all": (delta_cap2_all / stake_total) if stake_total > 0 else float("nan"),
                "delta_roi_on_stake_cap2_covered": (delta_cap2_cov / stake_covered) if stake_covered > 0 else float("nan"),
                "delta_roi_on_stake_uncapped_all": (delta_unc_all / stake_total) if stake_total > 0 else float("nan"),
                "delta_roi_on_stake_uncapped_covered": (delta_unc_cov / stake_covered) if stake_covered > 0 else float("nan"),
            }
        )

    detail = pd.concat(all_rows, axis=0, ignore_index=True) if all_rows else pd.DataFrame()
    weekly = pd.DataFrame(weekly_rows).sort_values("week")

    # resumo global (impacto estimado no OOS)
    stake_total = float(weekly["stake_total"].sum())
    stake_cov = float(weekly["stake_covered"].sum())
    d_cap2 = float(weekly["delta_profit_cap2"].sum())
    d_unc = float(weekly["delta_profit_uncapped"].sum())
    w_nonzero_stake = weekly.loc[weekly["stake_total"] > 0].copy()
    stake_per_week = float(w_nonzero_stake["stake_total"].mean()) if not w_nonzero_stake.empty else float("nan")

    # impacto “esperado” (por semana) usando delta ROI histórico × stake/semana típico do OOS
    delta_roi_cap2 = float(d_cap2 / stake_total) if stake_total > 0 else float("nan")
    exp_delta_week_cap2 = float(delta_roi_cap2 * stake_per_week) if np.isfinite(delta_roi_cap2) and np.isfinite(stake_per_week) else float("nan")

    summary = pd.DataFrame(
        [
            {
                "mode": MODE,
                "weeks": int(weekly.shape[0]),
                "weeks_with_stake": int((weekly["stake_total"] > 0).sum()),
                "stake_total": stake_total,
                "stake_covered": stake_cov,
                "stake_covered_pct": float(stake_cov / stake_total) if stake_total > 0 else float("nan"),
                "delta_profit_cap2_total": d_cap2,
                "delta_profit_uncapped_total": d_unc,
                "delta_roi_on_stake_cap2": delta_roi_cap2,
                "delta_roi_on_stake_uncapped": float(d_unc / stake_total) if stake_total > 0 else float("nan"),
                "stake_per_week_when_active": stake_per_week,
                "exp_delta_profit_cap2_per_week": exp_delta_week_cap2,
            }
        ]
    )

    out_sum = OUT_DIR / "slippage_portfolio_impact_summary.csv"
    out_wk = OUT_DIR / "slippage_portfolio_impact_weekly.csv"
    out_det = OUT_DIR / "slippage_portfolio_impact_detail.csv"
    summary.to_csv(out_sum, index=False)
    weekly.to_csv(out_wk, index=False)
    detail.to_csv(out_det, index=False)

    print(str(out_sum))
    print(str(out_wk))
    print(str(out_det))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

