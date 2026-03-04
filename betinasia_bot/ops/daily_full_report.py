from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from loguru import logger

from .accounting_daily_report import DailyCfg as AcctDailyCfg, run_daily as run_acct_daily
from .accounting_report import compute_pnl_report
from .execution_kpis import compute_kpis_from_lines


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _load_env_file(path: Path) -> None:
    """
    Carrega variáveis de um arquivo .env simples (KEY=VALUE), sem sobrescrever env já definido.
    Ajuda quando rodando manualmente fora do systemd (que usa EnvironmentFile=...).
    """
    try:
        if not path.exists():
            return
        for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            if not k or k in os.environ:
                continue
            os.environ[k] = v.strip()
    except Exception:
        return


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if not path.exists():
            return None
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _fmt_pct(x: Any, nd: int = 2) -> str:
    try:
        if x is None:
            return "—"
        return f"{float(x):.{nd}f}%"
    except Exception:
        return "—"


def _fmt_num(x: Any, nd: int = 2) -> str:
    try:
        if x is None:
            return "—"
        return f"{float(x):.{nd}f}"
    except Exception:
        return "—"


def _demote_h2_to_h3(md: str) -> str:
    # Usado para "embrulhar" o bloco in-sample sem reescrever o conteúdo.
    out = []
    for ln in (md or "").splitlines(True):
        if ln.startswith("## "):
            out.append("### " + ln[3:])
        else:
            out.append(ln)
    return "".join(out)


def _split_base_into_insample_and_oos(md: str) -> tuple[str, str]:
    """
    O relatório robusto escreve o bloco OOS no topo-nível:
      '## 12) OOS walk-forward ...'
    Tudo antes disso é o bloco in-sample.
    """
    key = "## 12) OOS walk-forward"
    i = (md or "").find(key)
    if i < 0:
        # fallback: não encontrou; trata tudo como in-sample
        return (md or ""), ""
    return (md or "")[:i], (md or "")[i:]


def _extract_md_block(md: str, *, start: str, until_any: list[str]) -> str:
    """
    Extrai um trecho de markdown começando em `start` até antes do primeiro marcador em `until_any`.
    Best-effort: se não achar `start`, retorna "".
    """
    txt = md or ""
    i = txt.find(start)
    if i < 0:
        return ""
    j = None
    for u in until_any:
        k = txt.find(u, i + len(start))
        if k >= 0:
            j = k if j is None else min(j, k)
    return txt[i : (j if j is not None else len(txt))].strip() + "\n"


def _tail_lines(path: Path, n: int) -> list[str]:
    try:
        xs = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        return xs[-n:] if n > 0 else xs
    except Exception:
        return []

def _week_start_iso(day_iso: str) -> Optional[str]:
    try:
        from datetime import date as _date, timedelta as _td

        d = _date.fromisoformat(str(day_iso))
        ws = d - _td(days=int(d.weekday()))
        return ws.isoformat()
    except Exception:
        return None


def _month_key(day_iso: str) -> Optional[str]:
    try:
        from datetime import date as _date

        d = _date.fromisoformat(str(day_iso))
        return f"{d.year:04d}-{d.month:02d}"
    except Exception:
        return None


def _agg_by_week(pnls_by_day: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for d, v in pnls_by_day.items():
        ws = _week_start_iso(d)
        if not ws:
            continue
        out[ws] = float(out.get(ws, 0.0)) + float(v or 0.0)
    return dict(sorted(out.items()))


def _agg_by_month(pnls_by_day: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for d, v in pnls_by_day.items():
        mk = _month_key(d)
        if not mk:
            continue
        out[mk] = float(out.get(mk, 0.0)) + float(v or 0.0)
    return dict(sorted(out.items()))


def _max_drawdown(pnls_by_day: Dict[str, float]) -> Dict[str, Any]:
    """
    Max drawdown em unidade monetária, usando curva de equity = cumsum(P&L diário).
    """
    days = sorted([d for d in pnls_by_day.keys() if str(d)])
    eq = 0.0
    peak = 0.0
    mdd = 0.0
    mdd_from = None
    mdd_to = None
    peak_day = None
    for d in days:
        eq += float(pnls_by_day.get(d) or 0.0)
        if eq >= peak:
            peak = eq
            peak_day = d
        dd = peak - eq
        if dd > mdd:
            mdd = dd
            mdd_from = peak_day
            mdd_to = d
    return {"mdd": float(mdd), "from_day": mdd_from, "to_day": mdd_to}


def _sharpe_annualized(pnls_by_day: Dict[str, float], *, bankroll_ref: float) -> Optional[float]:
    """
    Sharpe anualizado (sqrt(252)) usando retornos diários r = pnl / bankroll_ref.
    """
    try:
        br = float(bankroll_ref)
        if br <= 0:
            return None
        rs = [float(v) / br for _, v in sorted(pnls_by_day.items())]
        if len(rs) < 5:
            return None
        import statistics
        import math

        mu = statistics.fmean(rs)
        sd = statistics.pstdev(rs)
        if sd <= 0:
            return None
        return float((mu / sd) * math.sqrt(252.0))
    except Exception:
        return None

def _read_jsonl_last(path: Path, last: int) -> list[str]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if last > 0 and len(lines) > last:
        return lines[-last:]
    return lines


def _telegram_send_document(token: str, chat_id: str, *, file_path: Path, caption: str) -> bool:
    url = f"https://api.telegram.org/bot{token}/sendDocument"
    with file_path.open("rb") as f:
        files = {"document": (file_path.name, f, "application/pdf")}
        data = {"chat_id": chat_id, "caption": caption[:900]}
        r = requests.post(url, data=data, files=files, timeout=60)
        return bool(r.ok)


@dataclass
class DailyReportCfg:
    out_dir: Path = Path("logs/daily_reports")
    report_tz: str = "America/Sao_Paulo"
    # Alinhar com o relatório “v38” por default
    versions: str = os.getenv("DAILY_OOS_VERSIONS", "v4.0-api,v5.0-ws-only,v5.1-ws-gate-lay")
    direction: str = os.getenv("DAILY_OOS_DIRECTION", "up")
    # Alinha com o relatório “atual” (ex.: 21d) se o usuário não setar nada.
    lookback_days: str = os.getenv("DAILY_OOS_LOOKBACK_DAYS", "21")
    no_auto_exclude_days: bool = (os.getenv("DAILY_NO_AUTO_EXCLUDE_DAYS", "0").strip() in ("1", "true", "True", "yes", "YES"))
    report_mode: str = os.getenv("DAILY_REPORT_MODE", "oos_first")
    wf_policy_current: Path = Path(os.getenv("DAILY_WF_POLICY_CURRENT", "logs/wf_policy_current.json"))
    wf_policy_history_dir: Path = Path(os.getenv("DAILY_WF_POLICY_HISTORY_DIR", "logs/policy_history"))
    wf_policy_history_jsonl: Path = Path(os.getenv("DAILY_WF_POLICY_HISTORY_JSONL", "logs/wf_policy_history.jsonl"))
    # Walk-forward knobs (para casar com versões como leaguePre / AHgatePre / expanding)
    wf_train_mode: str = os.getenv("DAILY_WF_TRAIN_MODE", "expanding")
    wf_train_days: str = os.getenv("DAILY_WF_TRAIN_DAYS", "2")
    wf_test_days: str = os.getenv("DAILY_WF_TEST_DAYS", "2")
    wf_step_days: str = os.getenv("DAILY_WF_STEP_DAYS", "2")
    wf_key_by_league: bool = (os.getenv("DAILY_WF_KEY_BY_LEAGUE", "1").strip() in ("1", "true", "True", "yes", "YES"))
    wf_key_by_league_scope: str = os.getenv("DAILY_WF_KEY_BY_LEAGUE_SCOPE", "pre")
    wf_ah_max_abs_line: str = os.getenv("DAILY_WF_AH_MAX_ABS_LINE", "2.0")
    wf_ah_scope: str = os.getenv("DAILY_WF_AH_SCOPE", "pre")
    wf_liquidity_mode: str = os.getenv("DAILY_WF_LIQUIDITY_MODE", "none")
    wf_liquidity_scope: str = os.getenv("DAILY_WF_LIQUIDITY_SCOPE", "pre")
    wf_min_matches: str = os.getenv("DAILY_WF_MIN_MATCHES", "0")
    wf_shrinkage: bool = (os.getenv("DAILY_WF_SHRINKAGE", "1").strip() in ("1", "true", "True", "yes", "YES"))
    wf_exclude_exec_buckets_back: str = os.getenv("DAILY_WF_EXCLUDE_EXEC_BUCKETS_BACK", "10-20s")
    # Escala de banca/sizing (manter “10k etc.”)
    kelly_bankroll: str = os.getenv("DAILY_KELLY_BANKROLL", "10000")
    wf_bankroll_grid: str = os.getenv("DAILY_WF_BANKROLL_GRID", "").strip()
    executor_jsonl: Path = Path(os.getenv("EXECUTOR_JSONL", "logs/executor_live.jsonl"))
    exec_kpi_last: int = int(os.getenv("DAILY_EXEC_KPI_LAST", "50000"))
    send_telegram: bool = (os.getenv("DAILY_REPORT_TELEGRAM", "1").strip() not in ("0", "false", "False", "no", "NO"))

    def __post_init__(self) -> None:
        # Releitura de env em runtime (importante quando rodando manualmente e carregando .env em main()).
        self.versions = os.getenv("DAILY_OOS_VERSIONS", self.versions)
        self.direction = os.getenv("DAILY_OOS_DIRECTION", self.direction)
        self.lookback_days = os.getenv("DAILY_OOS_LOOKBACK_DAYS", self.lookback_days)
        self.report_mode = os.getenv("DAILY_REPORT_MODE", self.report_mode)
        self.wf_policy_current = Path(os.getenv("DAILY_WF_POLICY_CURRENT", str(self.wf_policy_current)))
        self.wf_policy_history_dir = Path(os.getenv("DAILY_WF_POLICY_HISTORY_DIR", str(self.wf_policy_history_dir)))
        self.wf_policy_history_jsonl = Path(os.getenv("DAILY_WF_POLICY_HISTORY_JSONL", str(self.wf_policy_history_jsonl)))
        self.executor_jsonl = Path(os.getenv("EXECUTOR_JSONL", str(self.executor_jsonl)))
        try:
            self.exec_kpi_last = int(os.getenv("DAILY_EXEC_KPI_LAST", str(self.exec_kpi_last)))
        except Exception:
            pass


async def run_daily_full(cfg: DailyReportCfg) -> Dict[str, Any]:
    ts = _utcnow()
    day = ts.astimezone(timezone.utc).strftime("%Y%m%d")
    day_dir = cfg.out_dir / day
    day_dir.mkdir(parents=True, exist_ok=True)

    # 1) Accounting snapshot + report
    acct_out = day_dir / "accounting_daily_report.json"
    acct: Dict[str, Any] = {}
    try:
        acct = await run_acct_daily(
            AcctDailyCfg(
                out_dir=Path(os.getenv("ACCOUNTING_OUT_DIR", "logs/accounting")),
                jsonl=Path(os.getenv("ACCOUNTING_JSONL", "logs/accounting_snapshots.jsonl")),
                tz_name=str(os.getenv("REPORT_TZ", cfg.report_tz)),
                report_out=acct_out,
                print_json=False,
            )
        )
    except Exception as e:
        # Não aborta o daily: ainda queremos OOS + KPIs + aderência mesmo sem login no accounting.
        acct = {"ts": ts.isoformat(), "error": f"ACCOUNTING_FAILED: {str(e)[:200]}"}
        try:
            acct_out.write_text(json.dumps(acct, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    # 2) Execution KPIs (all + success-only)
    exec_lines = []
    if cfg.executor_jsonl.exists():
        exec_lines = _read_jsonl_last(cfg.executor_jsonl, int(cfg.exec_kpi_last))
    kpi_all = compute_kpis_from_lines(exec_lines, path=str(cfg.executor_jsonl))
    kpi_ok = compute_kpis_from_lines(exec_lines, path=str(cfg.executor_jsonl), only_status=["LIVE_OK", "DRY_OK"])
    (day_dir / "execution_kpis_all.json").write_text(json.dumps(kpi_all, ensure_ascii=False, indent=2), encoding="utf-8")
    (day_dir / "execution_kpis_ok.json").write_text(json.dumps(kpi_ok, ensure_ascii=False, indent=2), encoding="utf-8")

    # 3) Rodar OOS (walk-forward) e exportar policy
    base_md = day_dir / "report_base.md"
    policy_hist = cfg.wf_policy_history_dir / f"wf_policy_{day}.json"
    cfg.wf_policy_history_dir.mkdir(parents=True, exist_ok=True)

    args = [
        sys.executable,
        str(Path(__file__).resolve().parent.parent / "analyze_contexto_operacao_b808_robust_report.py"),
        "--direction",
        str(cfg.direction),
        "--versions",
        str(cfg.versions),
        "--out",
        str(base_md),
        "--report-mode",
        str(cfg.report_mode),
        "--walkforward",
        "--wf-export-policy-json",
        str(policy_hist),
    ]
    if bool(cfg.no_auto_exclude_days):
        args += ["--no-auto-exclude-days"]
    if str(cfg.lookback_days).strip():
        args += ["--lookback-days", str(cfg.lookback_days).strip()]
    if str(cfg.kelly_bankroll).strip():
        args += ["--kelly-bankroll", str(cfg.kelly_bankroll).strip()]
    if str(cfg.wf_bankroll_grid).strip():
        args += ["--wf-bankroll-grid", str(cfg.wf_bankroll_grid).strip()]
    if str(cfg.wf_train_mode).strip():
        args += ["--wf-train-mode", str(cfg.wf_train_mode).strip()]
    if str(cfg.wf_train_days).strip():
        args += ["--wf-train-days", str(cfg.wf_train_days).strip()]
    if str(cfg.wf_test_days).strip():
        args += ["--wf-test-days", str(cfg.wf_test_days).strip()]
    if str(cfg.wf_step_days).strip():
        args += ["--wf-step-days", str(cfg.wf_step_days).strip()]
    if bool(cfg.wf_key_by_league):
        args += ["--wf-key-by-league"]
        if str(cfg.wf_key_by_league_scope).strip():
            args += ["--wf-key-by-league-scope", str(cfg.wf_key_by_league_scope).strip()]
    if str(cfg.wf_ah_max_abs_line).strip():
        args += ["--wf-ah-max-abs-line", str(cfg.wf_ah_max_abs_line).strip()]
        if str(cfg.wf_ah_scope).strip():
            args += ["--wf-ah-scope", str(cfg.wf_ah_scope).strip()]
    if str(cfg.wf_liquidity_mode).strip():
        args += ["--wf-liquidity-mode", str(cfg.wf_liquidity_mode).strip()]
        if str(cfg.wf_liquidity_scope).strip():
            args += ["--wf-liquidity-scope", str(cfg.wf_liquidity_scope).strip()]
    if str(cfg.wf_min_matches).strip():
        args += ["--wf-min-matches", str(cfg.wf_min_matches).strip()]
    if bool(cfg.wf_shrinkage):
        args += ["--wf-shrinkage"]
    if str(cfg.wf_exclude_exec_buckets_back).strip():
        args += ["--wf-exclude-exec-buckets-back", str(cfg.wf_exclude_exec_buckets_back).strip()]

    subprocess.run(args, check=True, cwd=str(Path(__file__).resolve().parent.parent))

    # Atualiza policy_current (atomic replace) e registra histórico (jsonl)
    cfg.wf_policy_current.parent.mkdir(parents=True, exist_ok=True)
    tmp = cfg.wf_policy_current.with_suffix(".tmp")
    tmp.write_text(policy_hist.read_text(encoding="utf-8"), encoding="utf-8")
    tmp.replace(cfg.wf_policy_current)

    active_keys = None
    active_keys_base = None
    try:
        pol = json.loads(policy_hist.read_text(encoding="utf-8"))
        steps = pol.get("steps") if isinstance(pol, dict) else []
        last = steps[-1] if isinstance(steps, list) and steps else {}
        if isinstance(last, dict):
            active_keys = last.get("active_keys")
            active_keys_base = last.get("active_keys_base")
        rec = {
            "ts": ts.isoformat(),
            "policy_path": str(policy_hist),
            "policy_current": str(cfg.wf_policy_current),
            "active_keys": active_keys,
            "active_keys_base": active_keys_base,
        }
        cfg.wf_policy_history_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with cfg.wf_policy_history_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass

    # 4) Relatórios auxiliares para seção 0/1 e apêndices (99.x)
    adh_json = day_dir / "oos_adherence.json"
    adh: Optional[Dict[str, Any]] = None
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "ops.oos_adherence_report",
                "--policy-json",
                str(cfg.wf_policy_current),
                "--executor-jsonl",
                str(cfg.executor_jsonl),
                "--tz",
                "UTC",
                "--days",
                str(os.getenv("DAILY_ADHERENCE_DAYS", "7")),
                "--out",
                str(adh_json),
            ],
            check=False,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        adh = _read_json(adh_json)
    except Exception:
        adh = None

    audit_json = day_dir / "audit_status_kpis.json"
    audit_rep: Optional[Dict[str, Any]] = None
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "ops.audit_status_kpis",
                "--hours",
                str(os.getenv("DAILY_AUDIT_KPI_HOURS", "24")),
                "--direction",
                str(cfg.direction),
                "--out",
                str(audit_json),
            ],
            check=False,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        audit_rep = _read_json(audit_json)
    except Exception:
        audit_rep = None

    # 5) Montagem do relatório final:
    # Ordem pedida: 0 (Resumo) -> 1 (Resultados reais) -> 2 (OOS) -> 3 (In-sample) -> 99 (apêndices operacionais)
    base_txt = ""
    try:
        base_txt = base_md.read_text(encoding="utf-8")
    except Exception:
        base_txt = ""
    insample_txt, oos_txt = _split_base_into_insample_and_oos(base_txt)
    if oos_txt:
        oos_txt = oos_txt.replace("## 12) OOS walk-forward", "## 2) OOS walk-forward")

    # Accounting: série por dia/mês a partir do CSV (quando disponível)
    acct_series = None
    try:
        bal_csv = Path(str(acct.get("balance_csv") or "")).expanduser()
        if bal_csv.exists():
            tz = timezone.utc
            try:
                from zoneinfo import ZoneInfo  # type: ignore

                tz = ZoneInfo(str(os.getenv("REPORT_TZ", cfg.report_tz)))
            except Exception:
                tz = timezone.utc
            acct_series = compute_pnl_report(bal_csv, tz=tz)
    except Exception:
        acct_series = None

    # --- Seção 0: Resumo / conclusões (executivo) ---
    s0 = []
    s0.append("## 0) Resumo e conclusões (executivo)\n\n")

    # performance “real” (accounting) quando houver
    if isinstance(acct, dict) and not acct.get("error"):
        s0.append(f"- **Banca real (saldo atual)**: `{acct.get('balance_current')}`\n")
        s0.append(f"- **P&L (hoje / semana / mês)**: `{acct.get('pnl_today')} / {acct.get('pnl_week')} / {acct.get('pnl_month')}`\n")
    else:
        s0.append("- **Accounting**: indisponível (ver apêndice 99.1)\n")

    # risco/estabilidade operacional (últimas 24h) via audit_status_kpis
    top_errs = []
    try:
        for it in (audit_rep or {}).get("error_rows") or []:
            if not isinstance(it, dict):
                continue
            n = int(it.get("n") or 0)
            if n <= 0:
                continue
            top_errs.append((n, str(it.get("audit_version") or ""), str(it.get("status") or ""), str(it.get("api_error") or "")))
        top_errs.sort(key=lambda x: x[0], reverse=True)
    except Exception:
        top_errs = []
    if top_errs:
        s0.append("\n**Principais causas de perda de throughput (24h)**\n\n")
        for n, ver, st, err in top_errs[:6]:
            err2 = (err[:160] + "…") if len(err) > 160 else err
            s0.append(f"- `{ver}`: **{st} ×{n}** — `{err2}`\n")

    # leitura executiva (heurística): onde atacar primeiro
    s0.append(
        "\n**Conclusões operacionais (prioridades)**\n\n"
        "- **Objetivo 1 (conversão)**: reduzir `API_FAILED` (especialmente `No PMMs received`) e `STALE_QUEUE_WAIT` para aumentar taxa de execução sem inflar risco.\n"
        "- **Objetivo 2 (governança de risco)**: consolidar sizing/limites (banca teórica vs banca real) e travas para evitar picos (`too_many_open_betslips`, rate limit, backoff).\n"
        "- **Objetivo 3 (qualidade de entrada)**: acompanhar slippage **com sinal** e seu impacto em ROI por bucket (negativo/flat/positivo) para validar edge e execução.\n\n"
    )

    # --- Seção 1: Resultados reais (shadow/live) ---
    s1 = []
    s1.append("## 1) Resultados reais (shadow/live)\n\n")

    # KPIs por recortes (diário/semana/mês) — quando há série
    if acct_series is not None:
        # preferir P&L filtrado (exclui depósitos/saques) quando existir
        pnls = acct_series.pnl_by_day_filtered or acct_series.pnl_by_day
        # semana corrente (por dia)
        try:
            now = _utcnow().astimezone(timezone.utc)
            # usa o maior dia presente como "today" do dataset para evitar mismatch tz
            days_sorted = sorted(pnls.keys())
            today = days_sorted[-1] if days_sorted else now.date().isoformat()
            ws = _week_start_iso(today) or today
            cur_week_days = [d for d in days_sorted if ws <= d <= today]
        except Exception:
            cur_week_days = []
            today = None
            ws = None

        s1.append("**P&L real por dia (semana corrente)**\n\n")
        s1.append("| Dia | P&L |\n|---|---:|\n")
        for d in (cur_week_days or [])[-14:]:
            s1.append(f"| {d} | {_fmt_num(pnls.get(d), 2)} |\n")
        s1.append("\n")

        # drawdown e sharpe (curto, usando a própria janela da série)
        dd = _max_drawdown({k: float(v) for k, v in pnls.items()})
        dd_w = _max_drawdown(_agg_by_week({k: float(v) for k, v in pnls.items()}))
        dd_m = _max_drawdown(_agg_by_month({k: float(v) for k, v in pnls.items()}))
        br_real = None
        try:
            br_real = float(acct.get("balance_current")) if isinstance(acct, dict) and acct.get("balance_current") is not None else None
        except Exception:
            br_real = None
        br_theo = None
        try:
            br_theo = float(cfg.kelly_bankroll) if str(cfg.kelly_bankroll).strip() else None
        except Exception:
            br_theo = None

        s1.append("**Risco/consistência (a partir do P&L diário)**\n\n")
        s1.append("| Métrica | Valor |\n|---|---:|\n")
        s1.append(f"| Max drawdown (diário, monetário) | {_fmt_num(dd.get('mdd'), 2)} |\n")
        s1.append(f"| Max drawdown (semanal, monetário) | {_fmt_num(dd_w.get('mdd'), 2)} |\n")
        s1.append(f"| Max drawdown (mensal, monetário) | {_fmt_num(dd_m.get('mdd'), 2)} |\n")
        if dd.get("from_day") and dd.get("to_day"):
            s1.append(f"| Janela do DD | {dd.get('from_day')} → {dd.get('to_day')} |\n")
        if br_real:
            sh = _sharpe_annualized(pnls, bankroll_ref=float(br_real))
            s1.append(f"| Sharpe anualizado (vs banca real) | {_fmt_num(sh, 2)} |\n")
            # ROI por banca (recortes simples)
            try:
                pnl_week = float(acct.get("pnl_filtered_week") if acct.get("pnl_filtered_week") is not None else acct.get("pnl_week") or 0.0)
                pnl_month = float(acct.get("pnl_filtered_month") if acct.get("pnl_filtered_month") is not None else acct.get("pnl_month") or 0.0)
                s1.append(f"| ROI/banca real (semana) | {_fmt_num((pnl_week/float(br_real))*100.0, 2)}% |\n")
                s1.append(f"| ROI/banca real (mês) | {_fmt_num((pnl_month/float(br_real))*100.0, 2)}% |\n")
            except Exception:
                pass
        if br_theo:
            sh2 = _sharpe_annualized(pnls, bankroll_ref=float(br_theo))
            s1.append(f"| Sharpe anualizado (vs banca teórica) | {_fmt_num(sh2, 2)} |\n")
            try:
                pnl_week = float(acct.get("pnl_filtered_week") if acct.get("pnl_filtered_week") is not None else acct.get("pnl_week") or 0.0)
                pnl_month = float(acct.get("pnl_filtered_month") if acct.get("pnl_filtered_month") is not None else acct.get("pnl_month") or 0.0)
                s1.append(f"| ROI/banca teórica (semana; ref={_fmt_num(br_theo,0)}) | {_fmt_num((pnl_week/float(br_theo))*100.0, 2)}% |\n")
                s1.append(f"| ROI/banca teórica (mês; ref={_fmt_num(br_theo,0)}) | {_fmt_num((pnl_month/float(br_theo))*100.0, 2)}% |\n")
            except Exception:
                pass
        s1.append("\n")

        # semanas fechadas do mês corrente (visão executiva)
        try:
            weeks = _agg_by_week({k: float(v) for k, v in pnls.items()})
            # identifica mês corrente pelo último dia do dataset
            days_sorted = sorted(pnls.keys())
            if days_sorted:
                mk_cur = _month_key(days_sorted[-1])
                # semanas cujo week_start está no mesmo mês e não é a semana corrente
                ws_cur = _week_start_iso(days_sorted[-1])
                rows = [(ws, val) for ws, val in weeks.items() if _month_key(ws) == mk_cur and ws != ws_cur]
                if rows:
                    s1.append("**Semanas anteriores fechadas (mês corrente)**\n\n")
                    s1.append("| Semana (start) | P&L |\n|---|---:|\n")
                    for ws, val in rows[-6:]:
                        s1.append(f"| {ws} | {_fmt_num(val, 2)} |\n")
                    s1.append("\n")
        except Exception:
            pass
    else:
        s1.append("_Sem série de accounting disponível para métricas diárias/Sharpe/DD (ver 99.1)._ \n\n")

    # execução (contagens + stake médio) via aderência (7 dias)
    if isinstance(adh, dict) and isinstance(adh.get("per_day"), list) and adh.get("per_day"):
        s1.append("**Execução (últimos dias; executor_jsonl + placares quando disponíveis)**\n\n")
        s1.append("| Dia | Exec rows | LIVE_OK | DRY_OK | API_FAILED | P&L Back | ROI Back | P&L Lay | ROI Lay/liab |\n")
        s1.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for it in adh.get("per_day") or []:
            if not isinstance(it, dict):
                continue
            ex = it.get("execution") if isinstance(it.get("execution"), dict) else {}
            sc = ex.get("status_counts") if isinstance(ex.get("status_counts"), dict) else {}
            back = ex.get("back") if isinstance(ex.get("back"), dict) else {}
            lay = ex.get("lay") if isinstance(ex.get("lay"), dict) else {}
            s1.append(
                f"| {it.get('day')} | {int(ex.get('n_exec_rows') or 0)} | {int(sc.get('LIVE_OK') or 0)} | {int(sc.get('DRY_OK') or 0)} | "
                f"{int(sc.get('API_FAILED') or 0)} | {_fmt_num(back.get('pnl_sum'),2)} | {_fmt_pct(back.get('roi_pct'))} | "
                f"{_fmt_num(lay.get('pnl_sum'),2)} | {_fmt_pct(lay.get('roi_pct_per_liability'))} |\n"
            )
        s1.append("\n")

        # slippage x ROI (3 buckets raw com sinal) — último dia com dados
        last = None
        try:
            last = (adh.get("per_day") or [])[-1]
        except Exception:
            last = None
        if isinstance(last, dict):
            ex = last.get("execution") if isinstance(last.get("execution"), dict) else {}
            rawblk = ex.get("slippage_vs_roi_raw") if isinstance(ex.get("slippage_vs_roi_raw"), dict) else {}
            if rawblk:
                s1.append(f"**Slippage × ROI por bucket (raw, com sinal) — exemplo do dia `{last.get('day')}`**\n\n")
                for side_key, title in (("back", "Back (ROI por stake)"), ("lay", "Lay (ROI por liability)")):
                    b = rawblk.get(side_key) if isinstance(rawblk.get(side_key), dict) else {}
                    buckets = b.get("buckets") if isinstance(b.get("buckets"), list) else []
                    if not buckets:
                        continue
                    s1.append(f"- **{title}**\n\n")
                    s1.append("| Bucket slippage_raw_pct | n | ROI mean |\n|---|---:|---:|\n")
                    for row in buckets:
                        if not isinstance(row, dict):
                            continue
                        s1.append(f"| {row.get('bucket')} | {int(row.get('n') or 0)} | {_fmt_pct(row.get('roi_mean'))} |\n")
                    s1.append("\n")

    # Funil (24h) por auditoria: total → OK/valid → erros principais
    if isinstance(audit_rep, dict) and isinstance(audit_rep.get("by_version"), list) and audit_rep.get("by_version"):
        s1.append("**Funil de oportunidades (últimas 24h; auditoria DB)**\n\n")
        s1.append("| audit_version | total | OK | OK_valid | GATE_NOT_ELIGIBLE | API_FAILED | STALE_QUEUE_WAIT |\n")
        s1.append("|---|---:|---:|---:|---:|---:|---:|\n")
        for v in audit_rep.get("by_version") or []:
            if not isinstance(v, dict):
                continue
            sc = v.get("status_counts") if isinstance(v.get("status_counts"), dict) else {}
            s1.append(
                f"| {v.get('audit_version')} | {int(v.get('total') or 0)} | {int(sc.get('OK') or 0)} | {int(v.get('ok_valid') or 0)} | "
                f"{int(sc.get('GATE_NOT_ELIGIBLE') or 0)} | {int(sc.get('API_FAILED') or 0)} | {int(sc.get('STALE_QUEUE_WAIT') or 0)} |\n"
            )
        s1.append("\n")

        # motivos top (api_error) agregados
        errs = []
        try:
            for it in (audit_rep.get("error_rows") or []):
                if not isinstance(it, dict):
                    continue
                n = int(it.get("n") or 0)
                if n <= 0:
                    continue
                errs.append((n, str(it.get("audit_version") or ""), str(it.get("status") or ""), str(it.get("api_error") or "")))
            errs.sort(key=lambda x: x[0], reverse=True)
        except Exception:
            errs = []
        if errs:
            s1.append("**Motivos principais de não-execução / falha (top)**\n\n")
            for n, ver, st, err in errs[:8]:
                err2 = (err[:180] + "…") if len(err) > 180 else err
                s1.append(f"- `{ver}`: {st} ×{n} — `{err2}`\n")
            s1.append("\n")

        s1.append("**Oportunidades identificadas / melhorias propostas (curto prazo)**\n\n")
        s1.append(
            "- **PMM/timeout**: se `No PMMs received` dominar, aumentar timeout efetivo e reduzir bursts (workers/queue) tende a elevar conversão sem mexer na estratégia.\n"
            "- **Betslips abertos**: `too_many_open_betslips` é um gargalo de throughput; manter caps/janelas e garantir cleanup rápido evita bloqueio global.\n"
            "- **Fila**: `STALE_QUEUE_WAIT` indica atraso interno; atacar latência/concorrência antes de aumentar volume/seleção.\n\n"
        )

    # Sensibilidade de banca (reusa tabela OOS existente)
    if oos_txt:
        sens = _extract_md_block(
            oos_txt,
            start="### 12.2b Sensibilidade por banca",
            until_any=["### 12.2c Sensibilidade por banca", "### 12.3 "],
        )
        if sens.strip():
            s1.append("**Estudo de sensibilidade (banca × lucro)**\n\n")
            s1.append(
                "_A tabela abaixo é reaproveitada do bloco OOS (mesmo layout). Ela responde “até onde a operação escala” antes de bater em caps/limites._\n\n"
            )
            s1.append(sens + "\n")

    # Histórico recente de policy (parâmetros “passados” do portfólio ativo)
    try:
        hist_lines = _tail_lines(cfg.wf_policy_history_jsonl, 12)
        recs = []
        for ln in hist_lines:
            try:
                recs.append(json.loads(ln))
            except Exception:
                continue
        recs = [r for r in recs if isinstance(r, dict)]
        if recs:
            s1.append("**Portfólio OOS: vigente vs histórico recente**\n\n")
            s1.append("| ts | n_active_keys |\n|---|---:|\n")
            for r in recs[-8:]:
                nkeys = None
                try:
                    ak = r.get("active_keys")
                    nkeys = len(ak) if isinstance(ak, list) else None
                except Exception:
                    nkeys = None
                s1.append(f"| {r.get('ts')} | {nkeys if nkeys is not None else '—'} |\n")
            s1.append("\n")
    except Exception:
        pass

    # parâmetros de negócio e técnicos: manter 99.6 como fonte, mas resumir aqui
    s1.append("**Parâmetros vigentes (visão executiva)**\n\n")
    # decomposição de active_keys (negócio)
    try:
        if isinstance(active_keys, list) and active_keys:
            def _cnt(prefix: str) -> int:
                return sum(1 for k in active_keys if str(k).startswith(prefix))
            by_league = sum(1 for k in active_keys if "__" in str(k))
            s1.append("| Dimensão | Valor |\n|---|---:|\n")
            s1.append(f"| active_keys (total) | {len(active_keys)} |\n")
            s1.append(f"| chaves por liga (suFIXO `__<League>`) | {by_league} |\n")
            s1.append(f"| Back_Pre | {_cnt('Back_Pre_')} |\n")
            s1.append(f"| Back_In | {_cnt('Back_In_')} |\n")
            s1.append(f"| Lay_Pre_Yes | {_cnt('Lay_Pre_Yes')} |\n")
            s1.append(f"| Lay_Pre_No | {_cnt('Lay_Pre_No')} |\n")
            s1.append(f"| Lay_In_Yes | {_cnt('Lay_In_Yes')} |\n")
            s1.append(f"| Lay_In_No | {_cnt('Lay_In_No')} |\n")
            s1.append("\n")
    except Exception:
        pass
    s1.append(
        "- **Combinações ativas (OOS)**: ver `99.3` (active_keys) e o bloco `2) OOS`.\n"
        "- **Stake sizing operacional (real)**: hoje é **FLAT** via `BRIDGE_STAKE` (ver `99.3` e `99.6`).\n"
        "- **Parâmetros técnicos efetivos** (executor/audit/bridge): ver `99.6 Filtros ativos`.\n\n"
    )

    # 6) Combinar markdown (base reordenado + blocos operacionais 99.x)
    extra = []
    extra.append("\n\n## 99) Operacional — saldo, P&L e execução\n\n")
    extra.append("### 99.1 Accounting (saldo + P&L)\n\n")
    extra.append(f"- Arquivo: `{acct_out}`\n")
    if acct.get("error"):
        extra.append(f"- **Erro**: **{acct.get('error')}**\n")
    extra.append(f"- Saldo atual: **{acct.get('balance_current')}**\n")
    extra.append(f"- P&L hoje/semana/mês: **{acct.get('pnl_today')} / {acct.get('pnl_week')} / {acct.get('pnl_month')}**\n")
    extra.append("\nMeses fechados:\n\n")
    extra.append("| Mês | P&L |\n|---|---:|\n")
    for k, v in (acct.get("closed_months") or {}).items():
        extra.append(f"| {k} | {v} |\n")

    extra.append("\n### 99.2 Execução (KPIs)\n\n")
    extra.append(f"- Fonte: `{cfg.executor_jsonl}`\n\n")

    # Status table
    extra.append("**Status (all)**\n\n")
    extra.append("| Status | N |\n|---|---:|\n")
    for k, v in (kpi_all.get("status_counts") or {}).items():
        extra.append(f"| {k} | {int(v)} |\n")
    extra.append("\n")

    def _timing_table(title: str, obj: Dict[str, Any]) -> str:
        def _row(name: str, a: dict) -> str:
            return (
                f"| {name} | {a.get('n')} | {a.get('p50')} | {a.get('p90')} | {a.get('p99')} | {a.get('mean')} |\n"
            )

        s = []
        s.append(f"**{title}**\n\n")
        s.append("| Métrica | n | p50 | p90 | p99 | mean |\n|---|---:|---:|---:|---:|---:|\n")
        for nm in ("queue_delay", "call_to_done", "post"):
            a = ((obj.get(nm) or {}) if isinstance(obj, dict) else {})
            s.append(_row(nm, a if isinstance(a, dict) else {}))
        s.append("\n")
        return "".join(s)

    timing_ok = (kpi_ok.get("timing_ms") or {}) if isinstance(kpi_ok, dict) else {}
    extra.append(_timing_table("Latência (somente LIVE_OK/DRY_OK) — ms", timing_ok))

    slip_ok = (kpi_ok.get("slippage") or {}) if isinstance(kpi_ok, dict) else {}
    extra.append("**Slippage (somente LIVE_OK/DRY_OK, quando houver odd_at_decision)**\n\n")
    extra.append(
        "- Definição: `slippage = odd_final - odd_at_decision` (em odds decimais) e `slippage_pct = slippage/odd_at_decision`.\n"
        "- Interpretação depende do lado:\n"
        "  - **Back**: slippage_pct **negativo** = piorou (odd caiu); **positivo** = melhorou.\n"
        "  - **Lay**: slippage_pct **positivo** = piorou (odd subiu); **negativo** = melhorou.\n\n"
    )
    extra.append("| Tipo | n | p50 | p90 | p99 | mean |\n|---|---:|---:|---:|---:|---:|\n")
    for nm in ("abs", "pct"):
        a = (slip_ok.get(nm) or {}) if isinstance(slip_ok, dict) else {}
        extra.append(f"| {nm} | {a.get('n')} | {a.get('p50')} | {a.get('p90')} | {a.get('p99')} | {a.get('mean')} |\n")
    extra.append("\n")

    # Slippage por lado (Back vs Lay)
    slip_by_side = (kpi_ok.get("slippage_by_side") or {}) if isinstance(kpi_ok, dict) else {}
    if isinstance(slip_by_side, dict) and slip_by_side:
        extra.append("**Slippage por lado (Back vs Lay)**\n\n")
        extra.append("| Lado | Métrica | n | p50 | p90 | p99 | mean |\n|---|---|---:|---:|---:|---:|---:|\n")
        for side, obj in slip_by_side.items():
            if not isinstance(obj, dict):
                continue
            for nm, label in (
                ("raw_pct", "slippage_pct (raw)"),
                ("cost_pct", "slippage_pct (custo, >=0)"),
            ):
                a = obj.get(nm) if isinstance(obj.get(nm), dict) else {}
                extra.append(
                    f"| {side} | {label} | {a.get('n')} | {a.get('p50')} | {a.get('p90')} | {a.get('p99')} | {a.get('mean')} |\n"
                )
        extra.append("\n")
    extra.append(
        "_Nota: o p90/p99 de `call_to_done_ms` explode quando inclui `NO_SESSION/API_FAILED` (timeouts/relogin). "
        "Por isso reportamos também o recorte apenas de sucessos._\n\n"
    )

    if active_keys:
        extra.append("\n### 99.3 Regras OOS ativas (último step)\n\n")
        extra.append(f"- active_keys: {len(active_keys) if isinstance(active_keys, list) else '—'}\n")
        extra.append("```json\n" + json.dumps(active_keys, ensure_ascii=False, indent=2) + "\n```\n\n")
        extra.append("**Como ler `active_keys` (regra de aprovação)**\n\n")
        extra.append(
            "- `active_keys` é o **portfólio aprovado** pelo walk-forward (OOS) no **último step** exportado.\n"
            "- O bridge (`ops.executor_bridge_audit`) só envia para o executor oportunidades cuja **chave operacional** (combinação) esteja ativa.\n"
            "- Mapeamento de chaves (simplificado):\n"
            "  - **Back**: `Back_Pre_Any` (pre) ou `Back_In_Any` (in). Se o walk-forward estiver com `key_by_league`, a chave pode ter sufixo `__<League>`.\n"
            "  - **Lay**: `Lay_Pre_Yes/No` (pre) ou `Lay_In_Yes/No` (in). Para **H3B**, `Yes` indica que o sinal envolve reversão (por definição da hipótese).\n\n"
        )
        extra.append("**Regras de execução atuais (stake sizing)**\n\n")
        extra.append(
            "- No operacional (shadow/live), o tamanho enviado pelo bridge é **FLAT** via `BRIDGE_STAKE`.\n"
            "- Em **Back**: stake = `BRIDGE_STAKE`.\n"
            "- Em **Lay**: o executor recebe stake, mas o risco relevante é a **liability**, aproximadamente `liability ≈ stake × (odd - 1)`.\n"
            "- Importante: o Kelly/caps que aparece no relatório OOS é **simulação/diagnóstico** do walk-forward; ele não está sendo aplicado no executor/bridge neste momento.\n\n"
        )
        extra.append("**Lay em `ws_gate_lay`: é só pós-reversão?**\n\n")
        extra.append(
            "- Sim: o audit `v5.1-ws-gate-lay` só abre ticket Lay quando passa pelo gate de **queda** (ex.: >2% em 5s): `WS(t+offset) < ratio × WS(t0)`.\n"
            "- Isso significa que, mesmo em shadow, a amostra Lay desse audit representa apenas casos em que houve a movimentação (reversão/queda) definida pela estratégia.\n\n"
        )

    # 99.6 Config efetiva (filtros ativos) — executor/audit/bridge/OOS
    extra.append("\n### 99.6 Filtros ativos (config efetiva)\n\n")
    extra.append(
        "_Nota: esta seção reflete as variáveis carregadas pelo `daily_full_report` (via `.env`). "
        "Services do systemd podem ter overrides (`Environment=`) que não aparecem aqui; use `systemctl show` para confirmar no VPS._\n\n"
    )
    def _env(k: str, default: str = "") -> str:
        v = os.getenv(k, default)
        return str(v) if v is not None else ""

    extra.append("**Executor**\n\n")
    extra.append("| chave | valor |\n|---|---|\n")
    for k in [
        "EXECUTOR_ALLOW_LIVE",
        "EXECUTOR_WORKERS",
        "EXECUTOR_QUEUE_MAX",
        "EXECUTOR_CAP_WINDOW_SEC",
        "EXECUTOR_CAP_MAX",
        "EXECUTOR_FAST_PMM",
        "EXECUTOR_PMM_TIMEOUT_SEC",
        "EXECUTOR_PMM_MIN_WAIT_SEC",
        "EXECUTOR_PMM_IDLE_TIMEOUT_SEC",
        "EXECUTOR_BETSLIP_CACHE_MAX_KEYS",
    ]:
        extra.append(f"| {k} | `{_env(k)}` |\n")
    extra.append("\n")

    extra.append("**Audit H3B**\n\n")
    extra.append("| chave | valor |\n|---|---|\n")
    for k in [
        "AUDIT_MODE",
        "AUDIT_API_SIDES",
        "AUDIT_EXECUTOR_WORKERS",
        "AUDIT_TEMPORAL_WORKERS",
        "AUDIT_MAX_QUEUE_DEPTH",
        "AUDIT_MAX_QUEUE_WAIT_MS",
        "GATE_DROP_OFFSET_SEC",
        "GATE_DROP_RATIO",
        "GATE_RISE_OFFSET_SEC",
        "GATE_RISE_RATIO",
        "GATE_OPEN_WINDOW_SEC",
        "GATE_OPEN_MAX",
        "GATE_MAX_LATE_SEC",
        "GATE_LAY_REFRESH_TIMES_SEC",
    ]:
        extra.append(f"| {k} | `{_env(k)}` |\n")
    extra.append("\n")

    extra.append("**Bridge**\n\n")
    extra.append("| chave | valor |\n|---|---|\n")
    for k in [
        "BRIDGE_MODE",
        "BRIDGE_EXEC_SIDE",
        "BRIDGE_STAKE",
        "BRIDGE_POLL_SEC",
        "BRIDGE_LOOKBACK_SEC",
        "BRIDGE_MAX_PER_CYCLE",
        "BRIDGE_PREMATCH_ONLY",
        "BRIDGE_POLICY_JSON",
        "BRIDGE_POLICY_RELOAD_SEC",
        "BRIDGE_POLICY_USE_BASE",
        "BRIDGE_MIN_LIMIT",
    ]:
        extra.append(f"| {k} | `{_env(k)}` |\n")
    extra.append("\n")

    extra.append("**OOS / Walk-forward (daily)**\n\n")
    extra.append("| chave | valor |\n|---|---|\n")
    for k in [
        "DAILY_OOS_DIRECTION",
        "DAILY_OOS_VERSIONS",
        "DAILY_OOS_LOOKBACK_DAYS",
        "DAILY_WF_TRAIN_MODE",
        "DAILY_WF_TRAIN_DAYS",
        "DAILY_WF_TEST_DAYS",
        "DAILY_WF_STEP_DAYS",
        "DAILY_WF_KEY_BY_LEAGUE",
        "DAILY_WF_KEY_BY_LEAGUE_SCOPE",
        "DAILY_WF_AH_MAX_ABS_LINE",
        "DAILY_WF_AH_SCOPE",
        "DAILY_WF_EXCLUDE_EXEC_BUCKETS_BACK",
    ]:
        extra.append(f"| {k} | `{_env(k)}` |\n")
    extra.append("\n")

    # 99.4 Aderência OOS (policy por dia × execução)
    try:
        if isinstance(adh, dict) and isinstance(adh.get("per_day"), list) and adh.get("per_day"):
            extra.append("\n### 99.4 Aderência OOS (portfolio por dia × execução)\n\n")
            extra.append(f"- Arquivo: `{adh_json}`\n")
            extra.append(f"- Policy current: `{cfg.wf_policy_current}`\n\n")

            extra.append("**Resumo (últimos dias)**\n\n")
            extra.append("| Dia | Ativas (keys) | Bridge rows | Skipped(not_active) | Exec rows | LIVE_OK | DRY_OK | P&L Back | ROI Back | P&L Lay | ROI Lay/liab | P&L total |\n")
            extra.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
            for it in adh.get("per_day") or []:
                if not isinstance(it, dict):
                    continue
                pol = it.get("policy") if isinstance(it.get("policy"), dict) else {}
                nkeys = pol.get("n_active_keys")
                bridge_rows = 0
                skipped_na = 0
                for b in (it.get("bridge") or []):
                    if isinstance(b, dict):
                        bridge_rows += int(b.get("n_rows") or 0)
                        skipped_na += int(b.get("n_not_active") or 0)
                ex = it.get("execution") if isinstance(it.get("execution"), dict) else {}
                sc = ex.get("status_counts") if isinstance(ex.get("status_counts"), dict) else {}
                live_ok = int(sc.get("LIVE_OK") or 0)
                dry_ok = int(sc.get("DRY_OK") or 0)
                back = ex.get("back") if isinstance(ex.get("back"), dict) else {}
                lay = ex.get("lay") if isinstance(ex.get("lay"), dict) else {}
                pnl_b = float(back.get("pnl_sum") or 0.0)
                pnl_l = float(lay.get("pnl_sum") or 0.0)
                extra.append(
                    f"| {it.get('day')} | {nkeys if nkeys is not None else '—'} | {bridge_rows} | {skipped_na} | "
                    f"{int(ex.get('n_exec_rows') or 0)} | {live_ok} | {dry_ok} | {_fmt_num(pnl_b,2)} | {_fmt_pct(back.get('roi_pct'))} | "
                    f"{_fmt_num(pnl_l,2)} | {_fmt_pct(lay.get('roi_pct_per_liability'))} | {_fmt_num(pnl_b + pnl_l,2)} |\n"
                )
            extra.append("\n")

            # Potencial (30d) pela banca que maximiza lucro na sensibilidade OOS (se disponível)
            try:
                sens = _extract_md_block(
                    oos_txt,
                    start="### 12.2b Sensibilidade por banca",
                    until_any=["### 12.2c Sensibilidade por banca", "### 12.3 "],
                )
                best = None
                if sens:
                    for ln in sens.splitlines():
                        if not ln.startswith("|") or ln.strip().startswith("|---"):
                            continue
                        cols = [c.strip() for c in ln.strip().strip("|").split("|")]
                        if len(cols) < 6 or cols[0].lower().startswith("banca"):
                            continue
                        def _f(s: str) -> Optional[float]:
                            try:
                                t = str(s).strip().replace(".", "").replace(",", ".")
                                t = t.replace("%", "")
                                return float(t)
                            except Exception:
                                return None
                        bank_ref = _f(cols[0])
                        turn_30 = _f(cols[1])
                        prof_30 = _f(cols[2])
                        bank_eff = _f(cols[3])
                        roi_bank = _f(cols[4])
                        dd_p95 = _f(cols[5])
                        if prof_30 is None:
                            continue
                        if best is None or float(prof_30) > float(best["profit_30d_exp"]):
                            best = {
                                "bank_ref": bank_ref,
                                "turn_30d": turn_30,
                                "profit_30d_exp": prof_30,
                                "bank_eff": bank_eff,
                                "roi_bank_30d": roi_bank,
                                "dd_p95": dd_p95,
                            }
                if best:
                    # share observado (últimos dias) para decompor back/lay como estimativa
                    pnl_b = pnl_l = 0.0
                    for it in adh.get("per_day") or []:
                        ex = it.get("execution") if isinstance(it.get("execution"), dict) else {}
                        back = ex.get("back") if isinstance(ex.get("back"), dict) else {}
                        lay = ex.get("lay") if isinstance(ex.get("lay"), dict) else {}
                        pnl_b += float(back.get("pnl_sum") or 0.0)
                        pnl_l += float(lay.get("pnl_sum") or 0.0)
                    tot = pnl_b + pnl_l
                    w_b = (pnl_b / tot) if tot else 0.5
                    w_l = 1.0 - w_b
                    extra.append("**Potencial de lucro (30d) pela banca ótima (sensibilidade OOS)**\n\n")
                    extra.append(f"- Banca ref (grid): `{_fmt_num(best.get('bank_ref'),2)}` | banca rec. (max): `{_fmt_num(best.get('bank_eff'),2)}`\n")
                    extra.append(f"- Lucro 30d (exp.): `{_fmt_num(best.get('profit_30d_exp'),2)}` | ROI/banca 30d (exp.): `{_fmt_num(best.get('roi_bank_30d'),2)}%` | DD p95 (30d): `{_fmt_num(best.get('dd_p95'),2)}`\n")
                    extra.append(
                        f"- Decomposição *estimada* por lado (proporcional ao P&L observado na janela): total `{_fmt_num(best.get('profit_30d_exp'),2)}` → "
                        f"Back `{_fmt_num(float(best.get('profit_30d_exp'))*w_b,2)}` | Lay `{_fmt_num(float(best.get('profit_30d_exp'))*w_l,2)}`\n\n"
                    )
            except Exception:
                pass

            # Slippage × ROI (com sinal): 3 buckets por lado (<=-2%, -2..2, >2)
            last = None
            try:
                last = (adh.get("per_day") or [])[-1]
            except Exception:
                last = None
            if isinstance(last, dict):
                ex = last.get("execution") if isinstance(last.get("execution"), dict) else {}
                rawblk = ex.get("slippage_vs_roi_raw") if isinstance(ex.get("slippage_vs_roi_raw"), dict) else {}
                if rawblk:
                    extra.append(f"**Slippage × ROI (raw, com sinal; 3 buckets) — exemplo do dia `{last.get('day')}`**\n\n")
                    for side_key, title in (("back", "Back (ROI por stake)"), ("lay", "Lay (ROI por liability)")):
                        blk = rawblk.get(side_key) if isinstance(rawblk.get(side_key), dict) else {}
                        buckets = blk.get("buckets") if isinstance(blk.get("buckets"), list) else []
                        if not buckets:
                            continue
                        extra.append(f"- **{title}**\n\n")
                        extra.append("| Bucket slippage_raw_pct | n | ROI mean |\n|---|---:|---:|\n")
                        for b in buckets:
                            if not isinstance(b, dict):
                                continue
                            extra.append(f"| {b.get('bucket')} | {int(b.get('n') or 0)} | {_fmt_pct(b.get('roi_mean'))} |\n")
                        extra.append("\n")
    except Exception:
        pass

    # 99.5 Auditoria (DB): motivos de no-OK por versão + qualidade dos OK
    try:
        rep = audit_rep
        if isinstance(rep, dict) and isinstance(rep.get("by_version"), list) and rep.get("by_version"):
            extra.append("\n### 99.5 Auditoria (DB) — motivos de no-OK (por versão)\n\n")
            extra.append(f"- Arquivo: `{audit_json}`\n")
            extra.append(f"- Janela: últimas **{rep.get('hours')}h** (desde `{rep.get('since_utc')}`)\n\n")
            extra.append("**Definições (colunas)**\n\n")
            extra.append(
                "- **OK**: `status='OK'` no `betslip_audit_results` (a auditoria concluiu com sucesso).\n"
                "- **OK com betslip_odd**: subset de OK em que `betslip_odd` está preenchido (houve snapshot do ticket/odds).\n"
                "- **OK valid**: subset de OK em que `is_valid_opportunity=true` (passou o critério operacional de “oportunidade executável”).\n"
                "  - Na prática, o `is_valid_opportunity` tende a cair quando `difference_pct` está fora do range aceito (edge muito pequeno <2% ou mismatch >10%) ou quando campos essenciais do ticket estão ausentes.\n\n"
            )
            extra.append("| audit_version | total | OK | OK com betslip_odd | OK valid | top no-OK |\n")
            extra.append("|---|---:|---:|---:|---:|---|\n")
            for v in rep.get("by_version") or []:
                if not isinstance(v, dict):
                    continue
                sc = v.get("status_counts") if isinstance(v.get("status_counts"), dict) else {}
                total = int(v.get("total") or 0)
                nok = 0
                try:
                    nok = int(sc.get("OK") or 0)
                except Exception:
                    nok = 0
                # top no-OK
                pairs = []
                for k, cnt in sc.items():
                    if str(k) == "OK":
                        continue
                    try:
                        pairs.append((str(k), int(cnt)))
                    except Exception:
                        continue
                pairs.sort(key=lambda x: x[1], reverse=True)
                top = ", ".join([f"{k}={c}" for k, c in pairs[:4]]) if pairs else "—"
                extra.append(
                    f"| {v.get('audit_version')} | {total} | {nok} | {int(v.get('ok_with_bs') or 0)} | {int(v.get('ok_valid') or 0)} | {top} |\n"
                )
            extra.append("\n")

            # Diagnóstico (OK): por que OK_with_bs >> OK_valid?
            extra.append("**Diagnóstico dos OK (por versão): buckets de |difference_pct|**\n\n")
            extra.append(
                "_Leitura: `OK valid` tende a ser aproximadamente o bucket `2% ≤ |difference_pct| ≤ 10%` (dependendo da regra vigente)._\n\n"
            )
            extra.append("| audit_version | OK diff nulo | OK |diff|<2% | OK 2–10% | OK |diff|>10% |\n")
            extra.append("|---|---:|---:|---:|---:|\n")
            for v in rep.get("by_version") or []:
                if not isinstance(v, dict):
                    continue
                extra.append(
                    f"| {v.get('audit_version')} | {int(v.get('ok_diff_null') or 0)} | {int(v.get('ok_absdiff_lt2') or 0)} | {int(v.get('ok_absdiff_2_10') or 0)} | {int(v.get('ok_absdiff_gt10') or 0)} |\n"
                )
            extra.append("\n")

            # Top api_error (quando houver) para explicar API_FAILED/NO_SESSION/etc.
            tev = rep.get("top_errors_by_version") if isinstance(rep.get("top_errors_by_version"), dict) else {}
            if tev:
                extra.append("**Top erros (api_error) por versão**\n\n")
                for ver, xs in tev.items():
                    if not isinstance(xs, list) or not xs:
                        continue
                    extra.append(f"- `{ver}`:\n")
                    for it in xs[:5]:
                        if not isinstance(it, dict):
                            continue
                        st = str(it.get("status") or "NA")
                        err = str(it.get("api_error") or "").strip()
                        n = int(it.get("n") or 0)
                        if err:
                            err = (err[:180] + "…") if len(err) > 180 else err
                            extra.append(f"  - {st} ×{n}: `{err}`\n")
                    extra.append("\n")
    except Exception:
        pass

    combined_md = day_dir / "report_daily.md"
    insample_wrapped = ""
    if insample_txt.strip():
        insample_wrapped = "## 3) In-sample (detalhe)\n\n" + _demote_h2_to_h3(insample_txt.strip() + "\n")

    combined_core = "".join(s0) + "".join(s1) + (oos_txt.strip() + "\n\n" if oos_txt.strip() else "") + insample_wrapped
    combined_md.write_text(combined_core + "".join(extra), encoding="utf-8")

    # 5) PDF
    pdf = day_dir / "report_daily.pdf"
    renderer = Path(__file__).resolve().parent.parent / "docs" / "render_markdown_to_pdf.py"
    subprocess.run([sys.executable, str(renderer), str(combined_md), str(pdf)], check=True)

    out = {
        "ts": ts.isoformat(),
        "day_dir": str(day_dir),
        "pdf": str(pdf),
        "policy_current": str(cfg.wf_policy_current),
    }

    # 6) Telegram
    if cfg.send_telegram:
        token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
        if token and chat_id and pdf.exists():
            ok = _telegram_send_document(token, chat_id, file_path=pdf, caption=f"Relatório diário BetinAsia ({day})")
            out["telegram_sent"] = bool(ok)
        else:
            out["telegram_sent"] = False

    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Relatório diário completo: OOS + execution KPIs + accounting + PDF + Telegram.")
    ap.add_argument("--out-dir", default=os.getenv("DAILY_REPORT_OUT_DIR", "logs/daily_reports"))
    args = ap.parse_args()

    logger.remove()
    logger.add(sys.stderr, level=os.getenv("LOG_LEVEL", "INFO"))

    # Se rodando manualmente, garante que .env seja carregado antes do cfg.
    _load_env_file(Path(os.getenv("ENV_FILE", ".env")))

    cfg = DailyReportCfg(out_dir=Path(str(args.out_dir)))
    import asyncio

    out = asyncio.run(run_daily_full(cfg))
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

