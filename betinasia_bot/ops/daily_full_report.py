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

    # v38: renomeia apenas o header do bloco OOS para "## 1) OOS ..." (o resto pode manter numbering interno)
    try:
        txt = base_md.read_text(encoding="utf-8")
        txt2 = txt.replace("## 12) OOS walk-forward", "## 1) OOS walk-forward")
        if txt2 != txt:
            base_md.write_text(txt2, encoding="utf-8")
    except Exception:
        pass

    # 4) Combinar markdown (base + blocos operacionais)
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
    extra.append("| Tipo | n | p50 | p90 | p99 | mean |\n|---|---:|---:|---:|---:|---:|\n")
    for nm in ("abs", "pct"):
        a = (slip_ok.get(nm) or {}) if isinstance(slip_ok, dict) else {}
        extra.append(f"| {nm} | {a.get('n')} | {a.get('p50')} | {a.get('p90')} | {a.get('p99')} | {a.get('mean')} |\n")
    extra.append("\n")
    extra.append(
        "_Nota: o p90/p99 de `call_to_done_ms` explode quando inclui `NO_SESSION/API_FAILED` (timeouts/relogin). "
        "Por isso reportamos também o recorte apenas de sucessos._\n\n"
    )

    if active_keys:
        extra.append("\n### 99.3 Regras OOS ativas (último step)\n\n")
        extra.append(f"- active_keys: {len(active_keys) if isinstance(active_keys, list) else '—'}\n")
        extra.append("```json\n" + json.dumps(active_keys, ensure_ascii=False, indent=2) + "\n```\n\n")

    # 99.4 Aderência OOS (policy por dia × execução)
    try:
        adh_json = day_dir / "oos_adherence.json"
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
        if isinstance(adh, dict) and isinstance(adh.get("per_day"), list) and adh.get("per_day"):
            extra.append("\n### 99.4 Aderência OOS (portfolio por dia × execução)\n\n")
            extra.append(f"- Arquivo: `{adh_json}`\n")
            extra.append(f"- Policy current: `{cfg.wf_policy_current}`\n\n")

            extra.append("**Resumo (últimos dias)**\n\n")
            extra.append("| Dia | Ativas (keys) | Bridge rows | Skipped(not_active) | Exec rows | LIVE_OK | DRY_OK | ROI Back (stake) | ROI Lay (liability) |\n")
            extra.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
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
                extra.append(
                    f"| {it.get('day')} | {nkeys if nkeys is not None else '—'} | {bridge_rows} | {skipped_na} | "
                    f"{int(ex.get('n_exec_rows') or 0)} | {live_ok} | {dry_ok} | {_fmt_pct(back.get('roi_pct'))} | {_fmt_pct(lay.get('roi_pct_per_liability'))} |\n"
                )
            extra.append("\n")
    except Exception:
        pass

    # 99.5 Auditoria (DB): motivos de no-OK por versão + qualidade dos OK
    try:
        audit_json = day_dir / "audit_status_kpis.json"
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
        rep = _read_json(audit_json)
        if isinstance(rep, dict) and isinstance(rep.get("by_version"), list) and rep.get("by_version"):
            extra.append("\n### 99.5 Auditoria (DB) — motivos de no-OK (por versão)\n\n")
            extra.append(f"- Arquivo: `{audit_json}`\n")
            extra.append(f"- Janela: últimas **{rep.get('hours')}h** (desde `{rep.get('since_utc')}`)\n\n")
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
    combined_md.write_text(base_md.read_text(encoding="utf-8") + "".join(extra), encoding="utf-8")

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

