#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnóstico do pipeline de resultados (ROI).

Objetivo:
- Verificar se a tabela `matches` está recebendo placares (home_score/away_score).
- Medir taxa de "match" entre jogos do BetinAsia e fixtures do API-Football.
- Apontar motivos típicos de falha: API sem retorno, nomes de time não batem, data errada (timezone), etc.

Uso:
  cd betinasia_bot
  python3 -m results.diagnose_results_pipeline --lookback-days 3 --sample 25

Requisitos:
- `DATABASE_URL` no .env
- `API_FOOTBALL_KEY` no ambiente (recomendado). Se ausente, usa o fallback em `results.update_results.API_KEY`.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple

from sqlalchemy import and_, select, text

from storage.database import Database
from storage.models import Match
from .api_football import APIFootballClient, MatchResult
from .update_results import match_teams, normalize_team_name, similarity_ratio, get_mapped_name


def _fmt_dt(dt: datetime) -> str:
    try:
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return str(dt)


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lookback-days", type=int, default=3, help="Janela para inspecionar matches recentes (default 3)")
    p.add_argument("--cutoff-hours", type=int, default=2, help="Considera jogo 'deveria ter resultado' se kickoff < now - cutoff (default 2h)")
    p.add_argument("--sample", type=int, default=30, help="N de jogos para testar matching com API (default 30)")
    p.add_argument("--date", default=None, help="Força uma data específica YYYY-MM-DD (opcional)")
    args = p.parse_args()

    db = Database()
    await db.connect()

    from .update_results import API_KEY
    api = APIFootballClient(api_key=API_KEY)
    await api.start()

    try:
        now = datetime.now(timezone.utc)
        lookback = now - timedelta(days=int(args.lookback_days))
        cutoff = now - timedelta(hours=int(args.cutoff_hours))

        print("=" * 80)
        print("DIAGNÓSTICO PIPELINE DE RESULTADOS / ROI")
        print("=" * 80)
        print(f"Agora: {now.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"Lookback: {args.lookback_days}d | Cutoff (kickoff < now - {args.cutoff_hours}h)")
        print()

        # status da API (se falhar aqui, já é um forte indício)
        try:
            status = await api.get_status()
        except Exception:
            status = None
        print(f"API-Football status: {status}" if status else "API-Football status: (falhou ao consultar /status)")
        print()

        async with db.async_session() as session:
            # 1) Cobertura de placares
            q_cov = text(
                """
                SELECT
                  COUNT(*) AS n_total,
                  COUNT(*) FILTER (WHERE kickoff_time >= :lookback) AS n_lookback,
                  COUNT(*) FILTER (WHERE kickoff_time >= :lookback AND kickoff_time < :cutoff) AS n_should_have_result,
                  COUNT(*) FILTER (WHERE kickoff_time >= :lookback AND kickoff_time < :cutoff AND home_score IS NOT NULL AND away_score IS NOT NULL) AS n_with_score,
                  COUNT(*) FILTER (WHERE kickoff_time >= :lookback AND kickoff_time < :cutoff AND status='finished') AS n_finished_flag
                FROM matches;
                """
            )
            row = (await session.execute(q_cov, {"lookback": lookback, "cutoff": cutoff})).fetchone()
            n_total, n_lookback, n_should, n_score, n_finished = row

            print("1) Cobertura de placares na tabela `matches`")
            print(f"- n_total matches: {n_total}")
            print(f"- n_lookback (kickoff >= lookback): {n_lookback}")
            print(f"- n_should_have_result (kickoff < cutoff): {n_should}")
            print(f"- n_with_score: {n_score}")
            print(f"- n_finished_flag: {n_finished}")
            if n_should and n_score == 0:
                print("  [ALERTA] Existem jogos que já deveriam ter resultado, mas nenhum tem placar preenchido.")
            print()

            # 2) Lista de jogos pendentes
            if args.date:
                dt = datetime.strptime(args.date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                start = dt
                end = dt + timedelta(days=1)
                pending_q = select(Match).where(
                    and_(
                        Match.kickoff_time >= start,
                        Match.kickoff_time < end,
                        Match.status != "finished",
                    )
                )
                print(f"2) Jogos pendentes para a data {args.date}")
            else:
                pending_q = select(Match).where(
                    and_(
                        Match.kickoff_time >= lookback,
                        Match.kickoff_time < cutoff,
                        Match.status != "finished",
                    )
                )
                print("2) Jogos pendentes (kickoff < cutoff, status != finished)")

            pending = (await session.execute(pending_q)).scalars().all()
            print(f"- pendentes: {len(pending)}")
            for m in pending[: min(10, len(pending))]:
                print(f"  - {m.league} | {_fmt_dt(m.kickoff_time)} | {m.home_team} vs {m.away_team} | status={m.status}")
            print()

            # 3) Teste de matching com API
            if pending:
                sample = pending[: min(int(args.sample), len(pending))]
            else:
                # Se não há pendentes, teste com jogos do lookback (mesmo que já finished)
                any_q = select(Match).where(Match.kickoff_time >= lookback).order_by(Match.kickoff_time.desc()).limit(int(args.sample))
                sample = (await session.execute(any_q)).scalars().all()

        print("3) Teste de matching BetinAsia -> API-Football (por data)")
        print(f"- sample size: {len(sample)}")
        if not sample:
            print("Sem jogos para testar.")
            return

        # cache por data (UTC)
        api_cache: Dict[str, list] = {}
        api_meta: Dict[str, dict] = {}
        reasons = Counter()

        def get_api(date_str: str):
            if date_str not in api_cache:
                api_cache[date_str] = []
            return api_cache[date_str]

        def parse_results(payload: dict) -> List[MatchResult]:
            if not payload:
                return []
            results: List[MatchResult] = []
            for fixture in payload.get("response", []) or []:
                try:
                    fixture_data = fixture.get("fixture", {}) or {}
                    teams = fixture.get("teams", {}) or {}
                    goals = fixture.get("goals", {}) or {}
                    score = fixture.get("score", {}) or {}
                    league = fixture.get("league", {}) or {}

                    status = (fixture_data.get("status", {}) or {}).get("short", "") or ""
                    if status not in ["FT", "AET", "PEN"]:
                        continue

                    results.append(
                        MatchResult(
                            fixture_id=int(fixture_data.get("id")),
                            home_team=str((teams.get("home", {}) or {}).get("name", "")),
                            away_team=str((teams.get("away", {}) or {}).get("name", "")),
                            home_score=int(goals.get("home", 0) or 0),
                            away_score=int(goals.get("away", 0) or 0),
                            status=str(status),
                            kickoff_time=datetime.fromisoformat(str(fixture_data.get("date", "")).replace("Z", "+00:00")),
                            league_name=str(league.get("name", "")),
                            league_country=str(league.get("country", "")),
                            home_score_ht=(score.get("halftime", {}) or {}).get("home"),
                            away_score_ht=(score.get("halftime", {}) or {}).get("away"),
                        )
                    )
                except Exception:
                    continue
            return results

        def _classify_api_meta(payload: dict) -> str:
            if not payload:
                return "api_falhou_sem_payload"
            errs = payload.get("errors") or {}
            if not errs:
                return ""
            txt = str(errs)
            if "Free plans do not have access to this date" in txt or "access to this date" in txt:
                return "api_sem_acesso_data_plano_free"
            return "api_com_erro"

        # busca fixtures (1 request por data) e parseia FT
        dates = sorted({m.kickoff_time.strftime("%Y-%m-%d") for m in sample})
        for ds in dates:
            payload = await api._request("/fixtures", {"date": ds})
            api_meta[ds] = payload
            api_cache[ds] = parse_results(payload or {})
        # também busca datas adjacentes para detectar mismatch de timezone/data
        for ds in dates:
            dt = datetime.strptime(ds, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            for adj in [(dt - timedelta(days=1)), (dt + timedelta(days=1))]:
                s = adj.strftime("%Y-%m-%d")
                if s not in api_cache:
                    payload = await api._request("/fixtures", {"date": s})
                    api_meta[s] = payload
                    api_cache[s] = parse_results(payload or {})

        matched = 0
        examples_not_found: List[Tuple[Match, str]] = []
        for m in sample:
            ds = m.kickoff_time.strftime("%Y-%m-%d")
            api_results = api_cache.get(ds) or []
            if not api_results:
                tag = _classify_api_meta(api_meta.get(ds) or {})
                reasons[tag or "api_sem_resultado_na_data"] += 1
                continue

            found = None
            for r in api_results:
                if match_teams(m.home_team, m.away_team, r.home_team, r.away_team):
                    found = r
                    break

            if found:
                matched += 1
            else:
                # tenta achar em dias adjacentes
                dt = datetime.strptime(ds, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                prev_ds = (dt - timedelta(days=1)).strftime("%Y-%m-%d")
                next_ds = (dt + timedelta(days=1)).strftime("%Y-%m-%d")
                prev = api_cache.get(prev_ds) or []
                nxt = api_cache.get(next_ds) or []
                found_adj = any(match_teams(m.home_team, m.away_team, r.home_team, r.away_team) for r in prev + nxt)
                if found_adj:
                    reasons["match_so_em_dia_adjacente"] += 1
                    if len(examples_not_found) < 5:
                        examples_not_found.append((m, "dia_adjacente"))
                else:
                    reasons["time_nao_bateu"] += 1
                    if len(examples_not_found) < 5:
                        examples_not_found.append((m, "time_nao_bateu"))

        print(f"- matched: {matched}/{len(sample)} ({matched/len(sample)*100:.1f}%)")
        if reasons:
            print("- principais motivos (aprox.):")
            for k, v in reasons.most_common(10):
                print(f"  - {k}: {v}")
        # datas com erro da API (ex.: plano free sem acesso)
        bad_dates = []
        for ds, payload in api_meta.items():
            tag = _classify_api_meta(payload or {})
            if tag:
                bad_dates.append((ds, tag, payload.get("errors") if payload else None))
        if bad_dates:
            print("\n- datas com erro reportado pela API:")
            for ds, tag, err in sorted(bad_dates)[:10]:
                print(f"  - {ds}: {tag} | errors={err}")
        print()

        if examples_not_found:
            print("3.1) Exemplos de jogos não encontrados (com melhor candidato por similaridade)")
            for m, reason in examples_not_found:
                ds = m.kickoff_time.strftime("%Y-%m-%d")
                api_results = (api_cache.get(ds) or []) + (api_cache.get((datetime.strptime(ds, '%Y-%m-%d').replace(tzinfo=timezone.utc)-timedelta(days=1)).strftime('%Y-%m-%d')) or []) + (api_cache.get((datetime.strptime(ds, '%Y-%m-%d').replace(tzinfo=timezone.utc)+timedelta(days=1)).strftime('%Y-%m-%d')) or [])
                # pick melhor candidato por similaridade média dos dois times
                best = None
                best_score = -1.0
                mh = get_mapped_name(m.home_team)
                ma = get_mapped_name(m.away_team)
                for r in api_results[:500]:
                    rh = get_mapped_name(r.home_team)
                    ra = get_mapped_name(r.away_team)
                    score = 0.5 * similarity_ratio(mh, rh) + 0.5 * similarity_ratio(ma, ra)
                    if score > best_score:
                        best_score = score
                        best = r
                print(f"- {reason} | {_fmt_dt(m.kickoff_time)} | {m.home_team} vs {m.away_team} | norm=({normalize_team_name(m.home_team)} vs {normalize_team_name(m.away_team)})")
                if best:
                    print(f"    melhor candidato: {best.home_team} vs {best.away_team} | score≈{best_score:.2f} | api_date={best.kickoff_time.strftime('%Y-%m-%d')}")
            print()

        print("4) Hipóteses prováveis quando ROI/placar = 0")
        print("- O job de resultados não está rodando em loop (systemd/cron) ou falha ao iniciar.")
        print("- `API_FOOTBALL_KEY` ausente/inválida (status/requests podem denunciar).")
        print("- Matching por time está falhando (variações de nomes / liga / data).")
        print()
        print("Próximo passo recomendado:")
        print("- Rodar `cd betinasia_bot && python3 -m results.update_results --dry-run` e ver taxa 'Atualizados vs Não encontrados'.")
        print("- Se o dry-run mostrar muitos 'NAO ENCONTRADO', precisamos ajustar matching (ex.: usar kickoff_time como filtro, data adjacente, liga).")

    finally:
        await api.close()
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())

