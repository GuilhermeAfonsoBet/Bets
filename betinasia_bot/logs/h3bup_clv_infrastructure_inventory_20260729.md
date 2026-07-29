# Inventário CLV / Fair Edge — 20260729

## Resumo
| Componente | Existe? | Localização | Reutilizável H3BUP? | Alteração necessária | Risco |
|---|---|---|---|---|---|
| Scheduler CLV genérico (jobs post_5m/15m/closing) | **Não encontrado** | — | N/A | criar | alto se betslips |
| Scheduler CLV DT dedicado | Não encontrado nesta VPS H3BUP | docs DT mencionam reports, não scheduler CLV betslip | parcial conceitual | adaptar | médio |
| Scheduler CLV H3BUP | **Não** | — | — | criar | — |
| Closing odd offline via best_odds_history | **Sim** | `results/update_hypothesis_results.py:get_closing_odd`, B808 report | **Sim (analytics)** | mapear order_id/policy | baixo |
| CLV raw formula | **Sim** | B808: `clv_bs=(bs_odd-closing_odd)/closing_odd*100` | Sim | amarrar a LIVE_OK | baixo |
| Fair edge / overround / de-vig | **Não confirmado** no B808 path inspecionado | grep sem hits fair_edge/overround | NÃO CONFIRMADO | possivelmente criar | médio |
| same-line strict | Parcial (closing busca mesma ah_line/side) | get_closing_odd | parcial | formalizar strict | médio |
| opposite side | Parcial em detector H3B down path | detectors.py | parcial | não no path H3BUP up | médio |
| Kickoff resolver | Parcial | `matches.kickoff_time` join; B808/best_odds | parcial | confidence/conflitos | médio |
| Betslip open for CLV snapshots | Não como scheduler H3BUP | executor dryrun existe | reutilizável com cuidado | obligation+cap | **alto** |
| Dedup/idempotency CLV jobs | N/A scheduler | — | — | desenhar | — |
| Kill switch / daily cap CLV | N/A | — | — | desenhar | — |
| Daily report CLV sections | Parcial | daily_full_report / B808 analytics | reporting only | section H3BUP | baixo |

## Respostas-chave
1. Scheduler CLV genérico operacional? **Não encontrado.**
2. DT específico? **Não confirmado nesta VPS como jobs de betslip CLV.**
3. H3BUP específico? **Não.**
4. Reutilizável? **Analytics de closing/CLV raw sim; path de betslip scheduling precisa ser criado/adaptado.**
5. Unidade atual de closing updates: eventos de hipótese/`Match`+`BestOddsHistory`, **não** order LIVE_OK H3BUP.
25–27. Overround/fair edge: **NÃO CONFIRMADO** no código inspecionado desta auditoria.
28. Sinal CLV B808: positivo se entry/bs > closing (definição `(bs-closing)/closing`).
31. Reutilizar sem alterar execução H3BUP? **Analytics sim; jobs de betslip não, sem desenho de caps.**

## Fórmulas confirmadas
```text
closing_odd = last BestOddsHistory odd BEFORE kickoff for same match/line(/side)
clv_bs = (bs_odd - closing_odd) / closing_odd * 100
```
