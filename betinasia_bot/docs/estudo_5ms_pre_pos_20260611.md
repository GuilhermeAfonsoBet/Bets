# Estudo 5Ms (estatistico) - Pre e Pos 25/05

Data de consolidacao: 2026-06-11
Escopo: estrategia Back Pre com slippage_pre_pct < 0, com P&L real vindo de accounting balance.

## 1) Objetivo

Consolidar os achados da analise estatistica 5Ms, com foco em:

- robustez do edge;
- concentracao de resultado;
- estabilidade e reproducibilidade temporal;
- comparacao de regime Pre 25/05 vs Pos 25/05 (mudanca E2E).

## 2) Base de dados e reconciliacao

- Universo base (audit+bridge): `/tmp/base_audit_bridge_ate_20260611.csv`
- Base inferida para analise: `/tmp/base_5ms_real_ate_20260611_inferred.csv`
- Accounting usado: `/home/betbot/Bets/betinasia_bot/logs/accounting/20260610_220229__balance.csv`
- Ultimo `post_date` disponivel no accounting: **2026-06-10 21:00:15.937240+00:00**

### 2.1 Cobertura observada

- Audit base (pre_slipneg): 589
- Audit com bridge expandido: 582 (98.8%)
- Exec key com order_id no balance via jsonl: 422 (sobre 450 exec keys no universo mapeado inicial)

Observacao: na consolidacao por `bet_id`, o recorte final ficou:

- Pre 25/05: 375 apostas
- Pos 25/05: 206 apostas

## 3) Alertas metodologicos importantes

- `execution_id` (UUID) nao casa com `bets.confirmation_id`.
- `order_id` no accounting nao representa 1:1 a unidade de aposta.
- A unidade operacional mais consistente no ledger foi `bet_id`.
- ROI por aposta pode ficar "digital" quando stake e inferida por sinal de P&L e `got price`.

## 4) Resultado 5Ms por regime

### 4.1 Pre 25/05

| M | Conceito | Valor auferido | Faixa robusta (alvo) | Status |
|---|---|---|---|---|
| M1 | Evidencia estatistica de edge (perm + bootstrap) | p_perm=0.4524, ci90_lo=-6.9973% | p_perm <= 0.10 e ci90_lo > 0 | FAIL |
| M2 | Robustez a concentracao (sem Top-3) | ROI sem Top-3=-2.2068%, top1_abs=2.03% | ROI sem Top-3 > 0 e top1_abs <= 35% | FAIL |
| M3 | Estabilidade semanal | pos_ratio=50.0%, mediana semanal=1.2881% | >=55% semanas positivas e mediana >0 | FAIL |
| M4 | Reprodutibilidade temporal (3 blocos) | r1=4.4338%, r2=-1.0332%, r3=-2.3190% | 2/3 blocos >0 e ultimo bloco >0 | FAIL |
| M5 | Economia de payoff | EV/aposta=+0.0475, payoff_ratio=1.0124 | EV>0 e payoff_ratio>=1.8 | FAIL |

Score Pre: **0/5**

### 4.2 Pos 25/05

| M | Conceito | Valor auferido | Faixa robusta (alvo) | Status |
|---|---|---|---|---|
| M1 | Evidencia estatistica de edge (perm + bootstrap) | p_perm=0.1220, ci90_lo=-3.5629% | p_perm <= 0.10 e ci90_lo > 0 | FAIL |
| M2 | Robustez a concentracao (sem Top-3) | ROI sem Top-3=-1.0969%, top1_abs=11.15% | ROI sem Top-3 > 0 e top1_abs <= 35% | FAIL |
| M3 | Estabilidade semanal | pos_ratio=100%, mediana semanal=10.2796% | >=55% semanas positivas e mediana >0 | OK |
| M4 | Reprodutibilidade temporal (3 blocos) | r1=-3.8204%, r2=20.8622%, r3=12.9702% | 2/3 blocos >0 e ultimo bloco >0 | OK |
| M5 | Economia de payoff | EV/aposta=+2.0094, payoff_ratio=0.8920 | EV>0 e payoff_ratio>=1.8 | FAIL |

Score Pos: **2/5**

## 5) Evolucao temporal (rolling 28d)

Timeline rodada com step=3 dias e min_bets=40:
`/tmp/ms5_timeline_ate_20260610_step3_min40.csv`

Resumo:

- 6 janelas
- score variando entre 0/5 e 1/5
- melhora pontual em M4 nas ultimas janelas
- M1/M2/M5 persistem sem validacao robusta

## 6) Media vs mediana (por aposta)

### 6.1 Resultado por aposta (bet_id)

| Regime | n_apostas | mean_pnl | median_pnl | mean_roi_pct_por_aposta | median_roi_pct_por_aposta | weighted_roi_pct_carteira |
|---|---:|---:|---:|---:|---:|---:|
| Pre 25/05 | 375 | +0.0397 | -1.18 | -5.3157% | -100.0% | +0.6064% |
| Pos 25/05 | 206 | +0.8194 | +0.78 | +12.8354% | +73.7% | +10.7725% |

Leitura:

- Pre: perfil assimetrico com mediana fraca e carteira quase neutra/levemente positiva.
- Pos: melhora clara de P&L medio e mediano, com carteira positiva.

## 7) Pontos fortes e vulnerabilidades

### Pontos fortes

- Melhora de regime no Pos 25/05 em estabilidade (M3) e reproducao temporal (M4).
- ROI de carteira positivo no Pos.
- Pipeline audit->bridge tecnicamente recuperado com boa cobertura.

### Vulnerabilidades principais

- M1 ainda nao fecha (sem evidenca estatistica robusta no corte atual).
- M2 negativo (resultado ainda sensivel a concentracao em cauda).
- M5 fraco (payoff_ratio < 1.8).
- Cobertura de reconciliacao P&L ainda parcial em alguns caminhos de chave.

## 8) Conclusao executiva

A operacao mostra **sinal promissor no Pos 25/05**, mas ainda **inconclusivo em robustez estatistica** no criterio atual dos 5Ms.
Recomendacao: manter monitoramento em rolling, ampliar cobertura de reconciliacao e reavaliar M1/M2/M5 com mais liquidados apos 10/06.

## 9) Proximos passos recomendados

- Regerar base quando houver accounting com `post_date` mais avancado.
- Fixar metodologia de unidade economica em `bet_id` para comparabilidade.
- Rodar teste de diferenca de regime (Pre vs Pos) com bootstrap/permutation da diferenca de ROI.
- Publicar painel simples com score 5Ms por janela (semaforo operacional).
