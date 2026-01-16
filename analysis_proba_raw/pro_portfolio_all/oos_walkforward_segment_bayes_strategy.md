## OOS walk-forward (segment_bayes) — estratégia completa
- Dataset: `/workspace/analysis_proba_raw/scored_dedup_proba_raw_all.csv`
- Semanas totais no dataset: **26**; semanas testadas OOS (WF): **16** (a partir de 10 semanas globais de treino; por-segmento exige >= 6)
- **Modo segment**: constraints avaliadas por segmento (equivalente ao que já rodamos antes), α=1.

### Performance OOS (cap2) — portfólio agregado
- **PnL semanal médio (bootstrap IC95%)**: **USD 249.6** (IC95% 0.4..525.5)
- **Desvio padrão semanal**: USD 554.4
- **P(semana < 0)**: 37.5%

### Performance OOS (cap2) — excluindo semanas sem trades (stake=0)
- **PnL semanal médio (bootstrap IC95%)**: **USD 266.2** (IC95% 2.7..562.9)
- **Desvio padrão semanal**: USD 569.7
- **P(semana < 0)**: 40.0%
- **ROI on stake agregado (ponderado)**: 0.0547

### Risco no OOS (teste) — portfólio agregado
- p80(soma stakes/dia) = USD 1933 (limite=USD 1610)
- VaR10%(PnL diário) = USD -161.9 (limite >= USD -575)
- P(PnL diário <= -25% banca) = 0.0% (limite <= 10%)

### Ajuste de stake global (α)
- α médio=1.000; p10=1.000; p50=1.000; p90=1.000; P(α<1)=0.0%

### Estabilidade OOS da decisão por segmento (frequência de ativação)
- Arquivo regras: `analysis_proba_raw/pro_portfolio_all/oos_walkforward_segment_bayes_selected_rules.csv`

- **FH | domingo**: active_rate=12.5%, ok_rate=12.5%, stake_frac_médio=0.12%, cutoff_médio=0.89
- **FH | quarta-feira**: active_rate=50.0%, ok_rate=50.0%, stake_frac_médio=1.19%, cutoff_médio=0.58
- **FH | quinta-feira**: active_rate=75.0%, ok_rate=75.0%, stake_frac_médio=3.31%, cutoff_médio=0.42
- **FH | segunda-feira**: active_rate=0.0%, ok_rate=0.0%, stake_frac_médio=0.00%, cutoff_médio=1.00
- **FH | sexta-feira**: active_rate=31.2%, ok_rate=31.2%, stake_frac_médio=0.31%, cutoff_médio=0.71
- **FH | sábado**: active_rate=81.2%, ok_rate=81.2%, stake_frac_médio=1.31%, cutoff_médio=0.43
- **FH | terça-feira**: active_rate=43.8%, ok_rate=43.8%, stake_frac_médio=0.88%, cutoff_médio=0.73
- **FT | domingo**: active_rate=50.0%, ok_rate=50.0%, stake_frac_médio=0.75%, cutoff_médio=0.63
- **FT | quarta-feira**: active_rate=75.0%, ok_rate=75.0%, stake_frac_médio=4.38%, cutoff_médio=0.55
- **FT | quinta-feira**: active_rate=12.5%, ok_rate=12.5%, stake_frac_médio=0.25%, cutoff_médio=0.90
- **FT | segunda-feira**: active_rate=56.2%, ok_rate=56.2%, stake_frac_médio=1.31%, cutoff_médio=0.55
- **FT | sexta-feira**: active_rate=0.0%, ok_rate=0.0%, stake_frac_médio=0.00%, cutoff_médio=1.00
- **FT | sábado**: active_rate=81.2%, ok_rate=81.2%, stake_frac_médio=4.12%, cutoff_médio=0.59
- **FT | terça-feira**: active_rate=62.5%, ok_rate=62.5%, stake_frac_médio=1.25%, cutoff_médio=0.50

### Segmentos mais estáveis no OOS (por lucro semanal)
- CSV: `analysis_proba_raw/pro_portfolio_all/oos_walkforward_segment_bayes_segment_stability.csv`
- **FT|terça-feira**: mean_week=65.6 (IC95% 4.3..137.1), P(semana>0)=43.8%
- **FT|segunda-feira**: mean_week=64.3 (IC95% -6.1..137.6), P(semana>0)=50.0%
- **FT|sábado**: mean_week=59.9 (IC95% -2.1..128.1), P(semana>0)=37.5%
- **FH|sábado**: mean_week=31.8 (IC95% -9.9..81.2), P(semana>0)=43.8%
- **FT|quarta-feira**: mean_week=30.6 (IC95% -80.2..144.1), P(semana>0)=37.5%
- **FH|quinta-feira**: mean_week=20.8 (IC95% -58.1..120.3), P(semana>0)=31.2%
- **FH|terça-feira**: mean_week=20.1 (IC95% -9.7..56.4), P(semana>0)=25.0%
- **FT|domingo**: mean_week=2.9 (IC95% -5.8..11.5), P(semana>0)=12.5%
