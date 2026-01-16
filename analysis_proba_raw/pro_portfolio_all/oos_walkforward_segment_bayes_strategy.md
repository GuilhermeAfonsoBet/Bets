## OOS walk-forward (segment_bayes) — estratégia completa
- Dataset: `/workspace/analysis_proba_raw/scored_dedup_proba_raw_all.csv`
- Semanas totais no dataset: **26**; semanas testadas OOS (WF): **16** (a partir de 10 semanas globais de treino; por-segmento exige >= 6)
- **Modo segment**: constraints avaliadas por segmento (equivalente ao que já rodamos antes), α=1.

### Performance OOS (cap2) — portfólio agregado
- **PnL semanal médio (bootstrap IC95%)**: **USD 234.9** (IC95% -22.6..501.2)
- **Desvio padrão semanal**: USD 550.6
- **P(semana < 0)**: 31.2%

### Performance OOS (cap2) — excluindo semanas sem trades (stake=0)
- **PnL semanal médio (bootstrap IC95%)**: **USD 268.5** (IC95% -16.7..567.1)
- **Desvio padrão semanal**: USD 583.2
- **P(semana < 0)**: 35.7%
- **ROI on stake agregado (ponderado)**: 0.0634

### Risco no OOS (teste) — portfólio agregado
- p80(soma stakes/dia) = USD 2001 (limite=USD 1610)
- VaR10%(PnL diário) = USD -197.1 (limite >= USD -575)
- P(PnL diário <= -25% banca) = 0.0% (limite <= 10%)

### Ajuste de stake global (α)
- α médio=1.000; p10=1.000; p50=1.000; p90=1.000; P(α<1)=0.0%

### Estabilidade OOS da decisão por segmento (frequência de ativação)
- Arquivo regras: `analysis_proba_raw/pro_portfolio_all/oos_walkforward_segment_bayes_selected_rules.csv`

- **FH | domingo**: active_rate=0.0%, ok_rate=0.0%, stake_frac_médio=0.00%, cutoff_médio=1.00
- **FH | quarta-feira**: active_rate=50.0%, ok_rate=50.0%, stake_frac_médio=1.19%, cutoff_médio=0.58
- **FH | quinta-feira**: active_rate=62.5%, ok_rate=62.5%, stake_frac_médio=2.69%, cutoff_médio=0.56
- **FH | segunda-feira**: active_rate=0.0%, ok_rate=0.0%, stake_frac_médio=0.00%, cutoff_médio=1.00
- **FH | sexta-feira**: active_rate=0.0%, ok_rate=0.0%, stake_frac_médio=0.00%, cutoff_médio=1.00
- **FH | sábado**: active_rate=75.0%, ok_rate=75.0%, stake_frac_médio=0.94%, cutoff_médio=0.43
- **FH | terça-feira**: active_rate=43.8%, ok_rate=43.8%, stake_frac_médio=0.88%, cutoff_médio=0.73
- **FT | domingo**: active_rate=31.2%, ok_rate=31.2%, stake_frac_médio=0.31%, cutoff_médio=0.78
- **FT | quarta-feira**: active_rate=75.0%, ok_rate=75.0%, stake_frac_médio=4.38%, cutoff_médio=0.55
- **FT | quinta-feira**: active_rate=12.5%, ok_rate=12.5%, stake_frac_médio=0.25%, cutoff_médio=0.89
- **FT | segunda-feira**: active_rate=56.2%, ok_rate=56.2%, stake_frac_médio=1.31%, cutoff_médio=0.55
- **FT | sexta-feira**: active_rate=18.8%, ok_rate=18.8%, stake_frac_médio=0.19%, cutoff_médio=0.85
- **FT | sábado**: active_rate=75.0%, ok_rate=75.0%, stake_frac_médio=1.50%, cutoff_médio=0.53
- **FT | terça-feira**: active_rate=56.2%, ok_rate=56.2%, stake_frac_médio=1.12%, cutoff_médio=0.55

### Segmentos mais estáveis no OOS (por lucro semanal)
- CSV: `analysis_proba_raw/pro_portfolio_all/oos_walkforward_segment_bayes_segment_stability.csv`
- **FT|segunda-feira**: mean_week=64.3 (IC95% -6.1..137.6), P(semana>0)=50.0%
- **FT|sábado**: mean_week=59.8 (IC95% 0.0..131.4), P(semana>0)=50.0%
- **FH|sábado**: mean_week=44.3 (IC95% 5.8..90.7), P(semana>0)=50.0%
- **FT|terça-feira**: mean_week=37.6 (IC95% -9.3..93.3), P(semana>0)=37.5%
- **FH|quinta-feira**: mean_week=36.3 (IC95% -36.2..113.2), P(semana>0)=31.2%
- **FT|quarta-feira**: mean_week=20.5 (IC95% -90.9..135.7), P(semana>0)=37.5%
- **FH|terça-feira**: mean_week=11.0 (IC95% -33.2..53.8), P(semana>0)=25.0%
- **FT|domingo**: mean_week=0.0 (IC95% -4.3..4.3), P(semana>0)=6.2%
