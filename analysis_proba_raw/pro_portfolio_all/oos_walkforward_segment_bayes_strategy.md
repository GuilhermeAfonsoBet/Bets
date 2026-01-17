## OOS walk-forward (segment_bayes) — estratégia completa
- Dataset: `/workspace/analysis_proba_raw/scored_dedup_proba_raw_all.csv`
- Semanas totais no dataset: **26**; semanas testadas OOS (WF): **16** (a partir de 10 semanas globais de treino; por-segmento exige >= 6)
- **Modo segment**: constraints avaliadas por segmento (equivalente ao que já rodamos antes), α=1.

### Performance OOS (cap2) — portfólio agregado
- **PnL semanal médio (bootstrap IC95%)**: **USD 254.5** (IC95% -12.7..532.9)
- **Desvio padrão semanal**: USD 571.7
- **P(semana < 0)**: 25.0%

### Performance OOS (cap2) — excluindo semanas sem trades (stake=0)
- **PnL semanal médio (bootstrap IC95%)**: **USD 290.9** (IC95% -4.7..603.5)
- **Desvio padrão semanal**: USD 604.7
- **P(semana < 0)**: 28.6%
- **ROI on stake agregado (ponderado)**: 0.0638

### Risco no OOS (teste) — portfólio agregado
- p80(soma stakes/dia) = USD 1996 (limite=USD 1610)
- VaR10%(PnL diário) = USD -205.0 (limite >= USD -575)
- P(PnL diário <= -25% banca) = 0.0% (limite <= 10%)

### Ajuste de stake global (α)
- α médio=1.000; p10=1.000; p50=1.000; p90=1.000; P(α<1)=0.0%

### Estabilidade OOS da decisão por segmento (frequência de ativação)
- Arquivo regras: `analysis_proba_raw/pro_portfolio_all/oos_walkforward_segment_bayes_selected_rules.csv`

- **FH | domingo**: active_rate=0.0%, ok_rate=0.0%, stake_frac_médio=0.00%, cutoff_médio=1.00
- **FH | quarta-feira**: active_rate=50.0%, ok_rate=50.0%, stake_frac_médio=1.56%, cutoff_médio=0.58
- **FH | quinta-feira**: active_rate=62.5%, ok_rate=62.5%, stake_frac_médio=2.69%, cutoff_médio=0.56
- **FH | segunda-feira**: active_rate=0.0%, ok_rate=0.0%, stake_frac_médio=0.00%, cutoff_médio=1.00
- **FH | sexta-feira**: active_rate=0.0%, ok_rate=0.0%, stake_frac_médio=0.00%, cutoff_médio=1.00
- **FH | sábado**: active_rate=75.0%, ok_rate=75.0%, stake_frac_médio=0.94%, cutoff_médio=0.42
- **FH | terça-feira**: active_rate=37.5%, ok_rate=37.5%, stake_frac_médio=0.94%, cutoff_médio=0.79
- **FT | domingo**: active_rate=31.2%, ok_rate=31.2%, stake_frac_médio=0.31%, cutoff_médio=0.78
- **FT | quarta-feira**: active_rate=75.0%, ok_rate=75.0%, stake_frac_médio=4.38%, cutoff_médio=0.55
- **FT | quinta-feira**: active_rate=12.5%, ok_rate=12.5%, stake_frac_médio=0.25%, cutoff_médio=0.89
- **FT | segunda-feira**: active_rate=56.2%, ok_rate=56.2%, stake_frac_médio=1.31%, cutoff_médio=0.55
- **FT | sexta-feira**: active_rate=25.0%, ok_rate=25.0%, stake_frac_médio=0.25%, cutoff_médio=0.80
- **FT | sábado**: active_rate=75.0%, ok_rate=75.0%, stake_frac_médio=1.50%, cutoff_médio=0.53
- **FT | terça-feira**: active_rate=50.0%, ok_rate=50.0%, stake_frac_médio=1.31%, cutoff_médio=0.61

### Segmentos mais estáveis no OOS (por lucro semanal)
- CSV: `analysis_proba_raw/pro_portfolio_all/oos_walkforward_segment_bayes_segment_stability.csv`
- **FT|terça-feira**: mean_week=73.9 (IC95% 12.8..143.4), P(semana>0)=37.5%
- **FT|segunda-feira**: mean_week=64.3 (IC95% -6.1..137.6), P(semana>0)=50.0%
- **FT|sábado**: mean_week=59.8 (IC95% 0.0..131.4), P(semana>0)=50.0%
- **FH|sábado**: mean_week=47.9 (IC95% 8.6..93.9), P(semana>0)=50.0%
- **FH|quinta-feira**: mean_week=30.5 (IC95% -43.0..108.7), P(semana>0)=31.2%
- **FT|quarta-feira**: mean_week=11.9 (IC95% -106.1..131.9), P(semana>0)=37.5%
- **FH|quarta-feira**: mean_week=4.0 (IC95% -66.7..68.7), P(semana>0)=18.8%
- **FT|domingo**: mean_week=0.0 (IC95% -4.3..4.3), P(semana>0)=6.2%
