## OOS walk-forward (segment_bayes) — estratégia completa
- Dataset: `/workspace/analysis_proba_raw/scored_dedup_proba_raw_all.csv`
- Semanas totais no dataset: **26**; semanas testadas OOS (WF): **16** (a partir de 10 semanas globais de treino; por-segmento exige >= 6)
- **Modo segment**: constraints avaliadas por segmento (equivalente ao que já rodamos antes), α=1.

### Performance OOS (cap2) — portfólio agregado
- **PnL semanal médio (bootstrap IC95%)**: **USD -59.0** (IC95% -221.1..97.4)
- **Desvio padrão semanal**: USD 333.0
- **P(semana < 0)**: 43.8%

### Performance OOS (cap2) — excluindo semanas sem trades (stake=0)
- **PnL semanal médio (bootstrap IC95%)**: **USD -78.7** (IC95% -292.6..129.1)
- **Desvio padrão semanal**: USD 386.7
- **P(semana < 0)**: 58.3%
- **ROI on stake agregado (ponderado)**: -0.0420

### Risco no OOS (teste) — portfólio agregado
- p80(soma stakes/dia) = USD 1033 (limite=USD 1610)
- VaR10%(PnL diário) = USD -255.7 (limite >= USD -575)
- P(PnL diário <= -25% banca) = 0.0% (limite <= 10%)

### Ajuste de stake global (α)
- α médio=1.000; p10=1.000; p50=1.000; p90=1.000; P(α<1)=0.0%

### Estabilidade OOS da decisão por segmento (frequência de ativação)
- Arquivo regras: `analysis_proba_raw/pro_portfolio_all/oos_walkforward_segment_bayes_selected_rules.csv`

- **FH | domingo**: active_rate=0.0%, ok_rate=0.0%, stake_frac_médio=0.00%, cutoff_médio=1.00
- **FH | quarta-feira**: active_rate=6.2%, ok_rate=6.2%, stake_frac_médio=0.06%, cutoff_médio=0.95
- **FH | quinta-feira**: active_rate=50.0%, ok_rate=50.0%, stake_frac_médio=1.69%, cutoff_médio=0.71
- **FH | segunda-feira**: active_rate=0.0%, ok_rate=0.0%, stake_frac_médio=0.00%, cutoff_médio=1.00
- **FH | sexta-feira**: active_rate=0.0%, ok_rate=0.0%, stake_frac_médio=0.00%, cutoff_médio=1.00
- **FH | sábado**: active_rate=0.0%, ok_rate=0.0%, stake_frac_médio=0.00%, cutoff_médio=1.00
- **FH | terça-feira**: active_rate=31.2%, ok_rate=31.2%, stake_frac_médio=0.31%, cutoff_médio=0.86
- **FT | domingo**: active_rate=0.0%, ok_rate=0.0%, stake_frac_médio=0.00%, cutoff_médio=1.00
- **FT | quarta-feira**: active_rate=68.8%, ok_rate=68.8%, stake_frac_médio=2.50%, cutoff_médio=0.62
- **FT | quinta-feira**: active_rate=6.2%, ok_rate=6.2%, stake_frac_médio=0.06%, cutoff_médio=0.94
- **FT | segunda-feira**: active_rate=50.0%, ok_rate=50.0%, stake_frac_médio=1.00%, cutoff_médio=0.59
- **FT | sexta-feira**: active_rate=0.0%, ok_rate=0.0%, stake_frac_médio=0.00%, cutoff_médio=1.00
- **FT | sábado**: active_rate=12.5%, ok_rate=12.5%, stake_frac_médio=0.19%, cutoff_médio=0.92
- **FT | terça-feira**: active_rate=31.2%, ok_rate=31.2%, stake_frac_médio=0.50%, cutoff_médio=0.75

### Segmentos mais estáveis no OOS (por lucro semanal)
- CSV: `analysis_proba_raw/pro_portfolio_all/oos_walkforward_segment_bayes_segment_stability.csv`
- **FT|terça-feira**: mean_week=21.7 (IC95% -7.6..53.5), P(semana>0)=25.0%
- **FT|segunda-feira**: mean_week=18.3 (IC95% -37.5..76.5), P(semana>0)=37.5%
- **FH|quarta-feira**: mean_week=-5.0 (IC95% -15.1..0.0), P(semana>0)=0.0%
- **FH|terça-feira**: mean_week=-5.6 (IC95% -12.9..0.0), P(semana>0)=0.0%
- **FT|sábado**: mean_week=-5.8 (IC95% -32.4..14.9), P(semana>0)=6.2%
- **FT|quinta-feira**: mean_week=-14.5 (IC95% -43.6..0.0), P(semana>0)=0.0%
- **FH|quinta-feira**: mean_week=-17.6 (IC95% -78.6..22.6), P(semana>0)=31.2%
- **FT|quarta-feira**: mean_week=-50.3 (IC95% -156.7..42.6), P(semana>0)=12.5%
