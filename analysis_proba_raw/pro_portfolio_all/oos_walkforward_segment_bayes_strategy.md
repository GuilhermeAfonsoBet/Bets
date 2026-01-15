## OOS walk-forward (segment_bayes) — estratégia completa
- Dataset: `/workspace/analysis_proba_raw/scored_dedup_proba_raw_all.csv`
- Semanas totais no dataset: **26**; semanas testadas OOS (WF): **16** (a partir de 10 semanas globais de treino; por-segmento exige >= 6)
- **Modo segment**: constraints avaliadas por segmento (equivalente ao que já rodamos antes), α=1.

### Performance OOS (cap2) — portfólio agregado
- **PnL semanal médio (bootstrap IC95%)**: **USD 104.2** (IC95% -82.7..317.3)
- **Desvio padrão semanal**: USD 418.5
- **P(semana < 0)**: 37.5%

### Performance OOS (cap2) — excluindo semanas sem trades (stake=0)
- **PnL semanal médio (bootstrap IC95%)**: **USD 119.1** (IC95% -93.1..356.5)
- **Desvio padrão semanal**: USD 447.5
- **P(semana < 0)**: 42.9%
- **ROI on stake agregado (ponderado)**: 0.0372

### Risco no OOS (teste) — portfólio agregado
- p80(soma stakes/dia) = USD 1600 (limite=USD 1610)
- VaR10%(PnL diário) = USD -179.5 (limite >= USD -575)
- P(PnL diário <= -25% banca) = 0.0% (limite <= 10%)

### Ajuste de stake global (α)
- α médio=1.000; p10=1.000; p50=1.000; p90=1.000; P(α<1)=0.0%

### Estabilidade OOS da decisão por segmento (frequência de ativação)
- Arquivo regras: `analysis_proba_raw/pro_portfolio_all/oos_walkforward_segment_bayes_selected_rules.csv`

- **FH | domingo**: active_rate=0.0%, ok_rate=0.0%, stake_frac_médio=0.00%, cutoff_médio=1.00
- **FH | quarta-feira**: active_rate=18.8%, ok_rate=18.8%, stake_frac_médio=0.44%, cutoff_médio=0.85
- **FH | quinta-feira**: active_rate=56.2%, ok_rate=56.2%, stake_frac_médio=1.69%, cutoff_médio=0.68
- **FH | segunda-feira**: active_rate=0.0%, ok_rate=0.0%, stake_frac_médio=0.00%, cutoff_médio=1.00
- **FH | sexta-feira**: active_rate=31.2%, ok_rate=31.2%, stake_frac_médio=0.31%, cutoff_médio=0.71
- **FH | sábado**: active_rate=50.0%, ok_rate=50.0%, stake_frac_médio=0.50%, cutoff_médio=0.62
- **FH | terça-feira**: active_rate=37.5%, ok_rate=37.5%, stake_frac_médio=0.56%, cutoff_médio=0.76
- **FT | domingo**: active_rate=18.8%, ok_rate=18.8%, stake_frac_médio=0.19%, cutoff_médio=0.87
- **FT | quarta-feira**: active_rate=75.0%, ok_rate=75.0%, stake_frac_médio=3.50%, cutoff_médio=0.49
- **FT | quinta-feira**: active_rate=12.5%, ok_rate=12.5%, stake_frac_médio=0.25%, cutoff_médio=0.89
- **FT | segunda-feira**: active_rate=56.2%, ok_rate=56.2%, stake_frac_médio=1.25%, cutoff_médio=0.50
- **FT | sexta-feira**: active_rate=0.0%, ok_rate=0.0%, stake_frac_médio=0.00%, cutoff_médio=1.00
- **FT | sábado**: active_rate=50.0%, ok_rate=50.0%, stake_frac_médio=1.62%, cutoff_médio=0.74
- **FT | terça-feira**: active_rate=43.8%, ok_rate=43.8%, stake_frac_médio=0.75%, cutoff_médio=0.63

### Segmentos mais estáveis no OOS (por lucro semanal)
- CSV: `analysis_proba_raw/pro_portfolio_all/oos_walkforward_segment_bayes_segment_stability.csv`
- **FT|segunda-feira**: mean_week=55.1 (IC95% -11.6..124.1), P(semana>0)=50.0%
- **FT|terça-feira**: mean_week=38.7 (IC95% -19.4..105.8), P(semana>0)=25.0%
- **FT|sábado**: mean_week=25.7 (IC95% -19.9..80.0), P(semana>0)=18.8%
- **FT|quarta-feira**: mean_week=8.7 (IC95% -69.2..98.5), P(semana>0)=25.0%
- **FH|quinta-feira**: mean_week=6.8 (IC95% -72.6..82.8), P(semana>0)=18.8%
- **FH|sábado**: mean_week=5.0 (IC95% -9.3..25.9), P(semana>0)=18.8%
- **FH|terça-feira**: mean_week=2.1 (IC95% -22.4..25.2), P(semana>0)=25.0%
- **FT|domingo**: mean_week=-1.4 (IC95% -12.9..8.6), P(semana>0)=6.2%
