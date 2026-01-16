## OOS walk-forward (segment_classic) — estratégia completa
- Dataset: `/workspace/analysis_proba_raw/scored_dedup_proba_raw_all.csv`
- Semanas totais no dataset: **26**; semanas testadas OOS (WF): **16** (a partir de 10 semanas globais de treino; por-segmento exige >= 6)
- **Modo segment**: constraints avaliadas por segmento (equivalente ao que já rodamos antes), α=1.

### Performance OOS (cap2) — portfólio agregado
- **PnL semanal médio (bootstrap IC95%)**: **USD 213.2** (IC95% -120.3..555.6)
- **Desvio padrão semanal**: USD 711.6
- **P(semana < 0)**: 43.8%

### Performance OOS (cap2) — excluindo semanas sem trades (stake=0)
- **PnL semanal médio (bootstrap IC95%)**: **USD 243.6** (IC95% -129.9..628.8)
- **Desvio padrão semanal**: USD 759.1
- **P(semana < 0)**: 50.0%
- **ROI on stake agregado (ponderado)**: 0.0465

### Risco no OOS (teste) — portfólio agregado
- p80(soma stakes/dia) = USD 2029 (limite=USD 1610)
- VaR10%(PnL diário) = USD -282.6 (limite >= USD -575)
- P(PnL diário <= -25% banca) = 1.7% (limite <= 10%)

### Ajuste de stake global (α)
- α médio=1.000; p10=1.000; p50=1.000; p90=1.000; P(α<1)=0.0%

### Estabilidade OOS da decisão por segmento (frequência de ativação)
- Arquivo regras: `analysis_proba_raw/pro_portfolio_all/oos_walkforward_segment_classic_selected_rules.csv`

- **FH | domingo**: active_rate=6.2%, ok_rate=6.2%, stake_frac_médio=0.06%, cutoff_médio=0.94
- **FH | quarta-feira**: active_rate=56.2%, ok_rate=56.2%, stake_frac_médio=2.56%, cutoff_médio=0.56
- **FH | quinta-feira**: active_rate=62.5%, ok_rate=62.5%, stake_frac_médio=3.69%, cutoff_médio=0.57
- **FH | segunda-feira**: active_rate=0.0%, ok_rate=0.0%, stake_frac_médio=0.00%, cutoff_médio=1.00
- **FH | sexta-feira**: active_rate=0.0%, ok_rate=0.0%, stake_frac_médio=0.00%, cutoff_médio=1.00
- **FH | sábado**: active_rate=75.0%, ok_rate=75.0%, stake_frac_médio=0.94%, cutoff_médio=0.41
- **FH | terça-feira**: active_rate=50.0%, ok_rate=50.0%, stake_frac_médio=2.94%, cutoff_médio=0.68
- **FT | domingo**: active_rate=37.5%, ok_rate=37.5%, stake_frac_médio=0.38%, cutoff_médio=0.72
- **FT | quarta-feira**: active_rate=75.0%, ok_rate=75.0%, stake_frac_médio=4.50%, cutoff_médio=0.56
- **FT | quinta-feira**: active_rate=25.0%, ok_rate=25.0%, stake_frac_médio=0.44%, cutoff_médio=0.78
- **FT | segunda-feira**: active_rate=56.2%, ok_rate=56.2%, stake_frac_médio=1.69%, cutoff_médio=0.56
- **FT | sexta-feira**: active_rate=18.8%, ok_rate=18.8%, stake_frac_médio=0.31%, cutoff_médio=0.85
- **FT | sábado**: active_rate=75.0%, ok_rate=75.0%, stake_frac_médio=1.56%, cutoff_médio=0.53
- **FT | terça-feira**: active_rate=68.8%, ok_rate=68.8%, stake_frac_médio=2.19%, cutoff_médio=0.47

### Segmentos mais estáveis no OOS (por lucro semanal)
- CSV: `analysis_proba_raw/pro_portfolio_all/oos_walkforward_segment_classic_segment_stability.csv`
- **FT|sábado**: mean_week=65.6 (IC95% 3.8..138.0), P(semana>0)=50.0%
- **FT|terça-feira**: mean_week=63.2 (IC95% -10.1..144.4), P(semana>0)=43.8%
- **FT|segunda-feira**: mean_week=56.6 (IC95% -13.8..130.4), P(semana>0)=43.8%
- **FT|quarta-feira**: mean_week=47.8 (IC95% -70.5..167.2), P(semana>0)=37.5%
- **FH|sábado**: mean_week=41.4 (IC95% 2.2..88.6), P(semana>0)=43.8%
- **FH|quinta-feira**: mean_week=18.0 (IC95% -71.4..105.5), P(semana>0)=31.2%
- **FT|domingo**: mean_week=-2.2 (IC95% -8.6..2.9), P(semana>0)=6.2%
- **FT|sexta-feira**: mean_week=-4.3 (IC95% -21.6..8.6), P(semana>0)=6.2%
