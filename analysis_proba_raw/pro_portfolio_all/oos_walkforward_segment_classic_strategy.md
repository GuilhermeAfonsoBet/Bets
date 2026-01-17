## OOS walk-forward (segment_classic) — estratégia completa
- Dataset: `/workspace/analysis_proba_raw/scored_dedup_proba_raw_all.csv`
- Semanas totais no dataset: **26**; semanas testadas OOS (WF): **16** (a partir de 10 semanas globais de treino; por-segmento exige >= 6)
- **Modo segment**: constraints avaliadas por segmento (equivalente ao que já rodamos antes), α=1.

### Performance OOS (cap2) — portfólio agregado
- **PnL semanal médio (bootstrap IC95%)**: **USD 196.2** (IC95% -143.3..542.4)
- **Desvio padrão semanal**: USD 721.7
- **P(semana < 0)**: 43.8%

### Performance OOS (cap2) — excluindo semanas sem trades (stake=0)
- **PnL semanal médio (bootstrap IC95%)**: **USD 224.2** (IC95% -150.6..618.8)
- **Desvio padrão semanal**: USD 770.8
- **P(semana < 0)**: 50.0%
- **ROI on stake agregado (ponderado)**: 0.0405

### Risco no OOS (teste) — portfólio agregado
- p80(soma stakes/dia) = USD 2026 (limite=USD 1610)
- VaR10%(PnL diário) = USD -298.0 (limite >= USD -575)
- P(PnL diário <= -25% banca) = 3.3% (limite <= 10%)

### Ajuste de stake global (α)
- α médio=1.000; p10=1.000; p50=1.000; p90=1.000; P(α<1)=0.0%

### Estabilidade OOS da decisão por segmento (frequência de ativação)
- Arquivo regras: `analysis_proba_raw/pro_portfolio_all/oos_walkforward_segment_classic_selected_rules.csv`

- **FH | domingo**: active_rate=6.2%, ok_rate=6.2%, stake_frac_médio=0.06%, cutoff_médio=0.94
- **FH | quarta-feira**: active_rate=56.2%, ok_rate=56.2%, stake_frac_médio=3.50%, cutoff_médio=0.52
- **FH | quinta-feira**: active_rate=62.5%, ok_rate=62.5%, stake_frac_médio=3.69%, cutoff_médio=0.57
- **FH | segunda-feira**: active_rate=0.0%, ok_rate=0.0%, stake_frac_médio=0.00%, cutoff_médio=1.00
- **FH | sexta-feira**: active_rate=0.0%, ok_rate=0.0%, stake_frac_médio=0.00%, cutoff_médio=1.00
- **FH | sábado**: active_rate=75.0%, ok_rate=75.0%, stake_frac_médio=0.94%, cutoff_médio=0.41
- **FH | terça-feira**: active_rate=50.0%, ok_rate=50.0%, stake_frac_médio=2.94%, cutoff_médio=0.69
- **FT | domingo**: active_rate=37.5%, ok_rate=37.5%, stake_frac_médio=0.38%, cutoff_médio=0.72
- **FT | quarta-feira**: active_rate=75.0%, ok_rate=75.0%, stake_frac_médio=4.50%, cutoff_médio=0.56
- **FT | quinta-feira**: active_rate=25.0%, ok_rate=25.0%, stake_frac_médio=0.44%, cutoff_médio=0.78
- **FT | segunda-feira**: active_rate=56.2%, ok_rate=56.2%, stake_frac_médio=1.69%, cutoff_médio=0.56
- **FT | sexta-feira**: active_rate=25.0%, ok_rate=25.0%, stake_frac_médio=0.38%, cutoff_médio=0.80
- **FT | sábado**: active_rate=75.0%, ok_rate=75.0%, stake_frac_médio=1.56%, cutoff_médio=0.53
- **FT | terça-feira**: active_rate=62.5%, ok_rate=62.5%, stake_frac_médio=1.75%, cutoff_médio=0.51

### Segmentos mais estáveis no OOS (por lucro semanal)
- CSV: `analysis_proba_raw/pro_portfolio_all/oos_walkforward_segment_classic_segment_stability.csv`
- **FT|terça-feira**: mean_week=83.4 (IC95% 9.4..165.2), P(semana>0)=50.0%
- **FT|sábado**: mean_week=65.6 (IC95% 3.8..138.0), P(semana>0)=50.0%
- **FT|segunda-feira**: mean_week=56.6 (IC95% -13.8..130.4), P(semana>0)=43.8%
- **FH|sábado**: mean_week=46.4 (IC95% 6.5..93.2), P(semana>0)=50.0%
- **FT|quarta-feira**: mean_week=27.7 (IC95% -98.4..156.1), P(semana>0)=37.5%
- **FH|quarta-feira**: mean_week=2.7 (IC95% -74.5..72.7), P(semana>0)=31.2%
- **FT|domingo**: mean_week=-2.2 (IC95% -8.6..2.9), P(semana>0)=6.2%
- **FT|sexta-feira**: mean_week=-2.9 (IC95% -20.1..10.1), P(semana>0)=12.5%
