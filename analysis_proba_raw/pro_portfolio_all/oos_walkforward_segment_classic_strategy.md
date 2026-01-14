## OOS walk-forward (segment_classic) — estratégia completa
- Dataset: `/workspace/analysis_proba_raw/scored_dedup_proba_raw_all.csv`
- Semanas totais no dataset: **26**; semanas testadas OOS (WF): **16** (a partir de 10 semanas globais de treino; por-segmento exige >= 6)
- **Modo segment**: constraints avaliadas por segmento (equivalente ao que já rodamos antes), α=1.

### Performance OOS (cap2) — portfólio agregado
- **PnL semanal médio (bootstrap IC95%)**: **USD 267.5** (IC95% -79.3..654.8)
- **Desvio padrão semanal**: USD 787.5
- **P(semana < 0)**: 50.0%

### Performance OOS (cap2) — excluindo semanas sem trades (stake=0)
- **PnL semanal médio (bootstrap IC95%)**: **USD 267.5** (IC95% -82.4..668.1)
- **Desvio padrão semanal**: USD 787.5
- **P(semana < 0)**: 50.0%
- **ROI on stake agregado (ponderado)**: 0.0518

### Risco no OOS (teste) — portfólio agregado
- p80(soma stakes/dia) = USD 1733 (limite=USD 1610)
- VaR10%(PnL diário) = USD -261.0 (limite >= USD -575)
- P(PnL diário <= -25% banca) = 0.0% (limite <= 10%)

### Ajuste de stake global (α)
- α médio=1.000; p10=1.000; p50=1.000; p90=1.000; P(α<1)=0.0%

### Estabilidade OOS da decisão por segmento (frequência de ativação)
- Arquivo regras: `analysis_proba_raw/pro_portfolio_all/oos_walkforward_segment_classic_selected_rules.csv`

- **FH | domingo**: active_rate=25.0%, ok_rate=25.0%, stake_frac_médio=1.06%, cutoff_médio=0.82
- **FH | quarta-feira**: active_rate=62.5%, ok_rate=62.5%, stake_frac_médio=2.25%, cutoff_médio=0.51
- **FH | quinta-feira**: active_rate=93.8%, ok_rate=93.8%, stake_frac_médio=6.31%, cutoff_médio=0.60
- **FH | segunda-feira**: active_rate=0.0%, ok_rate=0.0%, stake_frac_médio=0.00%, cutoff_médio=1.00
- **FH | sexta-feira**: active_rate=31.2%, ok_rate=31.2%, stake_frac_médio=0.31%, cutoff_médio=0.71
- **FH | sábado**: active_rate=93.8%, ok_rate=93.8%, stake_frac_médio=2.19%, cutoff_médio=0.38
- **FH | terça-feira**: active_rate=68.8%, ok_rate=68.8%, stake_frac_médio=4.75%, cutoff_médio=0.77
- **FT | domingo**: active_rate=87.5%, ok_rate=87.5%, stake_frac_médio=4.62%, cutoff_médio=0.45
- **FT | quarta-feira**: active_rate=75.0%, ok_rate=75.0%, stake_frac_médio=5.25%, cutoff_médio=0.62
- **FT | quinta-feira**: active_rate=93.8%, ok_rate=93.8%, stake_frac_médio=1.81%, cutoff_médio=0.72
- **FT | segunda-feira**: active_rate=56.2%, ok_rate=56.2%, stake_frac_médio=2.94%, cutoff_médio=0.59
- **FT | sexta-feira**: active_rate=37.5%, ok_rate=37.5%, stake_frac_médio=2.63%, cutoff_médio=0.84
- **FT | sábado**: active_rate=100.0%, ok_rate=100.0%, stake_frac_médio=6.06%, cutoff_médio=0.53
- **FT | terça-feira**: active_rate=56.2%, ok_rate=56.2%, stake_frac_médio=1.38%, cutoff_médio=0.57

### Segmentos mais estáveis no OOS (por lucro semanal)
- CSV: `analysis_proba_raw/pro_portfolio_all/oos_walkforward_segment_classic_segment_stability.csv`
- **FT|segunda-feira**: mean_week=79.4 (IC95% -31.1..214.2), P(semana>0)=37.5%
- **FT|quarta-feira**: mean_week=66.0 (IC95% -70.3..229.8), P(semana>0)=31.2%
- **FT|terça-feira**: mean_week=64.6 (IC95% 0.6..138.7), P(semana>0)=37.5%
- **FH|terça-feira**: mean_week=45.0 (IC95% -50.5..127.0), P(semana>0)=37.5%
- **FH|quinta-feira**: mean_week=32.6 (IC95% -54.1..133.7), P(semana>0)=31.2%
- **FH|sábado**: mean_week=29.0 (IC95% -13.5..78.5), P(semana>0)=43.8%
- **FT|sexta-feira**: mean_week=17.0 (IC95% 0.0..44.1), P(semana>0)=12.5%
- **FT|quinta-feira**: mean_week=10.0 (IC95% -110.0..130.1), P(semana>0)=12.5%
