## OOS walk-forward (global_bayes) — estratégia completa
- Dataset: `/workspace/analysis_proba_raw/scored_dedup_proba_raw_all.csv`
- Semanas totais no dataset: **26**; semanas testadas OOS (WF): **16** (a partir de 10 semanas globais de treino; por-segmento exige >= 6)
- **Modo global**: aplica um fator α (0..1) multiplicando todos os stakes para satisfazer constraints do portfólio agregado no treino de cada passo.

### Performance OOS (cap2) — portfólio agregado
- **PnL semanal médio (bootstrap IC95%)**: **USD 232.1** (IC95% 2.0..481.4)
- **Desvio padrão semanal**: USD 506.2
- **P(semana < 0)**: 31.2%

### Performance OOS (cap2) — excluindo semanas sem trades (stake=0)
- **PnL semanal médio (bootstrap IC95%)**: **USD 265.3** (IC95% 9.9..548.2)
- **Desvio padrão semanal**: USD 535.0
- **P(semana < 0)**: 35.7%
- **ROI on stake agregado (ponderado)**: 0.0714

### Risco no OOS (teste) — portfólio agregado
- p80(soma stakes/dia) = USD 1776 (limite=USD 1610)
- VaR10%(PnL diário) = USD -166.0 (limite >= USD -575)
- P(PnL diário <= -25% banca) = 0.0% (limite <= 10%)

### Ajuste de stake global (α)
- α médio=0.909; p10=0.759; p50=0.998; p90=1.000; P(α<1)=50.0%

### Estabilidade OOS da decisão por segmento (frequência de ativação)
- Arquivo regras: `analysis_proba_raw/pro_portfolio_all/oos_walkforward_global_bayes_selected_rules.csv`

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
- CSV: `analysis_proba_raw/pro_portfolio_all/oos_walkforward_global_bayes_segment_stability.csv`
- **FT|segunda-feira**: mean_week=60.2 (IC95% -9.2..133.0), P(semana>0)=50.0%
- **FT|sábado**: mean_week=59.2 (IC95% 4.0..128.8), P(semana>0)=50.0%
- **FT|terça-feira**: mean_week=37.6 (IC95% -8.8..93.2), P(semana>0)=37.5%
- **FH|sábado**: mean_week=37.2 (IC95% 5.5..74.3), P(semana>0)=50.0%
- **FH|quinta-feira**: mean_week=27.9 (IC95% -36.5..96.3), P(semana>0)=31.2%
- **FT|quarta-feira**: mean_week=18.8 (IC95% -86.3..127.6), P(semana>0)=37.5%
- **FH|terça-feira**: mean_week=12.9 (IC95% -25.1..51.8), P(semana>0)=25.0%
- **FT|domingo**: mean_week=-0.3 (IC95% -4.3..3.5), P(semana>0)=6.2%
