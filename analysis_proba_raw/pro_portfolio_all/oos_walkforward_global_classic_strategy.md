## OOS walk-forward (global_classic) — estratégia completa
- Dataset: `/workspace/analysis_proba_raw/scored_dedup_proba_raw_all.csv`
- Semanas totais no dataset: **26**; semanas testadas OOS (WF): **16** (a partir de 10 semanas globais de treino; por-segmento exige >= 6)
- **Modo global**: aplica um fator α (0..1) multiplicando todos os stakes para satisfazer constraints do portfólio agregado no treino de cada passo.

### Performance OOS (cap2) — portfólio agregado
- **PnL semanal médio (bootstrap IC95%)**: **USD 145.8** (IC95% -111.8..434.7)
- **Desvio padrão semanal**: USD 578.8
- **P(semana < 0)**: 43.8%

### Performance OOS (cap2) — excluindo semanas sem trades (stake=0)
- **PnL semanal médio (bootstrap IC95%)**: **USD 155.6** (IC95% -119.0..463.9)
- **Desvio padrão semanal**: USD 597.8
- **P(semana < 0)**: 46.7%
- **ROI on stake agregado (ponderado)**: 0.0294

### Risco no OOS (teste) — portfólio agregado
- p80(soma stakes/dia) = USD 1968 (limite=USD 1610)
- VaR10%(PnL diário) = USD -255.7 (limite >= USD -575)
- P(PnL diário <= -25% banca) = 1.6% (limite <= 10%)

### Ajuste de stake global (α)
- α médio=0.852; p10=0.665; p50=0.890; p90=1.000; P(α<1)=62.5%

### Estabilidade OOS da decisão por segmento (frequência de ativação)
- Arquivo regras: `analysis_proba_raw/pro_portfolio_all/oos_walkforward_global_classic_selected_rules.csv`

- **FH | domingo**: active_rate=12.5%, ok_rate=12.5%, stake_frac_médio=0.38%, cutoff_médio=0.89
- **FH | quarta-feira**: active_rate=56.2%, ok_rate=56.2%, stake_frac_médio=2.56%, cutoff_médio=0.56
- **FH | quinta-feira**: active_rate=75.0%, ok_rate=75.0%, stake_frac_médio=3.38%, cutoff_médio=0.42
- **FH | segunda-feira**: active_rate=0.0%, ok_rate=0.0%, stake_frac_médio=0.00%, cutoff_médio=1.00
- **FH | sexta-feira**: active_rate=31.2%, ok_rate=31.2%, stake_frac_médio=0.31%, cutoff_médio=0.71
- **FH | sábado**: active_rate=87.5%, ok_rate=87.5%, stake_frac_médio=1.50%, cutoff_médio=0.39
- **FH | terça-feira**: active_rate=50.0%, ok_rate=50.0%, stake_frac_médio=2.94%, cutoff_médio=0.68
- **FT | domingo**: active_rate=50.0%, ok_rate=50.0%, stake_frac_médio=2.19%, cutoff_médio=0.63
- **FT | quarta-feira**: active_rate=75.0%, ok_rate=75.0%, stake_frac_médio=4.50%, cutoff_médio=0.56
- **FT | quinta-feira**: active_rate=31.2%, ok_rate=31.2%, stake_frac_médio=0.50%, cutoff_médio=0.77
- **FT | segunda-feira**: active_rate=56.2%, ok_rate=56.2%, stake_frac_médio=1.69%, cutoff_médio=0.56
- **FT | sexta-feira**: active_rate=0.0%, ok_rate=0.0%, stake_frac_médio=0.00%, cutoff_médio=1.00
- **FT | sábado**: active_rate=93.8%, ok_rate=93.8%, stake_frac_médio=4.81%, cutoff_médio=0.53
- **FT | terça-feira**: active_rate=68.8%, ok_rate=68.8%, stake_frac_médio=2.19%, cutoff_médio=0.47

### Segmentos mais estáveis no OOS (por lucro semanal)
- CSV: `analysis_proba_raw/pro_portfolio_all/oos_walkforward_global_classic_segment_stability.csv`
- **FT|sábado**: mean_week=63.9 (IC95% -16.3..160.8), P(semana>0)=31.2%
- **FT|terça-feira**: mean_week=54.4 (IC95% 2.6..112.0), P(semana>0)=43.8%
- **FT|segunda-feira**: mean_week=39.2 (IC95% -20.3..96.2), P(semana>0)=43.8%
- **FT|quarta-feira**: mean_week=36.9 (IC95% -53.5..128.0), P(semana>0)=37.5%
- **FH|sábado**: mean_week=21.1 (IC95% -10.5..56.5), P(semana>0)=43.8%
- **FH|quinta-feira**: mean_week=16.8 (IC95% -46.4..99.3), P(semana>0)=31.2%
- **FH|sexta-feira**: mean_week=-6.5 (IC95% -39.2..28.9), P(semana>0)=6.2%
- **FT|quinta-feira**: mean_week=-9.2 (IC95% -98.3..89.4), P(semana>0)=6.2%
