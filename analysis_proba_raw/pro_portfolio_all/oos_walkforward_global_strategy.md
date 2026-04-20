## OOS walk-forward (global) — estratégia completa
- Dataset: `/workspace/analysis_proba_raw/scored_dedup_proba_raw_all.csv`
- Semanas totais no dataset: **26**; semanas testadas OOS (WF): **16** (a partir de 10 semanas globais de treino; por-segmento exige >= 6)
- **Modo global**: aplica um fator α (0..1) multiplicando todos os stakes para satisfazer constraints do portfólio agregado no treino de cada passo.

### Performance OOS (cap2) — portfólio agregado
- **PnL semanal médio (bootstrap IC95%)**: **USD 245.5** (IC95% -72.3..611.4)
- **Desvio padrão semanal**: USD 717.6
- **P(semana < 0)**: 50.0%

### Performance OOS (cap2) — excluindo semanas sem trades (stake=0)
- **PnL semanal médio (bootstrap IC95%)**: **USD 245.5** (IC95% -75.9..602.0)
- **Desvio padrão semanal**: USD 717.6
- **P(semana < 0)**: 50.0%
- **ROI on stake agregado (ponderado)**: 0.0526

### Ajuste de stake global (α)
- α médio=0.924; p10=0.774; p50=0.996; p90=1.000; P(α<1)=50.0%

### Estabilidade OOS da decisão por segmento (frequência de ativação)
- Arquivo regras: `analysis_proba_raw/pro_portfolio_all/oos_walkforward_global_selected_rules.csv`

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
