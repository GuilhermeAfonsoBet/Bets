## Estabilidade robusta (semanal) — portfólio proba_raw\n- Banca usada p/ sizing: **USD 2,300**; max 7% já embutido na regra\n- Treino: **2025-10-01..2025-12-31**; OOS: **>= 2026-01-01**\n- Bootstrap: **50,000** amostras (semana com reposição); drawdown paths: **10,000**\n
### Resultados por cenário (treino)

### Cenário: **raw**
- Semanas no treino: **14**; PnL semanal médio: **USD 5636**; std: **USD 3118**; P(semana<0): **0.0%**
- **52 semanas (bootstrap)**: média 293182, p05 258127, VaR5% 258127, CVaR5% 249762, P(<0) 0.0%

### Cenário: **winsor_p995**
- Semanas no treino: **14**; PnL semanal médio: **USD 5580**; std: **USD 3066**; P(semana<0): **0.0%**
- **52 semanas (bootstrap)**: média 290153, p05 255435, VaR5% 255435, CVaR5% 247156, P(<0) 0.0%

### Cenário: **cap2**
- Semanas no treino: **14**; PnL semanal médio: **USD 1199**; std: **USD 1429**; P(semana<0): **21.4%**
- **52 semanas (bootstrap)**: média 62317, p05 46184, VaR5% 46184, CVaR5% 42217, P(<0) 0.0%

### Cenário: **cap1**
- Semanas no treino: **14**; PnL semanal médio: **USD 593**; std: **USD 1372**; P(semana<0): **35.7%**
- **52 semanas (bootstrap)**: média 30831, p05 15175, VaR5% 15175, CVaR5% 10983, P(<0) 0.1%

### Drawdown (paths semanais, 52 semanas) — resumo
- **raw**: MaxDD USD p50=0, p95=0, p99=0; MaxDD% p95=0.0%; P(ruína end<=0)=0.00%
- **winsor_p995**: MaxDD USD p50=0, p95=0, p99=0; MaxDD% p95=0.0%; P(ruína end<=0)=0.00%
- **cap2**: MaxDD USD p50=1558, p95=3048, p99=3778; MaxDD% p95=5.4%; P(ruína end<=0)=0.00%
- **cap1**: MaxDD USD p50=3790, p95=7132, p99=9350; MaxDD% p95=32.4%; P(ruína end<=0)=0.05%
\n### Nota sobre o lucro alto vs análises anteriores\n- Se o cenário **raw** mostrar lucro irreal, os cenários **cap2/cap1** te dizem o quanto isso depende de outliers/erros de payout. Para uma mesa profissional, eu recomendaria usar **cap1/cap2** como stress test obrigatório e só aceitar o edge se ele sobreviver a isso.
