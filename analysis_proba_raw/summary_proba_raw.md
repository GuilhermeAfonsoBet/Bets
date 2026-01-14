## Portfólio reotimizado (proba_raw)
- Banca considerada: **USD 2,300**
- Max por aposta: **7.0%** (USD 161)
- Clip do score (calib_floor): **0.005**
- Treino: **2025-10-01..2025-12-31**; OOS: **>= 2026-01-01**

### Regras por dia (FT)
- **segunda-feira**: usar `proba_raw_segunda`, cutoff **0.19**, stake **7.0%**
- **terça-feira**: usar `proba_raw_terca`, cutoff **0.14**, stake **7.0%**
- **quarta-feira**: usar `proba_raw_quarta`, cutoff **0.06**, stake **7.0%**
- **quinta-feira**: usar `proba_raw_quarta`, cutoff **0.11**, stake **7.0%**

### Métricas agregadas (treino)
- **lucro mensal médio**: USD 26893.70
- **desvio-padrão mensal**: USD 6153.64
- **P(mês < 0)**: 0.0% (n=3)

### Métricas agregadas (OOS)
- **lucro mensal médio**: USD 2332.20
- **desvio-padrão mensal**: USD 0.00
- **P(mês < 0)**: 0.0% (n=1)

### Nota operacional (quinta-feira)
- O seu `score_logit_weekdays_cli.py` **não tem modelo de quinta**. Aqui eu avaliei quinta aplicando os modelos de seg/ter/qua e escolhendo o melhor no treino. Para operar quinta **sem mexer no Python**, a forma mais simples é o PAD preencher `Dia Semana Aposta (UTC)` como o dia-modelo escolhido (ex.: `quarta-feira`) nas quintas, ou então você passa a usar um CLI que tenha `model_logit_qui.joblib`/`model_logit_SegQui.joblib`.
