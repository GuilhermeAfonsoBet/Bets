## Portfólio reotimizado (proba_raw)
- Banca considerada: **USD 2,300**
- Max por aposta: **7.0%** (USD 161)
- Clip do score (calib_floor): **0.005**
- Treino: **2025-10-01..2025-12-31**; OOS: **>= 2026-01-01**

### Regras por dia (FT)
- **segunda-feira**: usar `proba_raw_segunda`, cutoff **0.19**, stake **7.0%**
- **terça-feira**: usar `proba_raw_terca`, cutoff **0.14**, stake **7.0%**
- **quarta-feira**: usar `proba_raw_quarta`, cutoff **0.06**, stake **7.0%**
- **quinta-feira**: usar `proba_raw_segqui`, cutoff **0.05**, stake **7.0%**

### Métricas agregadas (treino)
- **lucro mensal médio**: USD 26303.36
- **desvio-padrão mensal**: USD 6210.67
- **P(mês < 0)**: 0.0% (n=3)

### Métricas agregadas (OOS)
- **lucro mensal médio**: USD 2332.20
- **desvio-padrão mensal**: USD 0.00
- **P(mês < 0)**: 0.0% (n=1)

### Nota operacional (quinta-feira)
- Para quinta-feira, este portfólio usa o modelo **`model_logit_SegQui.joblib`** (score `proba_raw_segqui`). Isso permite operar quinta sem precisar "fingir" que é quarta no payload.
