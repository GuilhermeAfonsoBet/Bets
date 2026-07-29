# H3BUP Fase 2B Soak — Sumário executivo (2026-07-29)

## Classificação: E2E_SOAK_SUFFICIENT

60. Período: 2026-07-29T15:53:09+00:00 → 2026-07-29T20:07:33.532169+00:00
61. trace_ids: 2056
62. requests: 63
63. executor_received: 63
64. dryruns: 63
65. final_gates: 57
66. place_started: 10
67. place_finished: 10
68. LIVE_OK: 10
69. coverage WS→LIVE_OK: 0.49% (dos traces; orders LIVE_OK rastreados=10)
70. mediana WS→LIVE_OK: 9149.886131286621
71. p95 WS→LIVE_OK: 12742.152619361876
72. mediana audit→request: 3182.194948196411
73. mediana request→finished: 2709.3288898468018
74. mediana dry-run: 1397.424813
75. mediana place: 4547.489825
76. etapa dominante: dry-run / espera bridge (audit→request segundos)
77–83. dropped/write/missing/corrupt/dup/ordering/negativos: ver CSVs; clock_skew=64 ordering=64
84. overhead detect→audit mediana ~13.867324 ms
85–86. CPU/mem estáveis (serviços active)
87. telemetria bloqueou execução? **Não**
88. suficiência? **Sim** (10 LIVE_OK rastreados)
89. E2E_SOAK_SUFFICIENT
90. Rollback necessário? **Não**
