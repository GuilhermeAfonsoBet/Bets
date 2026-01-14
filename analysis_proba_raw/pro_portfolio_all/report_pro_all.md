## Portfólio mesa profissional — FT e FH (todos os dias)
- Banca: USD 2,300; max por aposta: 7.0%
- Exposição diária (métrica): reportamos p95 da exposição por dia (stake somado); não é hard-constraint. A restrição diária ativa é **P(PnL_dia <= -25% banca) <= 10%** (via VaR10%).

### FT
- **segunda-feira**: `proba_raw_segunda` ≥ **0.19**, stake **7.0%** (p95 exp dia ~USD 3619; wf_mean 466)
- **terça-feira**: `proba_raw_terca` ≥ **0.15**, stake **5.0%** (p95 exp dia ~USD 3574; wf_mean 290)
- **quarta-feira**: `proba_raw_quarta` ≥ **0.05**, stake **3.0%** (p95 exp dia ~USD 2927; wf_mean 117)
- **quinta-feira**: (removido) — no_candidate_stage2
- **sexta-feira**: (removido) — no_candidate_stage2
- **sábado**: `proba_raw_sexdom` ≥ **0.41**, stake **7.0%** (p95 exp dia ~USD 5590; wf_mean 296)
- **domingo**: `proba_raw_sexdom` ≥ **0.49**, stake **4.0%** (p95 exp dia ~USD 92; wf_mean 34)

### FH
- **segunda-feira**: (removido) — no_candidate_stage2
- **terça-feira**: `proba_raw_terca` ≥ **0.67**, stake **7.0%** (p95 exp dia ~USD 636; wf_mean 109)
- **quarta-feira**: (removido) — no_candidate_stage2
- **quinta-feira**: (removido) — no_candidate_stage2
- **sexta-feira**: `proba_raw_sexdom` ≥ **0.09**, stake **1.0%** (p95 exp dia ~USD 1725; wf_mean 51)
- **sábado**: `proba_raw_sexdom` ≥ **0.21**, stake **7.0%** (p95 exp dia ~USD 8964; wf_mean 793)
- **domingo**: `proba_raw_sexdom` ≥ **0.41**, stake **5.0%** (p95 exp dia ~USD 546; wf_mean 6)
