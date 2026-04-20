## Portfólio mesa profissional — FT e FH (todos os dias)
- Banca: USD 2,300; max por aposta: 7.0%
- Exposição diária (constraint): p80 do somatório de stakes no dia <= 70% da banca.
- Risco diário (constraint): P(PnL_dia <= -25% banca) <= 10% (via VaR10%).

### FT
- **segunda-feira**: `proba_raw_segunda` ≥ **0.21**, stake **3.0%** (p80 exp dia ~USD 1604; wf_mean 242)
- **terça-feira**: `proba_raw_terca` ≥ **0.25**, stake **3.0%** (p80 exp dia ~USD 1532; wf_mean 226)
- **quarta-feira**: `proba_raw_quarta` ≥ **0.31**, stake **3.0%** (p80 exp dia ~USD 1318; wf_mean 117)
- **quinta-feira**: (removido) — no_candidate_stage2
- **sexta-feira**: `proba_raw_sexdom` ≥ **0.31**, stake **2.0%** (p80 exp dia ~USD 1518; wf_mean 232)
- **sábado**: `proba_raw_sexdom` ≥ **0.57**, stake **7.0%** (p80 exp dia ~USD 1070; wf_mean 296)
- **domingo**: `proba_raw_sexdom` ≥ **0.49**, stake **4.0%** (p80 exp dia ~USD 92; wf_mean 34)

### FH
- **segunda-feira**: (removido) — no_candidate_stage2
- **terça-feira**: `proba_raw_terca` ≥ **0.67**, stake **7.0%** (p80 exp dia ~USD 467; wf_mean 109)
- **quarta-feira**: (removido) — no_candidate_stage2
- **quinta-feira**: (removido) — no_candidate_stage2
- **sexta-feira**: `proba_raw_sexdom` ≥ **0.09**, stake **1.0%** (p80 exp dia ~USD 1228; wf_mean 51)
- **sábado**: `proba_raw_sexdom` ≥ **0.23**, stake **1.0%** (p80 exp dia ~USD 1454; wf_mean 151)
- **domingo**: `proba_raw_sexdom` ≥ **0.41**, stake **5.0%** (p80 exp dia ~USD 460; wf_mean 6)
