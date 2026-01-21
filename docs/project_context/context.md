## Contexto do Projeto (resumo executivo)

### Objetivo
Construir e operar uma estratégia de apostas (“Mesa Profissional”) baseada em um portfólio otimizado via walk-forward, cuja execução depende criticamente de um **score operacional** confiável por aposta (coluna `ApostaLive.rf_prob` após merge).

O requisito operacional é **aderência auditável e determinística do score** (ideal: 100% de match do score gravado no Excel/base vs score calculado pelos modelos).

### Por que isso é crítico
O portfólio/estratégia usa thresholds de score por segmento (dia da semana × tipo de aposta). Se o score gravado diverge do score real do modelo, a execução pode:
- **ativar/desativar apostas erradas**
- **quebrar a validação OOS**
- **invalidar o portfólio otimizado**

### Componentes principais
- **Modelos e CLIs de scoring**:
  - `score_logit_weekdays_cli.py` (Seg–Qua): modelos por dia (`model_logit_segunda.joblib`, `model_logit_terca.joblib`, `model_logit_quarta.joblib`), com `calib_floor` (clip).
  - `score_logit_by_dow_cli.py` (SegQui vs SexDom): modelo por subset com calibração isotônica (usa `clv_calib_*.json` quando existir).
    - Para SexDom, o operacional usa frequentemente `model_logit_prod_SexDom.joblib`.
- **Excel/PAD (Power Automate Desktop)**:
  - o PAD escreve o score em uma coluna (`rf_prob`) e depois ocorre o merge para virar `ApostaLive.rf_prob`.
  - o PAD salva payloads do scoring e captura o stdout do CLI.
- **Macro (Excel)**:
  - Documento `Macro principal.docx` contém VBA para preencher lacunas em colunas (inclui coluna I).
  - A intenção é replicar o score dentro do mesmo `IDAposta` (coluna A) quando só a primeira linha recebe o score.
  - Auditoria em snapshot mostrou `rf_prob` consistente por `ID Aposta` (sem evidência de vazamento entre IDs), mas a macro antiga não valida ID (risco teórico caso haja quebra de contiguidade).

### Achados importantes já confirmados
- `ApostaLive.rf_prob` **não batia** com score reconstruído em alguns dias no histórico antigo (principalmente Ter/Qua).
- Parte do ruído veio de falta de trilha auditável (JSONL antigo sem `bet_id`/`payload_hash`).
- Correções aplicadas no repo:
  - Parsing numérico robusto (ponto/vírgula) no `score_logit_weekdays_cli.py`.
  - Inclusão de `bet_id` no log JSONL dos CLIs (weekdays e by_dow).
  - `score_logit_by_dow_cli.py` agora resolve `model_logit_{subset}.joblib` **ou** `model_logit_prod_{subset}.joblib`.
  - Compat patch sklearn (`_fill_dtype`) também em `score_logit_by_dow_cli.py`.
- Auditoria “artefatos RPA” (payload%betID% + stdout.csv):
  - Para os exemplos enviados (sábado / SexDom), o match entre **stdout capturado** e **reexecução do CLI** foi **100%** para todos os casos com stdout disponível.

### Atualizações recentes (Jan/2026)
- **Consistência real “Excel ↔ logs” (últimos dias)**:
  - Com a planilha `ResumoApostas_PBI_final_20.01.2026.xlsx` e os logs atualizados (`scoring_weekdays.jsonl`),
    foi possível fazer join por `bet_id` e validar que:
    - **2026-01-19 (segunda)**: match6 = **100%** (Excel `ApostaLive.rf_prob` vs `scoring_weekdays.jsonl`)
    - **2026-01-20 (terça)**: match6 = **100%**
  - Isso confirma que, quando o Excel está preenchido e o log tem `bet_id`, o processo é auditável e consistente.

- **Compatibilidade do stdout com PAD (weekdays)**:
  - `score_logit_weekdays_cli.py` voltou a emitir por padrão exatamente o formato legado (2 linhas):
    `proba,decision` + **apenas a última linha do payload**.
  - Para auditoria, existe `--stdout-all-rows` para imprimir todas as linhas do CSV.

- **Alinhamento “pipeline ↔ operacional” (feature `Dif Odds RB & BIA`)**:
  - `BetinAsia.got price` é **ex-post** e não pode ser usado para decidir no operacional.
  - O pipeline de estudo foi ajustado para calcular `Dif Odds RB & BIA` de forma **operacional-like** (op_sim), usando:
    - `Odd_RB := RebelBetting.Odds` (fallback `Odd Indicada no RB`)
    - `Odd_BIA := ApostaLive.Aux1 - maior odd / 1000` (quando `Aux1 > 10`, assume milésimos)
  - Isso garante que o score do estudo não dependa de informação que só existe após a execução.

- **Execução (slippage) — diagnóstico e impacto no portfólio**:
  - Medimos o slippage como \(\\Delta odd = got\\_price - Aux1/1000\\).
  - No dataset, `got price` aparece em ~16% das apostas (ex-post), então a análise de execução tem cobertura parcial.
  - No portfólio OOS (`global_bayes_roll12_robust_p10_p70`), o slippage histórico estimado adicionou aproximadamente:
    - **ΔPnL ≈ +USD 54** no período OOS (cap2)
    - **ΔROI/$ ≈ +0,00216** sobre o stake do portfólio
    - Cobertura por stake (apostas com got+aux): **~22%**
  - Leitura: o efeito foi pequeno e levemente positivo; o slippage deve ser monitorado como risco operacional, não como feature de decisão.

- **Evolução do motor WF (experimentos de escala e stake máximo)**:
  - `evaluate_oos_walkforward_strategy.py` ganhou:
    - `--bankroll` e `--out-suffix` para rodar walk-forward reotimizando por faixa de banca sem sobrescrever artefatos.
    - modo experimental `cap_bin` (house_cap como dimensão do segmento) para testar “stake máximo como feature de decisão”.

### Como o operacional deve funcionar (contrato)
1) PAD gera `payload%betID%.csv` com header e 1 linha.
2) PAD executa o CLI correto para o dia:
   - Seg–Qua → `score_logit_weekdays_cli.py`
   - Sex–Dom → `score_logit_by_dow_cli.py` (SexDom calibrado)
   - Qui → definir e estabilizar (SegQui ou SexDom) e versionar a escolha.
3) CLI imprime no stdout:
   - Header `proba,decision`
   - 1 linha com `proba` e `decision`
4) PAD grava `rf_prob` no Excel (e depois isso vira `ApostaLive.rf_prob` pós-merge).
5) CLI escreve log JSONL com pelo menos: `ts`, `bet_id`, `model_path`, `payload_hash`, `proba`, `decision`.

### Estado atual / Próximos passos imediatos
- **Objetivo de curto prazo**: obter 100% de aderência em produção (principalmente Ter/Qua, mas também outros dias).
- **Coleta**:
  - manter payload por betID (`payloads/payload%betID%.csv`)
  - registrar stdout (ex.: `stdout.csv`)
  - manter `scoring_weekdays.jsonl` e `scoring.jsonl` com `bet_id` (com as versões novas dos CLIs).
- **Diagnóstico**:
  - quando ocorrer discrepância, usar `bet_id` para:
    - reproduzir o score a partir do payload,
    - validar se o Excel gravou o mesmo valor do stdout,
    - validar se o modelo/script usado é o esperado (via `model_path` no JSONL).

- **Recomendação prática**:
  - Tratar `ApostaLive.rf_prob` como “valor auditável” apenas quando:
    - estiver em [0,1],
    - houver `bet_id` no log do CLI no mesmo dia,
    - e o join por `bet_id` bater em 6 casas (match6 ~ 100%).

