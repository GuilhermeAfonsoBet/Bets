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

- **Auditoria operacional (payload → CLI → log) — 21/01 e 22/01/2026**:
  - Foram adicionados ao repo (branch `main`) os ZIPs:
    - `payloads 21.01-22.01.zip` (amostra de payloads)
    - `payloads_22.01.26.zip` (payloads completos de 22/01/26)
  - Resultados confirmados:
    - **21/01/2026 (quarta / weekdays)**: na interseção disponível (n=12), `payload → score_logit_weekdays_cli.py` bateu com `scoring_weekdays.jsonl` em **100% (match@6dec)** e `payload_hash` bateu **100%**.
    - **22/01/2026 (quinta / SegQui)**: (n=40) `payload → score_logit_by_dow_cli.py` bateu com `scoring.jsonl` em **100%** para:
      - `proba_cal` (calibrado; equivalente ao `proba_cal_segqui` do portfólio)
      - `proba_raw` (diagnóstico adicional via `--skip-calib`)
      - `decision`
  - Artefatos gerados:
    - `analysis_proba_raw/pro_portfolio_all/audit_payload_cli_vs_log_2026-01-21_sample.csv`
    - `analysis_proba_raw/pro_portfolio_all/audit_payload_cli_vs_log_2026-01-22_segqui.csv`

- **Minipipeline (payload → “pipeline do estudo”) — 21/01 e 22/01/2026**:
  - Como o dataset completo do estudo (`scored_dedup_proba_raw_all.csv`) está limitado até **2026-01-20**, foi criado um dataset mínimo a partir dos payloads para auditar alinhamento do “pipeline” nesses dias.
  - Script reprodutível:
    - `build_minipipeline_payloads_2026_01_21_22.py`
  - Saída:
    - `analysis_proba_raw/pro_portfolio_all/minipipeline_payload_scores_2026-01-21_22.csv`
  - Resultado: `match@6dec` **100%** para 21/01 (n=12) e 22/01 (n=40), comparando com os logs.

- **Slippage (seção 3.5 do relatório) — semana 19–25/01/2026**:
  - O ΔPnL cap2 semanal alto (**+47,56**) está **correto** e veio de **1 única aposta coberta** (stake coberto=92) onde houve grande diferença entre `got price` e `Aux1/1000`.
  - O relatório `Relatorio_BayesGlobal_Mesa_Profissional_2026-01-22.pdf` foi atualizado para incluir na tabela semanal a métrica **ΔPnL / PnL** (impacto percentual do slippage sobre o PnL cap2 semanal do OOS).

- **Evolução do motor WF (experimentos de escala e stake máximo)**:
  - `evaluate_oos_walkforward_strategy.py` ganhou:
    - `--bankroll` e `--out-suffix` para rodar walk-forward reotimizando por faixa de banca sem sobrescrever artefatos.
    - modo experimental `cap_bin` (house_cap como dimensão do segmento) para testar “stake máximo como feature de decisão”.

- **Região (localidade) — estudo ex-ante (Jan/2026)**:
  - Contexto: `BetinAsia.event info competition name` é **ex-post** (só existe em parte das apostas executadas). Para estudar região sem vazamento, precisamos de um proxy **ex-ante**.
  - Abordagem usada:
    - Treinamos um classificador de região usando como “rótulo” uma heurística aplicada ao `competition name` (quando existe),
      mas usando como features **apenas** texto ex-ante: `Evento`, `Time Home`, `Time Away`, `RebelBetting.Bookmaker`.
    - Geramos `region_exante_pred.csv` com `region_pred` e uma confiança `region_pred_pmax` e aplicamos limiar (ex.: pmax>=0.70) para evitar ruído.
  - Resultado OOS (gating por região, sem mexer no score):
    - `oos_walkforward_region_gating_exantepred_summary.csv`: região-gating ficou **ligeiramente melhor** que o baseline no OOS completo, quando usamos o modelo ex-ante + limiar de confiança.
  - Resultado exploratório (modo fast, segmentando portfólio por DoW×Tipo×Região):
    - Em “fast mode” (últimas 8 semanas + Bayes barato + mínimos de evidência), a segmentação completa por região foi **instável/sensível ao limiar**:
      - com limiar alto (pmax>=0.85), a região vira “desconhecida” e o resultado colapsa para o baseline;
      - com limiar intermediário (pmax>=0.75) e thresholds mais altos, o portfólio por região ficou **pior que o baseline**, apesar de ainda positivo.
  - Leitura: região parece mais promissora como **gating/filtragem** (ou feature no score) do que como dimensão direta do portfólio (alto risco de overfit/escassez por segmento).

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

---

## Atualizações adicionais (25–27/01/2026) — relatórios e início de operação real

### Correções no `Relatorio_BayesGlobal_Estrutural_2026-01-25.pdf`
- **ROI/$ (forecast vs realizado) aparecendo igual**:
  - Diagnóstico: em parte por **arredondamento** (4 casas) e em parte por uma tabela “ex-post” que **reconcilia médias por construção**.
  - Ajuste: o relatório passou a imprimir ROI/$ com **5 casas** e incluiu explicitamente **Δ ROI/$ (Forecast on-line − Realizado teórico)**.
  - O bloco “ex-post” foi reclassificado como **diagnóstico/reconciliação** para não ser interpretado como forecast.

- **Seção 3.3 (Forecast máx) com sobreposição de colunas**:
  - Ajuste de layout (wrap/`Paragraph`, `colWidths`, fonte menor) e remoção do ROI/$ “ex-post corrigido” no máx (evita igualdade mecânica com realizado).

### Mudança relevante no estudo de escala de banca (3.4) — análise profunda + verificação
- A curva de banca (`stat_tests_bankroll_scaling.csv`) é **muito sensível à última semana OOS** quando ela tem volume/house_cap altos.
- Foi adicionada uma análise de sensibilidade **excluindo a última semana**:
  - `analysis_proba_raw/pro_portfolio_all/stat_tests_bankroll_scaling_excl_lastweek.csv`
- Leitura prática:
  - “Sem degradação” pode aparecer quando a **última semana** puxa a curva para cima.
  - Sem a última semana, a curva pode voltar a mostrar **degradação** em bancas maiores.

### Split do relatório em “Semanal” vs “Estrutural” (qualidade de leitura)
- O split foi ajustado para:
  - **Estrutural** começar na **Seção 3** e incluir **capa/introdução própria** (evita PDF começando no meio de tabela).
  - **Semanal** conter seções operacionais (2.*) e também ter capa/introdução própria.

### Nova dimensão de métricas: “realizado efetivo” (execução) — desde 24/01/2026
- Como a operação real começou em **24/01/2026**, foi criada uma camada de análise com:
  - **fill-rate** por número de apostas (selecionadas vs executadas)
  - **fill-rate por stake** (stake pretendido vs stake executado)
  - **PnL efetivo** e **ROI/$ efetivo** (usando stake/odds realizados)
  - **ΔPnL total** (efetivo − teórico) e um ΔPnL “em stake executado” (isolando sizing)
- Artefato semanal gerado:
  - `analysis_proba_raw/pro_portfolio_all/effective_realized_since_2026_01_24_weekly.csv`

