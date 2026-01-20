## Runbook — Scoring & Auditoria (PAD / Excel / Modelos)

### Objetivo
Garantir que o score operacional (`rf_prob` no Excel → `ApostaLive.rf_prob` após merge) seja:
- **determinístico**
- **reprodutível**
- **auditável por aposta (bet_id)**
- com **match esperado: 100%** entre “score calculado” e “score gravado”.

---

## Fluxo Operacional (end-to-end)

### A) Geração do payload (PAD → CSV)
Para cada aposta (betID), o PAD salva um arquivo:
- **`payloads/payload%betID%.csv`**

Contrato mínimo do payload:
- 1ª coluna: `IDAposta` (betID)
- demais colunas: **mesmo schema já usado no treino** (headers preservados)

Schema dos modelos (weekdays e by_dow):
- Numéricas:
  - `Número de casas disponíveis no momento da aposta`
  - `Dif percent maior odd e segunda maior` (ou alias com `%`)
  - `Dif percent maior odd e odd mediana` (ou alias com `%`)
  - `Dif Odds RB E BIA` (ou alias com `&`)
  - `MinutesToMatchStart`
  - `TempoApostas.Tempo total bot`
- Categóricas:
  - `Subtipo da Aposta`
  - `Dia Semana Aposta (UTC)`
  - `Turno Aposta (UTC)`
  - `Casa aposta vencedora`

Formato numérico:
- o CLI aceita **ponto OU vírgula decimal** (parsing robusto).

---

### B) Execução do scoring (PAD → CLI → stdout)
O PAD chama o CLI correto para o dia:

- **Seg–Qua**: `score_logit_weekdays_cli.py`
  - modelos: `model_logit_segunda.joblib`, `model_logit_terca.joblib`, `model_logit_quarta.joblib`
  - score: `predict_proba` + `clip` por `calib_floor`

- **Sex–Dom**: `score_logit_by_dow_cli.py`
  - subset: `SexDom`
  - modelo: `model_logit_prod_SexDom.joblib` (ou `model_logit_SexDom.joblib`)
  - score: `proba_cal` (isotônico + floor, quando calibrador existir)

- **Qui**:
  - precisa ser **fixado e versionado** (SegQui vs SexDom), pois historicamente houve transição.

Saída do CLI (stdout):
```
proba,decision
0.123456,False
```

Observação importante (compatibilidade com PAD):
- `score_logit_weekdays_cli.py` imprime por padrão **apenas 1 linha** (último registro do CSV), para manter compatibilidade com fluxos PAD que assumem 2 linhas.
- Para auditoria (payload multi-linha), use `--stdout-all-rows`.

---

### C) Persistência no Excel (PAD)
O PAD grava o score retornado em:
- coluna `rf_prob` do Excel

Após a etapa de merge, o campo final esperado é:
- `ApostaLive.rf_prob` (na base consolidada)

---

## Log & Auditoria (trilha mínima)

### 1) JSONL do CLI (obrigatório para auditoria)
Os CLIs devem escrever em JSONL:
- weekdays: `scoring_weekdays.jsonl`
- by_dow: `scoring.jsonl`

Campos mínimos recomendados:
- `ts` (UTC)
- `bet_id` (IDAposta)
- `model_path` (+ `model_stat` se possível)
- `payload_hash`
- `proba` (e, quando existir, `proba_raw/proba_cal`)
- `decision`
- `cutoff`, `calib_floor`

Observação:
- logs antigos sem `bet_id` não permitem join determinístico com o Excel.

---

## Como auditar e depurar divergências

### Caso 1: “Excel não bate com stdout”
Sintoma:
- `rf_prob` no Excel ≠ `proba` do stdout (da mesma execução)

Verificações:
- parsing do stdout no PAD (delimitadores, linha usada, conversão para número)
- escrita do valor na coluna correta do Excel
- alguma etapa posterior sobrescrevendo a célula (macro, fórmulas, merge)

### Caso 2: “stdout não bate com reexecução do CLI”
Sintoma:
- reexecutar o CLI com o `payload%betID%.csv` produz outro valor vs stdout capturado

Verificações:
- modelo diferente (comparar `model_path`/`mtime/size`)
- script diferente (versionamento do CLI)
- payload diferente do esperado (colunas/aliases/unidades)

### Caso 3: “Excel bate com stdout, mas não bate com o estudo”
Sintoma:
- operacional consistente internamente, mas diverge do pipeline de estudo

Verificações:
- usar a mesma lógica de scoring no estudo (coluna correta por dia)
- garantir que o estudo usa `proba_cal` no weekend e `raw+clip` nos weekdays
- confirmar qual lógica foi usada na quinta-feira para o período analisado

---

## Auditoria de consistência “Excel ↔ logs” (por bet_id)
Quando o Excel está atualizado e o CLI está logando `bet_id`, a checagem mais forte é:

1) Ler o Excel `ResumoApostas_PBI_final_*.xlsx` (sheet `ResumoApostas (2)`), obter:
   - `ID Aposta` (bet_id)
   - `ApostaLive.rf_prob` (probabilidade gravada)
   - `BIA_ApostaUTC`

2) Ler o JSONL do dia:
   - weekdays: `scoring_weekdays.jsonl`
   - by_dow: `scoring.jsonl`

3) Fazer join por `bet_id` e calcular:
   - `match6 = round(rf_prob,6) == round(proba_log,6)`
   - `MAE = mean(|rf_prob - proba_log|)`

Regra prática:
- Esperado em produção (com versões alinhadas): **match6 ≈ 100%**.

---

## Regras de Ouro (para garantir 100%)
- **Sempre salvar payload por betID** (nunca sobrescrever apenas “o último payload”).
- **Sempre registrar stdout capturado** por betID (para prova de execução).
- **Sempre logar `bet_id` no JSONL** (join determinístico).
- **Congelar versões**:
  - logar `model_path` + `model_stat`
  - adicionar uma string `version` no script e manter no log
- **Quando houver diferença**, investigar primeiro:
  1) modelo/script usado
  2) payload e unidades
  3) parsing/escrita no Excel

