## Pitch técnico (1 página) — Tese de edge em microestrutura + execução (BetinAsia / b808)

### 1) Oportunidade (o “trade”)
- **Mercados**: odds de futebol (principalmente pre‑match), com coleta por WebSocket + auditoria de execução via betslip.
- **Sinais candidatos**:
  - **Reversão após queda forte**: a odd cai por uma sequência de atualizações e **reverte para alta** (H3B `down→up`). Intuição: overshoot/pressão de fluxo e correção.
  - **Continuação de curtíssimo prazo**: após o gatilho, há drift em segundos; entradas com execução rápida capturam melhor esse regime.
- **O que medimos**:
  - **CLV (closing line value)** como métrica de qualidade de entrada (principalmente pre‑match).
  - ROI/P&L como métrica de monetização (dependente de cobertura de placar e sizing).

### 2) Por que pode existir edge (literatura / mecanismo)
- **Microestrutura / price discovery**: a informação chega/propaga com atraso; o preço “escorre” por alguns segundos/minutos (underreaction).
- **Persistência de fluxo**: ordem/pressão de um lado costuma ter autocorrelação no curtíssimo prazo.
- **Overreaction + correction**: movimentos rápidos (queda forte) podem exceder o “valor” e corrigir quando liquidez/informação se reequilibra.

### 3) Produto / execução (moat operacional)
- **Coletor contínuo**: captura mercados e persiste histórico (`best_odds_history`) para closing e séries temporais.
- **Auditoria de execução**: valida na prática o preço do betslip vs WS e registra latências e falhas.
- **Qualidade de dados**: filtros de betslip “confiável” e *quality gates* de closing (odds recentes pré‑kickoff).
- **Resiliência**: auto‑recovery contra ciclos “sem odds” e contra timeouts (coleta e SAVE).

### 4) Evidência atual (como um investidor deveria ler)
- **CLV pre‑match** é o núcleo: mede qualidade de entrada sem precisar esperar liquidação completa.
- Resultados sugerem que **subcoortes condicionais** (reversão + execução rápida / confirmação) concentram melhor CLV do que o universo.
- **ROI** deve ser tratado como *second‑order* no estágio atual (cobertura de resultados, ruído e sizing ainda “proxy”).

### 5) Principais riscos (e como mitigamos)
- **Risco de mensuração** (coverage de odds/closing e de resultados): reportar cobertura e condicionar análises a dados presentes.
- **Overfit / múltiplos testes**: validação walk‑forward e estabilidade a thresholds (streak/magnitude/Δt).
- **Execução**: suspensões, limits, slippage e falhas. Monitorar fill‑rate, latência, taxa de FAIL, e kill‑switch.
- **Risco de cauda (Lay)**: sizing por **liability**, caps, e governança por p95/p99/ES.

### 6) Plano de validação (marcos objetivos para captação)
- **Fase A (2–4 semanas)**: paper/live‑sim com regras ex‑ante, logs completos e métricas operacionais (SLA).
- **Fase B (4–8 semanas)**: capital pequeno, sizing conservador (p99/ES), metas de estabilidade.
- **Critérios de “go/no‑go”** (exemplos):
  - CLV pre‑match positivo e estável fora da amostra;
  - drawdown e cauda dentro do previsto;
  - pipeline com alta disponibilidade (ex.: >95% do tempo coletando, auditoria contínua).

### 7) O que pedimos (estrutura típica)
- **Capital em tranches** condicionado a métricas (ex.: tranche 1 para validação; tranche 2 para escala).
- **Uso**: infraestrutura/robustez operacional, monitoramento, e execução controlada (não “apostar grande cedo”).

