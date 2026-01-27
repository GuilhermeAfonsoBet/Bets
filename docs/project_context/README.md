## Documentação — Contexto e Auditoria

Esta pasta consolida o **contexto do projeto** e o **runbook de scoring/auditoria** para continuidade do trabalho sem dependência do histórico do chat.

- `context.md`: visão geral do projeto, decisões, status e próximos passos.
- `scoring_audit.md`: runbook operacional do scoring (payload → CLI → stdout → Excel/merge), formatos de dados e trilhas de auditoria.
- `Trabalhos_Chat_YYYY-MM-DD.pdf`: memória executiva do trabalho no chat (gerado automaticamente).

Artefatos úteis (fora desta pasta, mas referenciados na documentação):
- `build_minipipeline_payloads_2026_01_21_22.py`: gera um dataset mínimo (payload→pipeline) para auditar alinhamento com logs operacionais em dias recentes.
- `analysis_proba_raw/pro_portfolio_all/minipipeline_payload_scores_2026-01-21_22.csv`: saída do minipipeline (Jan/2026).

### Como atualizar o PDF “Trabalhos_Chat”
Execute:
`python3 /workspace/generate_chat_worklog_pdf.py`

Saída:
- `docs/project_context/Trabalhos_Chat_<YYYY-MM-DD>.pdf`

