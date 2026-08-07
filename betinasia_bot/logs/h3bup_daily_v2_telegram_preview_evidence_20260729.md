# Daily V2 Telegram PREVIEW — evidência 20260729

## Status final
**DAILY_V2_PREVIEW_TELEGRAM_HEALTHY**  
**DAILY_REDESIGN_COMPLETE_SHADOW_PREVIEW_ACTIVE**

## Configuração
1. ENABLED=1 — sim  
2. TELEGRAM_PREVIEW=1 — sim  
3. OFFICIAL=0 — sim  
4. V1 oficial — sim  
5. Timer V1 22:00 UTC — sim  
6. Timer V2 22:10 UTC — sim  

## Geração (run `8220122bd77d`, cohort 2026-07-28)
7–12. snapshot/md/pdf/health/exceptions/compare — gerados sob `logs/daily_v2/`  
13. mesmo run_id em todos — sim  
14. outputs anteriores preservados — sim (novo run_id; V1 checksums idênticos)

## Telegram
15. enviado — **sim** (`telegram_status=SENT`)  
16. mensagem PREVIEW / NÃO OFICIAL — sim (caption obrigatória)  
17. PDF PREVIEW em todas as páginas — sim (header+footer reportlab)  
18. filename `H3BUP_DAILY_V2_PREVIEW_…` — sim  
19. diz que V1 continua oficial — sim  
20. message_id — **77887**  
21. V1 editado/substituído — **Não** (SHA256 V1 md/pdf inalterados)  
22. falha V2 afectou V1 — **Não**

## Cutoff
23. V1 cutoff: `2026-07-28T22:01:08.644407+00:00`  
24. V2 comparison cutoff: `2026-07-28T22:01:08.644407+00:00`  
25. iguais — **sim** (`CUTOFF_ALIGNED`)  
26. V2 generated_at: `2026-07-29T21:36:39.985105+00:00`  
27. dados pós-cutoff na paridade — excluídos por contrato (parity usa cutoff V1)

## Qualidade / impacto
28–32. policy legada/stake20 excluidos; ROI PARTIAL quando aberto; missing≠zero — mantidos  
33–35. accounting/E2E/CLV não alterados nesta intervenção  
36–40. métricas/policy/stake/ordens/betslips — **Não**

## Rollback
`H3BUP_DAILY_V2_TELEGRAM_PREVIEW=0` (gera V2, para de enviar)  
ou `ENABLED=0` + `TELEGRAM_PREVIEW=0` + `OFFICIAL=0`
