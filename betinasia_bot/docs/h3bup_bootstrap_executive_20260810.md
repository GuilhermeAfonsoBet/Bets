# Executive — FASE 2E-A Bootstrap H3BUP_vNext

- **Status:** `BOOTSTRAP_ANALYSIS_COMPLETE_WITH_WARNINGS`
- **Classificação:** `NO_CLEAR_ROI_EDGE`
- **statistical_readiness:** `FIRST_READING`
- Void no denominador: **sim**

## Respostas (1–39)

1. Ordens resolvidas: **244**
2. Eventos únicos: **211** LIVE_OK / **189** resolvidas
3. Stake resolvida: **1176.00**
4. P&L: **-18.24**
5. ROI observado: **-1.55%**

### ORDER BOOTSTRAP
6. Mean: **-1.56%**
7. Median: **-1.55%**
8. IC90: **[-13.26%, 10.20%]**
9. IC95: **[-15.46%, 12.45%]**
10. P(ROI>0): **41.38%**
11. P(ROI>2%): **30.89%**
12. P(ROI>5%): **18.03%**
13. P(ROI>10%): **5.31%**
14. P(ROI<0): **58.62%**

### CLUSTER EVENT BOOTSTRAP (preferencial)
15. Mean: **-1.52%**
16. IC90: **[-11.21%, 8.32%]**
17. IC95: **[-13.05%, 10.19%]**
18. P(ROI>0): **39.68%**
19. P(ROI>5%): **13.51%**
20. P(ROI<0): **60.32%**

### ROBUSTEZ
21. P(ROI>0) sem maior evento: **33.59%**
22. sem top 3 eventos: **22.46%**
23. ROI permanece positivo? **não**
24. IC90 acima de zero? **não**
25. IC95 acima de zero? **não**

### FRIENDLY
26. ROI Friendly: **-1.59%**
27. P(ROI>0) Friendly: **42.18%**
28. ROI Non-Friendly: **-1.50%**
29. P(ROI>0) Non-Friendly: **43.63%**
30. P(ROI_NF > ROI_Friendly): **50.33%**

### CLV
31. P(CLV 5m mean>0): **0.00%**
32. P(CLV 15m mean>0): **0.00%**
33. P(CLV closing mean>0): **0.00%**
34. IC95 closing CLV: **[-3.16%, -1.33%]**
35. CLV e ROI mesma direção? **sim**

### TEMPORAL
36. P(ROI>0) últimas 25: **63.24%**
37. últimas 50: **88.61%**
38. últimas 100: **81.06%**
39. Evidência: **melhorando (rolling50 vs full)**

## Avisos
- statistical_readiness=FIRST_READING
- partial_settlement open=16 missing=11 coverage=0.900


## Fonte / cutoff
- Freeze VPS: `c86a5dec4503` · cutoff `2026-08-10T11:43:53Z` (consulta direta)
- N_LIVE_OK=271 · n_boot=100000 · seed=20260810
- Universo: policy_id=H3BUP_vNext · policy_version=H3BUP_vNext_20260629 · 0 mismatch
- Preferencial: **cluster event_id**

## Nota
Análise exclusivamente estatística/read-only. Não alterar exposição com base neste resultado.
Day-cluster: N_dias=13 · P(ROI>0)=32.51% · IC95=[-0.07524439918533607, 0.051126543209876536]
