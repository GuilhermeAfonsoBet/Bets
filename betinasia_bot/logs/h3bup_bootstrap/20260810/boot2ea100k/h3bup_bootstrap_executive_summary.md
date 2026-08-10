# Executive — FASE 2E-A Bootstrap H3BUP_vNext

- **Status:** `BOOTSTRAP_ANALYSIS_COMPLETE_WITH_WARNINGS`
- **Classificação:** `NEGATIVE_ROI_SIGNAL`
- **statistical_readiness:** `FIRST_READING`
- Void no denominador: **sim**

## Respostas (1–39)

1. Ordens resolvidas: **176**
2. Eventos únicos: **158** LIVE_OK / **135** resolvidas
3. Stake resolvida: **1040.00**
4. P&L: **-30.22**
5. ROI observado: **-2.91%**

### ORDER BOOTSTRAP
6. Mean: **-2.88%**
7. Median: **-2.87%**
8. IC90: **[-15.94%, 10.26%]**
9. IC95: **[-18.46%, 12.73%]**
10. P(ROI>0): **35.75%**
11. P(ROI>2%): **26.96%**
12. P(ROI>5%): **16.13%**
13. P(ROI>10%): **5.33%**
14. P(ROI<0): **64.24%**

### CLUSTER EVENT BOOTSTRAP (preferencial)
15. Mean: **-2.86%**
16. IC90: **[-13.64%, 8.04%]**
17. IC95: **[-15.70%, 10.05%]**
18. P(ROI>0): **32.92%**
19. P(ROI>5%): **11.68%**
20. P(ROI<0): **67.08%**

### ROBUSTEZ
21. P(ROI>0) sem maior evento: **27.00%**
22. sem top 3 eventos: **17.32%**
23. ROI permanece positivo? **não**
24. IC90 acima de zero? **não**
25. IC95 acima de zero? **não**

### FRIENDLY
26. ROI Friendly: **-2.60%**
27. P(ROI>0) Friendly: **38.00%**
28. ROI Non-Friendly: **-3.28%**
29. P(ROI>0) Non-Friendly: **37.64%**
30. P(ROI_NF > ROI_Friendly): **47.89%**

### CLV
31. P(CLV 5m mean>0): **0.00%**
32. P(CLV 15m mean>0): **0.00%**
33. P(CLV closing mean>0): **0.00%**
34. IC95 closing CLV: **[-3.42%, -1.29%]**
35. CLV e ROI mesma direção? **sim**

### TEMPORAL
36. P(ROI>0) últimas 25: **52.86%**
37. últimas 50: **82.40%**
38. últimas 100: **33.64%**
39. Evidência: **melhorando (rolling50 vs full)**

## Avisos
- statistical_readiness=FIRST_READING
- partial_settlement open=22 missing=10 coverage=0.846


## Fonte / cutoff
- Freeze oficial: `a27c1dc4ab52` · cutoff `2026-08-07T14:25:06Z`
- N_boot=100000 · seed=20260810
- VPS refresh 2026-08-10 **não disponível** (SSH key pending) — análise sobre freeze mais recente no repo
- Preferencial: **cluster event_id**

## Nota
Análise exclusivamente estatística/read-only. Não alterar exposição com base neste resultado.
Day-cluster: N_dias=11 · P(ROI>0)=18.91% · IC95=[-0.0957706093189964, 0.03532608695652173]
