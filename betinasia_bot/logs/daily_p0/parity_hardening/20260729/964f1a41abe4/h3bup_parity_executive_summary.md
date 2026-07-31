# Parity Hardening — Executive Summary

**Status:** `DAILY_PARITY_HARDENED_MATCH`

run_id: `964f1a41abe4`  
commit: `95d4af9` (branch `cursor/h3bup-daily-p0-47ee`)

## Universo (Q1–10)

1. Order_ids extra no V2: **1938082582**, **1938105954**
2. Policy: ambos `H3BUP_vNext_20260629`
3. Stake: ambos **10**
4. created_at: `22:21:51Z` e `22:30:18Z`
5. Fora do V1 porque **posteriores ao cutoff** `22:01:54.606850Z`
6. V1 errado? **Não**
7. V2 errado? **Não** (DAILY_CLOSED = dia completo)
8. Diferença intencional de escopo? **Sim** → `EXPECTED_SCOPE_DIFFERENCE`
9. Conjunto correcto para paridade: **22 order_ids** com `created_at <= parity_as_of`
10. Hashes V1 e V2 parity: **coincidem** (`order_set_match=true`)

## As-of (Q11–20)

11. parity_as_of: `2026-07-29T22:01:54.606850+00:00`
12. matured_as_of: `2026-07-30T20:23:40.223866+00:00`
13–16. Parity: open **6** / settled **15** / void **1** / missing **0**
17–19. Matured: open **1** / settled **21** / void **2**
20. Dados posteriores na paridade? **Não** (universo filtrado; accounting snapshot ≤cutoff)

## Performance (Q21–28)

21–23. Parity: stake_resolved **160** / P&L **-6.51** / ROI **-4.07%**
24–26. Matured: stake_resolved **230** / P&L **+6.60** / ROI **+2.87%**
27. Void no denominador? **Sim**
28. ROI ex-void mostrado? **Sim** (~3.14% matured)

## Paridade (Q29–37)

29–30. LIVE_OK / stake parity: **Sim** (22 / 220)
31–33. Open/settled/void no mesmo as-of: comparáveis na visão parity (health V1 n=12 era subset — PAR-003/004)
34–35. P&L/ROI: contratos separados parity vs matured
36. UNKNOWN? **Não**
37. Blockers publicação? **Não**

## Segurança (Q38–48)

Todas **Não**.

Dir: `logs/daily_p0/parity_hardening/20260729/964f1a41abe4/`
