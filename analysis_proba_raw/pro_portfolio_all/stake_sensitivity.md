## Stake sensitivity (empírico)
- Métrica de ROI usada: **ROI_cap2** (ROI Real capado em 2.0).
- Comparação principal: **uncapped vs capped** para o stake0 do próprio segmento.
- Relação adicional: Spearman(ROI_cap2, stake_hist) no conjunto selecionado com stake_hist>0.

### Período: **train**
- **FH | domingo** — stake0 **USD 115** (5.0%), n=14 (uncapped=14, capped=0). ROI_cap2 uncapped=0.343, capped=nan; Δ=nan (IC95% nan..nan), p_perm=nan. Spearman(ROI, stake_hist>0): rho=nan (p_perm=nan, n=0).
- **FH | sexta-feira** — stake0 **USD 23** (1.0%), n=483 (uncapped=483, capped=0). ROI_cap2 uncapped=-0.017, capped=nan; Δ=nan (IC95% nan..nan), p_perm=nan. Spearman(ROI, stake_hist>0): rho=-0.154 (p_perm=0.369, n=37).
- **FH | sábado** — stake0 **USD 23** (1.0%), n=563 (uncapped=562, capped=1). ROI_cap2 uncapped=0.029, capped=0.690; Δ=nan (IC95% nan..nan), p_perm=nan. Spearman(ROI, stake_hist>0): rho=-0.091 (p_perm=0.302, n=124).
- **FH | terça-feira** — stake0 **USD 161** (7.0%), n=43 (uncapped=9, capped=34). ROI_cap2 uncapped=0.431, capped=0.259; Δ=0.172 (IC95% -0.461..0.734), p_perm=0.608. Spearman(ROI, stake_hist>0): rho=0.110 (p_perm=0.760, n=11).
- **FT | domingo** — stake0 **USD 92** (4.0%), n=1 (uncapped=1, capped=0). ROI_cap2 uncapped=0.930, capped=nan; Δ=nan (IC95% nan..nan), p_perm=nan. Spearman(ROI, stake_hist>0): rho=nan (p_perm=nan, n=0).
- **FT | quarta-feira** — stake0 **USD 69** (3.0%), n=197 (uncapped=163, capped=34). ROI_cap2 uncapped=0.065, capped=0.093; Δ=-0.028 (IC95% -0.358..0.312), p_perm=0.875. Spearman(ROI, stake_hist>0): rho=0.325 (p_perm=0.039, n=39).
- **FT | segunda-feira** — stake0 **USD 69** (3.0%), n=152 (uncapped=121, capped=31). ROI_cap2 uncapped=0.098, capped=0.226; Δ=-0.128 (IC95% -0.524..0.272), p_perm=0.499. Spearman(ROI, stake_hist>0): rho=-0.284 (p_perm=0.381, n=12).
- **FT | sexta-feira** — stake0 **USD 46** (2.0%), n=313 (uncapped=311, capped=2). ROI_cap2 uncapped=-0.015, capped=0.100; Δ=nan (IC95% nan..nan), p_perm=nan. Spearman(ROI, stake_hist>0): rho=-0.184 (p_perm=0.247, n=41).
- **FT | sábado** — stake0 **USD 161** (7.0%), n=56 (uncapped=51, capped=5). ROI_cap2 uncapped=0.209, capped=-0.044; Δ=0.253 (IC95% -0.705..1.119), p_perm=0.597. Spearman(ROI, stake_hist>0): rho=-0.037 (p_perm=0.856, n=26).
- **FT | terça-feira** — stake0 **USD 69** (3.0%), n=214 (uncapped=174, capped=40). ROI_cap2 uncapped=0.090, capped=0.300; Δ=-0.210 (IC95% -0.507..0.092), p_perm=0.190. Spearman(ROI, stake_hist>0): rho=0.067 (p_perm=0.531, n=88).

### Período: **oos**
- **FH | sexta-feira** — stake0 **USD 23** (1.0%), n=41 (uncapped=41, capped=0). ROI_cap2 uncapped=0.093, capped=nan; Δ=nan (IC95% nan..nan), p_perm=nan. Spearman(ROI, stake_hist>0): rho=nan (p_perm=nan, n=0).
- **FH | sábado** — stake0 **USD 23** (1.0%), n=17 (uncapped=17, capped=0). ROI_cap2 uncapped=0.260, capped=nan; Δ=nan (IC95% nan..nan), p_perm=nan. Spearman(ROI, stake_hist>0): rho=nan (p_perm=nan, n=1).
- **FH | terça-feira** — stake0 **USD 161** (7.0%), n=8 (uncapped=2, capped=6). ROI_cap2 uncapped=0.000, capped=-0.080; Δ=nan (IC95% nan..nan), p_perm=nan. Spearman(ROI, stake_hist>0): rho=nan (p_perm=nan, n=1).
- **FT | quarta-feira** — stake0 **USD 69** (3.0%), n=10 (uncapped=10, capped=0). ROI_cap2 uncapped=0.208, capped=nan; Δ=nan (IC95% nan..nan), p_perm=nan. Spearman(ROI, stake_hist>0): rho=nan (p_perm=nan, n=5).
- **FT | segunda-feira** — stake0 **USD 69** (3.0%), n=6 (uncapped=6, capped=0). ROI_cap2 uncapped=-0.485, capped=nan; Δ=nan (IC95% nan..nan), p_perm=nan. Spearman(ROI, stake_hist>0): rho=nan (p_perm=nan, n=6).
- **FT | sexta-feira** — stake0 **USD 46** (2.0%), n=12 (uncapped=12, capped=0). ROI_cap2 uncapped=-0.137, capped=nan; Δ=nan (IC95% nan..nan), p_perm=nan. Spearman(ROI, stake_hist>0): rho=nan (p_perm=nan, n=0).
- **FT | sábado** — stake0 **USD 161** (7.0%), n=2 (uncapped=2, capped=0). ROI_cap2 uncapped=0.015, capped=nan; Δ=nan (IC95% nan..nan), p_perm=nan. Spearman(ROI, stake_hist>0): rho=nan (p_perm=nan, n=0).
- **FT | terça-feira** — stake0 **USD 69** (3.0%), n=73 (uncapped=63, capped=10). ROI_cap2 uncapped=0.100, capped=-0.239; Δ=0.339 (IC95% -0.250..0.865), p_perm=0.258. Spearman(ROI, stake_hist>0): rho=-0.018 (p_perm=0.894, n=63).

