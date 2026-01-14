## Stake sensitivity (empírico)
- Métrica de ROI usada: **ROI_cap2** (ROI Real capado em 2.0).
- Comparação principal: **uncapped vs capped** para o stake0 do próprio segmento.
- Relação adicional: Spearman(ROI_cap2, stake_hist) no conjunto selecionado com stake_hist>0.

### Período: **train**
- **FH | domingo** — stake0 **USD 115** (5.0%), n=14 (uncapped=14, capped=0). ROI_cap2 uncapped=0.464, capped=nan; Δ=nan (IC95% nan..nan), p_perm=nan. Spearman(ROI, stake_hist>0): rho=nan (p_perm=nan, n=0).
- **FH | sexta-feira** — stake0 **USD 23** (1.0%), n=483 (uncapped=483, capped=0). ROI_cap2 uncapped=0.070, capped=nan; Δ=nan (IC95% nan..nan), p_perm=nan. Spearman(ROI, stake_hist>0): rho=-0.148 (p_perm=0.391, n=37).
- **FH | sábado** — stake0 **USD 23** (1.0%), n=564 (uncapped=563, capped=1). ROI_cap2 uncapped=0.117, capped=1.000; Δ=nan (IC95% nan..nan), p_perm=nan. Spearman(ROI, stake_hist>0): rho=-0.131 (p_perm=0.143, n=125).
- **FH | terça-feira** — stake0 **USD 161** (7.0%), n=43 (uncapped=9, capped=34). ROI_cap2 uncapped=0.667, capped=0.397; Δ=0.270 (IC95% -0.471..0.941), p_perm=0.556. Spearman(ROI, stake_hist>0): rho=-0.030 (p_perm=0.951, n=11).
- **FT | domingo** — stake0 **USD 92** (4.0%), n=1 (uncapped=1, capped=0). ROI_cap2 uncapped=1.000, capped=nan; Δ=nan (IC95% nan..nan), p_perm=nan. Spearman(ROI, stake_hist>0): rho=nan (p_perm=nan, n=0).
- **FT | quarta-feira** — stake0 **USD 69** (3.0%), n=197 (uncapped=163, capped=34). ROI_cap2 uncapped=0.150, capped=0.250; Δ=-0.100 (IC95% -0.496..0.295), p_perm=0.613. Spearman(ROI, stake_hist>0): rho=0.301 (p_perm=0.062, n=39).
- **FT | segunda-feira** — stake0 **USD 69** (3.0%), n=152 (uncapped=121, capped=31). ROI_cap2 uncapped=0.207, capped=0.210; Δ=-0.003 (IC95% -0.401..0.393), p_perm=1.000. Spearman(ROI, stake_hist>0): rho=-0.064 (p_perm=1.000, n=12).
- **FT | sexta-feira** — stake0 **USD 46** (2.0%), n=314 (uncapped=312, capped=2). ROI_cap2 uncapped=0.075, capped=0.000; Δ=nan (IC95% nan..nan), p_perm=nan. Spearman(ROI, stake_hist>0): rho=-0.211 (p_perm=0.182, n=41).
- **FT | sábado** — stake0 **USD 161** (7.0%), n=56 (uncapped=51, capped=5). ROI_cap2 uncapped=0.265, capped=-0.100; Δ=0.365 (IC95% -0.504..1.182), p_perm=0.516. Spearman(ROI, stake_hist>0): rho=0.170 (p_perm=0.392, n=26).
- **FT | terça-feira** — stake0 **USD 69** (3.0%), n=213 (uncapped=173, capped=40). ROI_cap2 uncapped=0.162, capped=0.425; Δ=-0.263 (IC95% -0.603..0.067), p_perm=0.128. Spearman(ROI, stake_hist>0): rho=0.071 (p_perm=0.509, n=88).

### Período: **oos**
- **FH | sexta-feira** — stake0 **USD 23** (1.0%), n=36 (uncapped=36, capped=0). ROI_cap2 uncapped=0.194, capped=nan; Δ=nan (IC95% nan..nan), p_perm=nan. Spearman(ROI, stake_hist>0): rho=nan (p_perm=nan, n=0).
- **FH | sábado** — stake0 **USD 23** (1.0%), n=12 (uncapped=12, capped=0). ROI_cap2 uncapped=-0.083, capped=nan; Δ=nan (IC95% nan..nan), p_perm=nan. Spearman(ROI, stake_hist>0): rho=nan (p_perm=nan, n=1).
- **FH | terça-feira** — stake0 **USD 161** (7.0%), n=3 (uncapped=0, capped=3). ROI_cap2 uncapped=nan, capped=-0.167; Δ=nan (IC95% nan..nan), p_perm=nan. Spearman(ROI, stake_hist>0): rho=nan (p_perm=nan, n=1).
- **FT | quarta-feira** — stake0 **USD 69** (3.0%), n=6 (uncapped=6, capped=0). ROI_cap2 uncapped=0.333, capped=nan; Δ=nan (IC95% nan..nan), p_perm=nan. Spearman(ROI, stake_hist>0): rho=nan (p_perm=nan, n=2).
- **FT | segunda-feira** — stake0 **USD 69** (3.0%), n=6 (uncapped=6, capped=0). ROI_cap2 uncapped=-0.500, capped=nan; Δ=nan (IC95% nan..nan), p_perm=nan. Spearman(ROI, stake_hist>0): rho=nan (p_perm=nan, n=6).
- **FT | sexta-feira** — stake0 **USD 46** (2.0%), n=12 (uncapped=12, capped=0). ROI_cap2 uncapped=-0.125, capped=nan; Δ=nan (IC95% nan..nan), p_perm=nan. Spearman(ROI, stake_hist>0): rho=nan (p_perm=nan, n=0).
- **FT | sábado** — stake0 **USD 161** (7.0%), n=2 (uncapped=2, capped=0). ROI_cap2 uncapped=0.000, capped=nan; Δ=nan (IC95% nan..nan), p_perm=nan. Spearman(ROI, stake_hist>0): rho=nan (p_perm=nan, n=0).
- **FT | terça-feira** — stake0 **USD 69** (3.0%), n=64 (uncapped=55, capped=9). ROI_cap2 uncapped=0.009, capped=-0.111; Δ=0.120 (IC95% -0.508..0.721), p_perm=0.784. Spearman(ROI, stake_hist>0): rho=-0.042 (p_perm=0.784, n=54).

