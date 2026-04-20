## Testes estatísticos — edge e escala (MODE=global_bayes_roll12_robust_p10_p70)

### 1) Evidência formal de edge (banca 2.3k, PnL semanal OOS cap2)
- Semanas (inclui semanas sem trade=0): n=19
  - mean=56.1, IC95%(bootstrap)=[-52.6, 168.8]
  - p_perm(mean>0) (sign-flip)=0.1728
  - sign-test: k_pos=9/15, p=0.3036

- Apenas semanas com trade (stake>0): n=15
  - mean=71.1, IC95%(bootstrap)=[-67.6, 210.6]
  - p_perm(mean>0) (sign-flip)=0.1767
  - sign-test: k_pos=9/15, p=0.3036

### 2) Max (house_cap): variância vs modelo inadequado (online bias-adjusted)
Teste formal aqui é sobre o **erro** (y - mu_online) no cenário max.
- mean(error_online)=392.5, IC95%=[-2,510.3, 3,903.4]
- p_perm(mean_error<0) (sign-flip em -erro)=0.5969
- sign-test (erro<0): k=11/16, p=0.1051

### 3) Até que banca o edge permanece estatisticamente suportado?
- Critério (simples): p_perm(mean_week>0)<0.05 na curva de banca.
- Maior banca que passou no grid: —

CSV: `stat_tests_bankroll_scaling.csv`

CSV (sensibilidade, exclui última semana): `stat_tests_bankroll_scaling_excl_lastweek.csv`

### 4) ROI maior em stakes menores (teste formal)
Usamos apostas selecionadas no cenário max e testamos associação monotônica entre ROI_cap2 e log(1+house_cap).
- Spearman rho=-0.009, p_perm(two-sided)=0.8356

