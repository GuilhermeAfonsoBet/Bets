## Testes estatísticos — edge e escala (MODE=global_bayes_roll12_robust_p10_p70)

### 1) Evidência formal de edge (banca 2.3k, PnL semanal OOS cap2)
- Semanas (inclui semanas sem trade=0): n=17
  - mean=53.6, IC95%(bootstrap)=[-63.7, 171.5]
  - p_perm(mean>0) (sign-flip)=0.1984
  - sign-test: k_pos=8/14, p=0.3953

- Apenas semanas com trade (stake>0): n=14
  - mean=65.1, IC95%(bootstrap)=[-77.8, 204.4]
  - p_perm(mean>0) (sign-flip)=0.1995
  - sign-test: k_pos=8/14, p=0.3953

### 2) Max (house_cap): variância vs modelo inadequado (online bias-adjusted)
Teste formal aqui é sobre o **erro** (y - mu_online) no cenário max.
- mean(error_online)=397.9, IC95%=[-2,602.6, 3,800.6]
- p_perm(mean_error<0) (sign-flip em -erro)=0.6179
- sign-test (erro<0): k=9/14, p=0.2120

### 3) Até que banca o edge permanece estatisticamente suportado?
- Critério (simples): p_perm(mean_week>0)<0.05 na curva de banca.
- Maior banca que passou no grid: —

CSV: `stat_tests_bankroll_scaling.csv`

### 4) ROI maior em stakes menores (teste formal)
Usamos apostas selecionadas no cenário max e testamos associação monotônica entre ROI_cap2 e log(1+house_cap).
- Spearman rho=-0.021, p_perm(two-sided)=0.6221

