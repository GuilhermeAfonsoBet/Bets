## Comparação — In-sample vs GlobalBayes refinado (treino)
- Critérios refinados: P(post_mean_week>0) >= 0.80; q05(PnL_semana) >= -0.50 * mean_week
- Janela: 2025-10-01..2025-12-31

|segmento|in-sample?|passa refinado?|status|cutoff|stake|mean PnL/sem|p05 PnL/sem|P(mean>0)|
|-:|:-:|:-:|:-:|-:|-:|-:|-:|-:|
|`FH\\|domingo`|True|True|ok|0.410|0.070|87.0|0.0|1.000|
|`FH\\|quarta-feira`|False|False|no_candidate||||||
|`FH\\|quinta-feira`|False|True|ok|0.730|0.060|88.2|-13.5|1.000|
|`FH\\|segunda-feira`|False|False|no_candidate||||||
|`FH\\|sexta-feira`|True|False|no_candidate||||||
|`FH\\|sábado`|True|True|ok|0.230|0.010|118.0|-39.1|1.000|
|`FH\\|terça-feira`|True|False|no_candidate||||||
|`FT\\|domingo`|True|True|ok|0.490|0.070|13.4|0.0|1.000|
|`FT\\|quarta-feira`|True|True|ok|0.330|0.010|53.4|-26.0|0.999|
|`FT\\|quinta-feira`|False|False|no_candidate||||||
|`FT\\|segunda-feira`|True|True|ok|0.190|0.020|144.2|-23.0|1.000|
|`FT\\|sexta-feira`|True|False|no_candidate||||||
|`FT\\|sábado`|True|True|ok|0.910|0.070|37.2|0.0|1.000|
|`FT\\|terça-feira`|True|True|ok|0.250|0.030|212.8|-37.8|1.000|
