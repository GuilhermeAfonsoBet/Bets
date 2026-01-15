## Comparação — In-sample vs GlobalBayes refinado (treino) — P(mean>0) >= 0.80
- Critério: P(post_mean_week>0) >= 0.80 (sem lower bound)
- Janela: 2025-10-01..2025-12-31

|segmento|in-sample?|passa refinado?|status|cutoff|stake|mean PnL/sem|P(sem<0)|Sharpe sem|P(mean>0)|
|-:|:-:|:-:|:-:|-:|-:|-:|-:|-:|-:|
|`FH\|domingo`|True|True|ok|0.410|0.070|87.0|0.000|0.754|1.000|
|`FH\|quarta-feira`|False|False|no_candidate|||||||
|`FH\|quinta-feira`|False|True|ok|0.690|0.070|125.7|0.167|0.541|0.991|
|`FH\|segunda-feira`|False|False|no_candidate|||||||
|`FH\|sexta-feira`|True|True|ok|0.070|0.010|60.2|0.231|0.412|0.937|
|`FH\|sábado`|True|True|ok|0.230|0.010|118.0|0.308|0.747|1.000|
|`FH\|terça-feira`|True|True|ok|0.670|0.060|159.0|0.083|0.728|0.995|
|`FT\|domingo`|True|True|ok|0.490|0.070|13.4|0.000|0.289|1.000|
|`FT\|quarta-feira`|True|True|ok|0.390|0.050|205.5|0.308|0.691|0.998|
|`FT\|quinta-feira`|False|False|no_candidate|||||||
|`FT\|segunda-feira`|True|True|ok|0.190|0.020|144.2|0.091|0.983|1.000|
|`FT\|sexta-feira`|True|True|ok|0.310|0.020|84.7|0.231|0.478|0.967|
|`FT\|sábado`|True|True|ok|0.570|0.070|163.3|0.231|0.591|0.992|
|`FT\|terça-feira`|True|True|ok|0.250|0.030|212.8|0.231|0.765|1.000|
