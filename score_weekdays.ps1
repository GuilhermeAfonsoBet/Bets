# score_weekdays.ps1
# Chama o CLI de weekdays e sobrescreve um arquivo de saída com o score da ÚLTIMA aposta.

# ================== Config ==================
$python     = 'C:\Bets\venv\Scripts\python.exe'
$cli        = 'C:\Bets\ModelosEstatísticos\score_logit_weekdays_cli.py'
$modelsDir  = 'C:\Bets\ModelosEstatísticos'
$csvin      = 'C:\Bets\ModelosEstatísticos\payload.csv'
$cutoff     = '0'             # decisão real (threshold) fica no PAD
$calibFloor = '0.005'
$log        = 'C:\Bets\logs\scoring_weekdays.jsonl'

# Se você quiser um arquivo físico com o resultado:
$outFile    = 'C:\Bets\ModelosEstatísticos\score_weekday_out.csv'
# ================== Fim Config ==============

# Garante pasta do log
$logDir = Split-Path -Path $log -Parent
if (-not (Test-Path $logDir)) {
  New-Item -ItemType Directory -Path $logDir -Force | Out-Null
}

# Monta args do CLI
$args = @(
  $cli, '--models-dir', $modelsDir,
  '--csvin', $csvin,
  '--cutoff', $cutoff,
  '--calib-floor', $calibFloor,
  '--logfile', $log
)

# Executa o Python
$out = & $python @args 2>$null

# 1) Devolve no stdout (para o PAD capturar)
$out | Write-Output

# 2) Opcional: grava em arquivo, SEMPRE sobrescrevendo
$out | Set-Content -Path $outFile -Encoding UTF8
