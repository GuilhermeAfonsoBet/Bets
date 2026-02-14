#!/bin/bash
# ============================================================
# Health Check - Verifica se o coletor está saudável
# ============================================================
#
# Verifica:
# 1. Serviço está rodando
# 2. Últimos dados foram coletados recentemente
# 3. Banco está acessível
# 4. Disco não está cheio
#
# Pode ser usado com cron para alertas automáticos:
#   */5 * * * * /home/betbot/Bets/betinasia_bot/healthcheck.sh
#
# ============================================================

ALERT_FILE="/tmp/betinasia_alert_sent"
MAX_MINUTES_WITHOUT_DATA=15

# Verifica serviço
if ! systemctl is-active --quiet betinasia-collector; then
    echo "[$(date)] ALERTA: betinasia-collector NÃO está rodando!"
    
    # Tenta reiniciar automaticamente
    sudo systemctl restart betinasia-collector
    echo "[$(date)] Tentou reiniciar o serviço"
    exit 1
fi

# Verifica se banco está acessível e tem dados recentes
LAST_DATA=$(sudo -u betbot psql -d betinasia_bot -t -c "
    SELECT EXTRACT(EPOCH FROM (NOW() - MAX(scraped_at))) / 60.0
    FROM best_odds_history;
" 2>/dev/null | tr -d ' ')

if [ -z "$LAST_DATA" ] || [ "$LAST_DATA" = "" ]; then
    echo "[$(date)] AVISO: Não foi possível verificar o banco de dados"
    exit 1
fi

# Converte para inteiro
MINUTES_AGO=$(echo "$LAST_DATA" | cut -d'.' -f1)

if [ -n "$MINUTES_AGO" ] && [ "$MINUTES_AGO" -gt "$MAX_MINUTES_WITHOUT_DATA" ] 2>/dev/null; then
    echo "[$(date)] ALERTA: Última coleta há ${MINUTES_AGO} minutos (limite: ${MAX_MINUTES_WITHOUT_DATA})"
    
    # Reinicia o serviço
    sudo systemctl restart betinasia-collector
    echo "[$(date)] Serviço reiniciado automaticamente"
    exit 1
fi

# Verifica disco (alerta se > 90%)
DISK_USAGE=$(df / | awk 'NR==2{print $5}' | tr -d '%')
if [ "$DISK_USAGE" -gt 90 ]; then
    echo "[$(date)] ALERTA: Disco em ${DISK_USAGE}%! Considere limpar logs antigos."
    exit 1
fi

# Tudo OK
echo "[$(date)] OK - Serviço ativo, última coleta há ~${MINUTES_AGO:-0} min, disco ${DISK_USAGE}%"
exit 0
