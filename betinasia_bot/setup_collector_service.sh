#!/bin/bash
# Script para configurar o servico de coleta continua
# Executa como root: sudo bash setup_collector_service.sh

set -e

echo "=============================================="
echo "Configurando servico BetinAsia Collector"
echo "=============================================="

# Cria diretorio de logs
mkdir -p /home/betbot/Bets/betinasia_bot/logs
chown -R betbot:betbot /home/betbot/Bets/betinasia_bot/logs

# Copia arquivo de servico
cp /home/betbot/Bets/betinasia_bot/betinasia-collector.service /etc/systemd/system/

# Recarrega systemd
systemctl daemon-reload

# Habilita o servico para iniciar no boot
systemctl enable betinasia-collector.service

echo ""
echo "=============================================="
echo "Servico configurado com sucesso!"
echo "=============================================="
echo ""
echo "Comandos uteis:"
echo "  sudo systemctl start betinasia-collector   # Iniciar"
echo "  sudo systemctl stop betinasia-collector    # Parar"
echo "  sudo systemctl restart betinasia-collector # Reiniciar"
echo "  sudo systemctl status betinasia-collector  # Ver status"
echo "  sudo journalctl -u betinasia-collector -f  # Ver logs em tempo real"
echo ""
echo "Logs tambem disponiveis em:"
echo "  /home/betbot/Bets/betinasia_bot/logs/collector.log"
echo "  /home/betbot/Bets/betinasia_bot/logs/collector_error.log"
echo ""
