#!/bin/bash
# Configura cron job para atualizar resultados a cada 4 horas
#
# Uso: bash setup_results_cron.sh

echo "=============================================="
echo "Configurando cron job para resultados"
echo "=============================================="

# Cria script wrapper
cat > /home/betbot/Bets/betinasia_bot/run_results_update.sh << 'EOF'
#!/bin/bash
cd /home/betbot/Bets/betinasia_bot
source venv/bin/activate
python -m results.auto_update_results --once >> logs/results_cron.log 2>&1
EOF

chmod +x /home/betbot/Bets/betinasia_bot/run_results_update.sh

# Adiciona ao crontab (a cada 4 horas: 0, 4, 8, 12, 16, 20)
(crontab -l 2>/dev/null | grep -v "run_results_update"; echo "0 0,4,8,12,16,20 * * * /home/betbot/Bets/betinasia_bot/run_results_update.sh") | crontab -

echo ""
echo "Cron job configurado!"
echo ""
echo "Verificando crontab:"
crontab -l | grep results
echo ""
echo "O job vai rodar nos horarios: 00:00, 04:00, 08:00, 12:00, 16:00, 20:00 UTC"
echo ""
echo "Para ver logs: tail -f /home/betbot/Bets/betinasia_bot/logs/results_cron.log"
