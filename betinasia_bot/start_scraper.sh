#!/bin/bash
# Script para iniciar/reconectar ao scraper

SESSION_NAME="scraper"

# Verifica se já existe uma sessão
if screen -list | grep -q "$SESSION_NAME"; then
    echo "Reconectando à sessão existente..."
    screen -r $SESSION_NAME
else
    echo "Criando nova sessão e iniciando scraper..."
    screen -S $SESSION_NAME -d -m bash -c "cd ~/Bets/betinasia_bot && source venv/bin/activate && python main.py --collect"
    echo "Scraper iniciado em background!"
    echo ""
    echo "Para ver os logs: screen -r $SESSION_NAME"
    echo "Para sair sem matar: Ctrl+A, depois D"
fi
