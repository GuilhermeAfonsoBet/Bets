#!/bin/bash
# ============================================================
# DEPLOY BETINASIA BOT - Script Completo para VPS
# ============================================================
#
# Este script configura TUDO para rodar o bot em background
# independente da sua conexão SSH.
#
# Uso:
#   Como ROOT:      sudo bash deploy.sh
#   Primeira vez:   sudo bash deploy.sh --full
#   Atualização:    sudo bash deploy.sh --update
#   Só serviços:    sudo bash deploy.sh --services
#
# ============================================================

set -euo pipefail

# Cores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configurações
BOT_USER="betbot"
BOT_HOME="/home/${BOT_USER}"
PROJECT_DIR="${BOT_HOME}/Bets"
BOT_DIR="${PROJECT_DIR}/betinasia_bot"
VENV_DIR="${PROJECT_DIR}/venv"
LOGS_DIR="${BOT_DIR}/logs"

# ============================================================
# Funções auxiliares
# ============================================================

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERRO]${NC} $1"
}

log_step() {
    echo ""
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}============================================================${NC}"
}

check_root() {
    if [ "$EUID" -ne 0 ]; then
        log_error "Este script precisa ser executado como root (sudo)."
        exit 1
    fi
}

# ============================================================
# 1. Instalação de dependências do sistema
# ============================================================
install_system_deps() {
    log_step "1. Instalando dependências do sistema"

    apt-get update -qq

    # Python
    apt-get install -y -qq python3 python3-pip python3-venv python3-dev > /dev/null 2>&1
    log_info "Python instalado: $(python3 --version)"

    # PostgreSQL
    if ! command -v psql &> /dev/null; then
        apt-get install -y -qq postgresql postgresql-contrib > /dev/null 2>&1
        systemctl enable postgresql
        systemctl start postgresql
        log_info "PostgreSQL instalado e iniciado"
    else
        log_info "PostgreSQL já instalado: $(psql --version | head -1)"
    fi

    # Redis
    if ! command -v redis-cli &> /dev/null; then
        apt-get install -y -qq redis-server > /dev/null 2>&1
        systemctl enable redis-server
        systemctl start redis-server
        log_info "Redis instalado e iniciado"
    else
        log_info "Redis já instalado"
    fi

    # Xvfb (display virtual para Playwright)
    apt-get install -y -qq xvfb > /dev/null 2>&1
    log_info "Xvfb instalado"

    # Dependências do Playwright/Chromium
    apt-get install -y -qq \
        libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 \
        libcups2 libdrm2 libxkbcommon0 libxcomposite1 libxdamage1 \
        libxfixes3 libxrandr2 libgbm1 libasound2 libpango-1.0-0 \
        libpangocairo-1.0-0 libgtk-3-0 libx11-xcb1 libxshmfence1 \
        fonts-liberation libappindicator3-1 > /dev/null 2>&1
    log_info "Dependências do Chromium instaladas"

    # Ferramentas úteis
    apt-get install -y -qq git curl wget htop nano logrotate > /dev/null 2>&1
    log_info "Ferramentas auxiliares instaladas"
}

# ============================================================
# 2. Criar usuário do sistema
# ============================================================
setup_user() {
    log_step "2. Configurando usuário '${BOT_USER}'"

    if id "${BOT_USER}" &>/dev/null; then
        log_info "Usuário '${BOT_USER}' já existe"
    else
        adduser --disabled-password --gecos "" ${BOT_USER}
        usermod -aG sudo ${BOT_USER}
        log_info "Usuário '${BOT_USER}' criado"
    fi
}

# ============================================================
# 3. Configurar banco de dados PostgreSQL
# ============================================================
setup_database() {
    log_step "3. Configurando PostgreSQL"

    # Verifica se o usuário existe no PostgreSQL
    if sudo -u postgres psql -tAc "SELECT 1 FROM pg_roles WHERE rolname='${BOT_USER}'" | grep -q 1; then
        log_info "Usuário PostgreSQL '${BOT_USER}' já existe"
    else
        sudo -u postgres psql -c "CREATE USER ${BOT_USER} WITH PASSWORD 'betbot_secure_2026';"
        log_info "Usuário PostgreSQL '${BOT_USER}' criado"
    fi

    # Verifica se o banco existe
    if sudo -u postgres psql -tAc "SELECT 1 FROM pg_database WHERE datname='betinasia_bot'" | grep -q 1; then
        log_info "Banco de dados 'betinasia_bot' já existe"
    else
        sudo -u postgres psql -c "CREATE DATABASE betinasia_bot OWNER ${BOT_USER};"
        sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE betinasia_bot TO ${BOT_USER};"
        log_info "Banco de dados 'betinasia_bot' criado"
    fi
}

# ============================================================
# 4. Configurar Python e dependências
# ============================================================
setup_python() {
    log_step "4. Configurando ambiente Python"

    # Cria diretório do projeto se não existir
    mkdir -p "${PROJECT_DIR}"
    chown -R ${BOT_USER}:${BOT_USER} "${PROJECT_DIR}"

    # Cria venv
    if [ ! -d "${VENV_DIR}" ]; then
        sudo -u ${BOT_USER} python3 -m venv "${VENV_DIR}"
        log_info "Ambiente virtual criado em ${VENV_DIR}"
    else
        log_info "Ambiente virtual já existe"
    fi

    # Instala dependências
    sudo -u ${BOT_USER} "${VENV_DIR}/bin/pip" install --upgrade pip -q
    sudo -u ${BOT_USER} "${VENV_DIR}/bin/pip" install -r "${BOT_DIR}/requirements.txt" -q
    log_info "Dependências Python instaladas"

    # Instala Playwright browsers
    sudo -u ${BOT_USER} "${VENV_DIR}/bin/playwright" install chromium
    log_info "Playwright Chromium instalado"

    # Instala deps do sistema do Playwright
    "${VENV_DIR}/bin/playwright" install-deps chromium > /dev/null 2>&1 || true
    log_info "Dependências do Playwright instaladas"
}

# ============================================================
# 5. Configurar .env
# ============================================================
setup_env() {
    log_step "5. Configurando arquivo .env"

    if [ ! -f "${BOT_DIR}/.env" ]; then
        cp "${BOT_DIR}/.env.example" "${BOT_DIR}/.env"
        chown ${BOT_USER}:${BOT_USER} "${BOT_DIR}/.env"
        chmod 600 "${BOT_DIR}/.env"

        log_warn "Arquivo .env criado a partir do .env.example"
        log_warn "IMPORTANTE: Edite ${BOT_DIR}/.env com suas credenciais!"
        echo ""
        echo "  sudo -u ${BOT_USER} nano ${BOT_DIR}/.env"
        echo ""
        echo "  Preencha:"
        echo "    BETINASIA_USERNAME=seu_usuario"
        echo "    BETINASIA_PASSWORD=sua_senha"
        echo "    DATABASE_URL=postgresql://betbot:betbot_secure_2026@localhost:5432/betinasia_bot"
        echo ""
    else
        log_info "Arquivo .env já existe"
        chmod 600 "${BOT_DIR}/.env"
    fi
}

# ============================================================
# 6. Configurar logs e logrotate
# ============================================================
setup_logs() {
    log_step "6. Configurando logs e rotação"

    # Cria diretório de logs
    mkdir -p "${LOGS_DIR}"
    chown -R ${BOT_USER}:${BOT_USER} "${LOGS_DIR}"
    log_info "Diretório de logs: ${LOGS_DIR}"

    # Instala logrotate config
    cp "${BOT_DIR}/betinasia-logrotate.conf" /etc/logrotate.d/betinasia
    log_info "Logrotate configurado (rotação diária, 30 dias de retenção)"
}

# ============================================================
# 7. Configurar Xvfb (display virtual)
# ============================================================
setup_xvfb() {
    log_step "7. Configurando Xvfb (display virtual)"

    cp "${BOT_DIR}/betinasia-xvfb.service" /etc/systemd/system/
    systemctl daemon-reload
    systemctl enable betinasia-xvfb.service
    systemctl start betinasia-xvfb.service || true
    log_info "Xvfb configurado em DISPLAY=:99"
}

# ============================================================
# 8. Instalar e ativar serviços systemd
# ============================================================
setup_services() {
    log_step "8. Instalando serviços systemd"

    # Para serviços existentes (se houver)
    systemctl stop betinasia-collector.service 2>/dev/null || true
    systemctl stop betinasia-results.service 2>/dev/null || true

    # Copia service files
    cp "${BOT_DIR}/betinasia-collector.service" /etc/systemd/system/
    cp "${BOT_DIR}/betinasia-results.service" /etc/systemd/system/

    # Recarrega systemd
    systemctl daemon-reload

    # Habilita para iniciar no boot
    systemctl enable betinasia-collector.service
    systemctl enable betinasia-results.service

    log_info "Serviços instalados e habilitados para iniciar no boot"
    log_info "  - betinasia-collector (coleta contínua de odds)"
    log_info "  - betinasia-results  (atualização de resultados)"
}

# ============================================================
# 9. Iniciar serviços
# ============================================================
start_services() {
    log_step "9. Iniciando serviços"

    # Inicia Xvfb primeiro
    systemctl start betinasia-xvfb.service || true
    sleep 2

    # Inicia collector
    systemctl start betinasia-collector.service
    sleep 3

    # Verifica se está rodando
    if systemctl is-active --quiet betinasia-collector.service; then
        log_info "betinasia-collector: RODANDO ✓"
    else
        log_error "betinasia-collector: FALHOU ✗"
        echo "  Verifique os logs:"
        echo "    sudo journalctl -u betinasia-collector -n 50 --no-pager"
        echo "    cat ${LOGS_DIR}/collector_error.log"
    fi

    # Inicia results updater
    systemctl start betinasia-results.service
    sleep 2

    if systemctl is-active --quiet betinasia-results.service; then
        log_info "betinasia-results: RODANDO ✓"
    else
        log_warn "betinasia-results: FALHOU (pode ser normal se não tiver API key)"
    fi
}

# ============================================================
# 10. Criar script de monitoramento
# ============================================================
setup_monitoring() {
    log_step "10. Criando scripts de monitoramento"

    # Status script
    cat > "${BOT_DIR}/status.sh" << 'STATUSEOF'
#!/bin/bash
# Status rápido dos serviços BetinAsia

echo ""
echo "============================================================"
echo "  BETINASIA BOT - STATUS"
echo "  $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "============================================================"
echo ""

# Serviços
echo "SERVIÇOS:"
echo "------------------------------"
for svc in betinasia-xvfb betinasia-collector betinasia-results; do
    status=$(systemctl is-active $svc 2>/dev/null || echo "não instalado")
    uptime=""
    if [ "$status" = "active" ]; then
        uptime=" (uptime: $(systemctl show $svc --property=ActiveEnterTimestamp --value 2>/dev/null))"
        echo "  ✅ $svc: $status$uptime"
    else
        echo "  ❌ $svc: $status"
    fi
done

echo ""
echo "RECURSOS:"
echo "------------------------------"

# Memória
mem_used=$(free -m | awk '/Mem:/{print $3}')
mem_total=$(free -m | awk '/Mem:/{print $2}')
echo "  RAM: ${mem_used}MB / ${mem_total}MB"

# Disco
disk_used=$(df -h / | awk 'NR==2{print $3 "/" $2 " (" $5 ")"}')
echo "  Disco: $disk_used"

# Processos Python
python_procs=$(pgrep -c python 2>/dev/null || echo "0")
echo "  Processos Python: $python_procs"

echo ""
echo "LOGS (últimas 5 linhas do collector):"
echo "------------------------------"
tail -5 /home/betbot/Bets/betinasia_bot/logs/collector.log 2>/dev/null || echo "  (sem logs ainda)"

echo ""
echo "TAMANHO DOS LOGS:"
echo "------------------------------"
du -sh /home/betbot/Bets/betinasia_bot/logs/ 2>/dev/null || echo "  (sem logs)"

echo ""
echo "BANCO DE DADOS (contagem rápida):"
echo "------------------------------"
sudo -u betbot psql -d betinasia_bot -t -c "
  SELECT 'Jogos: ' || COUNT(*) FROM matches
  UNION ALL
  SELECT 'Odds: ' || COUNT(*) FROM best_odds_history
  UNION ALL
  SELECT 'Última coleta: ' || COALESCE(MAX(scraped_at)::text, 'nenhuma') FROM best_odds_history;
" 2>/dev/null || echo "  (banco não acessível)"

echo ""
echo "============================================================"
echo "COMANDOS ÚTEIS:"
echo "  sudo systemctl restart betinasia-collector  # Reiniciar"
echo "  sudo journalctl -u betinasia-collector -f   # Logs live"
echo "  python check_collector_status.py             # Status detalhado"
echo "============================================================"
echo ""
STATUSEOF

    chmod +x "${BOT_DIR}/status.sh"
    chown ${BOT_USER}:${BOT_USER} "${BOT_DIR}/status.sh"
    log_info "Script de status criado: ${BOT_DIR}/status.sh"
}

# ============================================================
# Resumo final
# ============================================================
show_summary() {
    log_step "DEPLOY CONCLUÍDO!"

    echo ""
    echo -e "  ${GREEN}Serviços rodando em background:${NC}"
    echo "    - betinasia-collector: coleta odds a cada ~10 segundos"
    echo "    - betinasia-results: atualiza resultados a cada 4 horas"
    echo "    - betinasia-xvfb: display virtual para Playwright"
    echo ""
    echo -e "  ${GREEN}Os serviços vão:${NC}"
    echo "    ✓ Rodar independente da sua conexão SSH"
    echo "    ✓ Reiniciar automaticamente se crashar"
    echo "    ✓ Iniciar automaticamente no boot do servidor"
    echo "    ✓ Rotacionar logs automaticamente (30 dias)"
    echo ""
    echo -e "  ${YELLOW}Comandos essenciais:${NC}"
    echo ""
    echo "    # Ver status de tudo"
    echo "    sudo bash ${BOT_DIR}/status.sh"
    echo ""
    echo "    # Ver logs em tempo real"
    echo "    sudo journalctl -u betinasia-collector -f"
    echo ""
    echo "    # Parar/Iniciar/Reiniciar coleta"
    echo "    sudo systemctl stop betinasia-collector"
    echo "    sudo systemctl start betinasia-collector"
    echo "    sudo systemctl restart betinasia-collector"
    echo ""
    echo "    # Ver status detalhado do banco"
    echo "    cd ${BOT_DIR} && source ${VENV_DIR}/bin/activate"
    echo "    python check_collector_status.py"
    echo ""
    echo "    # Atualizar código do GitHub"
    echo "    cd ${PROJECT_DIR} && git pull"
    echo "    sudo systemctl restart betinasia-collector"
    echo ""
    echo "    # Ver erros"
    echo "    cat ${LOGS_DIR}/collector_error.log"
    echo ""
}

# ============================================================
# MAIN
# ============================================================

check_root

MODE="${1:---full}"

case "$MODE" in
    --full)
        log_step "DEPLOY COMPLETO (primeira instalação)"
        install_system_deps
        setup_user
        setup_database
        setup_python
        setup_env
        setup_logs
        setup_xvfb
        setup_services
        start_services
        setup_monitoring
        show_summary
        ;;
    --update)
        log_step "ATUALIZAÇÃO (código + dependências)"
        setup_python
        setup_logs
        setup_services
        start_services
        show_summary
        ;;
    --services)
        log_step "REINSTALAÇÃO DE SERVIÇOS"
        setup_logs
        setup_xvfb
        setup_services
        start_services
        show_summary
        ;;
    --start)
        start_services
        ;;
    --stop)
        log_step "Parando serviços..."
        systemctl stop betinasia-collector.service 2>/dev/null || true
        systemctl stop betinasia-results.service 2>/dev/null || true
        log_info "Serviços parados"
        ;;
    *)
        echo "Uso: sudo bash deploy.sh [--full|--update|--services|--start|--stop]"
        echo ""
        echo "  --full      Instalação completa (primeira vez)"
        echo "  --update    Atualiza código e reinicia serviços"
        echo "  --services  Reinstala apenas os serviços systemd"
        echo "  --start     Inicia serviços"
        echo "  --stop      Para serviços"
        exit 1
        ;;
esac
