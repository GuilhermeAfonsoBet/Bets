# Guia de Setup da VPS - BetinAsia Bot

## Visão Geral

O sistema coleta odds de futebol do BetinAsia 24/7, rodando como serviço no Linux (systemd). Mesmo que você desconecte do SSH, o bot continua rodando.

**Serviços:**
- `betinasia-collector` — Coleta contínua de odds (a cada ~10 segundos)
- `betinasia-results` — Atualiza resultados dos jogos (a cada 4 horas)
- `betinasia-xvfb` — Display virtual para o Playwright
- `betinasia-report-b808.timer` — (opcional) Gera relatório PDF e faz push no GitHub

---

## Setup Rápido (Recomendado)

### 1. Conectar na VPS

```bash
ssh root@SEU_IP_AQUI
```

### 2. Clonar o repositório

```bash
# Criar usuário (se primeira vez)
adduser betbot
usermod -aG sudo betbot
su - betbot

# Clonar
cd ~
git clone https://github.com/SEU_USUARIO/Bets.git
cd Bets/betinasia_bot
exit  # Volta para root
```

### 3. Configurar credenciais

```bash
# Copiar e editar .env
sudo -u betbot cp /home/betbot/Bets/betinasia_bot/.env.example /home/betbot/Bets/betinasia_bot/.env
sudo -u betbot nano /home/betbot/Bets/betinasia_bot/.env
```

Preencha obrigatoriamente:
- `BETINASIA_USERNAME` — Seu usuário do BetinAsia
- `BETINASIA_PASSWORD` — Sua senha do BetinAsia
- `DATABASE_URL` — `postgresql://betbot:betbot_secure_2026@localhost:5432/betinasia_bot`
- `API_FOOTBALL_KEY` — chave do API-Football (api-sports.io) para sincronizar placares e habilitar ROI

### 4. Executar deploy

```bash
sudo bash /home/betbot/Bets/betinasia_bot/deploy.sh --full
```

**Pronto!** O bot está rodando em background.

---

## Comandos do Dia-a-Dia

### Ver status de tudo
```bash
sudo bash /home/betbot/Bets/betinasia_bot/status.sh
```

### Ver logs em tempo real
```bash
# Logs do coletor
sudo journalctl -u betinasia-collector -f

# Ou os arquivos de log
tail -f /home/betbot/Bets/betinasia_bot/logs/collector.log
```

### Controlar serviços
```bash
# Parar
sudo systemctl stop betinasia-collector

# Iniciar
sudo systemctl start betinasia-collector

# Reiniciar
sudo systemctl restart betinasia-collector

# Ver status detalhado
sudo systemctl status betinasia-collector
```

### Dica (systemd): quando `.env` não “pega”
Se um serviço usa `EnvironmentFile=.../.env` mas as variáveis parecem “travadas”, verifique se há overrides em `/etc/systemd/system/<service>.service.d/*.conf` (criados via `systemctl edit`). Linhas `Environment="X=..."` nesses drop-ins **sobrescrevem** o `.env`.

```bash
sudo systemctl show <service> -p EnvironmentFiles --no-pager
sudo systemctl show <service> -p Environment --no-pager
sudo systemctl cat <service> --no-pager | sed -n '1,160p'
sudo ls -la /etc/systemd/system/<service>.service.d/ || true
```

### Ver estatísticas do banco
```bash
cd /home/betbot/Bets/betinasia_bot
source /home/betbot/Bets/venv/bin/activate
python check_collector_status.py
```

### Atualizar código do GitHub
```bash
cd /home/betbot/Bets
git pull
sudo bash betinasia_bot/deploy.sh --update
```

---

## Relatório PDF automático (b808) com push no GitHub (opcional)

Objetivo: você abrir sempre o PDF direto do GitHub, sem ter que rodar manualmente.

### 1) Preparar autenticação git não-interativa (1x)

O `systemd` não pode ficar pedindo senha. Opções:

- **Opção A (recomendada)**: GitHub CLI com device login (não precisa lembrar senha)

```bash
sudo apt update && sudo apt install -y gh
gh auth login -h github.com -p https -w
gh auth setup-git
```

- **Opção B**: SSH (deploy key). Requer adicionar a chave no GitHub.

### 2) Instalar unit + timer

```bash
sudo cp /home/betbot/Bets/betinasia_bot/betinasia-report-b808.service /etc/systemd/system/betinasia-report-b808.service
sudo cp /home/betbot/Bets/betinasia_bot/betinasia-report-b808.timer /etc/systemd/system/betinasia-report-b808.timer
sudo systemctl daemon-reload
sudo systemctl enable --now betinasia-report-b808.timer
```

### 3) Ver status/logs

```bash
systemctl list-timers --all | grep betinasia-report-b808 || true
sudo systemctl status betinasia-report-b808.timer --no-pager
sudo journalctl -u betinasia-report-b808.service -n 200 --no-pager
```

O PDF será gerado em `betinasia_bot/docs/analise_contexto_operacao_b808_robusta.pdf` e o serviço tentará `commit + push`.

---

## Monitoramento Automático

### Health Check a cada 5 minutos

```bash
# Instalar cron de health check
(sudo crontab -l 2>/dev/null; echo "*/5 * * * * /home/betbot/Bets/betinasia_bot/healthcheck.sh >> /home/betbot/Bets/betinasia_bot/logs/healthcheck.log 2>&1") | sudo crontab -
```

O health check:
- Verifica se o serviço está rodando
- Reinicia automaticamente se parou
- Verifica se há dados recentes (alerta se > 15 min sem coleta)
- Verifica espaço em disco

---

## Backup do Banco

```bash
# Backup manual
sudo -u betbot pg_dump betinasia_bot > backup_$(date +%Y%m%d_%H%M).sql

# Restaurar
sudo -u betbot psql betinasia_bot < backup_XXXXXXXX_XXXX.sql
```

### Backup automático diário

```bash
# Cria script de backup
cat > /home/betbot/backup_db.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/home/betbot/backups"
mkdir -p $BACKUP_DIR
pg_dump betinasia_bot | gzip > "$BACKUP_DIR/betinasia_$(date +%Y%m%d).sql.gz"
# Remove backups com mais de 30 dias
find $BACKUP_DIR -name "*.sql.gz" -mtime +30 -delete
EOF
chmod +x /home/betbot/backup_db.sh

# Agenda no cron (diário às 3h)
(sudo -u betbot crontab -l 2>/dev/null; echo "0 3 * * * /home/betbot/backup_db.sh") | sudo -u betbot crontab -
```

---

## Troubleshooting

### Bot parou de coletar

```bash
# 1. Ver status
sudo systemctl status betinasia-collector

# 2. Ver últimos logs de erro
tail -50 /home/betbot/Bets/betinasia_bot/logs/collector_error.log

# 3. Ver logs do journald
sudo journalctl -u betinasia-collector -n 100 --no-pager

# 4. Reiniciar
sudo systemctl restart betinasia-collector
```

### Sessão expirou (login)

O Playwright faz login automático. Se persistir:

```bash
# Para o serviço
sudo systemctl stop betinasia-collector

# Faz login manual para salvar sessão
cd /home/betbot/Bets/betinasia_bot
source /home/betbot/Bets/venv/bin/activate
DISPLAY=:99 python -c "
from scraper.betinasia import BetinAsiaScraper
import asyncio
async def login():
    s = BetinAsiaScraper()
    await s.start()
    await s.login()
    await s.close()
asyncio.run(login())
"

# Reinicia
sudo systemctl start betinasia-collector
```

### Memória alta

```bash
# Ver uso de memória
htop

# Reiniciar serviço (limpa memória)
sudo systemctl restart betinasia-collector

# Limpar dados antigos do banco (opcional)
cd /home/betbot/Bets/betinasia_bot
source /home/betbot/Bets/venv/bin/activate
python cleanup_old_data.py
```

### PostgreSQL não conecta

```bash
# Verificar status
sudo systemctl status postgresql

# Reiniciar
sudo systemctl restart postgresql

# Ver logs
sudo tail -f /var/log/postgresql/postgresql-*-main.log
```

---

## Especificações Recomendadas da VPS

| Recurso | Mínimo | Recomendado |
|---------|--------|-------------|
| CPU     | 1 vCPU | 2 vCPUs     |
| RAM     | 2 GB   | 4 GB        |
| Disco   | 25 GB  | 50 GB       |
| OS      | Ubuntu 22.04+ | Ubuntu 24.04 |

**Estimativa de uso de disco por mês:**
- Banco PostgreSQL: ~500 MB/mês (depende do volume de jogos)
- Logs: ~200 MB/mês (com rotação configurada)
- Total: ~700 MB/mês

---

## Arquitetura

```
VPS (Ubuntu)
├── systemd
│   ├── betinasia-xvfb.service      # Display virtual
│   ├── betinasia-collector.service  # Coleta de odds (principal)
│   └── betinasia-results.service    # Atualização de resultados
├── logrotate
│   └── /etc/logrotate.d/betinasia   # Rotação de logs diária
├── cron
│   ├── healthcheck (5 em 5 min)     # Monitoramento
│   └── backup_db (diário 3h)        # Backup do banco
└── /home/betbot/Bets/
    ├── venv/                         # Ambiente virtual Python
    └── betinasia_bot/
        ├── .env                      # Credenciais (não commitado)
        ├── logs/                     # Logs do sistema
        ├── collector/                # Coletor contínuo
        ├── scraper/                  # Scraper BetinAsia
        ├── storage/                  # Banco de dados
        ├── hypothesis/               # Detectores de hipóteses
        └── results/                  # Atualização de resultados
```
