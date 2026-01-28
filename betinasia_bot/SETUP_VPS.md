# Guia de Setup da VPS - BetinAsia Bot

Este guia mostra como configurar sua VPS DigitalOcean do zero.

## 1. Primeiro Acesso à VPS

### 1.1 Conectando via SSH

Após criar o droplet, você receberá um email com o IP e senha.

**No Windows:**
- Baixe e instale o [PuTTY](https://www.putty.org/) ou use o Windows Terminal
- Ou use o PowerShell: `ssh root@SEU_IP_AQUI`

**No Mac/Linux:**
```bash
ssh root@SEU_IP_AQUI
```

Na primeira conexão, digite "yes" quando perguntar sobre fingerprint.

### 1.2 Mudando a senha (primeira vez)

O sistema vai pedir para mudar a senha no primeiro login.

---

## 2. Configuração Inicial do Servidor

### 2.1 Atualizar o sistema

```bash
apt update && apt upgrade -y
```

### 2.2 Instalar dependências do sistema

```bash
# Ferramentas essenciais
apt install -y git curl wget htop nano

# Python 3.11+
apt install -y python3 python3-pip python3-venv python3-dev

# PostgreSQL
apt install -y postgresql postgresql-contrib

# Redis
apt install -y redis-server

# Dependências do Playwright (navegador)
apt install -y libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 \
    libcups2 libdrm2 libxkbcommon0 libxcomposite1 libxdamage1 \
    libxfixes3 libxrandr2 libgbm1 libasound2
```

### 2.3 Verificar instalações

```bash
python3 --version    # Deve ser 3.10+
psql --version       # PostgreSQL
redis-cli ping       # Deve responder PONG
```

---

## 3. Configurar PostgreSQL

### 3.1 Criar usuário e banco de dados

```bash
# Acessar o PostgreSQL
sudo -u postgres psql
```

Dentro do prompt do PostgreSQL:

```sql
-- Criar usuário
CREATE USER betbot WITH PASSWORD 'COLOQUE_UMA_SENHA_SEGURA_AQUI';

-- Criar banco de dados
CREATE DATABASE betinasia_bot OWNER betbot;

-- Dar permissões
GRANT ALL PRIVILEGES ON DATABASE betinasia_bot TO betbot;

-- Sair
\q
```

### 3.2 Testar conexão

```bash
psql -U betbot -d betinasia_bot -h localhost
# Digite a senha quando pedir
# Se conectar, digite \q para sair
```

---

## 4. Configurar Redis

O Redis já deve estar rodando após a instalação.

```bash
# Verificar status
systemctl status redis

# Se não estiver rodando:
systemctl start redis
systemctl enable redis  # Iniciar automaticamente no boot
```

---

## 5. Criar Usuário da Aplicação (Segurança)

Não é boa prática rodar aplicações como root.

```bash
# Criar usuário
adduser betbot
# Responda as perguntas (ou pressione Enter para pular)

# Dar acesso sudo (opcional, para manutenção)
usermod -aG sudo betbot

# Mudar para o usuário
su - betbot
```

---

## 6. Clonar e Configurar o Projeto

### 6.1 Clonar do GitHub

```bash
# Como usuário betbot
cd ~
git clone https://github.com/SEU_USUARIO/Bets.git
cd Bets/betinasia_bot
```

### 6.2 Criar ambiente virtual Python

```bash
python3 -m venv venv
source venv/bin/activate
```

### 6.3 Instalar dependências

```bash
pip install --upgrade pip
pip install -r requirements.txt

# Instalar browsers do Playwright
playwright install chromium
playwright install-deps
```

### 6.4 Configurar variáveis de ambiente

```bash
# Copiar arquivo de exemplo
cp .env.example .env

# Editar com suas configurações
nano .env
```

Preencha:
- `BETINASIA_USERNAME` e `BETINASIA_PASSWORD`
- `DATABASE_URL` com a senha que criou
- `TELEGRAM_BOT_TOKEN` e `TELEGRAM_CHAT_ID` (opcional)

Salvar: `Ctrl+O`, Enter, `Ctrl+X`

---

## 7. Inicializar Banco de Dados

```bash
# Ativar ambiente virtual (se não estiver)
source venv/bin/activate

# Rodar script de migração
python -m storage.migrate
```

---

## 8. Testar a Aplicação

### 8.1 Teste rápido do scraper

```bash
python -m scraper.test_connection
```

### 8.2 Rodar em modo de desenvolvimento

```bash
python main.py --dry-run
```

---

## 9. Configurar para Rodar 24/7 (Systemd)

### 9.1 Criar arquivo de serviço

```bash
sudo nano /etc/systemd/system/betinasia-bot.service
```

Conteúdo:

```ini
[Unit]
Description=BetinAsia Bot
After=network.target postgresql.service redis.service

[Service]
Type=simple
User=betbot
WorkingDirectory=/home/betbot/Bets/betinasia_bot
Environment=PATH=/home/betbot/Bets/betinasia_bot/venv/bin
ExecStart=/home/betbot/Bets/betinasia_bot/venv/bin/python main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### 9.2 Ativar e iniciar serviço

```bash
sudo systemctl daemon-reload
sudo systemctl enable betinasia-bot
sudo systemctl start betinasia-bot
```

### 9.3 Verificar status

```bash
sudo systemctl status betinasia-bot
```

### 9.4 Ver logs em tempo real

```bash
sudo journalctl -u betinasia-bot -f
```

---

## 10. Comandos Úteis

### Gerenciar o serviço

```bash
# Parar
sudo systemctl stop betinasia-bot

# Reiniciar
sudo systemctl restart betinasia-bot

# Ver logs
sudo journalctl -u betinasia-bot -n 100

# Ver logs em tempo real
sudo journalctl -u betinasia-bot -f
```

### Monitorar o servidor

```bash
# Uso de CPU/RAM
htop

# Espaço em disco
df -h

# Processos Python
ps aux | grep python
```

### Atualizar código

```bash
cd ~/Bets
git pull
sudo systemctl restart betinasia-bot
```

---

## 11. Troubleshooting

### Erro de conexão com PostgreSQL

```bash
# Verificar se está rodando
sudo systemctl status postgresql

# Ver logs
sudo tail -f /var/log/postgresql/postgresql-*-main.log
```

### Erro de conexão com Redis

```bash
# Verificar se está rodando
sudo systemctl status redis

# Testar conexão
redis-cli ping
```

### Playwright não funciona

```bash
# Reinstalar dependências
playwright install-deps

# Testar browser
python -c "from playwright.sync_api import sync_playwright; p = sync_playwright().start(); b = p.chromium.launch(); print('OK'); b.close(); p.stop()"
```

### Ver uso de memória do bot

```bash
ps aux | grep python | grep main.py
```

---

## 12. Backup

### Backup do banco de dados

```bash
# Criar backup
pg_dump -U betbot betinasia_bot > backup_$(date +%Y%m%d).sql

# Restaurar backup
psql -U betbot betinasia_bot < backup_20260128.sql
```

### Backup das configurações

```bash
# Copiar .env para lugar seguro
cp .env ~/.env_backup
```

---

## Próximos Passos

1. Testar conexão com BetinAsia
2. Rodar coleta de dados por 2-4 semanas
3. Treinar modelo com dados coletados
4. Ativar execução de apostas (DRY_RUN=false)
