#!/bin/bash

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Автоматическая установка Galeries${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Проверка прав root
if [ "$EUID" -ne 0 ]; then 
    echo -e "${RED}Запустите скрипт с правами root: sudo ./auto-install.sh${NC}"
    exit 1
fi

# Установка системных зависимостей
echo -e "${YELLOW}[1/6] Установка системных зависимостей...${NC}"
apt-get update
apt-get install -y python3 python3-pip python3-venv git curl wget

# Переход в директорию python
cd python || exit 1

# Создание .env файла с предустановленными значениями
echo -e "${YELLOW}[2/6] Создание .env файла...${NC}"
cat > .env << 'EOF'
# Server Configuration
SERVER_HOST=0.0.0.0
SERVER_PORT=8001
SERVER_IP=23.88.61.20

# Google OAuth
GOOGLE_CLIENT_ID=
GOOGLE_CLIENT_SECRET=
GOOGLE_REDIRECT_URI=http://23.88.61.20:8001/auth/google/callback

# JWT Configuration
JWT_SECRET_KEY=R=U8"3{$wWtUF]=*[m%s1!ZW!Sl(4_TeRy6J8OU]9=$Gm=7+,Yw@!0-JEMP|?SG
JWT_ALGORITHM=HS256

# CORS Configuration
ALLOWED_ORIGINS=https://padelvalencia.vercel.app,http://localhost:3000

# Supabase Configuration
SUPABASE_URL=${SUPABASE_URL}
SUPABASE_KEY=${SUPABASE_SERVICE_ROLE_KEY}
EOF

echo -e "${GREEN}✓ .env файл создан${NC}"

# Создание виртуального окружения
echo -e "${YELLOW}[3/6] Создание виртуального окружения Python...${NC}"
python3 -m venv venv
source venv/bin/activate

# Установка Python зависимостей
echo -e "${YELLOW}[4/6] Установка Python зависимостей...${NC}"
pip install --upgrade pip
pip install -r requirements.txt

echo -e "${GREEN}✓ Зависимости установлены${NC}"

# Создание необходимых директорий
echo -e "${YELLOW}[5/6] Создание директорий...${NC}"
mkdir -p data models logs

# Создание systemd сервиса для автозапуска
echo -e "${YELLOW}[6/6] Создание systemd сервиса...${NC}"
cat > /etc/systemd/system/galeries-api.service << EOF
[Unit]
Description=Galeries Face Recognition API
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=$(pwd)
Environment="PATH=$(pwd)/venv/bin"
ExecStart=$(pwd)/venv/bin/python main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Перезагрузка systemd и запуск сервиса
systemctl daemon-reload
systemctl enable galeries-api.service
systemctl start galeries-api.service

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Установка завершена!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "FastAPI сервер запущен на: ${GREEN}http://23.88.61.20:8001${NC}"
echo -e "Документация API: ${GREEN}http://23.88.61.20:8001/docs${NC}"
echo ""
echo -e "Управление сервисом:"
echo -e "  Статус:  ${YELLOW}systemctl status galeries-api${NC}"
echo -e "  Стоп:    ${YELLOW}systemctl stop galeries-api${NC}"
echo -e "  Старт:   ${YELLOW}systemctl start galeries-api${NC}"
echo -e "  Рестарт: ${YELLOW}systemctl restart galeries-api${NC}"
echo -e "  Логи:    ${YELLOW}journalctl -u galeries-api -f${NC}"
echo ""
echo -e "${GREEN}Готово! Сервер работает и будет автоматически запускаться при перезагрузке.${NC}"
