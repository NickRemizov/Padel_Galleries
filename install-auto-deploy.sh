#!/bin/bash

# Скрипт установки автоматического деплоя как systemd сервис

set -e

if [ "$EUID" -ne 0 ]; then 
    echo "❌ ОШИБКА: Запустите скрипт с sudo"
    echo "   sudo ./install-auto-deploy.sh"
    exit 1
fi

echo "=========================================="
echo "  Установка Auto-Deploy Service"
echo "=========================================="
echo ""

# Копирование скрипта
echo "[1/4] Копирование auto-deploy.sh..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="/home/nickr"

if [ "$SCRIPT_DIR" != "$TARGET_DIR" ]; then
    cp "$SCRIPT_DIR/auto-deploy.sh" "$TARGET_DIR/auto-deploy.sh"
    echo "  ✓ Скрипт скопирован из $SCRIPT_DIR в $TARGET_DIR"
else
    echo "  ✓ Скрипт уже находится в $TARGET_DIR"
fi

chmod +x /home/nickr/auto-deploy.sh
chown nickr:nickr /home/nickr/auto-deploy.sh
echo "  ✓ Скрипт скопирован"

# Создание systemd service
echo "[2/4] Создание systemd service..."
cat > /etc/systemd/system/galeries-auto-deploy.service << 'EOF'
[Unit]
Description=Galeries Auto-Deploy Service
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/home/nickr
ExecStart=/home/nickr/auto-deploy.sh
Restart=always
RestartSec=10
StandardOutput=append:/home/nickr/auto-deploy.log
StandardError=append:/home/nickr/auto-deploy.log

[Install]
WantedBy=multi-user.target
EOF
echo "  ✓ Service файл создан"

# Перезагрузка systemd
echo "[3/4] Перезагрузка systemd..."
systemctl daemon-reload
echo "  ✓ systemd перезагружен"

# Запуск и включение автозапуска
echo "[4/4] Запуск сервиса..."
systemctl enable galeries-auto-deploy.service
systemctl start galeries-auto-deploy.service
echo "  ✓ Сервис запущен"

echo ""
echo "=========================================="
echo "  ✅ Установка завершена!"
echo "=========================================="
echo ""
echo "Сервис автоматического деплоя запущен и будет"
echo "автоматически стартовать при перезагрузке сервера."
echo ""
echo "Полезные команды:"
echo "  sudo systemctl status galeries-auto-deploy   # Статус"
echo "  sudo systemctl stop galeries-auto-deploy     # Остановить"
echo "  sudo systemctl start galeries-auto-deploy    # Запустить"
echo "  sudo systemctl restart galeries-auto-deploy  # Перезапустить"
echo "  tail -f /home/nickr/auto-deploy.log          # Логи"
echo ""
echo "Теперь просто загружайте galeries.zip в /home/nickr/"
echo "и деплой произойдет автоматически!"
echo ""
