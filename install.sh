#!/bin/bash

set -e

echo "=== Установка проекта Galeries ==="

# Проверка, что мы в правильной директории
if [ ! -d "python" ]; then
    echo "Ошибка: директория python не найдена!"
    echo "Убедитесь, что вы запускаете скрипт из корня проекта galeries"
    exit 1
fi

# Установка системных пакетов
echo "Установка системных зависимостей..."
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-venv python3-dev build-essential

# Переход в директорию Python
cd python

# Создание .env файла с РЕАЛЬНЫМИ значениями
echo "Создание .env файла..."
cat > .env << 'EOF'
SERVER_HOST=0.0.0.0
SERVER_PORT=8001
SERVER_IP=23.88.61.20

GOOGLE_CLIENT_ID=4
GOOGLE_CLIENT_SECRET=
GOOGLE_REDIRECT_URI=http://23.88.61.20:8001/auth/google/callback

JWT_SECRET_KEY=R=U8"3{$wWtUF]=*[m%s1!ZW!Sl(4_TeRy6J8OU]9=$Gm=7+,Yw@!0-JEMP|?SG
JWT_ALGORITHM=HS256

ALLOWED_ORIGINS=https://padelvalencia.vercel.app,http://localhost:3000

SUPABASE_URL=https://jczmumbzpqlckbkgznsd.supabase.co
SUPABASE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Impjem11bWJ6cHFsY2tia2d6bnNkIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjA1MDc1MDQsImV4cCI6MjA3NjA4MzUwNH0.qCrrVHh00ozgvAifwOFhC1qbnGdtWEL0II057FNRIdc
EOF

# Удаление старого venv если есть
if [ -d "venv" ]; then
    echo "Удаление старого виртуального окружения..."
    rm -rf venv
fi

# Создание виртуального окружения
echo "Создание виртуального окружения..."
python3 -m venv venv

echo "Установка Python библиотек..."
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

echo ""
echo "✓ Установка завершена!"
echo ""
echo "Для запуска сервера выполните:"
echo "  cd python"
echo "  ./start.sh"
echo ""
echo "Сервер будет доступен на http://23.88.61.20:8001"
