#!/bin/bash

set -e

echo "=========================================="
echo "  Установка FastAPI сервера Galeries"
echo "  Backend v3.2.1"
echo "=========================================="
echo ""

if [ "$EUID" -ne 0 ]; then 
    echo "❌ ОШИБКА: Запустите скрипт с sudo"
    echo "   sudo ./SETUP.sh"
    exit 1
fi

echo "[1/7] Остановка старого сервера (если запущен)..."
pkill -f "python main.py" 2>/dev/null || true
sleep 2
echo "  ✓ Старый сервер остановлен"

echo "[2/7] Установка системных зависимостей и Python 3.11..."
apt-get update -qq
apt-get install -y software-properties-common > /dev/null 2>&1
add-apt-repository ppa:deadsnakes/ppa -y > /dev/null 2>&1
apt-get update -qq
apt-get install -y python3.11 python3.11-venv python3.11-dev build-essential > /dev/null 2>&1
echo "  ✓ Python 3.11 установлен"

echo "[3/7] Переход в директорию python..."
cd python || { echo "❌ Папка python не найдена!"; exit 1; }

echo "[4/7] Создание .env файла..."
cat > .env << 'EOF'
SERVER_HOST=0.0.0.0
SERVER_PORT=8001
SERVER_IP=23.88.61.20

GOOGLE_CLIENT_ID=
GOOGLE_CLIENT_SECRET=

JWT_SECRET_KEY=

ALLOWED_ORIGINS=https://padelvalencia.vercel.app,http://localhost:3000

SUPABASE_URL=https://jczmumbzpqlckbkgznsd.supabase.co
SUPABASE_ANON_KEY=
SUPABASE_SERVICE_ROLE_KEY=

POSTGRES_URL=

BLOB_READ_WRITE_TOKEN=
EOF
echo "  ✓ Файл .env создан"

if [ -d "venv" ]; then
    echo "  Удаление старого виртуального окружения..."
    rm -rf venv
fi

echo "[5/7] Создание виртуального окружения..."
python3.11 -m venv venv || { echo "❌ Не удалось создать venv!"; exit 1; }
echo "  ✓ Виртуальное окружение создано"

echo "[6/7] Обновление pip..."
source venv/bin/activate
python -m pip install --upgrade pip setuptools wheel --quiet
echo "  ✓ pip обновлен"

echo "[7/7] Установка зависимостей (это займет 2-3 минуты)..."
pip install -r requirements.txt --quiet
echo "  ✓ Зависимости установлены"

echo "  Создание необходимых директорий..."
mkdir -p data/cache/photos
mkdir -p models
mkdir -p uploads
echo "  ✓ Директории созданы"

chmod +x *.sh
echo "  ✓ Права на выполнение установлены"

echo ""
echo "=========================================="
echo "  ✓ Установка завершена успешно!"
echo "  Backend v3.2.1 готов к запуску"
echo "=========================================="
echo ""
echo "Для запуска сервера выполните:"
echo "  cd python && ./start.sh"
echo ""
echo "Сервер будет доступен на:"
echo "  http://23.88.61.20:8001"
echo "  http://23.88.61.20:8001/docs (API документация)"
echo ""
