#!/bin/bash

echo "🚀 Запуск FastAPI сервера в фоновом режиме..."

# Проверка виртуального окружения
if [ ! -d "venv" ]; then
    echo "❌ Виртуальное окружение не найдено!"
    echo "Запустите сначала ./SETUP.sh"
    exit 1
fi

# Проверка .env файла
if [ ! -f .env ]; then
    echo "❌ Файл .env не найден!"
    echo "Запустите ./SETUP.sh для создания .env файла"
    exit 1
fi

# Загрузка переменных из .env
set -a
source .env
set +a

# Проверка обязательных переменных
if [ -z "$SUPABASE_URL" ] || [ -z "$SUPABASE_SERVICE_ROLE_KEY" ]; then
    echo "❌ Ошибка: SUPABASE_URL и SUPABASE_SERVICE_ROLE_KEY должны быть заданы в .env"
    echo "Отредактируйте файл .env:"
    echo "  nano .env"
    exit 1
fi

# Остановка старого процесса (если есть)
if [ -f "server.pid" ]; then
    OLD_PID=$(cat server.pid)
    if ps -p $OLD_PID > /dev/null 2>&1; then
        echo "🛑 Остановка старого процесса (PID: $OLD_PID)..."
        kill $OLD_PID
        sleep 2
    fi
    rm -f server.pid
fi

# Удаление старого лога
rm -f server.log

# Запуск сервера в фоновом режиме
echo "▶️  Запуск сервера на http://0.0.0.0:${SERVER_PORT:-8001}..."
nohup venv/bin/uvicorn main:app --host ${SERVER_HOST:-0.0.0.0} --port ${SERVER_PORT:-8001} > server.log 2>&1 &

# Сохранение PID
echo $! > server.pid

# Ждем запуска
sleep 3

# Проверяем что сервер запустился
if ps -p $(cat server.pid) > /dev/null 2>&1; then
    echo "✅ Сервер запущен! PID: $(cat server.pid)"
    echo ""
    echo "Для просмотра логов: ./logs.sh"
    echo "Для остановки сервера: ./stop.sh"
else
    echo "❌ Ошибка запуска сервера! Проверьте логи:"
    echo ""
    cat server.log
    rm -f server.pid
    exit 1
fi
