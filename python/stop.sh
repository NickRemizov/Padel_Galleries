#!/bin/bash

echo "🛑 Остановка FastAPI сервера..."

if [ ! -f "server.pid" ]; then
    echo "❌ Файл server.pid не найден. Сервер не запущен?"
    exit 1
fi

PID=$(cat server.pid)

if ps -p $PID > /dev/null 2>&1; then
    kill $PID
    echo "✅ Сервер остановлен (PID: $PID)"
    rm -f server.pid
else
    echo "⚠️  Процесс с PID $PID не найден"
    rm -f server.pid
fi
