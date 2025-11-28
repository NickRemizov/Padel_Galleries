#!/bin/bash

if [ ! -f "server.pid" ]; then
    echo "❌ Файл server.pid не найден. Сервер не запущен?"
    echo "Запустите сервер: ./start-daemon.sh"
    exit 1
fi

PID=$(cat server.pid)
if ! ps -p $PID > /dev/null 2>&1; then
    echo "❌ Сервер не запущен (PID $PID не найден)"
    echo "Запустите сервер: ./start-daemon.sh"
    exit 1
fi

if [ ! -f "server.log" ]; then
    echo "❌ Файл server.log не найден."
    exit 1
fi

echo "📋 Логи сервера (PID: $PID, Ctrl+C для выхода):"
echo ""
tail -f server.log
