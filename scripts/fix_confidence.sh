#!/bin/bash

# Скрипт для безопасного исправления полей confidence в таблице photo_faces
# Использование: ./scripts/fix_confidence.sh

set -e

echo "🔧 Исправление полей confidence в базе данных..."
echo ""

# Проверка что Python скрипт существует
if [ ! -f "/home/nickr/scripts/fix_confidence_fields.py" ]; then
    echo "❌ Ошибка: fix_confidence_fields.py не найден"
    exit 1
fi

# Активация виртуального окружения Python
if [ -d "/home/nickr/python/venv" ]; then
    source /home/nickr/python/venv/bin/activate
    echo "✅ Виртуальное окружение активировано"
else
    echo "❌ Виртуальное окружение не найдено в /home/nickr/python/venv"
    exit 1
fi

# Запуск Python скрипта исправления
cd /home/nickr
python3 scripts/fix_confidence_fields.py

# Деактивация venv
deactivate

echo ""
echo "✅ Исправление завершено"
