#!/bin/bash

# Скрипт для проверки полей confidence в таблице photo_faces
# Использование: ./scripts/check_confidence.sh

set -e

echo "🔍 Проверка полей confidence в базе данных..."
echo ""

# Проверка что Python скрипт существует
if [ ! -f "/home/nickr/scripts/check_confidence_fields.py" ]; then
    echo "❌ Ошибка: check_confidence_fields.py не найден"
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

# Запуск Python скрипта проверки
cd /home/nickr
python3 scripts/check_confidence_fields.py

# Деактивация venv
deactivate

echo ""
echo "✅ Проверка завершена"
