#!/bin/bash

echo "🔄 Перезапуск FastAPI бэкенда..."
echo ""

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 1. Убить все процессы uvicorn
echo "1️⃣ Остановка всех процессов uvicorn..."
pkill -f "uvicorn.*8001" 2>/dev/null
sleep 2

# Проверить что все убиты
if pgrep -f "uvicorn.*8001" > /dev/null; then
    echo -e "${RED}❌ Процессы все еще работают, убиваем force${NC}"
    pkill -9 -f "uvicorn.*8001"
    sleep 2
fi

if ! pgrep -f "uvicorn.*8001" > /dev/null; then
    echo -e "${GREEN}✅ Все процессы остановлены${NC}"
else
    echo -e "${RED}❌ Не удалось остановить процессы${NC}"
    ps aux | grep uvicorn
    exit 1
fi

echo ""

# 2. Перейти в директорию проекта
echo "2️⃣ Переход в директорию проекта..."
cd /home/nickr/scripts || exit 1
echo -e "${GREEN}✅ В директории: $(pwd)${NC}"
echo ""

# 3. Проверить структуру файлов
echo "3️⃣ Проверка структуры файлов..."
if [ ! -f "main.py" ]; then
    echo -e "${RED}❌ Файл main.py не найден!${NC}"
    ls -la
    exit 1
fi

if [ ! -d "python" ]; then
    echo -e "${RED}❌ Директория python/ не найдена!${NC}"
    ls -la
    exit 1
fi

echo -e "${GREEN}✅ Структура файлов корректна${NC}"
echo ""

# 4. Активировать виртуальное окружение
echo "4️⃣ Активация виртуального окружения..."
if [ -f "/home/nickr/python/venv/bin/activate" ]; then
    source /home/nickr/python/venv/bin/activate
    echo -e "${GREEN}✅ Виртуальное окружение активировано${NC}"
else
    echo -e "${YELLOW}⚠️  Виртуальное окружение не найдено, используем системный Python${NC}"
fi

# Проверить Python и uvicorn
python3 --version
which uvicorn || echo -e "${RED}❌ uvicorn не найден${NC}"
echo ""

# 5. Проверить переменные окружения
echo "5️⃣ Проверка переменных окружения..."
if [ -f ".env" ]; then
    echo -e "${GREEN}✅ Файл .env найден${NC}"
    export $(cat .env | grep -v '^#' | xargs)
else
    echo -e "${YELLOW}⚠️  Файл .env не найден${NC}"
fi

if [ -z "$PV_POSTGRES_URL" ]; then
    echo -e "${RED}❌ PV_POSTGRES_URL не установлен!${NC}"
else
    echo -e "${GREEN}✅ PV_POSTGRES_URL установлен${NC}"
fi
echo ""

# 6. Проверить импорты в main.py
echo "6️⃣ Проверка импортов Python..."
python3 -c "import sys; sys.path.insert(0, '.'); from main import app; print('✅ Импорты успешны')" 2>/tmp/python_check.log
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Ошибка импортов:${NC}"
    cat /tmp/python_check.log
    echo ""
    echo "Попытка запуска с python.main..."
fi
echo ""

# 7. Запустить FastAPI
echo "7️⃣ Запуск FastAPI..."
echo "Команда: python -m uvicorn main:app --host 0.0.0.0 --port 8001"
echo ""

# Очистить старый лог
> /tmp/fastapi.log

# Попробовать запустить
nohup python -m uvicorn main:app --host 0.0.0.0 --port 8001 > /tmp/fastapi.log 2>&1 &
FASTAPI_PID=$!

echo "PID процесса: $FASTAPI_PID"
echo "Ожидание запуска (5 секунд)..."
sleep 5

# 8. Проверить что запустился
echo ""
echo "8️⃣ Проверка статуса..."

if ps -p $FASTAPI_PID > /dev/null; then
    echo -e "${GREEN}✅ Процесс работает (PID: $FASTAPI_PID)${NC}"
else
    echo -e "${RED}❌ Процесс упал!${NC}"
    echo ""
    echo "Последние 30 строк лога:"
    tail -30 /tmp/fastapi.log
    exit 1
fi

# 9. Проверить доступность API
echo ""
echo "9️⃣ Проверка доступности API..."
sleep 2

for i in {1..10}; do
    if curl -s http://localhost:8001/health > /dev/null; then
        echo -e "${GREEN}✅ API доступен на http://localhost:8001${NC}"
        echo ""
        echo "✨ FastAPI успешно запущен!"
        echo ""
        echo "Проверить статус: ps aux | grep uvicorn"
        echo "Просмотр логов: tail -f /tmp/fastapi.log"
        echo "Остановка: kill $FASTAPI_PID"
        exit 0
    fi
    echo "Попытка $i/10..."
    sleep 1
done

echo -e "${RED}❌ API не отвечает после 10 секунд${NC}"
echo ""
echo "Последние 50 строк лога:"
tail -50 /tmp/fastapi.log
exit 1
