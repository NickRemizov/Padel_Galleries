#!/bin/bash

echo "🚀 Установка проекта Galeries на сервер..."
echo ""

# Проверка прав
if [ "$EUID" -ne 0 ]; then 
    echo "⚠️  Рекомендуется запускать с sudo для установки системных пакетов"
fi

# Обновление системы
echo "📦 Обновление системы..."
sudo apt-get update

# Установка Python 3 и pip
echo "🐍 Установка Python..."
sudo apt-get install -y python3 python3-pip python3-venv

# Установка Node.js и npm (если нет)
if ! command -v node &> /dev/null; then
    echo "📦 Установка Node.js..."
    curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
    sudo apt-get install -y nodejs
fi

# Установка системных зависимостей для OpenCV и InsightFace
echo "📦 Установка системных зависимостей..."
sudo apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libopenblas-dev

echo ""
echo "✅ Системные зависимости установлены"
echo ""

# Установка Next.js зависимостей
echo "📦 Установка Next.js зависимостей..."
npm install

echo ""
echo "✅ Next.js зависимости установлены"
echo ""

# Установка Python зависимостей
echo "🐍 Установка Python FastAPI сервера..."
cd python

# Создание виртуального окружения
echo "📦 Создание виртуального окружения..."
python3 -m venv venv

# Активация виртуального окружения
source venv/bin/activate

# Обновление pip
pip install --upgrade pip

# Установка зависимостей
echo "📥 Установка Python зависимостей (это может занять несколько минут)..."
pip install -r requirements.txt

# Создание .env файла
if [ ! -f .env ]; then
    echo "📝 Создание .env файла..."
    cp .env.example .env
    echo ""
    echo "⚠️  ВАЖНО: Отредактируйте файл python/.env и добавьте ваши настройки!"
    echo ""
fi

# Создание необходимых директорий
mkdir -p data
mkdir -p models

# Установка прав на выполнение скриптов
chmod +x setup.sh
chmod +x start.sh

cd ..

echo ""
echo "✅✅✅ Установка полностью завершена! ✅✅✅"
echo ""
echo "📋 Следующие шаги:"
echo ""
echo "1️⃣  Настройте переменные окружения в python/.env:"
echo "   nano python/.env"
echo ""
echo "   Обязательные переменные:"
echo "   - SERVER_HOST=0.0.0.0"
echo "   - SERVER_PORT=8001"
echo "   - SERVER_IP=23.88.61.20"
echo "   - SUPABASE_URL=ваш_supabase_url"
echo "   - SUPABASE_KEY=ваш_supabase_key"
echo "   - ALLOWED_ORIGINS=https://padelvalencia.vercel.app,http://localhost:3000"
echo "   - JWT_SECRET_KEY=ваш_секретный_ключ"
echo "   - GOOGLE_CLIENT_ID=ваш_google_client_id"
echo "   - GOOGLE_CLIENT_SECRET=ваш_google_client_secret"
echo ""
echo "2️⃣  Запустите FastAPI сервер:"
echo "   cd python && ./start.sh"
echo ""
echo "3️⃣  Сервер будет доступен на:"
echo "   http://23.88.61.20:8001"
echo ""
echo "4️⃣  Для автозапуска при перезагрузке сервера создайте systemd service"
echo ""
