#!/bin/bash

echo "🚀 Установка проекта Galeries (Next.js + FastAPI)..."
echo ""

# Установка Next.js зависимостей
echo "📦 Установка Next.js зависимостей..."
if command -v npm &> /dev/null; then
    npm install
    echo "✅ Next.js зависимости установлены"
else
    echo "❌ npm не найден. Установите Node.js"
    exit 1
fi

echo ""

# Установка Python зависимостей
echo "📦 Установка Python FastAPI сервера..."
cd python
chmod +x setup.sh
chmod +x start.sh
./setup.sh
cd ..

echo ""
echo "✅ Установка завершена!"
echo ""
echo "📋 Следующие шаги:"
echo ""
echo "1️⃣  Настройте переменные окружения:"
echo "   - Vercel: добавьте переменные в разделе 'Vars'"
echo "   - Python: отредактируйте python/.env"
echo ""
echo "2️⃣  Запустите проекты:"
echo "   - Next.js: npm run dev"
echo "   - FastAPI: cd python && ./start.sh"
echo ""
echo "3️⃣  Откройте в браузере:"
echo "   - Next.js: http://localhost:3000"
echo "   - FastAPI: http://localhost:8001"
echo ""
