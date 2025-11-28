#!/bin/bash

# ⚠️  Этот скрипт устарел. Используйте ./SETUP.sh
exit 0

echo "🔧 Настройка .env файла..."

# Создаем .env из .env.example если его нет
if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        cp .env.example .env
        echo "✅ Создан .env из .env.example"
    else
        echo "❌ .env.example не найден!"
        exit 1
    fi
fi

# Функция для обновления переменной в .env
update_env_var() {
    local key=$1
    local value=$2
    
    if [ -z "$value" ]; then
        echo "⚠️  $key не задана в окружении"
        return
    fi
    
    # Проверяем, есть ли уже эта переменная в .env
    if grep -q "^${key}=" .env; then
        # Обновляем существующую
        sed -i "s|^${key}=.*|${key}=${value}|" .env
        echo "✅ Обновлена $key"
    else
        # Добавляем новую
        echo "${key}=${value}" >> .env
        echo "✅ Добавлена $key"
    fi
}

echo ""
echo "📝 Заполнение переменных из окружения..."

# Supabase переменные
update_env_var "SUPABASE_URL" "$SUPABASE_URL"
update_env_var "SUPABASE_SERVICE_ROLE_KEY" "$SUPABASE_SERVICE_ROLE_KEY"
update_env_var "SUPABASE_ANON_KEY" "$SUPABASE_ANON_KEY"

# Google OAuth
update_env_var "GOOGLE_CLIENT_ID" "$GOOGLE_CLIENT_ID"
update_env_var "GOOGLE_CLIENT_SECRET" "$GOOGLE_CLIENT_SECRET"

# JWT Secret
update_env_var "JWT_SECRET_KEY" "$JWT_SECRET_KEY"

# Server config
update_env_var "SERVER_HOST" "$SERVER_HOST"
update_env_var "SERVER_PORT" "$SERVER_PORT"
update_env_var "SERVER_IP" "$SERVER_IP"

# CORS Origins
update_env_var "ALLOWED_ORIGINS" "$ALLOWED_ORIGINS"

echo ""
echo "✅ Настройка .env завершена!"
echo ""
echo "⚠️  ВАЖНО: Проверьте файл .env и убедитесь, что все переменные заполнены:"
echo "   nano .env"
echo ""
