# Развертывание FastAPI сервера v3.2.1

## Быстрый деплой на сервер (Рекомендуется)

### Автоматическая установка

1. Скачайте архив `galeries.zip` и загрузите на сервер в `/home/nickr/`

2. Выполните команды:
\`\`\`bash
cd /home/nickr
sudo rm -rf python
unzip -o galeries.zip
chmod +x SETUP.sh
sudo ./SETUP.sh
cd python && ./start.sh
\`\`\`

Сервер будет доступен на `http://23.88.61.20:8001`

### Что делает SETUP.sh

- Останавливает старый сервер (если запущен)
- Устанавливает Python 3.11 и системные зависимости
- Создает виртуальное окружение
- Устанавливает все Python пакеты из requirements.txt
- Создает .env файл с настройками
- Создает необходимые директории (data, models)

### Что делает start.sh

- Активирует виртуальное окружение
- Запускает FastAPI сервер на порту 8001

---

## Локальная разработка

1. Установите зависимости:
\`\`\`bash
cd python
pip install -r requirements.txt
\`\`\`

2. Создайте файл `.env` на основе `.env.example`:
\`\`\`bash
cp .env.example .env
\`\`\`

3. Настройте переменные окружения в `.env`:
- `SUPABASE_URL` - URL вашего Supabase проекта
- `SUPABASE_SERVICE_ROLE_KEY` - Service role key из Supabase
- `SERVER_HOST` - хост сервера (по умолчанию 0.0.0.0)
- `SERVER_PORT` - порт сервера (по умолчанию 8001)

4. Запустите сервер:
\`\`\`bash
python main.py
\`\`\`

---

## API Endpoints

### Training API (v2)

- `POST /api/v2/config` - Обновить конфигурацию распознавания
- `GET /api/v2/config` - Получить текущую конфигурацию
- `GET /api/v2/train/history` - История обучений
- `POST /api/v2/train/prepare` - Подготовка датасета
- `POST /api/v2/train/execute` - Запуск обучения
- `GET /api/v2/train/status/{session_id}` - Статус обучения

### Recognition API (v1)

- `POST /api/upload-photos` - Загрузка фото для распознавания
- `POST /api/group-players` - Группировка лиц
- `GET /api/health` - Проверка работоспособности

---

## Развертывание на Hetzner

### Вариант 1: Docker (рекомендуется)

1. Создайте `Dockerfile` в папке `python/`:
\`\`\`dockerfile
FROM python:3.10-slim

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Копирование requirements и установка зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование кода
COPY . .

# Создание директорий
RUN mkdir -p data/cache/photos uploads static

EXPOSE 8001

CMD ["python", "main.py"]
\`\`\`

2. Создайте `docker-compose.yml` в корне проекта:
\`\`\`yaml
version: '3.8'

services:
  nextjs:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    environment:
      - FASTAPI_URL=http://fastapi:8001
    depends_on:
      - fastapi

  fastapi:
    build:
      context: ./python
      dockerfile: Dockerfile
    ports:
      - "8001:8001"
    environment:
      - SUPABASE_URL=${SUPABASE_URL}
      - SUPABASE_SERVICE_KEY=${SUPABASE_SERVICE_ROLE_KEY}
    volumes:
      - ./python/data:/app/data
\`\`\`

3. Запустите на сервере:
\`\`\`bash
docker-compose up -d
\`\`\`

### Вариант 2: Systemd service

1. Создайте файл `/etc/systemd/system/fastapi-face-recognition.service`:
\`\`\`ini
[Unit]
Description=FastAPI Face Recognition Service
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/var/www/galeries/python
Environment="PATH=/var/www/galeries/python/venv/bin"
ExecStart=/var/www/galeries/python/venv/bin/python main.py
Restart=always

[Install]
WantedBy=multi-user.target
\`\`\`

2. Запустите сервис:
\`\`\`bash
sudo systemctl daemon-reload
sudo systemctl enable fastapi-face-recognition
sudo systemctl start fastapi-face-recognition
\`\`\`

---

## Nginx конфигурация

Добавьте в конфигурацию Nginx:

\`\`\`nginx
server {
    listen 80;
    server_name your-domain.com;

    # Next.js
    location / {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }

    # FastAPI
    location /api/v2/ {
        proxy_pass http://localhost:8001;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
\`\`\`

---

## Переменные окружения на Vercel

Если вы деплоите Next.js на Vercel, а FastAPI на Hetzner:

1. В Vercel добавьте переменную:
   - `FASTAPI_URL=http://23.88.61.20:8001`
   - `NEXT_PUBLIC_FASTAPI_URL=http://23.88.61.20:8001`

2. Убедитесь, что порт 8001 открыт в файрволе Hetzner:
\`\`\`bash
sudo ufw allow 8001/tcp
\`\`\`

---

## Проверка работоспособности

После деплоя проверьте:

1. **Health check:**
\`\`\`bash
curl http://23.88.61.20:8001/api/health
\`\`\`

Ожидаемый ответ:
\`\`\`json
{
  "status": "healthy",
  "service": "padel-recognition",
  "model_loaded": true
}
\`\`\`

2. **Проверка конфигурации:**
\`\`\`bash
curl http://23.88.61.20:8001/api/v2/config
\`\`\`

3. **Проверка истории обучений:**
\`\`\`bash
curl http://23.88.61.20:8001/api/v2/train/history?limit=10
\`\`\`

---

## Troubleshooting

### Сервер не запускается

1. Проверьте логи:
\`\`\`bash
cd /home/nickr/python
source venv/bin/activate
python main.py
\`\`\`

2. Проверьте .env файл:
\`\`\`bash
cat .env
\`\`\`

### 404 Not Found на /api/v2/config

Убедитесь, что:
1. Сервер запущен на порту 8001
2. В main.py есть строка: `app.include_router(training.router, prefix="/api/v2", tags=["training"])`
3. Файл `routers/training.py` существует и содержит endpoints

### Ошибки импорта модулей

Переустановите зависимости:
\`\`\`bash
cd /home/nickr/python
source venv/bin/activate
pip install -r requirements.txt --force-reinstall
\`\`\`

---

## Версия

**Backend v3.2.1**
- Добавлены training endpoints (/api/v2/*)
- Поддержка фильтрации качества лиц
- Интеграция с Supabase для обучения
