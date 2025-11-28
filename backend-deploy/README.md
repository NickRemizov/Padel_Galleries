# Galeries Backend v3.2.1

FastAPI сервер для распознавания лиц на турнирах по падел.

## Быстрая установка

\`\`\`bash
cd /home/nickr
sudo rm -rf python
unzip -o galeries.zip
chmod +x SETUP.sh
sudo ./SETUP.sh
cd python && ./start.sh
\`\`\`

## Что нового в v3.2.1

- ✅ Training API endpoints (`/api/v2/train/*`, `/api/v2/config`)
- ✅ Фильтрация качества лиц (det_score, размер, blur_score)
- ✅ Поддержка Supabase для хранения дескрипторов
- ✅ Оптимизированное обучение с переиспользованием дескрипторов
- ✅ Batch recognition для автоматического распознавания

## Структура

\`\`\`
galeries.zip
├── SETUP.sh              # Скрипт установки (запускать с sudo)
└── python/               # Папка с FastAPI кодом
    ├── main.py           # Главный файл приложения
    ├── start.sh          # Скрипт запуска сервера
    ├── requirements.txt  # Python зависимости
    ├── .env.example      # Пример конфигурации
    ├── routers/          # API endpoints
    │   ├── recognition.py
    │   └── training.py
    ├── services/         # Бизнес-логика
    │   ├── auth.py
    │   ├── database.py
    │   ├── face_recognition.py
    │   ├── supabase_client.py
    │   ├── supabase_database.py
    │   └── training_service.py
    └── models/           # Pydantic схемы
        └── schemas.py
\`\`\`

## Endpoints

### Recognition API (v1)
- `POST /api/detect-faces` - Детекция лиц на изображении
- `POST /api/recognize-face` - Распознавание лица по эмбеддингу
- `POST /api/cluster-unknown-faces` - Кластеризация неизвестных лиц
- `POST /api/regenerate-unknown-descriptors` - Регенерация дескрипторов
- `POST /api/rebuild-index` - Перестроение HNSWLIB индекса

### Training API (v2)
- `GET /api/v2/config` - Получить конфигурацию распознавания
- `PUT /api/v2/config` - Обновить конфигурацию
- `GET /api/v2/train/history` - История обучений
- `POST /api/v2/train/prepare` - Подготовка датасета
- `POST /api/v2/train/execute` - Запуск обучения
- `GET /api/v2/train/status/{sessionId}` - Статус обучения

## Требования

- Python 3.11
- Ubuntu/Debian Linux
- 4GB RAM минимум
- Supabase database

## Переменные окружения

Создаются автоматически в `python/.env` при запуске `SETUP.sh`:

\`\`\`env
SERVER_HOST=0.0.0.0
SERVER_PORT=8001
SUPABASE_URL=...
SUPABASE_SERVICE_ROLE_KEY=...
GOOGLE_CLIENT_ID=...
GOOGLE_CLIENT_SECRET=...
JWT_SECRET_KEY=...
ALLOWED_ORIGINS=...
\`\`\`

## Troubleshooting

### Сервер не запускается
\`\`\`bash
cd /home/nickr/python
source venv/bin/activate
python main.py
\`\`\`

### Проверка логов
\`\`\`bash
tail -f /home/nickr/python/server.log
\`\`\`

### Проверка процесса
\`\`\`bash
ps aux | grep python
\`\`\`

### Остановка сервера
\`\`\`bash
pkill -f "python main.py"
\`\`\`

## Версия

**Backend v3.2.1** - Training API + Quality Filters
