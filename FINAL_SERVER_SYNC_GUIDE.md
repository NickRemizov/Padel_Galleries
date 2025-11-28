# Финальная Документация: Синхронизация Backend Сервера

**Дата:** 20 ноября 2025
**Статус:** Миграция Supabase → PostgreSQL завершена на 98%

## ⚠️ КРИТИЧЕСКИ ВАЖНО

Все backend файлы должны быть синхронизированы между v0 проектом и сервером для целостности системы.

---

## 📋 Список файлов для замены на сервере

### 1. Файлы, которые УЖЕ ОБНОВЛЕНЫ на сервере (не трогать)

Эти файлы уже скопированы из v0 и работают:

- ✅ `python/services/postgres_client.py` - PostgreSQL клиент (19KB, обновлен 23:16)
- ✅ `python/routers/training.py` - роутер обучения (8.1KB, обновлен 10:45)
- ✅ `python/routers/recognition.py` - роутер распознавания (32KB, обновлен 00:26)
- ✅ `python/services/face_recognition.py` - сервис распознавания (47KB, обновлен 00:19)
- ✅ `python/routers/config.py` - роутер конфигурации (2.2KB, обновлен 23:29)
- ✅ `python/main.py` - главный файл FastAPI (скопирован)

### 2. Файлы, которые НУЖНО ЗАМЕНИТЬ (если еще не сделано)

**ВАЖНО:** Проверьте даты модификации файлов на сервере. Если файл старше 20 ноября 2025, замените его.

#### Проверка версий на сервере:

\`\`\`bash
cd /home/nickr/python

# Проверяем даты всех ключевых файлов
ls -lh services/postgres_client.py services/training_service.py services/face_recognition.py
ls -lh routers/training.py routers/recognition.py routers/config.py
ls -lh main.py

# Ожидаемые даты (20 ноября или позже):
# postgres_client.py: 19 Nov 23:16 или позже
# training_service.py: 19 Nov 23:44 или позже
# face_recognition.py: 20 Nov 00:19 или позже
# training.py: 20 Nov 10:45 или позже
# recognition.py: 20 Nov 00:26 или позже
# config.py: 19 Nov 23:29 или позже
# main.py: должен быть обновлен
\`\`\`

#### Если нужно обновить:

1. Скачайте ZIP архив из v0 проекта (кнопка "Download ZIP")
2. Извлеките файлы на сервер:

\`\`\`bash
cd /home/nickr/python

# Делаем бэкап ПЕРЕД заменой
tar -czf backup_before_sync_$(date +%Y%m%d_%H%M%S).tar.gz \
  services/postgres_client.py \
  services/training_service.py \
  services/face_recognition.py \
  routers/training.py \
  routers/recognition.py \
  routers/config.py \
  main.py

# Скопируйте файлы из ZIP архива v0 в соответствующие папки

# После копирования перезапустите FastAPI:
pkill -9 -f uvicorn
source venv/bin/activate
nohup uvicorn main:app --host 0.0.0.0 --port 8001 --reload > fastapi.log 2>&1 &
sleep 5
tail -30 fastapi.log
\`\`\`

### 3. Файлы, которые НУЖНО УДАЛИТЬ/ПЕРЕИМЕНОВАТЬ

Старые Supabase файлы больше не используются:

\`\`\`bash
cd /home/nickr/python

# Эти файлы уже переименованы (не трогать если уже сделано):
# services/supabase_client_old.py
# services/supabase_database.py (используется для локального кэша, не трогать)

# Если supabase_client.py все еще существует (НЕ с суффиксом _old):
if [ -f services/supabase_client.py ]; then
  echo "WARNING: supabase_client.py still exists without _old suffix!"
  mv services/supabase_client.py services/supabase_client_backup_$(date +%Y%m%d).py
fi
\`\`\`

---

## 🔍 Проверка упоминаний Supabase в коде

### Разрешенные упоминания:

- ✅ `python/routers/training.py:196` - комментарий "# Try from Supabase first"
- ✅ `python/services/face_recognition.py:32` - `self.supabase_db = None` (инициализируется как None)
- ✅ `python/services/supabase_client.py` - старый файл (переименован в _old.py)
- ✅ `python/services/supabase_database.py` - используется для ЛОКАЛЬНОГО SQLite кэша (не Supabase API)

### Недопустимые упоминания:

❌ Активное использование `SupabaseClient` в коде
❌ Импорты `from services.supabase_client import`
❌ Обращения к `self.supabase` в методах

---

## ✅ Тестирование после синхронизации

### 1. FastAPI запущен и работает

\`\`\`bash
# Проверяем что процесс работает
ps aux | grep uvicorn | grep -v grep

# Проверяем логи
tail -30 /home/nickr/python/fastapi.log | grep -E "ERROR|startup|PostgreSQL"

# Ожидаемый вывод:
# INFO:     Application startup complete.
# [PostgresClient] Initialized
# [v0] Using PostgreSQL client for face recognition
\`\`\`

### 2. API эндпоинты работают

\`\`\`bash
# Тест 1: Config
curl -s http://localhost:8001/api/v2/config
# Ожидается: JSON с конфигурацией

# Тест 2: Statistics
curl -s http://localhost:8001/api/v2/statistics
# Ожидается: {"people_count":104,"total_faces":1138,"unique_photos":942}

# Тест 3: Docs доступны
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8001/docs
# Ожидается: 200
\`\`\`

### 3. Нет ошибок Supabase в логах

\`\`\`bash
# Проверяем что нет ошибок Supabase
tail -100 /home/nickr/python/fastapi.log | grep -i supabase

# Ожидается: пусто или только старые записи
\`\`\`

---

## 🎯 API Эндпоинты (PostgreSQL)

### Работающие эндпоинты:

\`\`\`
GET  /api/v2/config              - Получить конфигурацию распознавания
PUT  /api/v2/config              - Обновить конфигурацию
GET  /api/v2/statistics          - Статистика обученных данных
POST /api/v2/train/prepare       - Подготовка датасета
POST /api/v2/train/execute       - Запуск обучения
GET  /api/v2/train/status/{id}   - Статус обучения
GET  /api/v2/train/history       - История обучений
POST /api/v2/recognize/batch     - Пакетное распознавание
POST /recognize/detect-faces     - Детекция лиц
POST /recognize/batch-recognize  - Пакетное распознавание с фильтрами
POST /recognize/cluster-unknown-faces - Кластеризация неизвестных лиц
\`\`\`

### Правильные URL для фронтенда:

\`\`\`typescript
// .env.local
FASTAPI_URL=http://23.88.61.20:8001
NEXT_PUBLIC_FASTAPI_URL=http://23.88.61.20:8001
NEXT_PUBLIC_API_BASE=http://23.88.61.20:8001
\`\`\`

---

## 📦 Архивы и бэкапы

### На сервере:

\`\`\`bash
ls -lh /home/nickr/python/*.tar.gz

# Должны быть:
# working_postgres_files_complete.tar.gz - архив рабочих файлов с PostgreSQL
# backup_before_sync_*.tar.gz - бэкап перед последней синхронизацией
\`\`\`

### Восстановление из бэкапа (если нужно):

\`\`\`bash
cd /home/nickr/python

# Найти последний бэкап
ls -lt *.tar.gz | head -3

# Восстановить из бэкапа
tar -xzf backup_before_sync_YYYYMMDD_HHMMSS.tar.gz

# Перезапустить FastAPI
pkill -9 -f uvicorn
source venv/bin/activate
nohup uvicorn main:app --host 0.0.0.0 --port 8001 --reload > fastapi.log 2>&1 &
\`\`\`

---

## 🚀 Статус проекта

### Завершено:

1. ✅ PostgreSQL клиент создан и протестирован (450+ строк кода)
2. ✅ Все сервисы обновлены на PostgreSQL
3. ✅ Все роутеры обновлены на PostgreSQL
4. ✅ FastAPI запущен на порту 8001 и работает
5. ✅ API эндпоинты протестированы (config, statistics)
6. ✅ Нет активных ссылок на Supabase в коде

### Осталось:

1. 🔄 Полное тестирование всех эндпоинтов (detect-faces, batch-recognize, train)
2. 🔄 Тестирование интеграции фронтенд ↔ backend
3. 🔄 Упаковка в Docker (опционально)

---

## 🆘 Troubleshooting

### Проблема: "SupabaseClient not found"

\`\`\`bash
# Проверить что старый файл переименован
ls -lh services/supabase_client*

# Должно быть:
# supabase_client_old.py или supabase_client_backup_*.py
\`\`\`

### Проблема: "Address already in use" (порт 8001)

\`\`\`bash
# Убить все процессы на порту 8001
lsof -ti:8001 | xargs kill -9 2>/dev/null

# Перезапустить
source venv/bin/activate
nohup uvicorn main:app --host 0.0.0.0 --port 8001 --reload > fastapi.log 2>&1 &
\`\`\`

### Проблема: "Module not found: asyncpg"

\`\`\`bash
cd /home/nickr/python
source venv/bin/activate
pip install asyncpg==0.29.0
\`\`\`

---

## 📝 Для нового чата

Когда будешь передавать контекст в новый чат, используй эту документацию как основу. Все критические детали, пути к файлам, команды и статус проекта здесь.

**Ключевые файлы для чтения в новом чате:**

1. `FINAL_SERVER_SYNC_GUIDE.md` (этот файл)
2. `python/services/postgres_client.py` - главный PostgreSQL клиент
3. `python/main.py` - точка входа FastAPI
4. `python/routers/training.py` - API обучения
5. `python/routers/recognition.py` - API распознавания

**Команда для быстрой проверки:**

\`\`\`bash
cd /home/nickr/python
curl -s http://localhost:8001/api/v2/statistics && echo "✅ Backend работает" || echo "❌ Backend не отвечает"
\`\`\`

---

**Конец документации**
