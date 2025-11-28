# Полный анализ Flow системы Galeries

## 1. Архитектура системы

\`\`\`
┌─────────────────────────────────────────────────────────────┐
│                     FRONTEND (Next.js)                       │
│                   Порт: 3000 (Vercel)                       │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           │ API calls через apiClient.ts
                           │ URL: env.FASTAPI_URL
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                   NEXT.JS API ROUTES                         │
│           /api/admin/training/* (Server-side)                │
│         Использует apiFetch() для связи с FastAPI           │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           │ HTTP requests to FASTAPI_URL
                           │ (server-side only)
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                  FASTAPI BACKEND                             │
│                    Порт: 8001                                │
│          IP: http://23.88.61.20:8001                        │
│                                                              │
│  Эндпоинты:                                                  │
│  ├─ /api/v2/train/* (training.py router)                   │
│  ├─ /api/v2/config (config.py router)                      │
│  └─ /detect-faces, /batch-recognize (recognition.py)       │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           │ PostgreSQL connection
                           │ via postgres_client (db_client)
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                    POSTGRESQL DATABASE                        │
│                   Порт: 5432 (Neon)                         │
│                                                              │
│  Таблицы:                                                    │
│  ├─ people (real_name, avatar_url)                         │
│  ├─ galleries (shoot_date, title)                          │
│  ├─ gallery_images (image_url)                             │
│  ├─ photo_faces (insightface_descriptor vector(512))       │
│  ├─ face_training_sessions                                  │
│  └─ face_recognition_config                                 │
└─────────────────────────────────────────────────────────────┘
\`\`\`

## 2. Переменные окружения

### Frontend (.env.local - НЕ СУЩЕСТВУЕТ!)
\`\`\`env
# ❌ ПРОБЛЕМА: .env.local не найден в проекте!
# Переменные должны быть в Vercel Dashboard или .env файле

NEXT_PUBLIC_FASTAPI_URL=http://23.88.61.20:8001  # ⚠️ HARDCODED в компонентах!
FASTAPI_URL=http://23.88.61.20:8001              # Server-side only
\`\`\`

### Backend (python/.env)
\`\`\`env
DATABASE_URL=postgresql://...                      # PostgreSQL connection
SERVER_HOST=0.0.0.0
SERVER_PORT=8001                                   # ✅ Правильный порт!
\`\`\`

## 3. Критические находки

### ❌ ПРОБЛЕМА 1: Hardcoded FastAPI URL
**Файл:** `components/admin/face-training-manager.tsx:14`
\`\`\`typescript
const FASTAPI_URL = process.env.NEXT_PUBLIC_FASTAPI_URL || "http://23.88.61.20:8001"
\`\`\`
- Fallback на HTTP (не HTTPS)
- IP хардкоден

**Файл:** `components/admin/face-tagging-dialog.tsx:18`
\`\`\`typescript
const FASTAPI_URL = process.env.NEXT_PUBLIC_FASTAPI_URL || "http://23.88.61.20:8001"
\`\`\`

### ✅ ПРАВИЛЬНО: apiClient.ts
**Файл:** `lib/apiClient.ts:24`
\`\`\`typescript
const url = `${env.FASTAPI_URL}${normalizedPath}`
\`\`\`
- Использует env schema validation
- Server-side only (безопасно)

### ❌ ПРОБЛЕМА 2: Дублирование логики API вызовов
**Фронтенд делает 2 типа вызовов:**
1. Напрямую к FastAPI (components) - ⚠️ опасно
2. Через Next.js API routes (/api/admin/*) - ✅ правильно

## 4. Flow данных по операциям

### 4.1 Обучение модели (Training)

\`\`\`
User clicks "Запустить обучение"
  │
  ▼
FaceTrainingManager.startTraining()
  │
  ▼
fetch("/api/admin/training/execute")  [Next.js API Route]
  │
  ▼
apiFetch("/api/v2/train/execute")  [apiClient.ts]
  │
  ▼
FastAPI POST /api/v2/train/execute  [training.py router]
  │
  ▼
TrainingService.train_model_on_verified_faces()
  │
  ▼
await db_client.connect()
await db_client.get_verified_faces_with_descriptors()
  │
  ▼
PostgreSQL SELECT from:
  - photo_faces (insightface_descriptor)
  - people (real_name)
  - gallery_images (image_url)
  - galleries (shoot_date)
  │
  ▼
FaceRecognizer.train_from_descriptors()
  │
  ▼
await db_client.update_training_session()
  │
  ▼
Response → Next.js → Frontend
\`\`\`

### 4.2 Распознавание лиц (Recognition)

\`\`\`
User uploads photos
  │
  ▼
API POST /detect-faces  [recognition.py]
  │
  ▼
await db_client.connect()
FaceRecognizer.detect_faces()
  │
  ▼
await db_client.save_photo_face()
  │
  ▼
PostgreSQL INSERT into photo_faces:
  - photo_id
  - insightface_bbox
  - insightface_descriptor (vector 512)
  - person_id (NULL if unknown)
\`\`\`

### 4.3 Получение конфигурации

\`\`\`
Frontend loads config
  │
  ▼
fetch("/api/admin/training/config")  [Next.js API]
  │
  ▼
apiFetch("/api/v2/config")  [apiClient.ts]
  │
  ▼
FastAPI GET /api/v2/config  [config.py]
  │
  ▼
await db_client.get_recognition_config()
  │
  ▼
PostgreSQL SELECT from face_recognition_config
WHERE key = 'recognition_settings'
  │
  ▼
Returns JSON with quality_filters, confidence_thresholds
\`\`\`

## 5. Эндпоинты FastAPI

### Training Router (/api/v2)
- POST `/train/prepare` - подготовка датасета
- POST `/train/execute` - запуск обучения
- GET `/train/status/{session_id}` - статус сессии
- GET `/train/history` - история обучений
- POST `/recognize/batch` - пакетное распознавание
- GET `/config` - получить конфигурацию
- PUT `/config` - обновить конфигурацию

### Recognition Router (/)
- POST `/detect-faces` - детекция лиц
- POST `/recognize-face` - распознавание одного лица
- POST `/batch-recognize` - пакетное распознавание
- POST `/cluster-unknown-faces` - кластеризация неизвестных
- POST `/reject-face-cluster` - отклонение кластера
- POST `/generate-descriptors` - генерация дескрипторов
- POST `/rebuild-index` - пересборка индекса
- POST `/regenerate-unknown-descriptors` - регенерация для неизвестных

### Config Router (/api/v2)
- GET `/config` - получить настройки
- PUT `/config` - обновить настройки

## 6. PostgreSQL Schema (критические таблицы)

### people
\`\`\`sql
id UUID PRIMARY KEY
real_name TEXT NOT NULL          -- ⚠️ НЕ "name"!
avatar_url TEXT
created_at TIMESTAMP
\`\`\`

### galleries
\`\`\`sql
id UUID PRIMARY KEY
title TEXT NOT NULL
shoot_date DATE                   -- ⚠️ НЕ "event_date"!
location_id UUID
created_at TIMESTAMP
\`\`\`

### photo_faces
\`\`\`sql
id UUID PRIMARY KEY
photo_id UUID                     -- ⚠️ НЕ "image_id"!
person_id UUID                    -- NULL если неизвестен
insightface_bbox JSONB
insightface_confidence FLOAT
insightface_descriptor VECTOR(512)  -- ⚠️ Дескрипторы ЗДЕСЬ, не в отдельной таблице!
recognition_confidence FLOAT
verified BOOLEAN DEFAULT FALSE
training_used BOOLEAN DEFAULT FALSE
face_category TEXT
\`\`\`

### face_training_sessions
\`\`\`sql
id UUID PRIMARY KEY
status TEXT
started_at TIMESTAMP
completed_at TIMESTAMP
people_count INTEGER
total_faces INTEGER
model_accuracy FLOAT
error_message TEXT
\`\`\`

### face_recognition_config
\`\`\`sql
key TEXT PRIMARY KEY
value JSONB                       -- Хранит recognition_settings
updated_at TIMESTAMP
\`\`\`

## 7. Проблемы и рекомендации

### 🔴 КРИТИЧЕСКИЕ ПРОБЛЕМЫ

1. **Нет .env.local файла**
   - Frontend не имеет локальных env переменных
   - FASTAPI_URL хардкоден в компонентах

2. **HTTP вместо HTTPS**
   - `http://23.88.61.20:8001` не защищен
   - FastAPI может требовать HTTPS

3. **Прямые вызовы к FastAPI из клиента**
   - `face-training-manager.tsx` делает прямые fetch к FASTAPI_URL
   - Должны идти через Next.js API routes

### 🟡 СРЕДНИЕ ПРОБЛЕМЫ

4. **Дублирование эндпоинтов**
   - `/api/v2/config` и `/config` делают одно и то же
   - Нужна унификация

5. **Отсутствие .env validation на сервере**
   - Python не проверяет наличие DATABASE_URL при старте

### 🟢 РЕКОМЕНДАЦИИ

6. **Создать .env.local**
   \`\`\`env
   NEXT_PUBLIC_FASTAPI_URL=https://23.88.61.20:8001
   \`\`\`

7. **Убрать hardcoded URL из компонентов**
   - Использовать только env.FASTAPI_URL

8. **Все клиентские вызовы через Next.js API**
   - Клиент → Next.js API → FastAPI → PostgreSQL
   - Безопаснее и проще для CORS

## 8. Тестовый чеклист

### Backend (FastAPI на 8001)
- [ ] FastAPI запущен и отвечает на `/docs` (200 OK) ✅
- [ ] PostgreSQL клиент подключается
- [ ] `/api/v2/config` возвращает настройки
- [ ] `/api/v2/train/prepare` работает
- [ ] `/detect-faces` обрабатывает фото

### Frontend (Next.js)
- [ ] env.FASTAPI_URL правильно настроен
- [ ] `/api/admin/training/config` работает
- [ ] FaceTrainingManager загружается без ошибок
- [ ] Можно запустить обучение
- [ ] Можно сохранить настройки

### Database
- [ ] Таблица `photo_faces` имеет insightface_descriptor
- [ ] Таблица `people` использует `real_name`
- [ ] Таблица `galleries` использует `shoot_date`
- [ ] Есть verified faces для обучения

## 9. Следующие шаги

1. ✅ Создать .env.local с FASTAPI_URL
2. ⏳ Обновить компоненты для использования env
3. ⏳ Протестировать полный цикл обучения
4. ⏳ Проверить сохранение конфигурации
5. ⏳ Упаковать FastAPI в Docker
