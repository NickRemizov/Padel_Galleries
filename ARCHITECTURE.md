# Архитектура проекта Galeries

## Обзор

Проект Galeries - фотогалерея с распознаванием лиц. **Весь проект работает на одном сервере.**

## Архитектура

\`\`\`
┌─────────────────────────────────────────────────────────────┐
│                    СЕРВЕР (Hetzner)                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐         ┌─────────────┐                   │
│  │  Next.js    │ ──────► │  Python     │                   │
│  │  (порт 3000)│  HTTP   │  FastAPI    │                   │
│  │             │         │  (порт 8001)│                   │
│  └─────────────┘         └──────┬──────┘                   │
│        │                        │                          │
│        │                        │ SQL                      │
│        │                        ▼                          │
│        │                 ┌─────────────┐                   │
│        │                 │  PostgreSQL │                   │
│        │                 │  (порт 5432)│                   │
│        │                 └─────────────┘                   │
│        │                                                   │
│        │                 ┌─────────────┐                   │
│        └───────────────► │    MinIO    │                   │
│          (статика)       │  (порт 9200)│                   │
│                          └─────────────┘                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
\`\`\`

## Компоненты

### Next.js Frontend

**Задачи:**
- UI и навигация
- Server Actions (прокси к Python API)
- Статические страницы

**Правило:** НЕ работает с БД напрямую!

### Python FastAPI

**Задачи:**
- ВСЯ работа с PostgreSQL
- Распознавание лиц (InsightFace)
- Кластеризация (HDBSCAN)
- Прокси к MinIO
- **Генерация embedding** (512-мерные векторы)

**Эндпоинты:**
- `/api/crud/*` - CRUD операции (people, galleries, images)
- `/api/faces/*` - операции с лицами (save, delete, get, tags)
- `/api/recognition/*` - распознавание лиц
- `/api/training/*` - обучение и кластеризация
- `/api/s3-proxy/*` - файлы из MinIO

### PostgreSQL

**Таблицы:**
- `galleries` - галереи событий
- `gallery_images` - фотографии
- `people` - игроки
- `photo_faces` - лица на фото (с `insightface_descriptor vector(512)`)
- `face_descriptors` - дескрипторы для распознавания
- `photographers`, `locations`, `organizers`

### MinIO

**Bucket:** `galleries`

**Структура:** `/galleries/{год}/{месяц}/{filename}`

**Доступ:** через Python API proxy

## Поток данных

### Загрузка фото
\`\`\`
Админ → Upload → Python API → MinIO + PostgreSQL
\`\`\`

### Распознавание лиц
\`\`\`
Админ → Python API → InsightFace → PostgreSQL
\`\`\`

### Сохранение тегов лиц (v0.7.0+)
\`\`\`
Админ выбирает игроков для лиц
    ↓
Frontend: saveFaceTagsAction(photoId, imageUrl, tags)
    ↓
Python API: POST /api/faces/save-face-tags
    ↓
Backend генерирует embedding из imageUrl + bbox
    ↓
PostgreSQL: INSERT INTO photo_faces (insightface_descriptor...)
\`\`\`

**ВАЖНО:** Frontend НЕ работает с embedding напрямую!

### Просмотр галереи
\`\`\`
Пользователь → Next.js → Python API → PostgreSQL/MinIO
\`\`\`

## Ключевые файлы

\`\`\`
python/
├── main.py                     # Точка входа FastAPI
├── routers/
│   ├── crud.py                 # CRUD endpoints (people, galleries)
│   ├── faces.py                # Face operations (save, delete, tags)
│   ├── recognition.py          # Распознавание
│   ├── training.py             # Обучение
│   └── s3_proxy.py            # Прокси MinIO
├── services/
│   ├── postgres_client.py      # Работа с БД
│   └── face_recognition.py     # ML логика
└── models/
    └── schemas.py              # Pydantic модели

app/
├── admin/
│   └── actions.ts              # Server Actions (вызывают Python API)
└── api/                        # API routes (прокси)

lib/
└── api/
    └── index.ts                # API клиент (peopleApi, facesApi)
\`\`\`

## Мигрированные Actions (v0.7.0)

| Action | Python Endpoint |
|--------|-----------------|
| `getPeopleAction` | `GET /api/crud/people` |
| `getPersonAction` | `GET /api/crud/people/{id}` |
| `addPersonAction` | `POST /api/crud/people` |
| `deletePersonAction` | `DELETE /api/crud/people/{id}` |
| `createPersonFromClusterAction` | `POST /api/crud/people/from-cluster` |
| `getPhotoFacesAction` | `POST /api/faces/get-photo-faces` |
| `getBatchPhotoFacesAction` | `POST /api/faces/get-batch-photo-faces` |
| `saveFaceTagsAction` | `POST /api/faces/save-face-tags` |
| `deletePhotoFaceAction` | `DELETE /api/faces/delete/{id}` |

## Внешние URL

- **API:** `https://api.vlcpadel.com`
- **Frontend:** `https://galeries.vlcpadel.com`

## Внутренние URL

- **Next.js:** `http://localhost:3000`
- **FastAPI:** `http://localhost:8001`
- **PostgreSQL:** `localhost:5432`
- **MinIO:** `localhost:9200`

---

**Версия:** 1.1.0
**Дата:** 27 ноября 2025
