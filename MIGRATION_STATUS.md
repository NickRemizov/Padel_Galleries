# Статус миграции проекта Galeries

**Дата обновления:** 25 ноября 2025

---

## Общий прогресс: ~40%

## Архитектура

**Текущая (временная):**
\`\`\`
Vercel (фронтенд) → Python API (Hetzner) → PostgreSQL/MinIO
                 → Supabase (auth + legacy) ← УДАЛИТЬ
\`\`\`

**Целевая:**
\`\`\`
Hetzner (один сервер):
Nginx → Next.js → Python API → PostgreSQL/MinIO
\`\`\`

---

## Завершённые этапы

### ✅ Python CRUD API

**Файл:** `python/routers/crud.py`

Эндпоинты:
- `GET/POST/PUT/DELETE /api/crud/galleries`
- `GET/POST/PUT/DELETE /api/crud/photographers`
- `GET/POST/PUT/DELETE /api/crud/locations`
- `GET/POST/PUT/DELETE /api/crud/organizers`
- `GET/POST/PUT/DELETE /api/crud/people`
- `GET /api/crud/stats/recognition`

### ✅ S3/MinIO Proxy

**Файл:** `python/routers/s3_proxy.py`

Эндпоинт:
- `GET /api/s3-proxy/{path}` - проксирует запросы к MinIO

### ✅ Next.js API клиент

**Файл:** `lib/api/index.ts`

Функции:
- `galleriesApi.getAll()`, `.create()`, `.update()`, `.delete()`
- `photographersApi.*`
- `locationsApi.*`
- `organizersApi.*`
- `peopleApi.*`

### ✅ Миграция данных

- Файлы перенесены из Vercel Blob в MinIO
- URL в базе обновлены на `api.vlcpadel.com/api/s3-proxy/...`
- Имена файлов в MinIO исправлены (убран URL-encoding)

---

## В процессе / Осталось

### ⏳ Supabase ещё используется для:

1. **Авторизация админов** (Supabase Auth)
   - `app/admin/login/page.tsx`
   - `app/admin/page.tsx`
   - `app/admin/actions.ts` - signIn, signUp, signOut
   - `lib/supabase/middleware.ts`

2. **Face Recognition** (~50 вызовов в `app/admin/actions.ts`)
   - photo_faces
   - face_descriptors
   - RPC функции статистики

3. **Пользовательский функционал**
   - `app/api/comments/` - комментарии
   - `app/api/favorites/` - избранное
   - `app/api/likes/` - лайки
   - `app/api/auth/telegram/` - Telegram авторизация

4. **Страницы с прямым доступом**
   - `app/gallery/[id]/page.tsx`
   - `app/players/page.tsx`
   - `app/favorites/page.tsx`

### 📋 План следующих этапов

См. `SUPABASE_REMOVAL_PLAN.md` для полного плана.

---

## Команды

\`\`\`bash
# Перезапуск Python
cd /home/nickr/python && ./run.sh

# Проверка логов
tail -100 /tmp/fastapi.log

# База данных
psql "postgresql://galeries_user:galeries_strong_pass_2025@localhost:5432/galleries"
