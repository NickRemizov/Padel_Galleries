# План удаления Supabase

> **Статус:** В ПРОЦЕССЕ
> **Цель:** Полное удаление зависимости от Supabase, перенос всего на свой сервер

## Текущее состояние

### Что УЖЕ перенесено на Python API:
- [x] CRUD операции: galleries, photographers, locations, organizers, people
- [x] Файлы галерей → MinIO
- [x] Обложки галерей → MinIO

### Что ещё использует Supabase:

#### 1. Авторизация (КРИТИЧНО)
- `app/admin/actions.ts` - signIn, signUp, signOut
- `app/admin/auth/callback/route.ts` - OAuth callback
- `app/admin/login/page.tsx` - проверка сессии
- `app/admin/page.tsx` - проверка сессии
- `lib/supabase/middleware.ts` - обновление токенов
- `lib/auth/serverGuard.ts` - проверка админ-прав

#### 2. Face Recognition (много кода)
- `app/admin/actions.ts` - ~50+ вызовов supabase для:
  - photo_faces
  - face_descriptors
  - people (частично)
- `app/api/admin/face-statistics/route.ts`
- `app/api/batch-face-recognition/route.ts`
- `lib/face-recognition/face-storage.ts`

#### 3. Пользовательский функционал
- `app/api/comments/[imageId]/route.ts` - комментарии
- `app/api/favorites/route.ts` - избранное
- `app/api/likes/[imageId]/route.ts` - лайки
- `app/api/auth/telegram/route.ts` - Telegram авторизация
- `app/api/images/[imageId]/people/route.ts` - люди на фото

#### 4. Страницы с прямым доступом к БД
- `app/gallery/[id]/page.tsx`
- `app/players/page.tsx`
- `app/players/[id]/page.tsx`
- `app/favorites/page.tsx`

---

## План миграции по этапам

### Этап 1: Авторизация (ПРИОРИТЕТ)
**Цель:** Своя JWT авторизация вместо Supabase Auth

1. Создать таблицу `admin_users` в PostgreSQL
2. Добавить эндпоинты в Python API:
   - `POST /api/auth/login`
   - `POST /api/auth/logout`
   - `GET /api/auth/me`
   - `POST /api/auth/refresh`
3. Мигрировать middleware на проверку своих JWT токенов
4. Удалить `lib/supabase/` после полного перехода

### Этап 2: Face Recognition
**Цель:** Все операции с лицами через Python API

1. Добавить эндпоинты:
   - `GET/POST/DELETE /api/photo-faces`
   - `GET/POST/DELETE /api/face-descriptors`
   - `GET /api/face-statistics`
2. Мигрировать `app/admin/actions.ts` - заменить прямые вызовы supabase

### Этап 3: Пользовательский функционал
**Цель:** Comments, Likes, Favorites через Python API

1. Добавить эндпоинты:
   - `GET/POST/DELETE /api/comments`
   - `GET/POST/DELETE /api/likes`
   - `GET/POST/DELETE /api/favorites`
2. Мигрировать пользовательскую авторизацию (Telegram)

### Этап 4: Финализация
1. Удалить все `@supabase` зависимости
2. Удалить `lib/supabase/`
3. Удалить переменные окружения Supabase
4. Перенести фронтенд на Hetzner

---

## Целевая архитектура (один сервер)

\`\`\`
┌─────────────────────────────────────────────────────┐
│                   Hetzner VPS                        │
├─────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │
│  │   Nginx     │  │   Next.js   │  │ Python API  │  │
│  │  (reverse   │──│  (port 3000)│  │ (port 8001) │  │
│  │   proxy)    │  └─────────────┘  └─────────────┘  │
│  └─────────────┘          │               │         │
│         │                 │               │         │
│         ▼                 ▼               ▼         │
│  ┌─────────────────────────────────────────────┐   │
│  │              PostgreSQL (port 5432)          │   │
│  └─────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────┐   │
│  │              MinIO (port 9200)               │   │
│  └─────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
\`\`\`

**Nginx конфигурация:**
- `/` → Next.js (localhost:3000)
- `/api/` → Python API (localhost:8001)
- Статика из MinIO кэшируется

---

## Файлы к удалению после миграции

\`\`\`
lib/supabase/client.ts
lib/supabase/middleware.ts
lib/supabase/server.ts
\`\`\`

## Переменные окружения к удалению

\`\`\`
NEXT_PUBLIC_SUPABASE_URL
NEXT_PUBLIC_SUPABASE_ANON_KEY
SUPABASE_SERVICE_ROLE_KEY
SUPABASE_URL
SUPABASE_JWT_SECRET
SUPABASE_ANON_KEY
NEXT_PUBLIC_DEV_SUPABASE_REDIRECT_URL
