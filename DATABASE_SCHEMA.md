# Database Schema - PostgreSQL (Hetzner)

Актуальная схема базы данных проекта Galeries.

**Сервер:** PostgreSQL на Hetzner
**База:** galleries
**Пользователь:** galeries_user
**Доступ:** Только через Python API (FastAPI)

---

## Важно: Архитектура доступа к данным

\`\`\`
Frontend (Vercel)
    │
    │ HTTP requests
    ↓
Python API (FastAPI, port 8001)
    │
    │ asyncpg
    ↓
PostgreSQL (Hetzner)
\`\`\`

**Supabase SQL (`await sql`) НЕ ИСПОЛЬЗУЕТСЯ для доступа к данным!**

Все операции идут через Python API endpoints:
- `/api/crud/*` - CRUD операции
- `/api/faces/*` - операции с лицами
- `/api/recognition/*` - распознавание

---

## Таблицы

### galleries
Галереи/события с фотографиями.

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| id | uuid | NOT NULL | gen_random_uuid() | PK |
| title | text | NOT NULL | | Название галереи |
| description | text | | | Описание |
| shoot_date | date | | | Дата съёмки |
| cover_image_url | text | | | URL обложки |
| photographer_id | uuid | | | FK → photographers |
| location_id | uuid | | | FK → locations |
| organizer_id | uuid | | | FK → organizers |
| created_at | timestamptz | | now() | Дата создания |

**Indexes:** `galleries_pkey`, `idx_galleries_shoot_date`

---

### gallery_images
Фотографии в галереях.

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| id | uuid | NOT NULL | gen_random_uuid() | PK |
| gallery_id | uuid | NOT NULL | | FK → galleries |
| image_url | text | NOT NULL | | URL изображения |
| original_url | text | | | Оригинальный URL |
| original_filename | text | | | Имя файла |
| file_size | integer | | | Размер в байтах |
| width | integer | | | Ширина в px |
| height | integer | | | Высота в px |
| display_order | integer | | | Порядок отображения |
| download_count | integer | | 0 | Счётчик скачиваний |
| has_been_processed | boolean | | false | Обработано ли распознаванием |
| created_at | timestamptz | | now() | Дата создания |

**Indexes:** `gallery_images_pkey`, `idx_gallery_images_gallery`, `idx_gallery_images_display_order`, `idx_gallery_images_processed`

---

### people
Люди (игроки, участники).

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| id | uuid | NOT NULL | gen_random_uuid() | PK |
| full_name | text | NOT NULL | | Полное имя |
| telegram_id | text | | | Telegram ID |
| telegram_username | text | | | Telegram username |
| phone | text | | | Телефон |
| email | text | | | Email |
| photo_url | text | | | URL фото профиля |
| created_at | timestamptz | | now() | Дата создания |

**Indexes:** `people_pkey`, `idx_people_telegram_id`

---

### photo_faces
Лица, найденные на фотографиях.

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| id | uuid | NOT NULL | gen_random_uuid() | PK |
| photo_id | uuid | NOT NULL | | FK → gallery_images |
| person_id | uuid | | | FK → people (NULL = неизвестный) |
| insightface_bbox | jsonb | | | Координаты лица {x,y,width,height} |
| insightface_descriptor | vector(512) | | | Дескриптор лица |
| insightface_confidence | real | | | Уверенность детекции |
| recognition_confidence | real | | | Уверенность распознавания |
| verified | boolean | | false | Подтверждено вручную |
| created_at | timestamptz | | now() | Дата создания |

**Indexes:** `photo_faces_pkey`, `idx_photo_faces_person`, `idx_photo_faces_photo`, `idx_photo_faces_verified`

---

### face_descriptors
Дескрипторы лиц для распознавания.

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| id | uuid | NOT NULL | gen_random_uuid() | PK |
| person_id | uuid | NOT NULL | | FK → people (CASCADE) |
| source_image_id | uuid | | | FK → gallery_images (SET NULL) |
| descriptor | jsonb | NOT NULL | | Дескриптор лица |
| created_at | timestamptz | | now() | Дата создания |

**Indexes:** `face_descriptors_pkey`, `idx_face_descriptors_person`

---

### photographers
Фотографы.

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| id | uuid | NOT NULL | gen_random_uuid() | PK |
| name | text | NOT NULL | | Имя |
| contact_info | text | | | Контакты |
| created_at | timestamptz | | now() | Дата создания |

---

### locations
Места съёмки.

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| id | uuid | NOT NULL | gen_random_uuid() | PK |
| name | text | NOT NULL | | Название |
| address | text | | | Адрес |
| city | text | | | Город |
| created_at | timestamptz | | now() | Дата создания |

---

### organizers
Организаторы мероприятий.

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| id | uuid | NOT NULL | gen_random_uuid() | PK |
| name | text | NOT NULL | | Название |
| contact_info | text | | | Контакты |
| created_at | timestamptz | | now() | Дата создания |

---

### users
Пользователи системы.

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| id | uuid | NOT NULL | gen_random_uuid() | PK |
| telegram_id | text | | | Telegram ID |
| telegram_username | text | | | Telegram username |
| first_name | text | | | Имя |
| last_name | text | | | Фамилия |
| photo_url | text | | | URL фото |
| is_admin | boolean | | false | Админ? |
| created_at | timestamptz | | now() | Дата создания |

---

### comments
Комментарии к фото.

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| id | uuid | NOT NULL | gen_random_uuid() | PK |
| gallery_image_id | uuid | NOT NULL | | FK → gallery_images |
| user_id | uuid | NOT NULL | | FK → users |
| content | text | NOT NULL | | Текст комментария |
| created_at | timestamptz | | now() | Дата создания |

---

### likes
Лайки фотографий.

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| id | uuid | NOT NULL | gen_random_uuid() | PK |
| image_id | uuid | NOT NULL | | FK → gallery_images |
| user_id | uuid | NOT NULL | | FK → users |
| created_at | timestamptz | | now() | Дата создания |

**Unique:** `(image_id, user_id)`

---

### favorites
Избранные фото пользователей.

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| id | uuid | NOT NULL | gen_random_uuid() | PK |
| gallery_image_id | uuid | NOT NULL | | FK → gallery_images |
| user_id | uuid | NOT NULL | | FK → users |
| created_at | timestamptz | | now() | Дата создания |

**Unique:** `(gallery_image_id, user_id)`

---

### face_training_sessions
Сессии обучения распознавания.

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| id | uuid | NOT NULL | gen_random_uuid() | PK |
| gallery_id | uuid | | | FK → galleries |
| status | text | | | Статус сессии |
| started_at | timestamptz | | | Начало |
| completed_at | timestamptz | | | Завершение |
| photos_processed | integer | | 0 | Обработано фото |
| faces_detected | integer | | 0 | Найдено лиц |
| error_message | text | | | Сообщение об ошибке |

---

### face_recognition_config
Конфигурация распознавания.

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| id | uuid | NOT NULL | gen_random_uuid() | PK |
| key | text | NOT NULL | | Ключ настройки |
| value | jsonb | | | Значение |
| updated_at | timestamptz | | now() | Дата обновления |

---

### rejected_faces
Отклонённые варианты распознавания.

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| id | uuid | NOT NULL | gen_random_uuid() | PK |
| photo_id | uuid | NOT NULL | | FK → gallery_images |
| person_id | uuid | NOT NULL | | FK → people |
| rejected_at | timestamptz | | now() | Дата отклонения |

---

### gallery_co_occurrence
Совместные появления людей в галереях.

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| person1_id | uuid | NOT NULL | | FK → people |
| person2_id | uuid | NOT NULL | | FK → people |
| gallery_id | uuid | NOT NULL | | FK → galleries |
| co_occurrence_count | integer | | 1 | Счётчик |

**PK:** `(person1_id, person2_id, gallery_id)`

---

### tournament_results
Результаты турниров.

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| id | uuid | NOT NULL | gen_random_uuid() | PK |
| gallery_id | uuid | | | FK → galleries |
| person_id | uuid | | | FK → people |
| place | integer | | | Место |
| category | text | | | Категория |

---

## Связи (Foreign Keys)

\`\`\`
galleries.photographer_id → photographers.id
galleries.location_id → locations.id
galleries.organizer_id → organizers.id
gallery_images.gallery_id → galleries.id (CASCADE)
photo_faces.photo_id → gallery_images.id (CASCADE)
photo_faces.person_id → people.id (SET NULL)
face_descriptors.person_id → people.id (CASCADE)
face_descriptors.source_image_id → gallery_images.id (SET NULL)
comments.gallery_image_id → gallery_images.id (CASCADE)
comments.user_id → users.id
likes.image_id → gallery_images.id (CASCADE)
likes.user_id → users.id
favorites.gallery_image_id → gallery_images.id (CASCADE)
favorites.user_id → users.id
\`\`\`

---

## Хранилище файлов

**MinIO** на порту 9200:
- Bucket: `galleries`
- Путь: `galleries/{year}/{month}/{filename}.jpg`
- Доступ через прокси: `https://api.vlcpadel.com/api/s3-proxy/galleries/...`
