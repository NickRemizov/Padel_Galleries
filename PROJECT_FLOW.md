# PROJECT FLOW - Galeries v0.8.2

Полная документация по потокам данных в системе распознавания лиц для фотогалерей.

## Архитектура системы

\`\`\`
┌─────────────────────────────────────────────────────────────────┐
│                         ПОЛЬЗОВАТЕЛЬ                            │
└─────────────────────────────────────────────────────────────────┘
                               ↓
        ┌──────────────────────┼──────────────────────┐
        ↓                      ↓                      ↓
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  ПУБЛИЧНЫЕ   │    │    АДМИН-    │    │  TELEGRAM    │
│  ГАЛЕРЕИ     │    │    ПАНЕЛЬ    │    │     БОТ      │
│ (Next.js)    │    │  (Next.js)   │    │   (Webhook)  │
└──────────────┘    └──────────────┘    └──────────────┘
        │                      │                      │
        └──────────────────────┼──────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────┐
│                      FastAPI Backend (Hetzner)                  │
│                      http://api.vlcpadel.com:8001               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │  /api/crud  │  │ /api/faces  │  │/api/recogn- │            │
│  │   people    │  │   save      │  │   ition     │            │
│  │  galleries  │  │   delete    │  │   detect    │            │
│  │   images    │  │   tags      │  │   recognize │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
└─────────────────────────────────────────────────────────────────┘
                               ↓
        ┌──────────────────────┼──────────────────────┐
        ↓                      ↓                      ↓
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  PostgreSQL  │    │    MinIO     │    │  InsightFace │
│   (Hetzner)  │    │   Storage    │    │    Model     │
│   (данные)   │    │   (фото)     │    │ (распознавание)
└──────────────┘    └──────────────┘    └──────────────┘
\`\`\`

**ВАЖНО: Supabase больше НЕ используется для данных. Все операции идут через Python API.**

## 1. ЗАГРУЗКА ФОТО

### Процесс
\`\`\`
Админ открывает галерею
    ↓
Загружает фото (drag & drop / кнопка)
    ↓
uploadAction() в app/admin/actions.ts
    ↓
MinIO Storage (через Python API)
    │
    ├─ Сохранение изображения
    └─ Получение url: api.vlcpadel.com/api/s3-proxy/galleries/...
    ↓
PostgreSQL (Hetzner): gallery_images
    │
    ├─ gallery_id: uuid
    ├─ image_url: text
    └─ has_been_processed: FALSE
    ↓
UI показывает фото БЕЗ БЕЙДЖА (новое)
\`\`\`

### Файлы
- `app/admin/actions.ts` → `uploadAction()`
- `components/admin/gallery-images-manager.tsx` → Upload UI

## 2. РАСПОЗНАВАНИЕ ЛИЦ

### Процесс автоматического распознавания
\`\`\`
Админ нажимает "Распознать фото"
    ↓
AutoRecognitionDialog открывается
    ↓
Для каждого фото (has_been_processed = false):
    │
    ├─ ДЕТЕКЦИЯ ЛИЦ
    │   │
    │   ├─ Next.js API: /api/face-detection/detect
    │   │       ↓
    │   │   FastAPI: POST /api/recognition/detect
    │   │       ↓
    │   │   InsightFace детекция:
    │   │       • app.get(image) → находит лица
    │   │       • Для каждого лица:
    │   │           ├─ bbox (координаты)
    │   │           ├─ det_score (уверенность детекции)
    │   │           ├─ blur_score (резкость, Laplacian)
    │   │           ├─ face_size (ширина bbox)
    │   │           └─ embedding (512-мерный вектор) - НЕ ОТПРАВЛЯЕТСЯ НА ФРОНТЕНД
    │   │
    │   │   ВАЖНО: Embedding остаётся на бэкенде, фронтенд получает только:
    │   │       {
    │   │           bbox: {x, y, width, height},
    │   │           insightface_confidence: det_score,
    │   │           blur_score: float,
    │   │           face_id: uuid (для последующего сохранения)
    │   │       }
    │
    └─ РАСПОЗНАВАНИЕ ЛИЦ
        │
        ├─ FastAPI: POST /api/recognition/recognize
        │       ↓
        │   Поиск похожих лиц:
        │       1. Загрузка верифицированных эмбеддингов из PostgreSQL
        │       2. HNSWLIB индекс (быстрый поиск)
        │       3. Порог для verified: 0.6
        │       4. Порог для unverified: 0.75
        │       ↓
        │   Возврат результата:
        │       {
        │           person_id: "uuid" | null,
        │           person_name: "Имя Фамилия" | null,
        │           confidence: float
        │       }
        │       ↓
        │   Сохранение в photo_faces через savePhotoFaceAction:
        │       INSERT INTO photo_faces (
        │           photo_id,
        │           person_id,
        │           insightface_bbox,
        │           insightface_descriptor,
        │           insightface_confidence,     // ← От детектора (0.7-1.0)
        │           recognition_confidence,     // ← От распознавания (0.0-1.0)
        │           verified
        │       )
        │       ↓
        │   ВАЖНО: Два разных поля confidence:
        │       • insightface_confidence - "Это точно лицо?" (качество обнаружения)
        │       • recognition_confidence - "Это точно человек X?" (качество идентификации)
        │       
        │   Примеры:
        │       • Хорошее обнаружение + плохое распознавание:
        │         insightface_confidence: 0.98, recognition_confidence: 0.45
        │       • Плохое обнаружение + хорошее распознавание:
        │         insightface_confidence: 0.70, recognition_confidence: 0.92
\`\`\`

### Файлы
- `components/admin/auto-recognition-dialog.tsx` → UI автораспознавания
- `app/api/face-detection/detect/route.ts` → API детекции
- `python/routers/recognition.py` → FastAPI endpoints
- `python/services/face_recognition.py` → Логика InsightFace

## 3. КЛАСТЕРИЗАЦИЯ НЕИЗВЕСТНЫХ ЛИЦ

### Процесс
\`\`\`
Админ нажимает "Неизвестные лица"
    ↓
UnknownFacesReviewDialog открывается
    ↓
Next.js API: /training/cluster-unverified-faces
    ↓
FastAPI: POST /api/v2/training/cluster-unverified-faces
    ↓
Загрузка неизвестных лиц:
    SELECT * FROM face_descriptors
    WHERE gallery_image_id IN (
        SELECT id FROM gallery_images 
        WHERE gallery_id = ?
    )
    AND person_id IS NULL
    AND verified = false
    ↓
HDBSCAN кластеризация:
    • min_cluster_size = 2
    • min_samples = 1
    • metric = 'cosine'
    ↓
Группировка похожих лиц:
    Cluster 0: [лицо1, лицо2, лицо5]
    Cluster 1: [лицо3, лицо7, лицо9]
    Cluster -1: [лицо4, лицо6] (шум)
    ↓
Возврат кластеров в UI (отсортировано по размеру)
    ↓
UI показывает кластеры:
    • Сетка фото с синими рамками
    • Статистика: "Всего: X | Активных: Y"
    • Кнопки: "Создать игрока", "Выбрать игрока", "Удалить"
\`\`\`

### Файлы
- `components/admin/unknown-faces-review-dialog.tsx` → UI кластеризации
- `app/api/training/cluster-unverified-faces/route.ts` → API кластеризации
- `python/routers/training.py` → FastAPI endpoint
- `python/services/training_service.py` → Логика HDBSCAN

## 4. СОЗДАНИЕ/НАЗНАЧЕНИЕ ИГРОКА

### Вариант A: Создать нового игрока
\`\`\`
Админ нажимает "Создать игрока"
    ↓
AddPersonDialog открывается
    ↓
Заполнение формы:
    • Имя, Фамилия
    • Telegram username
    • Instagram, VK
    • Рейтинг
    • Аватар (первое фото из кластера)
    ↓
createPersonAction() → PostgreSQL (Hetzner)
    ↓
people {
    id: uuid (НОВЫЙ)
    name: text
    telegram_id: bigint
    rating: integer
}
    ↓
Callback с person_id → assignClusterToPersonAction()
\`\`\`

### Вариант B: Выбрать существующего
\`\`\`
Админ нажимает "Выбрать игрока"
    ↓
Combobox с поиском по имени
    ↓
Выбор игрока → assignClusterToPersonAction()
\`\`\`

### Назначение кластера
\`\`\`
assignClusterToPersonAction(cluster, person_id):
    ↓
Для каждого АКТИВНОГО фото в кластере:
    ↓
    UPDATE face_descriptors SET
        person_id = ?,
        confidence = 1.0,
        verified = [ЛОГИКА]
    ↓
    ЛОГИКА verified:
        IF (на фото только 1 лицо):
            verified = true
        ELSE IF (все остальные лица verified=true OR confidence=1.0):
            verified = true
        ELSE:
            verified = false
    ↓
Перестроение HNSWLIB индекса:
    • POST /rebuild-index
    • Загрузка всех verified эмбеддингов
    • Создание нового индекса в памяти
    ↓
UI обновляется → бейджи меняются
\`\`\`

### Файлы
- `components/admin/add-person-dialog.tsx` → Форма создания игрока
- `app/admin/actions.ts` → `assignClusterToPersonAction()`
- `app/api/training/rebuild-index/route.ts` → Перестроение индекса

## 5. РУЧНАЯ ВЕРИФИКАЦИЯ ФОТО

### Процесс
\`\`\`
Админ кликает на фото
    ↓
FaceTaggingDialog открывается
    ↓
Проверка: есть ли уже лица в БД?
    │
    ├─ ДА: Показать существующие теги
    │       ↓
    │       Кнопка "Распознать заново без настроек"
    │           ↓
    │           Удалить все photo_faces для этого фото
    │           ↓
    │           /api/recognition/detect с apply_quality_filters=false
    │           ↓
    │           Детальное окно с метриками:
    │               • Размер лица: XX px
    │               • Blur score: XX.X
    │               • Detection score: 0.XX (insightface_confidence)
    │               • Качество эмбеддинга: XX.X
    │               • Distance to nearest: 0.XX
    │               • Top-3 похожих лиц
    │
    └─ НЕТ: Автоматически запустить распознавание
            ↓
            /api/recognition/detect + /api/recognition/recognize
            ↓
UI показывает все лица на фото:
    • Фото с синими рамками
    • Combobox для каждого лица
    • Поиск игрока по имени
    ↓
Админ выбирает игрока для каждого лица
    ↓
Кнопка "Сохранить"
    ↓
saveFaceTagsAction(photoId, imageUrl, tags):
    ↓
    FastAPI: POST /api/faces/save-face-tags
    {
        photo_id: uuid,
        image_url: string,  // ← Для генерации embedding на бэкенде
        tags: [
            {
                face_id: uuid | null,
                person_id: uuid | null,
                bbox: {x, y, width, height},
                verified: boolean
            }
        ]
    }
    ↓
    Бэкенд:
        1. Удаляет все существующие photo_faces для этого фото
        2. Удаляет связанные face_descriptors
        3. Для каждого тега:
            - Генерирует embedding из image_url + bbox (InsightFace)
            - Сохраняет в photo_faces с insightface_descriptor
            - Если verified=true, добавляет в face_descriptors
        4. Устанавливает has_been_processed = true
        5. Перестраивает HNSWLIB индекс
    ↓
    Если tags.length === 0:
        - Устанавливает has_been_processed = true
        - Фото получает статус NFD (No Faces Detected)
    ↓
Бейдж фото обновляется
\`\`\`

### Важно: Embedding на бэкенде
**Фронтенд НЕ работает с embedding напрямую!**
- При детекции embedding остаётся на бэкенде
- При сохранении передаётся `image_url` + `bbox`
- Бэкенд сам генерирует embedding из изображения

### Файлы
- `components/admin/face-tagging-dialog.tsx` → UI тегирования
- `app/admin/actions.ts` → `saveFaceTagsAction()`

## 6. ПУБЛИЧНЫЕ ГАЛЕРЕИ

### Главная страница
\`\`\`
Пользователь открывает /
    ↓
Next.js SSR: app/page.tsx
    ↓
PostgreSQL запрос:
    SELECT galleries.*, 
           locations.name,
           COUNT(DISTINCT face_descriptors.person_id) as players,
           COUNT(gallery_images.id) as photos
    FROM galleries
    WHERE galleries.published = true
    GROUP BY galleries.id
    ORDER BY event_date DESC
    ↓
Отображение карточек галерей:
    • Название, дата, локация
    • Превью (первое фото)
    • Количество игроков и фото
\`\`\`

### Галерея события
\`\`\`
Пользователь кликает на галерею → /gallery/[id]
    ↓
Next.js SSR: app/gallery/[id]/page.tsx
    ↓
PostgreSQL запросы:
    1. Информация о галерее
    2. Список игроков (только verified)
    3. Все фото (только verified)
    ↓
Отображение:
    • Заголовок с информацией
    • Список игроков с аватарами
    • Сетка фотографий
    ↓
Клик на игрока → /players/[id]
\`\`\`

### Галерея игрока
\`\`\`
/players/[id]
    ↓
Next.js SSR: app/players/[id]/page.tsx
    ↓
PostgreSQL запрос:
    SELECT gallery_images.*
    FROM face_descriptors
    JOIN gallery_images ON face_descriptors.gallery_image_id = gallery_images.id
    WHERE face_descriptors.person_id = ?
    AND (
        face_descriptors.verified = true
        OR face_descriptors.confidence >= 0.80
    )
    ↓
Отображение всех фото игрока
\`\`\`

### Файлы
- `app/page.tsx` → Главная страница
- `app/gallery/[id]/page.tsx` → Галерея события
- `app/players/[id]/page.tsx` → Галерея игрока

## 7. TELEGRAM БОТ

### Отправка фото игроку
\`\`\`
Админ в галерее игрока
    ↓
Нажимает "Отправить в Telegram"
    ↓
sendPhotosToTelegramAction(person_id, photo_ids)
    ↓
Telegram Bot API:
    POST https://api.telegram.org/bot{token}/sendPhoto
    {
        chat_id: person.telegram_id,
        photo: blob_url,
        caption: "Твои фото с турнира!"
    }
    ↓
Пользователь получает фото в Telegram
\`\`\`

### Webhook обработка
\`\`\`
Telegram отправляет webhook → /api/telegram/webhook
    ↓
Обработка команд:
    /start → Приветствие
    /photos → Ссылка на галереи
    ↓
Ответ пользователю
\`\`\`

### Файлы
- `app/api/telegram/webhook/route.ts` → Webhook обработчик
- `app/admin/actions.ts` → `sendPhotosToTelegramAction()`

## Ключевые настройки

### Quality Filters (из БД)
\`\`\`javascript
{
    min_face_size: 80,           // пиксели
    min_blur_score: 100.0,       // резкость (Laplacian)
    min_detection_score: 0.7     // уверенность детекции
}
\`\`\`

### Recognition Thresholds (жестко в коде)
\`\`\`javascript
{
    verified_threshold: 0.6,     // для verified лиц
    unverified_threshold: 0.75,  // для unverified
    context_weight: 0.1          // бонус за контекст
}
\`\`\`

### Gallery Display
\`\`\`javascript
{
    min_confidence_for_gallery: 0.80  // порог для публичных галерей
}
\`\`\`

## Статусы фото (бейджи)

| Бейдж | Цвет | Условие | Значение |
|-------|------|---------|----------|
| Без бейджа | - | `has_been_processed = false` | Новое фото |
| **NFD** | серый | Нет лиц ИЛИ все отфильтрованы | No Faces Detected |
| **XX/YY** | оранжевый | Есть `person_id = NULL` | XX неопознанных из YY |
| **XX%** | синий | Все назначены, но `verified = false` | Минимальная уверенность |
| **✓** | зеленый | Все `verified = true` ИЛИ `confidence = 1.0` | Верифицировано |

## Технологии

- **Frontend**: Next.js 15, React 19, TailwindCSS v4, shadcn/ui
- **Backend**: FastAPI, Python 3.11 (Hetzner VPS)
- **ML**: InsightFace (antelopev2), HNSWLIB, HDBSCAN
- **Database**: PostgreSQL (Hetzner) - основная БД
- **Storage**: MinIO (Hetzner) через S3 proxy
- **Deployment**: Vercel (frontend), Hetzner (backend + DB + storage)

**НЕ ИСПОЛЬЗУЕТСЯ:**
- ~~Supabase~~ - полностью удалён из data flow
- ~~Vercel Blob~~ - заменён на MinIO
