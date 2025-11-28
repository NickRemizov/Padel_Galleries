## [0.7.0] - 2025-11-27

### Критические изменения - Полная миграция на Python API

**ВАЖНО: Все операции с данными теперь идут через Python API (FastAPI на Hetzner)**

#### Мигрированные actions (app/admin/actions.ts):
- `getPeopleAction` → `peopleApi.getAll()` через Python API
- `getPeopleWithStatsAction` → `peopleApi.getAll({ include_stats: true })`
- `getPersonAction` → Python API `/api/crud/people/{id}`
- `getBatchPhotoFacesAction` → Python API `/api/faces/get-batch-photo-faces`
- `deletePhotoFaceAction` → Python API `/api/faces/delete/{id}`
- `saveFaceTagsAction` → Python API `/api/faces/save-face-tags`
- `addPersonAction` → `peopleApi.create()` через Python API
- `deletePersonAction` → `peopleApi.delete()` через Python API
- `createPersonFromClusterAction` → Python API `/api/crud/people/from-cluster`

#### Новые Python endpoints (python/routers/):
- `POST /api/crud/people/from-cluster` - создание человека из кластера с дескрипторами
- `POST /api/faces/get-batch-photo-faces` - получение лиц для нескольких фото
- `DELETE /api/faces/delete/{face_id}` - удаление лица
- `POST /api/faces/save-face-tags` - batch сохранение тегов с генерацией embedding на бэкенде

#### Архитектурные изменения:
- **Embedding генерируется ТОЛЬКО на бэкенде** - фронтенд больше не работает с embedding
- `/api/faces/save` принимает `image_url` вместо `insightface_descriptor` - бэкенд сам генерирует embedding
- `face-tagging-dialog.tsx` использует `saveFaceTagsAction` вместо цикла delete+save

#### Исправления NFD (No Faces Detected):
- При удалении всех лиц с фото вызывается `saveFaceTagsAction` с пустым массивом tags
- Бэкенд устанавливает `has_been_processed = true` и удаляет все записи photo_faces
- Фото получает статус NFD корректно

### Файлы для синхронизации на сервер:
- `python/routers/faces.py`
- `python/routers/crud.py`
- `python/services/postgres_client.py`
- `python/models/schemas.py`

## [0.6.6] - 2025-02-02

### Исправлено
- **КРИТИЧЕСКОЕ: Парсинг JSONB bbox при загрузке из БД**
  - `getPhotoFacesAction` теперь проверяет тип `insightface_bbox` и парсит JSON строку в объект
  - Проблема: PostgreSQL возвращал JSONB как строку `"{\"x\":476,...}"` → код обращался к `bbox.x` → undefined
  - Исправление: добавлена проверка `typeof face.insightface_bbox === 'string'` с JSON.parse()
  - **Результат:** Боксы корректно отображаются при повторном открытии фото

- **NFD статус при удалении всех лиц с фото**
  - Когда админ удаляет все лица и сохраняет (tags.length === 0), фото должно получить статус "No Faces Detected"
  - Проблема: фото оставалось с оранжевым бейджем 0/1/1 вместо NFD
  - Исправление: добавлена ранняя проверка в `saveFaceTagsAction` для установки `processing_status = 'NFD'` и `verified = true`
  - **Результат:** Фото без лиц корректно маркируется как NFD

- **Улучшено логирование процесса сохранения дескрипторов**
  - Добавлены детальные console.log для отслеживания backend descriptor vs embedding
  - Явное использование дескриптора из backend response: `const backendDescriptor = generatedDescriptors[i]?.descriptor`
  - Логи показывают: наличие backend дескриптора, длину массива, успех/неудачу сохранения в face_descriptors

### Технические детали

**bbox парсинг (getPhotoFacesAction):**
\`\`\`typescript
// Проверка типа и парсинг JSONB строки
let parsedBbox = face.insightface_bbox

if (typeof face.insightface_bbox === 'string') {
  try {
    parsedBbox = JSON.parse(face.insightface_bbox)
  } catch (e) {
    console.error('[v0] Failed to parse bbox:', face.insightface_bbox)
    parsedBbox = null
  }
}
\`\`\`

**NFD статус (saveFaceTagsAction):**
\`\`\`typescript
// Ранний выход если нет тегов
if (tags.length === 0) {
  await sql`UPDATE gallery_images SET processing_status = 'NFD', verified = true WHERE id = ${photoId}`
  return { success: true, message: "All faces removed, photo marked as NFD" }
}
\`\`\`

**Логирование дескрипторов:**
\`\`\`typescript
const backendDescriptor = generatedDescriptors[i]?.descriptor
const descriptor = backendDescriptor || tag.embedding
console.log(`[v0] Processing tag ${i}: has backend descriptor:`, !!backendDescriptor, "length:", descriptor?.length)
\`\`\`

## [0.6.5] - 2025-02-02
