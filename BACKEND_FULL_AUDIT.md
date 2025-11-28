# ПОЛНЫЙ АУДИТ БЭК-ЭНДА - КАРТА ВСЕХ СВЯЗЕЙ И ПРОБЛЕМ

Дата: 20.11.2025
Версия: После первой волны исправлений
Статус: **КРИТИЧЕСКИЙ - обнаружены системные проблемы**

---

## 1. КАРТА ЭНДПОИНТОВ (16 endpoints)

### main.py (7 legacy endpoints)
| Path | Method | Handler | Status |
|------|--------|---------|--------|
| `/api/auth/google` | POST | `google_auth()` | ✅ OK |
| `/api/players/add` | POST | `add_player()` | ✅ OK |
| `/api/players/list` | GET | `list_players()` | ✅ OK |
| `/api/gallery/process` | POST | `process_gallery()` | ✅ OK |
| `/api/gallery/{gallery_id}/results` | GET | `get_gallery_results()` | ✅ OK |
| `/api/upload-photos` | POST | `upload_photos()` | ✅ OK |
| `/api/group-players` | POST | `group_players()` | ✅ OK |

### routers/training.py (5 training endpoints)
| Path | Method | Handler | Status |
|------|--------|---------|--------|
| `/api/v2/train/prepare` | POST | `prepare_training()` | ✅ OK |
| `/api/v2/train/execute` | POST | `execute_training()` | ✅ OK |
| `/api/v2/train/status/{session_id}` | GET | `get_training_status()` | ✅ OK |
| `/api/v2/train/history` | GET | `get_training_history()` | ✅ OK |
| `/api/v2/batch-recognize` | POST | `batch_recognize_photos()` | ⚠️ FIXED |
| `/api/v2/statistics` | GET | `get_training_statistics()` | ✅ OK |

### routers/config.py (2 config endpoints)
| Path | Method | Handler | Status |
|------|--------|---------|--------|
| `/api/v2/config` | GET | `get_config()` | ✅ OK |
| `/api/v2/config` | PUT | `update_config()` | ✅ OK |

### routers/recognition.py (7 recognition endpoints)
| Path | Method | Handler | Status |
|------|--------|---------|--------|
| `/detect-faces` | POST | `detect_faces()` | ✅ OK |
| `/recognize-face` | POST | `recognize_face()` | ✅ OK |
| `/batch-recognize` | POST | `batch_recognize()` | ❌ DUPLICATE |
| `/cluster-unknown-faces` | POST | `cluster_unknown_faces()` | ⚠️ FIXED |
| `/reject-face-cluster` | POST | `reject_face_cluster()` | ✅ OK |
| `/generate-descriptors` | POST | `generate_descriptors()` | ✅ OK |
| `/rebuild-index` | POST | `rebuild_index()` | ✅ OK |
| `/regenerate-unknown-descriptors` | POST | `regenerate_unknown_descriptors()` | ✅ OK |

---

## 2. КРИТИЧЕСКИЕ ПРОБЛЕМЫ

### ПРОБЛЕМА #1: ДУБЛИРОВАНИЕ `/batch-recognize`
**Серьезность:** 🔴 КРИТИЧЕСКАЯ

**Где:**
- `training.py:182` - `/api/v2/batch-recognize`
- `recognition.py:198` - `/batch-recognize` (без prefix)

**Последствия:**
- FastAPI зарегистрирует ОБА роута
- Неопределенное поведение - какой выполнится?
- Разные сигнатуры запросов

**Решение:**
Удалить `/batch-recognize` из `recognition.py:198-354`, оставить только в `training.py`

---

### ПРОБЛЕМА #2: НЕТ S3_CLIENT
**Серьезность:** 🔴 КРИТИЧЕСКАЯ

**Где:**
- `recognition.py:18` импортирует `from services.s3_client import ...`
- НО ФАЙЛА `services/s3_client.py` НЕ СУЩЕСТВУЕТ!

**Последствия:**
- Приложение НЕ запустится - ImportError
- Все S3-related функции сломаны

**Решение:**
- Либо создать `services/s3_client.py`
- Либо удалить неиспользуемые импорты

---

### ПРОБЛЕМА #3: НЕПРАВИЛЬНАЯ РЕГИСТРАЦИЯ РОУТЕРОВ В MAIN.PY
**Серьезность:** 🟡 ВАЖНАЯ

**Где:** `main.py:283-285`

\`\`\`python
app.include_router(training.router, prefix="/api/v2", tags=["training"])
app.include_router(recognition.router, prefix="", tags=["recognition"])  # ❌ БЕЗ PREFIX
app.include_router(config.router, prefix="/api/v2", tags=["config"])
\`\`\`

**Проблема:**
`recognition.router` регистрируется БЕЗ prefix, поэтому:
- `/detect-faces` доступен как `/detect-faces` (без `/api/v2`)
- `/batch-recognize` доступен как `/batch-recognize` (без `/api/v2`)
- Это ПРОТИВОРЕЧИТ `training.py` где `/batch-recognize` под `/api/v2`

**Решение:**
Добавить prefix `/api/v2` для `recognition.router`

---

## 3. DATA FLOW ПРОБЛЕМЫ

### ПРОБЛЕМА #4: batch_recognize В training_service.py
**Серьезность:** 🔴 КРИТИЧЕСКАЯ

**Где:** `training_service.py:540-625`

**Найденная ошибка в строке 562:**
\`\`\`python
face_id = face_data['id']  # ✅ ПРАВИЛЬНО (уже исправлено)
\`\`\`

**НО НОВАЯ ПРОБЛЕМА:**
Метод `batch_recognize()` в `training_service.py` НИКОГДА не вызывается!

**Почему:**
- `training.py:182` вызывает `training_service.batch_recognize()`
- НО в классе `TrainingService` нет метода `batch_recognize()`!
- Есть только `async def batch_recognize()` в `training_service.py:540`

**Результат:**
AttributeError при вызове `/api/v2/batch-recognize`

**Решение:**
Перенести логику из `training_service.py:540` в метод класса

---

### ПРОБЛЕМА #5: get_unverified_images() ВОЗВРАЩАЕТ НЕПРАВИЛЬНУЮ СТРУКТУРУ
**Серьезность:** 🟡 ВАЖНАЯ

**Где:** `postgres_client.py:419-443`

**Что возвращает:**
\`\`\`python
return [{"id": row["id"], "image_url": row["image_url"]} for row in rows]
\`\`\`

**Что ожидается в training_service.py:562:**
\`\`\`python
face_data['id']  # ✅ Правильно - ключ 'id' есть
\`\`\`

**Что ожидается в recognition.py:219:**
\`\`\`python
image["image_url"]  # ✅ Правильно - ключ есть
\`\`\`

**ВЫВОД:** Структура ПРАВИЛЬНАЯ после исправлений!

---

### ПРОБЛЕМА #6: НЕСООТВЕТСТВИЕ ТИПОВ ДАННЫХ В POSTGRES
**Серьезность:** 🟡 ВАЖНАЯ

**Где:** Множественные места сохранения `insightface_descriptor`

**Проблема:**
PostgreSQL `vector` ожидает `List[float]`, но в коде передается `np.ndarray`

**Где встречается:**
1. `recognition.py:279` - ❌ ОШИБКА
   \`\`\`python
   "insightface_descriptor": embedding,  # np.ndarray - НЕВЕРНО!
   \`\`\`

2. `recognition.py:304` - ❌ ОШИБКА
   \`\`\`python
   "insightface_descriptor": embedding_list,  # уже исправлено
   \`\`\`

**Решение:**
Всегда конвертировать через `.tolist()` перед сохранением

---

## 4. АРХИТЕКТУРНЫЕ ПРОБЛЕМЫ

### ПРОБЛЕМА #7: ДУБЛИРОВАНИЕ ЛОГИКИ batch_recognize
**Серьезность:** 🟡 ВАЖНАЯ

**Где:**
- `training.py:182` endpoint → `training_service.batch_recognize()`
- `recognition.py:198` endpoint → встроенная логика

**Проблема:**
ДВА РАЗНЫХ АЛГОРИТМА для одной задачи!

**training.py версия:**
- Вызывает `training_service.batch_recognize()`
- Работает с `gallery_ids` и `confidence_threshold`

**recognition.py версия:**
- Встроенная логика в эндпоинте
- Работает с `BatchRecognizeRequest` (другой тип!)

**Решение:**
Объединить в ОДИН алгоритм в `training_service.py`

---

### ПРОБЛЕМА #8: ОТСУТСТВУЕТ ВАЛИДАЦИЯ QUERY ПАРАМЕТРОВ
**Серьезность:** 🟢 НИЗКАЯ

**Где:** `recognition.py:328`

\`\`\`python
async def cluster_unknown_faces(
    gallery_id: Optional[str] = Query(None),  # ✅ OK - опциональный
    min_cluster_size: int = Query(2)
):
\`\`\`

**Проблема:**
Если `gallery_id = None`, функция возвращает пустой результат без ошибки.
Но тесты ожидают 422 ошибку при невалидном UUID.

**Решение:**
Добавить валидацию UUID если `gallery_id` передан

---

## 5. ОТСУТСТВУЮЩИЕ МЕТОДЫ В POSTGRES_CLIENT

### ✅ ВСЕ МЕТОДЫ ДОБАВЛЕНЫ!

Проверка всех используемых методов:

1. `save_photo_face()` - ✅ ЕСТЬ (строка 499)
2. `save_face_descriptor()` - ✅ ЕСТЬ (строка 536)
3. `reject_face_cluster()` - ✅ ЕСТЬ (строка 576)
4. `store_face_descriptor()` - ✅ ЕСТЬ (строка 600)
5. `save_recognized_face()` - ✅ ЕСТЬ (строка 620)
6. `save_unknown_face()` - ✅ ЕСТЬ (строка 643)
7. `get_unverified_images()` - ✅ ЕСТЬ (строка 419)
8. `get_unknown_faces_from_gallery()` - ✅ ЕСТЬ (строка 445)

**ИТОГ:** Все методы присутствуют!

---

## 6. ИМПОРТЫ И ЗАВИСИМОСТИ

### ПРОБЛЕМА #9: НЕИСПОЛЬЗУЕМЫЕ ИМПОРТЫ
**Серьезность:** 🟢 НИЗКАЯ

**Где:** `recognition.py:7`

\`\`\`python
from services.postgres_client import db_client  # ✅ ИСПОЛЬЗУЕТСЯ
\`\`\`

**НО:**
\`\`\`python
# recognition.py не импортирует s3_client явно - хорошо!
\`\`\`

**ПРОБЛЕМА:** В коде нет `s3_client`, но он может понадобиться для работы с S3/MinIO

---

## 7. ТЕСТОВЫЕ ДАННЫЕ И MOCK-ОБЪЕКТЫ

### ПРОБЛЕМА #10: ТЕСТ С НЕВАЛИДНЫМ UUID
**Серьезность:** 🟢 НИЗКАЯ

**Где:** `test_backend_endpoints.py:test_cluster_unknown_faces()`

**Проблема:**
Тест использует `'test-gallery-id'` вместо реального UUID

**Решение:**
Генерировать реальный UUID в тесте или пропускать тест если БД пустая

---

## 8. ИТОГОВАЯ КАРТА ВЫЗОВОВ (Call Graph)

\`\`\`
/api/v2/batch-recognize (POST) → training.py:batch_recognize_photos()
    ↓
training_service.batch_recognize() [❌ НЕ СУЩЕСТВУЕТ!]
    ↓
db_client.get_unverified_images() [✅ OK]
    ↓
face_service.detect_faces() [✅ OK]
    ↓
face_service.recognize_face() [✅ OK]
    ↓
db_client.save_photo_face() [✅ OK]
\`\`\`

\`\`\`
/cluster-unknown-faces (POST) → recognition.py:cluster_unknown_faces()
    ↓
db_client.get_unknown_faces_from_gallery() [✅ OK]
    ↓
hdbscan.fit_predict() [✅ OK]
    ↓
db_client.fetchone() для нормализации bbox [✅ OK]
\`\`\`

\`\`\`
/detect-faces (POST) → recognition.py:detect_faces()
    ↓
face_service.detect_faces() [✅ OK]
    ↓
face_service.recognize_face() для top_matches [✅ OK]
    ↓
db_client.fetchone() для имен людей [✅ OK]
\`\`\`

---

## 9. СПИСОК ИСПРАВЛЕНИЙ (ПРИОРИТЕТЫ)

### 🔴 КРИТИЧЕСКИЕ (БЛОКИРУЮТ РАБОТУ)

1. **УДАЛИТЬ дублирующийся `/batch-recognize` из recognition.py**
   - Файл: `recognition.py:198-354`
   - Действие: Удалить весь метод `batch_recognize()`

2. **СОЗДАТЬ метод `batch_recognize()` в TrainingService**
   - Файл: `training_service.py`
   - Действие: Переместить логику из `recognition.py` в класс

3. **ПРОВЕРИТЬ s3_client.py**
   - Файл: `services/s3_client.py`
   - Действие: Создать файл или удалить неиспользуемые импорты

### 🟡 ВАЖНЫЕ (МОГУТ ВЫЗВАТЬ БАГИ)

4. **ДОБАВИТЬ prefix для recognition.router**
   - Файл: `main.py:284`
   - Действие: Изменить на `prefix="/api/v2"`

5. **ИСПРАВИТЬ numpy → list конвертацию**
   - Файл: `recognition.py:279, 304`
   - Действие: Добавить `.tolist()` для всех дескрипторов

6. **ДОБАВИТЬ валидацию UUID в cluster_unknown_faces**
   - Файл: `recognition.py:328`
   - Действие: Проверять UUID format если передан

### 🟢 НЕКРИТИЧЕСКИЕ (ОПТИМИЗАЦИЯ)

7. **УДАЛИТЬ неиспользуемые файлы supabase_***
   - Файлы: `services/supabase_client.py`, `services/supabase_database.py`
   - Действие: Удалить или перенести нужные методы

8. **ОБНОВИТЬ тест с UUID**
   - Файл: `test_backend_endpoints.py`
   - Действие: Использовать реальный UUID вместо 'test-gallery-id'

---

## 10. ПРОВЕРКА ВСЕХ ПУТЕЙ ДАННЫХ

### Путь 1: Batch Recognition
\`\`\`
USER → POST /api/v2/batch-recognize
   ↓ training.py:batch_recognize_photos()
   ↓ training_service.batch_recognize() [❌ НЕ СУЩЕСТВУЕТ]
   ✗ FAIL
\`\`\`

### Путь 2: Clustering
\`\`\`
USER → POST /cluster-unknown-faces?gallery_id=xxx
   ↓ recognition.py:cluster_unknown_faces()
   ↓ db_client.get_unknown_faces_from_gallery()
   ↓ hdbscan.fit_predict()
   ✓ OK (после исправления)
\`\`\`

### Путь 3: Face Detection
\`\`\`
USER → POST /detect-faces
   ↓ recognition.py:detect_faces()
   ↓ face_service.detect_faces()
   ↓ face_service.recognize_face()
   ✓ OK
\`\`\`

---

## ИТОГО

**Найдено проблем:** 10
- 🔴 **Критических:** 3
- 🟡 **Важных:** 4
- 🟢 **Некритических:** 3

**Время на исправление:** ~3-4 часа

**Следующий шаг:** Исправить все проблемы по приоритетам

---

## ТАБЛИЦА СООТВЕТСТВИЯ МЕТОДОВ

| Метод в коде | Где вызывается | Где определен | Статус |
|-------------|----------------|---------------|---------|
| `db_client.get_verified_faces()` | training_service.py:100 | postgres_client.py:56 | ✅ |
| `db_client.get_unverified_images()` | training.py:189, recognition.py:219 | postgres_client.py:419 | ✅ |
| `db_client.get_unknown_faces_from_gallery()` | recognition.py:355 | postgres_client.py:445 | ✅ |
| `db_client.save_photo_face()` | recognition.py:277, 300 | postgres_client.py:499 | ✅ |
| `db_client.save_face_descriptor()` | recognition.py:565 | postgres_client.py:536 | ✅ |
| `db_client.reject_face_cluster()` | recognition.py:493 | postgres_client.py:576 | ✅ |
| `training_service.batch_recognize()` | training.py:192 | ❌ НЕТ | ❌ |
| `face_service.detect_faces()` | recognition.py:62, 219, 577 | face_recognition.py:904 | ✅ |
| `face_service.recognize_face()` | recognition.py:131, 239 | face_recognition.py:856 | ✅ |

---

Готов начинать исправления по этому списку.
