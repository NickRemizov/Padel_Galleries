# Backend Testing Guide

Комплексное руководство по тестированию FastAPI бэкенда после миграции на PostgreSQL.

## Оглавление

1. [Быстрый старт](#быстрый-старт)
2. [Автоматическое тестирование](#автоматическое-тестирование)
3. [Ручное тестирование](#ручное-тестирование)
4. [Тестирование интеграции с фронтендом](#тестирование-интеграции-с-фронтендом)
5. [Проверка производительности](#проверка-производительности)
6. [Troubleshooting](#troubleshooting)

---

## Быстрый старт

### Предварительные требования

- Python 3.10+
- FastAPI сервер запущен на `http://23.88.61.20:8001`
- PostgreSQL база данных настроена и доступна
- Установлен `requests` и `colorama`: `pip install requests colorama`

### Запуск тестов

\`\`\`bash
# Перейти в папку проекта
cd /home/nickr/scripts

# Активировать виртуальное окружение (если есть)
source venv/bin/activate  # Или пропустить если venv нет

# Запустить тесты
python test_backend_endpoints.py

# С подробным выводом
python test_backend_endpoints.py --verbose
\`\`\`

### Ожидаемый результат

\`\`\`
================================================================================
Test Summary: FastAPI Backend
Duration: 45.23s
Total: 42 | Passed: 40 | Failed: 0 | Warnings: 2
================================================================================

✓ All tests passed!
\`\`\`

---

## Автоматическое тестирование

### Что тестируется

#### 1. Health Check
- ✅ Сервер отвечает
- ✅ Модель InsightFace загружена
- ✅ Статус "healthy"

#### 2. Configuration
- ✅ GET `/api/v2/config` - получение текущей конфигурации
- ✅ PUT `/api/v2/config` - обновление параметров качества
- ✅ Структура quality_filters корректна

#### 3. Database Connection
- ✅ PostgreSQL доступен через FastAPI
- ✅ Статистика возвращается корректно
- ⚠️ Есть данные в базе (warning если пусто)

#### 4. Face Detection
- ✅ POST `/detect-faces` - детекция лиц на изображении
- ✅ Возвращает bbox, confidence, blur_score
- ✅ Embedding 512-мерный
- ✅ Quality filters применяются

#### 5. Face Recognition  
- ✅ POST `/recognize-face` - распознавание по embedding
- ✅ Возвращает person_id и confidence
- ⚠️ Может не найти совпадений (expected для тестового фото)

#### 6. Index Rebuild
- ✅ POST `/rebuild-index` - пересборка HNSWLIB индекса
- ✅ Возвращает old_count, new_count, unique_people

#### 7. Training & Statistics
- ✅ GET `/api/v2/statistics` - общая статистика
- ✅ GET `/api/v2/train/history` - история обучений
- ✅ Структура сессий корректна

#### 8. Batch Recognition
- ✅ POST `/api/v2/batch-recognize` - массовое распознавание
- ✅ Возвращает processed, recognized, filtered_out

#### 9. Clustering
- ✅ POST `/cluster-unknown-faces` - кластеризация неизвестных лиц
- ✅ Endpoint доступен (требует реальный gallery_id)

---

## Ручное тестирование

### Использование curl

#### 1. Проверка здоровья сервера

\`\`\`bash
curl http://23.88.61.20:8001/api/health
\`\`\`

**Ожидаемый ответ:**
\`\`\`json
{
  "status": "healthy",
  "service": "padel-recognition",
  "model_loaded": true
}
\`\`\`

#### 2. Получение конфигурации

\`\`\`bash
curl http://23.88.61.20:8001/api/v2/config
\`\`\`

**Ожидаемый ответ:**
\`\`\`json
{
  "quality_filters": {
    "min_detection_score": 0.7,
    "min_face_size": 80,
    "min_blur_score": 100
  }
}
\`\`\`

#### 3. Детекция лиц

\`\`\`bash
curl -X POST http://23.88.61.20:8001/detect-faces \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://example.com/photo.jpg",
    "apply_quality_filters": true
  }'
\`\`\`

**Ожидаемый ответ:**
\`\`\`json
{
  "faces": [
    {
      "insightface_bbox": {"x": 100, "y": 150, "width": 200, "height": 200},
      "confidence": 0.98,
      "blur_score": 120.5,
      "embedding": [0.123, -0.456, ...],  // 512 чисел
      "distance_to_nearest": 0.45,
      "top_matches": [
        {"person_id": "uuid", "confidence": 0.75}
      ]
    }
  ]
}
\`\`\`

#### 4. Распознавание лица

\`\`\`bash
curl -X POST http://23.88.61.20:8001/recognize-face \
  -H "Content-Type: application/json" \
  -d '{
    "embedding": [0.123, -0.456, ...],  // 512-мерный вектор
    "confidence_threshold": 0.60
  }'
\`\`\`

**Ожидаемый ответ (найдено):**
\`\`\`json
{
  "person_id": "uuid-of-person",
  "confidence": 0.85
}
\`\`\`

**Ожидаемый ответ (не найдено):**
\`\`\`json
{
  "person_id": null,
  "confidence": null
}
\`\`\`

#### 5. Пересборка индекса

\`\`\`bash
curl -X POST http://23.88.61.20:8001/rebuild-index
\`\`\`

**Ожидаемый ответ:**
\`\`\`json
{
  "success": true,
  "old_count": 150,
  "new_count": 162,
  "unique_people": 45
}
\`\`\`

#### 6. Статистика

\`\`\`bash
curl http://23.88.61.20:8001/api/v2/statistics
\`\`\`

**Ожидаемый ответ:**
\`\`\`json
{
  "people_count": 45,
  "total_faces": 162,
  "unique_photos": 98
}
\`\`\`

#### 7. История обучений

\`\`\`bash
curl "http://23.88.61.20:8001/api/v2/train/history?limit=5"
\`\`\`

**Ожидаемый ответ:**
\`\`\`json
{
  "sessions": [
    {
      "id": "session-uuid",
      "created_at": "2025-01-20T10:30:00",
      "training_mode": "full",
      "status": "completed",
      "people_count": 45,
      "faces_count": 162,
      "metrics": {
        "accuracy": 0.92
      }
    }
  ],
  "total": 12
}
\`\`\`

#### 8. Batch Recognition

\`\`\`bash
curl -X POST http://23.88.61.20:8001/api/v2/batch-recognize \
  -H "Content-Type: application/json" \
  -d '{
    "confidence_threshold": 0.60,
    "apply_quality_filters": true
  }'
\`\`\`

**Ожидаемый ответ:**
\`\`\`json
{
  "processed": 250,
  "recognized": 180,
  "filtered_out": 70
}
\`\`\`

---

## Тестирование интеграции с фронтендом

### 1. Проверка переменных окружения

В Next.js проект должны быть установлены:

\`\`\`env
FASTAPI_URL=http://23.88.61.20:8001
NEXT_PUBLIC_FASTAPI_URL=http://23.88.61.20:8001
\`\`\`

### 2. Проверка API routes Next.js

#### GET /api/admin/training/config

\`\`\`bash
curl http://localhost:3000/api/admin/training/config \
  -H "Cookie: auth-token=YOUR_TOKEN"
\`\`\`

Должен проксировать запрос к FastAPI и вернуть конфигурацию.

#### PUT /api/admin/training/config

\`\`\`bash
curl -X PUT http://localhost:3000/api/admin/training/config \
  -H "Content-Type: application/json" \
  -H "Cookie: auth-token=YOUR_TOKEN" \
  -d '{
    "quality_filters": {
      "min_detection_score": 0.75,
      "min_face_size": 90,
      "min_blur_score": 110
    }
  }'
\`\`\`

### 3. Тестирование UI компонентов

#### Face Training Manager

1. Откройте админ панель: `http://localhost:3000/admin`
2. Перейдите на вкладку "Обучение модели"
3. Проверьте:
   - ✅ Загрузка конфигурации без ошибок
   - ✅ Слайдеры для min_blur_score, min_detection_score, min_face_size
   - ✅ Сохранение настроек работает
   - ✅ История обучений отображается
   - ✅ Статистика загружается

#### Face Tagging Dialog

1. Откройте любую галерею
2. Кликните на фото для тегирования
3. Проверьте:
   - ✅ Детекция лиц работает
   - ✅ Распознавание лиц работает
   - ✅ Кнопка "Распознать без фильтров" вызывает FastAPI
   - ✅ Метрики (blur_score, det_score, recognition_confidence) отображаются
   - ✅ Сохранение тегов работает

---

## Проверка производительности

### 1. Response Time

\`\`\`bash
# Health check должен ответить < 100ms
time curl http://23.88.61.20:8001/api/health

# Детекция лиц < 3s на одно фото
time curl -X POST http://23.88.61.20:8001/detect-faces \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://example.com/photo.jpg", "apply_quality_filters": true}'

# Распознавание < 100ms
time curl -X POST http://23.88.61.20:8001/recognize-face \
  -H "Content-Type: application/json" \
  -d '{"embedding": [...], "confidence_threshold": 0.60}'
\`\`\`

### 2. Concurrent Requests

\`\`\`bash
# Используйте Apache Bench
ab -n 100 -c 10 http://23.88.61.20:8001/api/health

# Ожидаемый результат:
# - 100% success rate
# - Mean time per request < 200ms
\`\`\`

### 3. Memory Usage

\`\`\`bash
# SSH на сервер
ssh user@23.88.61.20

# Проверьте использование памяти FastAPI процессом
ps aux | grep uvicorn

# Должно быть < 2GB RAM в idle
# < 4GB RAM при активной обработке
\`\`\`

---

## Troubleshooting

### Проблема: "Connection refused"

**Причина:** FastAPI сервер не запущен

**Решение:**
\`\`\`bash
ssh user@23.88.61.20
cd /path/to/backend
source venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
\`\`\`

### Проблема: "Database connection failed"

**Причина:** PostgreSQL недоступен или неверные учетные данные

**Решение:**
\`\`\`bash
# Проверьте подключение к БД
psql $PV_POSTGRES_URL

# Проверьте переменные окружения FastAPI
echo $PV_POSTGRES_URL
\`\`\`

### Проблема: "Model not loaded"

**Причина:** InsightFace модель не инициализирована

**Решение:**
1. Проверьте логи FastAPI: `tail -f /path/to/backend/logs/app.log`
2. Убедитесь что модель antelopev2 скачана и распакована
3. Перезапустите FastAPI

### Проблема: "CORS error" в браузере

**Причина:** CORS не настроен для вашего домена

**Решение:**
Добавьте домен в `ALLOWED_ORIGINS` в `.env` FastAPI:
\`\`\`env
ALLOWED_ORIGINS=http://localhost:3000,https://vlcpadel.com
\`\`\`

### Проблема: Tests fail with "quality_filters missing"

**Причина:** База данных не содержит запись конфигурации

**Решение:**
\`\`\`bash
# Создайте дефолтную конфигурацию через API
curl -X PUT http://23.88.61.20:8001/api/v2/config \
  -H "Content-Type: application/json" \
  -d '{
    "min_face_size": 80,
    "min_detection_score": 0.7,
    "min_blur_score": 100
  }'
\`\`\`

---

## Чек-лист готовности к production

- [ ] Все 9 групп тестов проходят успешно
- [ ] Response time < 3s для детекции
- [ ] Response time < 100ms для распознавания
- [ ] Статистика показывает > 0 людей в базе
- [ ] HNSWLIB индекс пересобирается без ошибок
- [ ] Фронтенд успешно загружает конфигурацию
- [ ] Face Tagging Dialog работает со всеми функциями
- [ ] Batch recognition обрабатывает > 100 фото
- [ ] Логи FastAPI не содержат ERROR (только INFO/WARNING)
- [ ] Memory usage стабильно < 4GB
- [ ] PostgreSQL connection pool не исчерпывается

---

## Следующие шаги

После успешного прохождения всех тестов:

1. **Deploy на production:**
   - Обновите `FASTAPI_URL` на production домен
   - Настройте HTTPS через Nginx/Traefik
   - Добавьте мониторинг (Prometheus/Grafana)

2. **Настройте CI/CD:**
   - Добавьте `test_backend_endpoints.py` в GitHub Actions
   - Запускайте тесты перед каждым деплоем
   - Автоматический rollback при провале тестов

3. **Оптимизация:**
   - Кэширование результатов распознавания
   - CDN для изображений через MinIO
   - Redis для session store

4. **Мониторинг:**
   - Алерты при failure rate > 1%
   - Response time metrics в Grafana
   - Database connection pool metrics
