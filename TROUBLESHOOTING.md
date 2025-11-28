# Troubleshooting - Galeries v0.8.2

Решение частых проблем и ошибок.

## Проблемы с распознаванием лиц

### Лица не детектируются

**Симптомы:**
- Фото обрабатывается, но бейдж показывает "NFD" (No Faces Detected)
- В логах ошибок нет

**Причины и решения:**

1. **Слишком строгие фильтры качества**
   - Проверьте настройки в `face_recognition_config`
   - Попробуйте снизить `min_face_size`, `min_blur_score`, `min_detection_score`
   - Или используйте "Распознать заново без настроек" в FaceTaggingDialog

2. **Лица слишком маленькие**
   - Проверьте разрешение фото
   - Минимальный размер лица по умолчанию: 80px
   - Уменьшите `min_face_size` в настройках

3. **Фото размытое**
   - `blur_score` ниже порога (default: 100.0)
   - Попробуйте отключить фильтры качества

**Отладка:**
\`\`\`typescript
// В FaceTaggingDialog смотрите детальные метрики:
console.log("[v0] Face metrics:", {
  face_size,
  blur_score,
  det_score
});
\`\`\`

---

### Лица детектируются, но не распознаются

**Симптомы:**
- Бейдж показывает "X/Y" (есть неопознанные лица)
- Уверенность распознавания низкая

**Причины и решения:**

1. **Недостаточно обучающих данных**
   - Нужно верифицировать больше лиц этого игрока
   - Используйте кластеризацию для группировки похожих лиц
   - Вручную назначьте игрока нескольким фото

2. **Порог распознавания слишком высокий**
   - Verified threshold: 0.6
   - Unverified threshold: 0.75
   - Измените в `python/services/face_recognition.py`

3. **Индекс HNSWLIB не обновлен**
   - После назначения лиц нужно пересобрать индекс
   - Это должно происходить автоматически, но можно вручную:
   \`\`\`bash
   curl -X POST http://your-server:8001/rebuild-index
   \`\`\`

---

### Неправильное распознавание

**Симптомы:**
- Лицо распознано как другой игрок
- Высокая уверенность, но неправильный человек

**Причины и решения:**

1. **Похожие лица в базе**
   - Используйте ручную верификацию (FaceTaggingDialog)
   - Проверьте "Top-3 похожих лиц" в детальном окне

2. **Плохое качество эмбеддинга**
   - Проверьте `det_score` и `blur_score`
   - Переснимите фото или используйте другое

3. **Ошибка в обучении**
   - Найдите неправильно верифицированные лица:
   \`\`\`sql
   SELECT * FROM face_descriptors
   WHERE person_id = 'wrong-person-id'
   AND verified = true;
   \`\`\`
   - Исправьте вручную и пересоберите индекс

---

## Проблемы с базой данных

### Ошибка подключения к Supabase

**Симптомы:**
- `Error: Failed to fetch data from Supabase`
- Страницы не загружаются

**Решения:**

1. **Проверьте переменные окружения:**
   \`\`\`bash
   POSTGRES_URL="postgresql://..."
   SUPABASE_URL="https://..."
   SUPABASE_ANON_KEY="..."
   \`\`\`

2. **Проверьте RLS (Row Level Security):**
   - Убедитесь, что политики настроены правильно
   - Для админа должен быть доступ ко всем данным

3. **Проверьте статус Supabase:**
   - Откройте Supabase Dashboard
   - Проверьте, не превышен ли лимит запросов

---

### Медленные запросы

**Симптомы:**
- Галереи загружаются долго
- Таймауты при распознавании

**Решения:**

1. **Добавьте индексы:**
   \`\`\`sql
   CREATE INDEX IF NOT EXISTS idx_face_descriptors_embedding 
   ON face_descriptors USING ivfflat (embedding vector_cosine_ops);
   \`\`\`

2. **Оптимизируйте запросы:**
   - Используйте `EXPLAIN ANALYZE` для анализа
   - Добавьте `LIMIT` для больших таблиц

3. **Увеличьте ресурсы Supabase:**
   - Перейдите на более мощный план
   - Или оптимизируйте данные (удалите старые галереи)

---

## Проблемы с FastAPI бэкендом

### FastAPI недоступен

**Симптомы:**
- `Error: Failed to connect to FastAPI backend`
- Распознавание не работает

**Решения:**

1. **Проверьте, запущен ли сервер:**
   \`\`\`bash
   ssh your-server
   cd /home/nickr/python
   ./start.sh
   \`\`\`

2. **Проверьте переменные окружения:**
   \`\`\`bash
   FASTAPI_URL="http://your-server-ip:8001"
   NEXT_PUBLIC_FASTAPI_URL="http://your-server-ip:8001"
   \`\`\`

3. **Проверьте логи:**
   \`\`\`bash
   tail -f /home/nickr/python/app.log
   \`\`\`

4. **Проверьте firewall:**
   \`\`\`bash
   sudo ufw status
   sudo ufw allow 8001
   \`\`\`

---

### Ошибки InsightFace

**Симптомы:**
- `Error: Failed to load InsightFace model`
- `Error: Unable to detect faces`

**Решения:**

1. **Переустановите модель:**
   \`\`\`bash
   cd /home/nickr/python
   python3 -c "from insightface.app import FaceAnalysis; app = FaceAnalysis(); app.prepare(ctx_id=0)"
   \`\`\`

2. **Проверьте зависимости:**
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

3. **Проверьте память сервера:**
   \`\`\`bash
   free -h
   # Если мало памяти, перезапустите сервер
   sudo reboot
   \`\`\`

---

### HNSWLIB индекс не работает

**Симптомы:**
- Распознавание очень медленное
- Все лица распознаются как "неизвестные"

**Решения:**

1. **Пересоберите индекс:**
   \`\`\`bash
   curl -X POST http://your-server:8001/rebuild-index
   \`\`\`

2. **Проверьте логи:**
   \`\`\`bash
   tail -f /home/nickr/python/app.log | grep "index"
   \`\`\`

3. **Проверьте, есть ли верифицированные лица:**
   \`\`\`sql
   SELECT COUNT(*) FROM face_descriptors WHERE verified = true;
   \`\`\`
   - Если 0, нужно сначала верифицировать несколько лиц

---

## Проблемы с Vercel Blob

### Фото не загружаются

**Симптомы:**
- Ошибка при загрузке фото
- `Error: Failed to upload to Blob Storage`

**Решения:**

1. **Проверьте токен:**
   \`\`\`bash
   BLOB_READ_WRITE_TOKEN="..."
   \`\`\`

2. **Проверьте лимиты Vercel Blob:**
   - Бесплатный план: 1GB
   - Проверьте использование в Vercel Dashboard

3. **Проверьте размер файлов:**
   - Максимальный размер файла: 4.5MB (Next.js default)
   - Увеличьте в `next.config.js`:
   \`\`\`javascript
   experimental: {
     serverActions: {
       bodySizeLimit: '10mb'
     }
   }
   \`\`\`

---

## Проблемы с Telegram ботом

### Бот не отвечает

**Симптомы:**
- Команды не работают
- Фото не отправляются

**Решения:**

1. **Проверьте токен:**
   \`\`\`bash
   TELEGRAM_BOT_TOKEN="..."
   \`\`\`

2. **Проверьте webhook:**
   \`\`\`bash
   curl https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getWebhookInfo
   \`\`\`

3. **Установите webhook заново:**
   \`\`\`bash
   curl -X POST "https://api.telegram.org/bot<YOUR_BOT_TOKEN>/setWebhook?url=https://your-domain.com/api/telegram/webhook"
   \`\`\`

4. **Проверьте логи Vercel:**
   - Откройте Vercel Dashboard → Functions → Logs
   - Найдите ошибки в `/api/telegram/webhook`

---

### Фото не отправляются игрокам

**Симптомы:**
- Кнопка "Отправить в Telegram" не работает
- Ошибка: "Failed to send photos"

**Решения:**

1. **Проверьте telegram_id игрока:**
   \`\`\`sql
   SELECT telegram_id FROM people WHERE id = 'your-person-id';
   \`\`\`
   - Должен быть числовой ID, не NULL

2. **Проверьте, запустил ли пользователь бота:**
   - Пользователь должен отправить `/start` боту
   - Иначе бот не может отправить ему сообщения

3. **Проверьте URL фото:**
   - Telegram должен иметь доступ к Blob URL
   - Проверьте, что URL публичный

---

## Общие проблемы

### Приложение не запускается локально

**Решения:**

1. **Установите зависимости:**
   \`\`\`bash
   npm install
   \`\`\`

2. **Создайте `.env.local`:**
   \`\`\`bash
   cp .env.example .env.local
   # Заполните переменные
   \`\`\`

3. **Запустите dev сервер:**
   \`\`\`bash
   npm run dev
   \`\`\`

---

### Ошибки типов TypeScript

**Решения:**

1. **Обновите типы:**
   \`\`\`bash
   npm run build
   \`\`\`

2. **Проверьте версии:**
   - Next.js: 15.x
   - React: 19.x
   - TypeScript: 5.x

---

## Получение помощи

Если проблема не решена:

1. **Проверьте логи:**
   - Vercel: Dashboard → Functions → Logs
   - FastAPI: `tail -f /home/nickr/python/app.log`
   - Browser: Console (F12)

2. **Создайте issue:**
   - Опишите проблему
   - Приложите логи
   - Укажите версию (v0.8.2)

3. **Обратитесь к документации:**
   - README.md
   - ARCHITECTURE.md
   - RECOGNITION_PROCESS_DOCUMENTATION.md
\`\`\`

```md file="AUTO-DEPLOY-README.md" isDeleted="true"
...deleted...
