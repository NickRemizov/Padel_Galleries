# Инструкции по деплою версии 0.7.9

## Что исправлено

Исправлена путаница между полями `insightface_confidence` (уверенность детектора что это лицо) и `recognition_confidence` (уверенность в идентификации конкретного человека).

## Измененные файлы

1. `app/admin/actions.ts` - обновлены `savePhotoFaceAction` и `updatePhotoFaceAction`
2. `components/admin/auto-recognition-dialog.tsx` - исправлена передача `recognition_confidence`
3. `components/admin/face-tagging-dialog.tsx` - добавлена передача `recognitionConfidence: 1.0` для ручной верификации
4. `scripts/010_check_confidence_fields.sql` - новый скрипт проверки БД
5. `PROJECT_FLOW.md` - обновлена документация по потокам данных

---

## ДЕПЛОЙ - Пошаговая инструкция

### Шаг 1: Остановить backend (если нужно обновить Python)

Этот фикс НЕ требует обновления Python backend, но если хотите:

\`\`\`bash
ssh nickr@23.88.61.20  # Или ваш SSH

# Остановка FastAPI
pkill -f "python.*main.py"
\`\`\`

### Шаг 2: Скопировать измененные файлы

#### Вариант A: Через Git (если у вас настроен Git на сервере)

\`\`\`bash
# На вашем компе в папке проекта
git add .
git commit -m "v0.7.9: fix confidence fields confusion"
git push

# На production сервере
cd /path/to/galeries
git pull
\`\`\`

#### Вариант B: Копирование файлов вручную (БЕЗ GIT)

**На вашем компе:**

\`\`\`bash
# 1. Скопировать actions.ts
scp app/admin/actions.ts nickr@23.88.61.20:/home/nickr/galeries/app/admin/

# 2. Скопировать auto-recognition-dialog.tsx
scp components/admin/auto-recognition-dialog.tsx nickr@23.88.61.20:/home/nickr/galeries/components/admin/

# 3. Скопировать face-tagging-dialog.tsx
scp components/admin/face-tagging-dialog.tsx nickr@23.88.61.20:/home/nickr/galeries/components/admin/

# 4. Скопировать SQL скрипт
scp scripts/010_check_confidence_fields.sql nickr@23.88.61.20:/home/nickr/galeries/scripts/

# 5. Скопировать документацию
scp PROJECT_FLOW.md nickr@23.88.61.20:/home/nickr/galeries/

# 6. Скопировать инструкции по деплою
scp DEPLOY_INSTRUCTIONS_v0.7.9.md nickr@23.88.61.20:/home/nickr/galeries/
\`\`\`

### Шаг 3: Деплой на Vercel

\`\`\`bash
# На вашем компе
cd /path/to/galeries

# Если используете Vercel CLI
vercel --prod

# ИЛИ просто коммит в GitHub (если настроен auto-deploy)
# git push → автоматический деплой на Vercel
\`\`\`

Vercel автоматически:
1. Соберет Next.js проект
2. Задеплоит на production
3. Обновит переменные окружения из настроек проекта

### Шаг 4: Проверка базы данных

Подключитесь к PostgreSQL и запустите проверочный скрипт:

\`\`\`bash
# На сервере где PostgreSQL
ssh nickr@23.88.61.20

# Запуск SQL скрипта
psql -U galeries_user -d galleries -f /home/nickr/galeries/scripts/010_check_confidence_fields.sql

# ИЛИ вручную подключиться и выполнить проверку
psql -U galeries_user -d galleries

-- В psql консоли:
-- Проверка 1: Существуют ли оба поля
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'photo_faces' 
  AND column_name IN ('insightface_confidence', 'recognition_confidence');

-- Проверка 2: Ошибки в данных
SELECT COUNT(*) as error_count
FROM photo_faces
WHERE recognition_confidence IS NOT NULL AND person_id IS NULL;
-- Должно быть 0

-- Проверка 3: Статистика
SELECT 
    COUNT(*) as total_faces,
    AVG(insightface_confidence) as avg_detection,
    AVG(recognition_confidence) as avg_recognition,
    COUNT(*) FILTER (WHERE verified = true) as verified_count
FROM photo_faces;

\q  -- выход из psql
\`\`\`

### Шаг 5: Перезапуск backend (если останавливали)

\`\`\`bash
# На сервере
cd /home/nickr/python
./start.sh  # Или ваш скрипт запуска

# Проверка что запустился
ps aux | grep "python.*main.py"
\`\`\`

---

## ТЕСТИРОВАНИЕ после деплоя

### Тест 1: Ручное тегирование с верификацией

1. Откройте админ-панель: `https://vlcpadel.com/admin`
2. Выберите любую галерею
3. Откройте фото с лицами (клик на фото)
4. **Ожидаемое поведение:**
   - FaceTaggingDialog открывается
   - Лица обнаруживаются автоматически
   - Выберите игрока для каждого лица
   - Нажмите "Сохранить"
5. **Проверка в БД:**
   \`\`\`sql
   SELECT 
       id, person_id, 
       insightface_confidence,    -- Должно быть ~0.9
       recognition_confidence,    -- Должно быть 1.0
       verified                   -- Должно быть true
   FROM photo_faces 
   WHERE photo_id = 'ВАШЕ_ФОТО_ID'
   ORDER BY created_at DESC;
   \`\`\`
6. **Ожидаемый результат:**
   - `insightface_confidence` ≈ 0.90-0.99 (от детектора)
   - `recognition_confidence` = 1.0 (ручная верификация)
   - `verified` = true

### Тест 2: Автоматическое распознавание

1. В админ-панели откройте галерею
2. Нажмите "Автоматическое распознавание"
3. Установите threshold (например, 0.7)
4. Нажмите "Начать обработку"
5. **Откройте консоль браузера (F12)**
6. **Проверьте логи:**
   \`\`\`
   [v0] Auto-recognition: detection confidence = 0.92
   [v0] Auto-recognition: recognition confidence = 0.78
   \`\`\`
7. **Проверка в БД:**
   \`\`\`sql
   SELECT 
       id, person_id,
       insightface_confidence,    -- От детектора
       recognition_confidence,    -- От распознавания
       verified                   -- false
   FROM photo_faces 
   WHERE verified = false 
     AND recognition_confidence IS NOT NULL
   ORDER BY created_at DESC 
   LIMIT 5;
   \`\`\`
8. **Ожидаемый результат:**
   - `insightface_confidence` ≈ 0.90+ (качество обнаружения)
   - `recognition_confidence` ≈ 0.70-0.90 (качество распознавания)
   - `verified` = false (автоматически)

### Тест 3: Проверка старых записей

Убедитесь что старые записи НЕ сломались:

\`\`\`sql
-- Найти старые записи с insightface_confidence но без recognition_confidence
SELECT 
    COUNT(*) as old_records_count,
    AVG(insightface_confidence) as avg_conf
FROM photo_faces
WHERE insightface_confidence IS NOT NULL
  AND recognition_confidence IS NULL
  AND created_at < NOW() - INTERVAL '1 day';
\`\`\`

Старые записи могут иметь только `insightface_confidence` - это нормально.

---

## ПРОБЛЕМЫ И РЕШЕНИЯ

### Проблема 1: "Cannot read property 'confidence'"

**Причина:** Старый код пытается читать несуществующее поле

**Решение:** Убедитесь что все файлы обновлены (особенно `face-tagging-dialog.tsx`)

### Проблема 2: В БД обе confidence = NULL

**Причина:** Backend не возвращает данные или запрос не выполнился

**Проверка:**
\`\`\`bash
# Откройте консоль браузера и проверьте Network tab
# Найдите запрос к /api/face-detection/detect
# Проверьте Response - должен быть массив с confidence
\`\`\`

**Решение:** Перезапустите FastAPI backend

### Проблема 3: Verified faces имеют recognition_confidence < 1.0

**Причина:** Старые записи до фикса

**Решение:** Обновите старые записи:
\`\`\`sql
UPDATE photo_faces
SET recognition_confidence = 1.0
WHERE verified = true 
  AND (recognition_confidence IS NULL OR recognition_confidence < 1.0);
\`\`\`

---

## ОТКАТ (если что-то пошло не так)

### Быстрый откат через Git

\`\`\`bash
# На сервере
cd /path/to/galeries
git log --oneline  # Найдите предыдущий коммит
git revert HEAD    # Откатить последний коммит

# На Vercel - автоматически задеплоится откат
\`\`\`

### Ручной откат файлов

Скопируйте старые версии файлов обратно:

\`\`\`bash
# Если сохранили бэкапы
scp backup/app/admin/actions.ts nickr@23.88.61.20:/home/nickr/galeries/app/admin/
scp backup/components/admin/auto-recognition-dialog.tsx nickr@23.88.61.20:/home/nickr/galeries/components/admin/
scp backup/components/admin/face-tagging-dialog.tsx nickr@23.88.61.20:/home/nickr/galeries/components/admin/
\`\`\`

---

## ВАЖНЫЕ ЗАМЕЧАНИЯ

1. **Этот фикс НЕ требует изменений в Python backend** - все изменения только в Next.js frontend

2. **Старые данные останутся работать** - новая логика обратно совместима

3. **Два поля НУЖНЫ ОБА:**
   - `insightface_confidence` - качество обнаружения лица (от детектора)
   - `recognition_confidence` - качество идентификации человека (от поиска)

4. **Для ручной верификации** `recognition_confidence` всегда = 1.0 (максимальная уверенность)

5. **После деплоя следите за консолью браузера** (F12) - там будут логи `[v0]` с информацией

---

## КОНТРОЛЬНЫЙ ЧЕКЛИСТ

После деплоя убедитесь:

- [ ] Все файлы скопированы на сервер
- [ ] Vercel успешно задеплоил проект
- [ ] SQL скрипт проверки выполнен
- [ ] Оба поля `insightface_confidence` и `recognition_confidence` существуют в БД
- [ ] Тест 1 (ручное тегирование) прошел успешно
- [ ] Тест 2 (автораспознавание) прошел успешно
- [ ] Нет ошибок в консоли браузера (F12)
- [ ] Нет ошибок в логах backend (`pm2 logs` или аналог)

---

**Версия:** 0.7.9
**Дата:** 2025-01-31
**Статус:** Готово к деплою
