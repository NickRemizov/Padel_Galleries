-- Синхронизация verified и confidence в таблице photo_faces
-- 1. Для всех записей с verified=true устанавливаем confidence=1
-- 2. Для всех записей с confidence=1 устанавливаем verified=true

-- Шаг 1: Обновляем confidence=1 для всех verified=true записей
UPDATE photo_faces
SET confidence = 1.0
WHERE verified = true AND confidence != 1.0;

-- Получаем количество обновленных записей
DO $$
DECLARE
  updated_count INTEGER;
BEGIN
  GET DIAGNOSTICS updated_count = ROW_COUNT;
  RAISE NOTICE 'Обновлено записей (verified=true -> confidence=1): %', updated_count;
END $$;

-- Шаг 2: Обновляем verified=true для всех confidence=1 записей
UPDATE photo_faces
SET verified = true
WHERE confidence = 1.0 AND verified = false;

-- Получаем количество обновленных записей
DO $$
DECLARE
  updated_count INTEGER;
BEGIN
  GET DIAGNOSTICS updated_count = ROW_COUNT;
  RAISE NOTICE 'Обновлено записей (confidence=1 -> verified=true): %', updated_count;
END $$;

-- Проверка результатов
SELECT 
  COUNT(*) FILTER (WHERE verified = true AND confidence = 1.0) as verified_and_confident,
  COUNT(*) FILTER (WHERE verified = true AND confidence != 1.0) as verified_but_not_confident,
  COUNT(*) FILTER (WHERE verified = false AND confidence = 1.0) as confident_but_not_verified,
  COUNT(*) as total_records
FROM photo_faces;
