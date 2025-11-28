-- Синхронизация полей verified и confidence в таблице photo_faces
-- Исправление для случаев, когда confidence = NULL

-- 1. Установить confidence=1 для всех записей с verified=true (включая NULL confidence)
UPDATE photo_faces
SET confidence = 1
WHERE verified = true 
  AND (confidence IS NULL OR confidence != 1);

-- 2. Установить verified=true для всех записей с confidence=1
UPDATE photo_faces
SET verified = true
WHERE confidence = 1 
  AND verified = false;

-- 3. Вывести статистику
DO $$
DECLARE
  total_records INTEGER;
  verified_with_conf_1 INTEGER;
  conf_1_with_verified INTEGER;
  null_confidence INTEGER;
BEGIN
  -- Общее количество записей
  SELECT COUNT(*) INTO total_records FROM photo_faces;
  
  -- Записи с verified=true и confidence=1
  SELECT COUNT(*) INTO verified_with_conf_1 
  FROM photo_faces 
  WHERE verified = true AND confidence = 1;
  
  -- Записи с confidence=1 и verified=true
  SELECT COUNT(*) INTO conf_1_with_verified 
  FROM photo_faces 
  WHERE confidence = 1 AND verified = true;
  
  -- Записи с NULL confidence
  SELECT COUNT(*) INTO null_confidence 
  FROM photo_faces 
  WHERE confidence IS NULL;
  
  RAISE NOTICE '=== Статистика синхронизации ===';
  RAISE NOTICE 'Всего записей: %', total_records;
  RAISE NOTICE 'Записей с verified=true и confidence=1: %', verified_with_conf_1;
  RAISE NOTICE 'Записей с confidence=1 и verified=true: %', conf_1_with_verified;
  RAISE NOTICE 'Записей с NULL confidence: %', null_confidence;
  
  IF null_confidence > 0 THEN
    RAISE WARNING 'Внимание: Найдено % записей с NULL confidence!', null_confidence;
  END IF;
END $$;

-- 4. Показать примеры записей, которые могут требовать внимания
SELECT 
  pf.id,
  gi.original_filename,
  p.real_name,
  pf.verified,
  pf.confidence
FROM photo_faces pf
JOIN gallery_images gi ON pf.photo_id = gi.id
LEFT JOIN people p ON pf.person_id = p.id
WHERE pf.confidence IS NULL OR (pf.verified = true AND pf.confidence != 1) OR (pf.confidence = 1 AND pf.verified = false)
LIMIT 10;
