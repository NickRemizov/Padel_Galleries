-- Скрипт проверки полей confidence в таблице photo_faces
-- Версия: 0.7.9
-- Дата: 2025-01-31

-- 1. Проверка существования обоих полей
SELECT 
    column_name, 
    data_type,
    is_nullable
FROM information_schema.columns 
WHERE table_name = 'photo_faces' 
  AND column_name IN ('insightface_confidence', 'recognition_confidence')
ORDER BY column_name;

-- Ожидаемый результат:
-- column_name              | data_type | is_nullable
-- -------------------------|-----------|------------
-- insightface_confidence   | real      | YES
-- recognition_confidence   | real      | YES


-- 2. Проверка на ошибки: лица с recognition_confidence но без person_id
SELECT 
    COUNT(*) as error_count,
    'Лица с recognition_confidence но без person_id (ОШИБКА!)' as description
FROM photo_faces
WHERE recognition_confidence IS NOT NULL 
  AND person_id IS NULL;

-- Ожидаемый результат: error_count = 0


-- 3. Проверка verified faces: должны иметь recognition_confidence = 1.0
SELECT 
    COUNT(*) as mismatch_count,
    'Verified faces без recognition_confidence = 1.0' as description
FROM photo_faces
WHERE verified = true 
  AND (recognition_confidence IS NULL OR ABS(recognition_confidence - 1.0) > 0.01);

-- Ожидаемый результат: mismatch_count = 0 (после новых сохранений)


-- 4. Статистика по confidence полям
SELECT 
    COUNT(*) as total_faces,
    COUNT(*) FILTER (WHERE insightface_confidence IS NOT NULL) as have_detection_conf,
    COUNT(*) FILTER (WHERE recognition_confidence IS NOT NULL) as have_recognition_conf,
    ROUND(AVG(insightface_confidence)::numeric, 2) as avg_detection_conf,
    ROUND(AVG(recognition_confidence)::numeric, 2) as avg_recognition_conf,
    COUNT(*) FILTER (WHERE verified = true) as verified_faces,
    COUNT(*) FILTER (WHERE verified = false) as auto_recognized
FROM photo_faces
WHERE insightface_confidence IS NOT NULL OR recognition_confidence IS NOT NULL;

-- Ожидаемый результат (пример):
-- total_faces | have_detection_conf | have_recognition_conf | avg_detection_conf | avg_recognition_conf | verified_faces | auto_recognized
-- ------------|---------------------|----------------------|-------------------|---------------------|---------------|---------------
-- 1250        | 1200                | 980                  | 0.92              | 0.81                | 450           | 800


-- 5. Проверка старых записей с путаницей полей
SELECT 
    id,
    person_id,
    insightface_confidence,
    recognition_confidence,
    verified,
    created_at
FROM photo_faces
WHERE 
    -- Случай 1: verified=false но recognition_confidence=1.0 (СТРАННО!)
    (verified = false AND ABS(recognition_confidence - 1.0) < 0.01)
    OR
    -- Случай 2: verified=true но recognition_confidence < 1.0 (ОШИБКА!)
    (verified = true AND recognition_confidence IS NOT NULL AND recognition_confidence < 0.99)
ORDER BY created_at DESC
LIMIT 20;

-- Если есть результаты - это записи с путаницей


-- 6. Разбивка по типам confidence
SELECT 
    CASE
        WHEN insightface_confidence IS NULL AND recognition_confidence IS NULL THEN 'Оба NULL'
        WHEN insightface_confidence IS NOT NULL AND recognition_confidence IS NULL THEN 'Только detection'
        WHEN insightface_confidence IS NULL AND recognition_confidence IS NOT NULL THEN 'Только recognition'
        ELSE 'Оба заполнены'
    END as confidence_status,
    COUNT(*) as count,
    ROUND(AVG(insightface_confidence)::numeric, 2) as avg_detection,
    ROUND(AVG(recognition_confidence)::numeric, 2) as avg_recognition
FROM photo_faces
GROUP BY confidence_status
ORDER BY count DESC;


-- 7. Топ-10 фото с самыми низкими recognition_confidence (потенциальные ошибки)
SELECT 
    pf.id,
    pf.photo_id,
    pf.person_id,
    p.real_name,
    pf.insightface_confidence,
    pf.recognition_confidence,
    pf.verified,
    pf.created_at
FROM photo_faces pf
LEFT JOIN people p ON pf.person_id = p.id
WHERE pf.recognition_confidence IS NOT NULL
  AND pf.recognition_confidence < 0.70
ORDER BY pf.recognition_confidence ASC
LIMIT 10;


-- Сводка по результатам проверки
SELECT 
    'ИТОГИ ПРОВЕРКИ' as title,
    '===================' as separator;

SELECT 
    'Всего лиц в БД:' as metric,
    COUNT(*)::text as value
FROM photo_faces;

SELECT 
    'Лиц с обоими confidence:' as metric,
    COUNT(*)::text as value
FROM photo_faces
WHERE insightface_confidence IS NOT NULL 
  AND recognition_confidence IS NOT NULL;

SELECT 
    'Проблемных записей:' as metric,
    COUNT(*)::text as value
FROM photo_faces
WHERE 
    (recognition_confidence IS NOT NULL AND person_id IS NULL)
    OR
    (verified = true AND recognition_confidence IS NOT NULL AND recognition_confidence < 0.99);
