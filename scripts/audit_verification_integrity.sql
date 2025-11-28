-- Проверка целостности данных верификации
-- Все верифицированные лица ДОЛЖНЫ иметь confidence = 1.0

-- 1. Найти все верифицированные лица с confidence != 1.0 (ОШИБКА!)
SELECT 
    pf.id,
    gi.image_url,
    p.real_name as person_name,
    pf.confidence,
    pf.verified,
    pf.created_at
FROM photo_faces pf
JOIN gallery_images gi ON pf.photo_id = gi.id
JOIN people p ON pf.person_id = p.id
WHERE pf.verified = true 
  AND pf.confidence != 1.0
ORDER BY pf.created_at DESC;

-- 2. Статистика по верифицированным лицам
SELECT 
    CASE 
        WHEN confidence = 1.0 THEN 'Correct (100%)'
        WHEN confidence >= 0.9 THEN '90-99%'
        WHEN confidence >= 0.8 THEN '80-89%'
        ELSE 'Below 80%'
    END as confidence_range,
    COUNT(*) as count
FROM photo_faces
WHERE verified = true
GROUP BY 
    CASE 
        WHEN confidence = 1.0 THEN 'Correct (100%)'
        WHEN confidence >= 0.9 THEN '90-99%'
        WHEN confidence >= 0.8 THEN '80-89%'
        ELSE 'Below 80%'
    END
ORDER BY confidence_range;

-- 3. Исправить все верифицированные лица с неправильным confidence
-- ВНИМАНИЕ: Раскомментируйте только после проверки результатов выше!
-- UPDATE photo_faces 
-- SET confidence = 1.0 
-- WHERE verified = true AND confidence != 1.0;
