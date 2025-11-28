-- Исправление всех верифицированных лиц с неправильным confidence
-- Это скрипт для одноразового исправления существующих данных

-- Показать что будет исправлено
SELECT 
    pf.id,
    gi.image_url,
    p.real_name as person_name,
    pf.confidence as old_confidence,
    1.0 as new_confidence,
    pf.verified,
    pf.created_at
FROM photo_faces pf
JOIN gallery_images gi ON pf.photo_id = gi.id
JOIN people p ON pf.person_id = p.id
WHERE pf.verified = true 
  AND pf.confidence != 1.0;

-- Исправить все верифицированные лица
UPDATE photo_faces 
SET confidence = 1.0 
WHERE verified = true AND confidence != 1.0;

-- Проверить результат
SELECT 
    COUNT(*) as total_verified,
    COUNT(CASE WHEN confidence = 1.0 THEN 1 END) as correct_confidence,
    COUNT(CASE WHEN confidence != 1.0 THEN 1 END) as incorrect_confidence
FROM photo_faces
WHERE verified = true;
