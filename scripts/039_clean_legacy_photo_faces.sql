-- Скрипт для очистки записей с данными в legacy полях
-- Эти записи будут пересозданы заново с правильными InsightFace полями

-- 1. Показываем записи, которые будут удалены
SELECT 
    'Records to be deleted' as status,
    COUNT(*) as count
FROM photo_faces
WHERE bounding_box IS NOT NULL 
  AND insightface_bbox IS NULL;

-- 2. Показываем детали записей для проверки
SELECT 
    id,
    photo_id,
    person_id,
    bounding_box,
    confidence,
    insightface_bbox,
    insightface_confidence,
    verified,
    created_at
FROM photo_faces
WHERE bounding_box IS NOT NULL 
  AND insightface_bbox IS NULL
ORDER BY created_at DESC
LIMIT 20;

-- 3. Удаляем записи с данными в legacy полях
DELETE FROM photo_faces
WHERE bounding_box IS NOT NULL 
  AND insightface_bbox IS NULL;

-- 4. Показываем итоговое состояние
SELECT 
    'After cleanup' as status,
    COUNT(*) as total_records,
    COUNT(CASE WHEN insightface_bbox IS NOT NULL THEN 1 END) as records_with_insightface,
    COUNT(CASE WHEN bounding_box IS NOT NULL THEN 1 END) as records_with_legacy
FROM photo_faces;

-- 5. Показываем оставшиеся записи
SELECT 
    id,
    photo_id,
    person_id,
    insightface_bbox IS NOT NULL as has_insightface_bbox,
    insightface_confidence,
    verified,
    created_at
FROM photo_faces
ORDER BY created_at DESC
LIMIT 10;
