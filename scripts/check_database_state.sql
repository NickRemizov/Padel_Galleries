-- Проверка состояния базы данных для распознавания лиц

-- 1. Количество игроков
SELECT 'Total people count:' as info, COUNT(*) as count FROM people;

-- 2. Количество лиц с эмбеддингами InsightFace
SELECT 'Photo faces with InsightFace descriptors:' as info, 
       COUNT(*) as count 
FROM photo_faces 
WHERE insightface_descriptor IS NOT NULL;

-- 3. Количество лиц без эмбеддингов
SELECT 'Photo faces WITHOUT InsightFace descriptors:' as info, 
       COUNT(*) as count 
FROM photo_faces 
WHERE insightface_descriptor IS NULL;

-- 4. Количество верифицированных лиц
SELECT 'Verified photo faces:' as info, 
       COUNT(*) as count 
FROM photo_faces 
WHERE verified = true;

-- 5. Распределение лиц по игрокам (топ 10)
SELECT p.real_name, 
       p.telegram_name,
       COUNT(pf.id) as faces_count,
       COUNT(CASE WHEN pf.insightface_descriptor IS NOT NULL THEN 1 END) as with_embeddings,
       COUNT(CASE WHEN pf.verified = true THEN 1 END) as verified_count
FROM people p
LEFT JOIN photo_faces pf ON p.id = pf.person_id
GROUP BY p.id, p.real_name, p.telegram_name
ORDER BY faces_count DESC
LIMIT 10;

-- 6. Общая статистика
SELECT 
    (SELECT COUNT(*) FROM people) as total_people,
    (SELECT COUNT(*) FROM photo_faces) as total_faces,
    (SELECT COUNT(*) FROM photo_faces WHERE insightface_descriptor IS NOT NULL) as faces_with_embeddings,
    (SELECT COUNT(*) FROM photo_faces WHERE verified = true) as verified_faces,
    (SELECT COUNT(DISTINCT person_id) FROM photo_faces WHERE insightface_descriptor IS NOT NULL) as people_with_embeddings;

-- 7. Проверка старой таблицы face_descriptors
SELECT 'Face descriptors (old table):' as info, COUNT(*) as count FROM face_descriptors;
