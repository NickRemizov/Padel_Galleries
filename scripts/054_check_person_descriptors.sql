-- Проверка дескрипторов для конкретного человека

-- 1. Найти ID человека "Александр Сметана"
SELECT id, real_name, telegram_name 
FROM people 
WHERE real_name ILIKE '%Сметана%' OR real_name ILIKE '%Александр%';

-- 2. Проверить все верифицированные лица этого человека
SELECT 
    pf.id as face_id,
    pf.photo_id,
    pf.verified,
    pf.confidence,
    gi.image_url,
    pf.created_at,
    pf.updated_at
FROM photo_faces pf
JOIN gallery_images gi ON gi.id = pf.photo_id
JOIN people p ON p.id = pf.person_id
WHERE p.real_name ILIKE '%Сметана%'
ORDER BY pf.updated_at DESC;

-- 3. Проверить дескрипторы для этого человека
SELECT 
    fd.id as descriptor_id,
    fd.person_id,
    p.real_name,
    fd.source_image_id,
    gi.image_url as source_image,
    fd.created_at,
    LENGTH(fd.descriptor::text) as descriptor_length
FROM face_descriptors fd
JOIN people p ON p.id = fd.person_id
LEFT JOIN gallery_images gi ON gi.id = fd.source_image_id
WHERE p.real_name ILIKE '%Сметана%'
ORDER BY fd.created_at DESC;

-- 4. Найти верифицированные лица БЕЗ дескрипторов для этого человека
SELECT 
    pf.id as face_id,
    pf.photo_id,
    p.real_name,
    gi.image_url,
    pf.verified,
    pf.updated_at
FROM photo_faces pf
JOIN people p ON p.id = pf.person_id
JOIN gallery_images gi ON gi.id = pf.photo_id
LEFT JOIN face_descriptors fd ON fd.person_id = pf.person_id AND fd.source_image_id = pf.photo_id
WHERE p.real_name ILIKE '%Сметана%'
    AND pf.verified = true
    AND fd.id IS NULL
ORDER BY pf.updated_at DESC;

-- 5. Общая статистика
SELECT 
    p.real_name,
    COUNT(DISTINCT pf.id) as total_faces,
    COUNT(DISTINCT CASE WHEN pf.verified THEN pf.id END) as verified_faces,
    COUNT(DISTINCT fd.id) as total_descriptors
FROM people p
LEFT JOIN photo_faces pf ON pf.person_id = p.id
LEFT JOIN face_descriptors fd ON fd.person_id = p.id
WHERE p.real_name ILIKE '%Сметана%'
GROUP BY p.id, p.real_name;
