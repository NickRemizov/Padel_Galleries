-- Проверка: какие верифицированные лица не имеют дескрипторов

-- Исправлено: p.name → p.real_name, gi.blob_url → gi.image_url
SELECT 
  pf.id as photo_face_id,
  pf.photo_id,
  pf.person_id,
  p.real_name as person_name,
  gi.image_url,
  pf.verified,
  COUNT(fd.id) as descriptor_count
FROM photo_faces pf
JOIN people p ON p.id = pf.person_id
JOIN gallery_images gi ON gi.id = pf.photo_id
LEFT JOIN face_descriptors fd ON fd.person_id = pf.person_id AND fd.source_image_id = pf.photo_id
WHERE pf.verified = true
GROUP BY pf.id, pf.photo_id, pf.person_id, p.real_name, gi.image_url, pf.verified
HAVING COUNT(fd.id) = 0
ORDER BY p.real_name, gi.image_url;

-- Статистика
SELECT 
  'Total verified faces' as metric,
  COUNT(*) as count
FROM photo_faces
WHERE verified = true

UNION ALL

SELECT 
  'Verified faces WITH descriptors' as metric,
  COUNT(DISTINCT pf.id) as count
FROM photo_faces pf
JOIN face_descriptors fd ON fd.person_id = pf.person_id AND fd.source_image_id = pf.photo_id
WHERE pf.verified = true

UNION ALL

SELECT 
  'Verified faces WITHOUT descriptors' as metric,
  COUNT(*) as count
FROM photo_faces pf
LEFT JOIN face_descriptors fd ON fd.person_id = pf.person_id AND fd.source_image_id = pf.photo_id
WHERE pf.verified = true AND fd.id IS NULL;
