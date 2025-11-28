-- Statistics on verified faces per person
-- This helps determine the minimum faces threshold for training

-- Overall statistics
SELECT 
  'Total Statistics' as category,
  COUNT(DISTINCT person_id) as total_people,
  COUNT(*) as total_verified_faces,
  ROUND(AVG(face_count), 2) as avg_faces_per_person,
  MIN(face_count) as min_faces,
  MAX(face_count) as max_faces
FROM (
  SELECT 
    person_id,
    COUNT(*) as face_count
  FROM photo_faces
  WHERE verified = true 
    AND person_id IS NOT NULL
  GROUP BY person_id
) as person_stats;

-- Distribution by face count thresholds
SELECT 
  'People with >= 1 faces' as threshold,
  COUNT(*) as people_count,
  SUM(face_count) as total_faces
FROM (
  SELECT 
    person_id,
    COUNT(*) as face_count
  FROM photo_faces
  WHERE verified = true 
    AND person_id IS NOT NULL
  GROUP BY person_id
  HAVING COUNT(*) >= 1
) as stats
UNION ALL
SELECT 
  'People with >= 3 faces' as threshold,
  COUNT(*) as people_count,
  SUM(face_count) as total_faces
FROM (
  SELECT 
    person_id,
    COUNT(*) as face_count
  FROM photo_faces
  WHERE verified = true 
    AND person_id IS NOT NULL
  GROUP BY person_id
  HAVING COUNT(*) >= 3
) as stats
UNION ALL
SELECT 
  'People with >= 5 faces' as threshold,
  COUNT(*) as people_count,
  SUM(face_count) as total_faces
FROM (
  SELECT 
    person_id,
    COUNT(*) as face_count
  FROM photo_faces
  WHERE verified = true 
    AND person_id IS NOT NULL
  GROUP BY person_id
  HAVING COUNT(*) >= 5
) as stats
UNION ALL
SELECT 
  'People with >= 10 faces' as threshold,
  COUNT(*) as people_count,
  SUM(face_count) as total_faces
FROM (
  SELECT 
    person_id,
    COUNT(*) as face_count
  FROM photo_faces
  WHERE verified = true 
    AND person_id IS NOT NULL
  GROUP BY person_id
  HAVING COUNT(*) >= 10
) as stats
UNION ALL
SELECT 
  'People with >= 15 faces' as threshold,
  COUNT(*) as people_count,
  SUM(face_count) as total_faces
FROM (
  SELECT 
    person_id,
    COUNT(*) as face_count
  FROM photo_faces
  WHERE verified = true 
    AND person_id IS NOT NULL
  GROUP BY person_id
  HAVING COUNT(*) >= 15
) as stats;

-- Detailed breakdown per person (top 20 by face count)
SELECT 
  p.real_name,
  p.telegram_name,
  COUNT(pf.id) as verified_faces_count,
  COUNT(DISTINCT pf.photo_id) as unique_photos
FROM photo_faces pf
JOIN people p ON pf.person_id = p.id
WHERE pf.verified = true
GROUP BY p.id, p.real_name, p.telegram_name
ORDER BY verified_faces_count DESC
LIMIT 20;

-- Distribution histogram
SELECT 
  face_range,
  COUNT(*) as people_count,
  SUM(face_count) as total_faces
FROM (
  SELECT 
    person_id,
    COUNT(*) as face_count,
    CASE 
      WHEN COUNT(*) = 1 THEN '1 face'
      WHEN COUNT(*) = 2 THEN '2 faces'
      WHEN COUNT(*) BETWEEN 3 AND 4 THEN '3-4 faces'
      WHEN COUNT(*) BETWEEN 5 AND 9 THEN '5-9 faces'
      WHEN COUNT(*) BETWEEN 10 AND 14 THEN '10-14 faces'
      WHEN COUNT(*) BETWEEN 15 AND 19 THEN '15-19 faces'
      WHEN COUNT(*) >= 20 THEN '20+ faces'
    END as face_range
  FROM photo_faces
  WHERE verified = true 
    AND person_id IS NOT NULL
  GROUP BY person_id
) as person_stats
GROUP BY face_range
ORDER BY 
  CASE face_range
    WHEN '1 face' THEN 1
    WHEN '2 faces' THEN 2
    WHEN '3-4 faces' THEN 3
    WHEN '5-9 faces' THEN 4
    WHEN '10-14 faces' THEN 5
    WHEN '15-19 faces' THEN 6
    WHEN '20+ faces' THEN 7
  END;
