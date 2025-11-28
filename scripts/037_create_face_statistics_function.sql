-- Create function to get overall face statistics for training
CREATE OR REPLACE FUNCTION get_face_statistics_overall()
RETURNS TABLE (
  total_people bigint,
  total_verified_faces bigint,
  people_with_min_faces bigint,
  avg_faces_per_person numeric,
  faces_by_count jsonb
) AS $$
BEGIN
  RETURN QUERY
  WITH face_counts AS (
    SELECT 
      person_id,
      COUNT(*) as face_count
    FROM photo_faces
    WHERE verified = true AND person_id IS NOT NULL
    GROUP BY person_id
  ),
  count_distribution AS (
    SELECT
      CASE
        WHEN face_count BETWEEN 3 AND 4 THEN '3-4'
        WHEN face_count BETWEEN 5 AND 9 THEN '5-9'
        WHEN face_count BETWEEN 10 AND 14 THEN '10-14'
        WHEN face_count BETWEEN 15 AND 19 THEN '15-19'
        WHEN face_count >= 20 THEN '20+'
        ELSE 'other'
      END as range,
      COUNT(*) as count
    FROM face_counts
    WHERE face_count >= 3
    GROUP BY range
  )
  SELECT
    (SELECT COUNT(DISTINCT person_id) FROM photo_faces WHERE verified = true AND person_id IS NOT NULL)::bigint as total_people,
    (SELECT COUNT(*) FROM photo_faces WHERE verified = true AND person_id IS NOT NULL)::bigint as total_verified_faces,
    (SELECT COUNT(*) FROM face_counts WHERE face_count >= 3)::bigint as people_with_min_faces,
    (SELECT ROUND(AVG(face_count), 1) FROM face_counts)::numeric as avg_faces_per_person,
    (SELECT jsonb_object_agg(range, count) FROM count_distribution) as faces_by_count;
END;
$$ LANGUAGE plpgsql;
