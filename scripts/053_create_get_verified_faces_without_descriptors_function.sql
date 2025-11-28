-- Function to get all verified faces without descriptors
CREATE OR REPLACE FUNCTION get_verified_faces_without_descriptors()
RETURNS TABLE (
  photo_face_id UUID,
  photo_id UUID,
  person_id UUID,
  person_name TEXT,
  image_url TEXT,
  bounding_box JSONB,
  verified BOOLEAN
) AS $$
BEGIN
  RETURN QUERY
  SELECT 
    pf.id AS photo_face_id,
    pf.photo_id,
    pf.person_id,
    p.real_name AS person_name,
    gi.image_url,
    pf.insightface_bbox AS bounding_box,
    pf.verified
  FROM photo_faces pf
  JOIN people p ON pf.person_id = p.id
  JOIN gallery_images gi ON pf.photo_id = gi.id
  LEFT JOIN face_descriptors fd ON fd.person_id = pf.person_id 
    AND fd.source_image_id = pf.photo_id
  WHERE pf.verified = true
    AND fd.id IS NULL
  ORDER BY p.real_name, gi.image_url;
END;
$$ LANGUAGE plpgsql;
