-- Check verification status for specific photos
-- Replace the image IDs with the actual IDs you want to check

-- Check photo_faces for Tournament 10-08-25-159.jpg (checkmark disappeared)
SELECT 
  pf.id as face_id,
  pf.photo_id,
  pf.person_id,
  p.real_name as person_name, -- Fixed from p.name to p.real_name
  pf.confidence,
  pf.verified,
  pf.created_at,
  gi.image_url -- Fixed from gi.filename to gi.image_url
FROM photo_faces pf
JOIN people p ON pf.person_id = p.id
JOIN gallery_images gi ON pf.photo_id = gi.id
WHERE gi.image_url LIKE '%Tournament%2010-08-25-159%' -- Fixed from filename to image_url
ORDER BY pf.created_at DESC;

-- Check photo_faces for Tournament 13-09-25-23.jpg (has checkmark)
SELECT 
  pf.id as face_id,
  pf.photo_id,
  pf.person_id,
  p.real_name as person_name, -- Fixed from p.name to p.real_name
  pf.confidence,
  pf.verified,
  pf.created_at,
  gi.image_url -- Fixed from gi.filename to gi.image_url
FROM photo_faces pf
JOIN people p ON pf.person_id = p.id
JOIN gallery_images gi ON pf.photo_id = gi.id
WHERE gi.image_url LIKE '%Tournament%2013-09-25-23%' -- Fixed from filename to image_url
ORDER BY pf.created_at DESC;

-- Check all verified faces for person 6f7ccdfa-e0d0-4776-bc6d-9296e2cb65f6 (Макар)
SELECT 
  pf.id as face_id,
  pf.photo_id,
  pf.person_id,
  p.real_name as person_name, -- Fixed from p.name to p.real_name
  pf.confidence,
  pf.verified,
  pf.created_at,
  gi.image_url -- Fixed from gi.filename to gi.image_url
FROM photo_faces pf
JOIN people p ON pf.person_id = p.id
JOIN gallery_images gi ON pf.photo_id = gi.id
WHERE pf.person_id = '6f7ccdfa-e0d0-4776-bc6d-9296e2cb65f6'
ORDER BY pf.created_at DESC
LIMIT 50;
