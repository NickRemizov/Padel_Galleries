-- Fix NULL recognition_confidence for verified faces
-- These should have been set to 1.0 when manually verified

UPDATE photo_faces
SET recognition_confidence = 1.0
WHERE verified = true 
  AND recognition_confidence IS NULL;

-- Log the changes
DO $$
DECLARE
  updated_count INTEGER;
BEGIN
  GET DIAGNOSTICS updated_count = ROW_COUNT;
  RAISE NOTICE 'Updated % verified faces with NULL recognition_confidence to 1.0', updated_count;
END $$;
