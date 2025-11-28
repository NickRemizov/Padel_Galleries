-- Migration: Make bounding_box nullable (legacy field from Face API)
-- Date: 2025-10-27
-- Description: Remove NOT NULL constraint from bounding_box field as it's legacy from Face API
--              and should not be used with InsightFace. Use insightface_bbox instead.

-- Make bounding_box nullable
ALTER TABLE photo_faces 
ALTER COLUMN bounding_box DROP NOT NULL;

-- Add comment explaining the field is legacy
COMMENT ON COLUMN photo_faces.bounding_box IS 'Legacy field from Azure Face API. Use insightface_bbox for InsightFace detections. Nullable for backward compatibility.';

-- Verify the change
SELECT 
    column_name,
    data_type,
    is_nullable,
    column_default
FROM information_schema.columns
WHERE table_name = 'photo_faces' 
AND column_name IN ('bounding_box', 'insightface_bbox')
ORDER BY column_name;
