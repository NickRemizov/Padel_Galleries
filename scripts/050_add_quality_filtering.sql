-- Migration 050: Add quality filtering parameters
-- Adds min_detection_score, min_face_size, min_blur_score to face_recognition_config
-- Adds blur_score and insightface_det_score fields to photo_faces for quality filtering

BEGIN;

-- Add quality filtering parameters to config
INSERT INTO face_recognition_config (key, value) VALUES
('quality_filters', '{
  "min_detection_score": 0.7,
  "min_face_size": 80,
  "min_blur_score": 100.0
}'::jsonb)
ON CONFLICT (key) DO UPDATE
SET value = EXCLUDED.value,
    updated_at = NOW();

-- Add both blur_score and insightface_det_score columns
ALTER TABLE photo_faces 
ADD COLUMN IF NOT EXISTS blur_score FLOAT,
ADD COLUMN IF NOT EXISTS insightface_det_score FLOAT;

COMMENT ON COLUMN photo_faces.blur_score IS 'Laplacian variance for blur detection (higher = sharper, typical range 0-500)';
COMMENT ON COLUMN photo_faces.insightface_det_score IS 'InsightFace detection score (0-1, higher = better quality detection)';

-- Create index for filtering by quality
CREATE INDEX IF NOT EXISTS idx_photo_faces_quality 
ON photo_faces(blur_score, insightface_det_score) 
WHERE blur_score IS NOT NULL AND insightface_det_score IS NOT NULL;

-- Log completion
DO $$
BEGIN
  RAISE NOTICE 'Quality filtering parameters added to face_recognition_config';
  RAISE NOTICE 'blur_score and insightface_det_score columns added to photo_faces';
END $$;

COMMIT;
