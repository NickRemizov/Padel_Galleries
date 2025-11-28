-- Add missing fields for recognition confidence and verification tracking

ALTER TABLE photo_faces
ADD COLUMN IF NOT EXISTS recognition_confidence DOUBLE PRECISION,
ADD COLUMN IF NOT EXISTS verified_at TIMESTAMP WITH TIME ZONE,
ADD COLUMN IF NOT EXISTS verified_by UUID;

-- Add comment for clarity
COMMENT ON COLUMN photo_faces.confidence IS 'Detection confidence from face-api.js or InsightFace';
COMMENT ON COLUMN photo_faces.recognition_confidence IS 'Classification confidence from HNSWLIB/SVM (0-1)';
COMMENT ON COLUMN photo_faces.verified_at IS 'Timestamp when manually verified by admin';
COMMENT ON COLUMN photo_faces.verified_by IS 'Admin user ID who verified this face';

-- Create index for faster queries
CREATE INDEX IF NOT EXISTS idx_photo_faces_verified ON photo_faces(verified) WHERE verified = true;
CREATE INDEX IF NOT EXISTS idx_photo_faces_recognition_confidence ON photo_faces(recognition_confidence);
