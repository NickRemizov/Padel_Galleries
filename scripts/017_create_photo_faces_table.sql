-- Drop existing table and related objects to ensure clean creation
-- Note: This script depends on people table (015) and gallery_images table (004)
DROP TABLE IF EXISTS photo_faces CASCADE;
DROP FUNCTION IF EXISTS update_photo_faces_updated_at() CASCADE;

-- Create photo_faces table for detected faces in photos
CREATE TABLE photo_faces (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  photo_id UUID NOT NULL REFERENCES gallery_images(id) ON DELETE CASCADE,
  person_id UUID REFERENCES people(id) ON DELETE SET NULL,
  bounding_box JSONB NOT NULL,
  confidence FLOAT8,
  verified BOOLEAN DEFAULT false,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for faster queries
CREATE INDEX idx_photo_faces_photo_id ON photo_faces(photo_id);
CREATE INDEX idx_photo_faces_person_id ON photo_faces(person_id);
CREATE INDEX idx_photo_faces_verified ON photo_faces(verified);

-- Enable RLS
ALTER TABLE photo_faces ENABLE ROW LEVEL SECURITY;

-- RLS Policies - Anyone can read photo faces
CREATE POLICY "Anyone can read photo faces"
  ON photo_faces FOR SELECT
  USING (true);

-- Authenticated users (admin) can manage photo faces
CREATE POLICY "Authenticated users can insert photo faces"
  ON photo_faces FOR INSERT
  TO authenticated
  WITH CHECK (true);

CREATE POLICY "Authenticated users can update photo faces"
  ON photo_faces FOR UPDATE
  TO authenticated
  USING (true);

CREATE POLICY "Authenticated users can delete photo faces"
  ON photo_faces FOR DELETE
  TO authenticated
  USING (true);

-- Create trigger to update updated_at
CREATE FUNCTION update_photo_faces_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER photo_faces_updated_at
  BEFORE UPDATE ON photo_faces
  FOR EACH ROW
  EXECUTE FUNCTION update_photo_faces_updated_at();

-- Add comments
COMMENT ON TABLE photo_faces IS 'Detected and tagged faces in gallery photos';
COMMENT ON COLUMN photo_faces.bounding_box IS 'Face bounding box: {"x": 0, "y": 0, "width": 100, "height": 100}';
COMMENT ON COLUMN photo_faces.confidence IS 'Face recognition confidence score (0-1)';
COMMENT ON COLUMN photo_faces.verified IS 'Whether the face tag has been verified by admin';
