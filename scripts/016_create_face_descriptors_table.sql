-- Drop existing table to recreate with correct schema
DROP TABLE IF EXISTS face_descriptors CASCADE;

-- Create face_descriptors table for storing face embeddings
-- Note: This script depends on people table from script 015

CREATE TABLE face_descriptors (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  person_id UUID NOT NULL REFERENCES people(id) ON DELETE CASCADE,
  descriptor JSONB NOT NULL,
  source_image_id UUID REFERENCES gallery_images(id) ON DELETE CASCADE,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for faster queries
CREATE INDEX idx_face_descriptors_person_id ON face_descriptors(person_id);
CREATE INDEX idx_face_descriptors_source_image_id ON face_descriptors(source_image_id);
CREATE INDEX idx_face_descriptors_created_at ON face_descriptors(created_at DESC);

-- Enable RLS
ALTER TABLE face_descriptors ENABLE ROW LEVEL SECURITY;

-- RLS Policies - Anyone can read descriptors
CREATE POLICY "Anyone can read face descriptors"
  ON face_descriptors FOR SELECT
  USING (true);

-- Authenticated users (admin) can manage descriptors
CREATE POLICY "Authenticated users can insert face descriptors"
  ON face_descriptors FOR INSERT
  TO authenticated
  WITH CHECK (true);

CREATE POLICY "Authenticated users can update face descriptors"
  ON face_descriptors FOR UPDATE
  TO authenticated
  USING (true);

CREATE POLICY "Authenticated users can delete face descriptors"
  ON face_descriptors FOR DELETE
  TO authenticated
  USING (true);

-- Add comments
COMMENT ON TABLE face_descriptors IS 'Face embeddings (128-dimensional vectors) for face recognition';
COMMENT ON COLUMN face_descriptors.descriptor IS 'Face-api.js 128-dimensional face descriptor stored as JSON array';
COMMENT ON COLUMN face_descriptors.source_image_id IS 'Reference to the gallery image where this face was detected';
