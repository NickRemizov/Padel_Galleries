-- Create table for rejected (not interesting) faces
CREATE TABLE IF NOT EXISTS rejected_faces (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  descriptor VECTOR(512) NOT NULL,
  gallery_id UUID REFERENCES galleries(id) ON DELETE CASCADE,
  photo_id UUID,
  rejected_by UUID REFERENCES auth.users(id),
  rejected_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  reason TEXT,
  
  -- Index for fast similarity search
  CONSTRAINT rejected_faces_descriptor_check CHECK (vector_dims(descriptor) = 512)
);

-- Create index for vector similarity search
CREATE INDEX IF NOT EXISTS rejected_faces_descriptor_idx 
ON rejected_faces 
USING ivfflat (descriptor vector_cosine_ops)
WITH (lists = 100);

-- Create index for gallery lookups
CREATE INDEX IF NOT EXISTS rejected_faces_gallery_idx 
ON rejected_faces(gallery_id);

COMMENT ON TABLE rejected_faces IS 'Stores face descriptors that admin marked as not interesting - these should not be shown in future recognition';
COMMENT ON COLUMN rejected_faces.descriptor IS '512-dimensional face embedding from InsightFace';
COMMENT ON COLUMN rejected_faces.reason IS 'Optional reason why face was rejected (e.g., "not a player", "spectator", etc.)';
