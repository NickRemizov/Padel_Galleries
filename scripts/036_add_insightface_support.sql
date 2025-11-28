-- Add InsightFace support to face recognition system
-- This migration adds new fields for InsightFace descriptors and creates tables for training management

-- Enable pgvector extension first (required for vector type)
CREATE EXTENSION IF NOT EXISTS vector;

-- 1. Add InsightFace fields to photo_faces table
ALTER TABLE photo_faces 
ADD COLUMN IF NOT EXISTS insightface_descriptor vector(512),
ADD COLUMN IF NOT EXISTS insightface_confidence FLOAT8,
ADD COLUMN IF NOT EXISTS insightface_bbox JSONB,
ADD COLUMN IF NOT EXISTS training_used BOOLEAN DEFAULT false,
ADD COLUMN IF NOT EXISTS training_context JSONB;

-- Create indexes for InsightFace fields
CREATE INDEX IF NOT EXISTS idx_photo_faces_insightface_confidence ON photo_faces(insightface_confidence);
CREATE INDEX IF NOT EXISTS idx_photo_faces_training_used ON photo_faces(training_used);

-- Add comments for new fields
COMMENT ON COLUMN photo_faces.insightface_descriptor IS 'InsightFace 512-dimensional face descriptor (ArcFace embedding)';
COMMENT ON COLUMN photo_faces.insightface_confidence IS 'InsightFace recognition confidence score (0-1)';
COMMENT ON COLUMN photo_faces.insightface_bbox IS 'InsightFace bounding box: {"x": 0, "y": 0, "width": 100, "height": 100}';
COMMENT ON COLUMN photo_faces.training_used IS 'Whether this face was used for model training';
COMMENT ON COLUMN photo_faces.training_context IS 'Training context metadata: {"event_id": "...", "co_occurring_people": [...], "training_session_id": "..."}';

-- 2. Create face_training_sessions table
CREATE TABLE IF NOT EXISTS face_training_sessions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at TIMESTAMPTZ DEFAULT NOW(),
  completed_at TIMESTAMPTZ,
  model_version TEXT NOT NULL,
  training_mode TEXT NOT NULL CHECK (training_mode IN ('full', 'incremental')),
  faces_count INTEGER NOT NULL,
  people_count INTEGER NOT NULL,
  context_weight FLOAT8 DEFAULT 0.1,
  min_faces_per_person INTEGER DEFAULT 3,
  metrics JSONB,
  status TEXT NOT NULL DEFAULT 'running' CHECK (status IN ('running', 'completed', 'failed')),
  error_message TEXT
);

-- Create indexes for training sessions
CREATE INDEX IF NOT EXISTS idx_training_sessions_created_at ON face_training_sessions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_training_sessions_status ON face_training_sessions(status);
CREATE INDEX IF NOT EXISTS idx_training_sessions_model_version ON face_training_sessions(model_version);

-- Enable RLS for training sessions
ALTER TABLE face_training_sessions ENABLE ROW LEVEL SECURITY;

-- RLS Policies - Anyone can read training sessions
CREATE POLICY "Anyone can read training sessions"
  ON face_training_sessions FOR SELECT
  USING (true);

-- Authenticated users (admin) can manage training sessions
CREATE POLICY "Authenticated users can insert training sessions"
  ON face_training_sessions FOR INSERT
  TO authenticated
  WITH CHECK (true);

CREATE POLICY "Authenticated users can update training sessions"
  ON face_training_sessions FOR UPDATE
  TO authenticated
  USING (true);

-- Add comments for training sessions table
COMMENT ON TABLE face_training_sessions IS 'History of face recognition model training sessions';
COMMENT ON COLUMN face_training_sessions.model_version IS 'Model version identifier (e.g., "v1.0", "v1.1")';
COMMENT ON COLUMN face_training_sessions.training_mode IS 'Training mode: "full" (complete retrain) or "incremental" (add new faces)';
COMMENT ON COLUMN face_training_sessions.context_weight IS 'Weight for context-aware recognition (0.0-0.5)';
COMMENT ON COLUMN face_training_sessions.metrics IS 'Training metrics: {"accuracy": 0.95, "precision": 0.93, "recall": 0.97, "problematic_people": [...]}';

-- 3. Create face_recognition_config table
CREATE TABLE IF NOT EXISTS face_recognition_config (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  key TEXT UNIQUE NOT NULL,
  value JSONB NOT NULL,
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create index for config lookups
CREATE INDEX IF NOT EXISTS idx_face_recognition_config_key ON face_recognition_config(key);

-- Enable RLS for config
ALTER TABLE face_recognition_config ENABLE ROW LEVEL SECURITY;

-- RLS Policies - Anyone can read config
CREATE POLICY "Anyone can read face recognition config"
  ON face_recognition_config FOR SELECT
  USING (true);

-- Authenticated users (admin) can manage config
CREATE POLICY "Authenticated users can insert config"
  ON face_recognition_config FOR INSERT
  TO authenticated
  WITH CHECK (true);

CREATE POLICY "Authenticated users can update config"
  ON face_recognition_config FOR UPDATE
  TO authenticated
  USING (true);

-- Add comments for config table
COMMENT ON TABLE face_recognition_config IS 'Configuration parameters for face recognition system';
COMMENT ON COLUMN face_recognition_config.key IS 'Configuration key (e.g., "confidence_thresholds", "context_weight")';
COMMENT ON COLUMN face_recognition_config.value IS 'Configuration value as JSON';

-- 4. Insert default configuration values
INSERT INTO face_recognition_config (key, value) VALUES
  ('confidence_thresholds', '{"low_data": 0.75, "medium_data": 0.65, "high_data": 0.55}'::jsonb),
  ('context_weight', '0.1'::jsonb),
  ('min_faces_per_person', '3'::jsonb),
  ('auto_retrain_threshold', '25'::jsonb),
  ('auto_retrain_percentage', '0.1'::jsonb),
  ('model_version', '"v1.0"'::jsonb),
  ('last_full_training', 'null'::jsonb),
  ('faces_since_last_training', '0'::jsonb)
ON CONFLICT (key) DO NOTHING;

-- 5. Create function to update config timestamp
CREATE OR REPLACE FUNCTION update_config_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER config_updated_at
  BEFORE UPDATE ON face_recognition_config
  FOR EACH ROW
  EXECUTE FUNCTION update_config_updated_at();

-- 6. Add per-person confidence override to people table
ALTER TABLE people
ADD COLUMN IF NOT EXISTS custom_confidence_threshold FLOAT8,
ADD COLUMN IF NOT EXISTS use_custom_confidence BOOLEAN DEFAULT false;

-- Create index for custom confidence
CREATE INDEX IF NOT EXISTS idx_people_use_custom_confidence ON people(use_custom_confidence);

-- Add comments
COMMENT ON COLUMN people.custom_confidence_threshold IS 'Custom confidence threshold for this person (overrides global settings)';
COMMENT ON COLUMN people.use_custom_confidence IS 'Whether to use custom confidence threshold instead of global settings';

-- Output summary
DO $$
DECLARE
  total_faces INTEGER;
  verified_faces INTEGER;
BEGIN
  SELECT COUNT(*) INTO total_faces FROM photo_faces;
  SELECT COUNT(*) INTO verified_faces FROM photo_faces WHERE verified = true;
  
  RAISE NOTICE '=== InsightFace Migration Complete ===';
  RAISE NOTICE 'Total photo_faces records: %', total_faces;
  RAISE NOTICE 'Verified faces ready for training: %', verified_faces;
  RAISE NOTICE 'New fields added to photo_faces: insightface_descriptor, insightface_confidence, insightface_bbox, training_used, training_context';
  RAISE NOTICE 'New tables created: face_training_sessions, face_recognition_config';
  RAISE NOTICE 'Default configuration inserted with min_faces_per_person=3';
END $$;
