-- Migration script for creating complete database schema on new PostgreSQL server
-- Excludes legacy Face-API fields (bounding_box, descriptor, confidence)
-- Date: 2025-11-18

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create ENUM types
DO $$ BEGIN
    CREATE TYPE person_category AS ENUM ('player', 'organizer', 'staff', 'other');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE face_category AS ENUM ('unknown', 'player', 'organizer', 'staff', 'other', 'rejected');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- ===================================
-- PHOTOGRAPHERS
-- ===================================
CREATE TABLE IF NOT EXISTS photographers (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_photographers_name ON photographers(name);

-- ===================================
-- LOCATIONS
-- ===================================
CREATE TABLE IF NOT EXISTS locations (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_locations_name ON locations(name);

-- ===================================
-- ORGANIZERS
-- ===================================
CREATE TABLE IF NOT EXISTS organizers (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_organizers_name ON organizers(name);

-- ===================================
-- GALLERIES
-- ===================================
CREATE TABLE IF NOT EXISTS galleries (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  title TEXT NOT NULL,
  shoot_date DATE NOT NULL,
  gallery_url TEXT NOT NULL,
  cover_image_url TEXT,
  cover_image_square_url TEXT,
  external_gallery_url TEXT,
  sort_order TEXT DEFAULT 'filename',
  photographer_id UUID REFERENCES photographers(id) ON DELETE SET NULL,
  location_id UUID REFERENCES locations(id) ON DELETE SET NULL,
  organizer_id UUID REFERENCES organizers(id) ON DELETE SET NULL,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_galleries_shoot_date ON galleries(shoot_date DESC);
CREATE INDEX IF NOT EXISTS idx_galleries_photographer ON galleries(photographer_id);
CREATE INDEX IF NOT EXISTS idx_galleries_location ON galleries(location_id);
CREATE INDEX IF NOT EXISTS idx_galleries_organizer ON galleries(organizer_id);

-- ===================================
-- GALLERY IMAGES
-- ===================================
CREATE TABLE IF NOT EXISTS gallery_images (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  gallery_id UUID REFERENCES galleries(id) ON DELETE CASCADE NOT NULL,
  image_url TEXT NOT NULL,
  original_url TEXT,
  original_filename TEXT,
  file_size INTEGER,
  width INTEGER,
  height INTEGER,
  display_order INTEGER,
  download_count INTEGER DEFAULT 0,
  has_been_processed BOOLEAN DEFAULT FALSE,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_gallery_images_gallery ON gallery_images(gallery_id);
CREATE INDEX IF NOT EXISTS idx_gallery_images_display_order ON gallery_images(gallery_id, display_order);
CREATE INDEX IF NOT EXISTS idx_gallery_images_processed ON gallery_images(has_been_processed);

-- ===================================
-- PEOPLE
-- ===================================
CREATE TABLE IF NOT EXISTS people (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  real_name TEXT,
  telegram_name TEXT,
  telegram_nickname TEXT,
  telegram_profile_url TEXT,
  instagram_profile_url TEXT,
  facebook_profile_url TEXT,
  avatar_url TEXT,
  category person_category DEFAULT 'player',
  paddle_ranking INTEGER,
  tournament_results JSONB,
  show_photos_in_galleries BOOLEAN DEFAULT TRUE,
  show_in_players_gallery BOOLEAN DEFAULT TRUE,
  use_custom_confidence BOOLEAN DEFAULT FALSE,
  custom_confidence_threshold DOUBLE PRECISION,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_people_real_name ON people(real_name);
CREATE INDEX IF NOT EXISTS idx_people_telegram_name ON people(telegram_name);
CREATE INDEX IF NOT EXISTS idx_people_category ON people(category);

-- ===================================
-- PHOTO FACES (without legacy Face-API fields)
-- ===================================
CREATE TABLE IF NOT EXISTS photo_faces (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  photo_id UUID REFERENCES gallery_images(id) ON DELETE CASCADE NOT NULL,
  person_id UUID REFERENCES people(id) ON DELETE SET NULL,
  face_category face_category DEFAULT 'unknown',
  -- InsightFace fields (current)
  insightface_descriptor vector(512),
  insightface_confidence DOUBLE PRECISION,
  insightface_bbox JSONB,
  insightface_det_score DOUBLE PRECISION,
  -- Recognition and training
  recognition_confidence DOUBLE PRECISION,
  training_used BOOLEAN DEFAULT FALSE,
  training_context JSONB,
  -- Quality metrics
  blur_score DOUBLE PRECISION,
  -- Verification
  verified BOOLEAN DEFAULT FALSE,
  verified_at TIMESTAMP WITH TIME ZONE,
  verified_by UUID,
  -- Timestamps
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_photo_faces_photo ON photo_faces(photo_id);
CREATE INDEX IF NOT EXISTS idx_photo_faces_person ON photo_faces(person_id);
CREATE INDEX IF NOT EXISTS idx_photo_faces_category ON photo_faces(face_category);
CREATE INDEX IF NOT EXISTS idx_photo_faces_verified ON photo_faces(verified);
CREATE INDEX IF NOT EXISTS idx_photo_faces_training ON photo_faces(training_used);

-- Create HNSW index for vector similarity search
CREATE INDEX IF NOT EXISTS idx_photo_faces_descriptor_hnsw 
ON photo_faces USING hnsw (insightface_descriptor vector_cosine_ops)
WHERE insightface_descriptor IS NOT NULL;

-- ===================================
-- FACE DESCRIPTORS
-- ===================================
CREATE TABLE IF NOT EXISTS face_descriptors (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  person_id UUID REFERENCES people(id) ON DELETE CASCADE NOT NULL,
  source_image_id UUID REFERENCES gallery_images(id) ON DELETE SET NULL,
  descriptor JSONB NOT NULL,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_face_descriptors_person ON face_descriptors(person_id);

-- ===================================
-- FACE TRAINING SESSIONS
-- ===================================
CREATE TABLE IF NOT EXISTS face_training_sessions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  status TEXT DEFAULT 'pending',
  training_mode TEXT DEFAULT 'full',
  model_version TEXT,
  faces_count INTEGER DEFAULT 0,
  people_count INTEGER DEFAULT 0,
  min_faces_per_person INTEGER DEFAULT 3,
  context_weight DOUBLE PRECISION DEFAULT 0.3,
  metrics JSONB,
  error_message TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  completed_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX IF NOT EXISTS idx_training_sessions_status ON face_training_sessions(status);
CREATE INDEX IF NOT EXISTS idx_training_sessions_created ON face_training_sessions(created_at DESC);

-- ===================================
-- FACE RECOGNITION CONFIG
-- ===================================
CREATE TABLE IF NOT EXISTS face_recognition_config (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  key TEXT UNIQUE NOT NULL,
  value JSONB NOT NULL,
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_face_config_key ON face_recognition_config(key);

-- ===================================
-- REJECTED FACES
-- ===================================
CREATE TABLE IF NOT EXISTS rejected_faces (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  photo_id UUID REFERENCES gallery_images(id) ON DELETE CASCADE,
  gallery_id UUID REFERENCES galleries(id) ON DELETE CASCADE,
  descriptor vector(512),
  reason TEXT,
  rejected_by UUID,
  rejected_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_rejected_faces_photo ON rejected_faces(photo_id);
CREATE INDEX IF NOT EXISTS idx_rejected_faces_gallery ON rejected_faces(gallery_id);

-- ===================================
-- GALLERY CO-OCCURRENCE
-- ===================================
CREATE TABLE IF NOT EXISTS gallery_co_occurrence (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  person_id_1 UUID REFERENCES people(id) ON DELETE CASCADE NOT NULL,
  person_id_2 UUID REFERENCES people(id) ON DELETE CASCADE NOT NULL,
  gallery_id UUID REFERENCES galleries(id) ON DELETE CASCADE NOT NULL,
  co_occurrence_count INTEGER DEFAULT 1,
  last_seen_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  UNIQUE(person_id_1, person_id_2, gallery_id)
);

CREATE INDEX IF NOT EXISTS idx_co_occurrence_person1 ON gallery_co_occurrence(person_id_1);
CREATE INDEX IF NOT EXISTS idx_co_occurrence_person2 ON gallery_co_occurrence(person_id_2);
CREATE INDEX IF NOT EXISTS idx_co_occurrence_gallery ON gallery_co_occurrence(gallery_id);

-- ===================================
-- TOURNAMENT RESULTS
-- ===================================
CREATE TABLE IF NOT EXISTS tournament_results (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  gallery_id UUID REFERENCES galleries(id) ON DELETE CASCADE NOT NULL,
  person_id UUID REFERENCES people(id) ON DELETE CASCADE NOT NULL,
  place INTEGER NOT NULL,
  notes TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  UNIQUE(gallery_id, person_id)
);

CREATE INDEX IF NOT EXISTS idx_tournament_results_gallery ON tournament_results(gallery_id);
CREATE INDEX IF NOT EXISTS idx_tournament_results_person ON tournament_results(person_id);

-- ===================================
-- USERS (Telegram users)
-- ===================================
CREATE TABLE IF NOT EXISTS users (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  telegram_id BIGINT UNIQUE NOT NULL,
  username TEXT,
  first_name TEXT,
  last_name TEXT,
  photo_url TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_users_telegram ON users(telegram_id);

-- ===================================
-- LIKES
-- ===================================
CREATE TABLE IF NOT EXISTS likes (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES users(id) ON DELETE CASCADE NOT NULL,
  image_id UUID REFERENCES gallery_images(id) ON DELETE CASCADE NOT NULL,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  UNIQUE(user_id, image_id)
);

CREATE INDEX IF NOT EXISTS idx_likes_user ON likes(user_id);
CREATE INDEX IF NOT EXISTS idx_likes_image ON likes(image_id);

-- ===================================
-- FAVORITES
-- ===================================
CREATE TABLE IF NOT EXISTS favorites (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES users(id) ON DELETE CASCADE NOT NULL,
  gallery_image_id UUID REFERENCES gallery_images(id) ON DELETE CASCADE NOT NULL,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  UNIQUE(user_id, gallery_image_id)
);

CREATE INDEX IF NOT EXISTS idx_favorites_user ON favorites(user_id);
CREATE INDEX IF NOT EXISTS idx_favorites_image ON favorites(gallery_image_id);

-- ===================================
-- COMMENTS
-- ===================================
CREATE TABLE IF NOT EXISTS comments (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES users(id) ON DELETE CASCADE NOT NULL,
  gallery_image_id UUID REFERENCES gallery_images(id) ON DELETE CASCADE NOT NULL,
  content TEXT NOT NULL,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_comments_user ON comments(user_id);
CREATE INDEX IF NOT EXISTS idx_comments_image ON comments(gallery_image_id);
CREATE INDEX IF NOT EXISTS idx_comments_created ON comments(created_at DESC);

-- ===================================
-- RLS POLICIES (PUBLIC ACCESS)
-- ===================================
-- Note: These policies are placeholders since we're migrating away from Supabase Auth
-- You'll need to implement authentication via NextAuth.js

ALTER TABLE photographers ENABLE ROW LEVEL SECURITY;
ALTER TABLE locations ENABLE ROW LEVEL SECURITY;
ALTER TABLE organizers ENABLE ROW LEVEL SECURITY;
ALTER TABLE galleries ENABLE ROW LEVEL SECURITY;
ALTER TABLE people ENABLE ROW LEVEL SECURITY;
ALTER TABLE photo_faces ENABLE ROW LEVEL SECURITY;
ALTER TABLE face_descriptors ENABLE ROW LEVEL SECURITY;
ALTER TABLE face_training_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE face_recognition_config ENABLE ROW LEVEL SECURITY;
ALTER TABLE likes ENABLE ROW LEVEL SECURITY;
ALTER TABLE favorites ENABLE ROW LEVEL SECURITY;
ALTER TABLE comments ENABLE ROW LEVEL SECURITY;

-- Public read policies (everyone can read)
CREATE POLICY "Public read photographers" ON photographers FOR SELECT USING (true);
CREATE POLICY "Public read locations" ON locations FOR SELECT USING (true);
CREATE POLICY "Public read organizers" ON organizers FOR SELECT USING (true);
CREATE POLICY "Public read galleries" ON galleries FOR SELECT USING (true);
CREATE POLICY "Public read people" ON people FOR SELECT USING (true);
CREATE POLICY "Public read photo_faces" ON photo_faces FOR SELECT USING (true);
CREATE POLICY "Public read face_descriptors" ON face_descriptors FOR SELECT USING (true);
CREATE POLICY "Public read training_sessions" ON face_training_sessions FOR SELECT USING (true);
CREATE POLICY "Public read face_config" ON face_recognition_config FOR SELECT USING (true);
CREATE POLICY "Public read likes" ON likes FOR SELECT USING (true);
CREATE POLICY "Public read comments" ON comments FOR SELECT USING (true);

-- Admin write policies (authenticated users via application logic)
-- These will be enforced in your Next.js middleware/API routes
CREATE POLICY "Admin manage photographers" ON photographers FOR ALL USING (true);
CREATE POLICY "Admin manage locations" ON locations FOR ALL USING (true);
CREATE POLICY "Admin manage organizers" ON organizers FOR ALL USING (true);
CREATE POLICY "Admin manage galleries" ON galleries FOR ALL USING (true);
CREATE POLICY "Admin manage people" ON people FOR ALL USING (true);
CREATE POLICY "Admin manage photo_faces" ON photo_faces FOR ALL USING (true);
CREATE POLICY "Admin manage face_descriptors" ON face_descriptors FOR ALL USING (true);
CREATE POLICY "Admin manage training_sessions" ON face_training_sessions FOR ALL USING (true);
CREATE POLICY "Admin manage face_config" ON face_recognition_config FOR ALL USING (true);

-- User-specific policies
CREATE POLICY "Users manage own likes" ON likes FOR ALL USING (true);
CREATE POLICY "Users manage own favorites" ON favorites FOR ALL USING (true);
CREATE POLICY "Users manage own comments" ON comments FOR ALL USING (true);

-- ===================================
-- GRANT PERMISSIONS
-- ===================================
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO galeries_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO galeries_user;

-- Complete!
