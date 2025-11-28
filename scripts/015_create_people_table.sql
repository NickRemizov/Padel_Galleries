-- Drop existing table and related objects to ensure clean creation
DROP TABLE IF EXISTS people CASCADE;
DROP FUNCTION IF EXISTS update_people_updated_at() CASCADE;

-- Create people table for face recognition
CREATE TABLE people (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  real_name TEXT NOT NULL,
  telegram_name TEXT,
  telegram_nickname TEXT,
  telegram_profile_url TEXT,
  facebook_profile_url TEXT,
  instagram_profile_url TEXT,
  paddle_ranking INTEGER,
  tournament_results JSONB DEFAULT '[]'::jsonb,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for faster queries
CREATE INDEX idx_people_real_name ON people(real_name);
CREATE INDEX idx_people_telegram_nickname ON people(telegram_nickname);
CREATE INDEX idx_people_paddle_ranking ON people(paddle_ranking);

-- Enable RLS
ALTER TABLE people ENABLE ROW LEVEL SECURITY;

-- RLS Policies - Anyone can read people
CREATE POLICY "Anyone can read people"
  ON people FOR SELECT
  USING (true);

-- Authenticated users (admin) can manage people
CREATE POLICY "Authenticated users can insert people"
  ON people FOR INSERT
  TO authenticated
  WITH CHECK (true);

CREATE POLICY "Authenticated users can update people"
  ON people FOR UPDATE
  TO authenticated
  USING (true);

CREATE POLICY "Authenticated users can delete people"
  ON people FOR DELETE
  TO authenticated
  USING (true);

-- Create trigger to update updated_at
CREATE FUNCTION update_people_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER people_updated_at
  BEFORE UPDATE ON people
  FOR EACH ROW
  EXECUTE FUNCTION update_people_updated_at();

-- Add comments
COMMENT ON TABLE people IS 'People database for face recognition with paddle rankings and social profiles';
COMMENT ON COLUMN people.tournament_results IS 'JSON array of tournament results: [{"tournament": "Tournament Name", "place": 1, "date": "2024-01-01"}]';
