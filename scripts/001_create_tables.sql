-- Create photographers table
CREATE TABLE IF NOT EXISTS photographers (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create locations table
CREATE TABLE IF NOT EXISTS locations (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create galleries table
CREATE TABLE IF NOT EXISTS galleries (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  title TEXT NOT NULL,
  shoot_date DATE NOT NULL,
  gallery_url TEXT NOT NULL,
  cover_image_url TEXT NOT NULL,
  photographer_id UUID REFERENCES photographers(id) ON DELETE SET NULL,
  location_id UUID REFERENCES locations(id) ON DELETE SET NULL,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index for faster queries
CREATE INDEX IF NOT EXISTS idx_galleries_shoot_date ON galleries(shoot_date DESC);
CREATE INDEX IF NOT EXISTS idx_galleries_photographer ON galleries(photographer_id);
CREATE INDEX IF NOT EXISTS idx_galleries_location ON galleries(location_id);

-- Enable Row Level Security
ALTER TABLE photographers ENABLE ROW LEVEL SECURITY;
ALTER TABLE locations ENABLE ROW LEVEL SECURITY;
ALTER TABLE galleries ENABLE ROW LEVEL SECURITY;

-- Create policies for public read access
CREATE POLICY "Allow public read access to photographers"
  ON photographers FOR SELECT
  USING (true);

CREATE POLICY "Allow public read access to locations"
  ON locations FOR SELECT
  USING (true);

CREATE POLICY "Allow public read access to galleries"
  ON galleries FOR SELECT
  USING (true);

-- Create policies for authenticated users (admin) to manage data
CREATE POLICY "Allow authenticated users to insert photographers"
  ON photographers FOR INSERT
  TO authenticated
  WITH CHECK (true);

CREATE POLICY "Allow authenticated users to update photographers"
  ON photographers FOR UPDATE
  TO authenticated
  USING (true);

CREATE POLICY "Allow authenticated users to delete photographers"
  ON photographers FOR DELETE
  TO authenticated
  USING (true);

CREATE POLICY "Allow authenticated users to insert locations"
  ON locations FOR INSERT
  TO authenticated
  WITH CHECK (true);

CREATE POLICY "Allow authenticated users to update locations"
  ON locations FOR UPDATE
  TO authenticated
  USING (true);

CREATE POLICY "Allow authenticated users to delete locations"
  ON locations FOR DELETE
  TO authenticated
  USING (true);

CREATE POLICY "Allow authenticated users to insert galleries"
  ON galleries FOR INSERT
  TO authenticated
  WITH CHECK (true);

CREATE POLICY "Allow authenticated users to update galleries"
  ON galleries FOR UPDATE
  TO authenticated
  USING (true);

CREATE POLICY "Allow authenticated users to delete galleries"
  ON galleries FOR DELETE
  TO authenticated
  USING (true);
