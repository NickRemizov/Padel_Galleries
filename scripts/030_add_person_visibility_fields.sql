-- Add visibility fields to people table
ALTER TABLE people 
ADD COLUMN IF NOT EXISTS show_in_players_gallery BOOLEAN DEFAULT true,
ADD COLUMN IF NOT EXISTS show_photos_in_galleries BOOLEAN DEFAULT true;

-- Add indexes for faster queries
CREATE INDEX IF NOT EXISTS idx_people_show_in_players_gallery ON people(show_in_players_gallery);
CREATE INDEX IF NOT EXISTS idx_people_show_photos_in_galleries ON people(show_photos_in_galleries);

-- Add comments
COMMENT ON COLUMN people.show_in_players_gallery IS 'Whether to show this person in the players gallery';
COMMENT ON COLUMN people.show_photos_in_galleries IS 'Whether to show photos of this person in event galleries';
