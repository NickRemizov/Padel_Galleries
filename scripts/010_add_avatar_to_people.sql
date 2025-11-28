-- Add avatar_url column to people table
ALTER TABLE people ADD COLUMN IF NOT EXISTS avatar_url TEXT;

-- Add comment to explain the column
COMMENT ON COLUMN people.avatar_url IS 'URL to person avatar image, can be from social media or uploaded photo';
