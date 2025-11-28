-- Add square cover image field to galleries table
ALTER TABLE galleries
ADD COLUMN IF NOT EXISTS cover_image_square_url TEXT;
