-- Add original_filename column to gallery_images table
ALTER TABLE gallery_images
ADD COLUMN original_filename TEXT;

-- Update existing records with a default value
UPDATE gallery_images
SET original_filename = 'image.jpg'
WHERE original_filename IS NULL;
