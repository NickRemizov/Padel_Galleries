-- Add file_size column to gallery_images table
ALTER TABLE gallery_images ADD COLUMN IF NOT EXISTS file_size INTEGER;

-- Add comment
COMMENT ON COLUMN gallery_images.file_size IS 'File size in bytes';
