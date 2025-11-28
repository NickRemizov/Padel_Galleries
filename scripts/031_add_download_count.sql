-- Add download_count column to gallery_images table
ALTER TABLE gallery_images
ADD COLUMN IF NOT EXISTS download_count INTEGER DEFAULT 0 NOT NULL;

-- Create index for better query performance
CREATE INDEX IF NOT EXISTS idx_gallery_images_download_count ON gallery_images(download_count);

-- Add comment
COMMENT ON COLUMN gallery_images.download_count IS 'Number of times this image has been downloaded';
