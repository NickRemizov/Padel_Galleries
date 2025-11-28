-- Add external_gallery_url field to galleries table
ALTER TABLE galleries ADD COLUMN IF NOT EXISTS external_gallery_url TEXT;

COMMENT ON COLUMN galleries.external_gallery_url IS 'Optional external gallery URL for galleries without uploaded images';
