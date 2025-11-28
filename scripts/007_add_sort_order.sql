-- Add sort_order column to galleries table
ALTER TABLE galleries ADD COLUMN IF NOT EXISTS sort_order TEXT DEFAULT 'filename';

-- Add comment
COMMENT ON COLUMN galleries.sort_order IS 'Sort order for gallery images: filename, created, or added';
