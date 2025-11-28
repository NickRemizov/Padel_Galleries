-- Add width and height columns to gallery_images table
ALTER TABLE gallery_images
ADD COLUMN width INTEGER,
ADD COLUMN height INTEGER;
