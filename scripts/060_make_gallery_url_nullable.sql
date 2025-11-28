-- Make gallery_url nullable since it's not used in the application
-- All gallery links are generated dynamically as /gallery/{id}

ALTER TABLE galleries ALTER COLUMN gallery_url DROP NOT NULL;

-- Update existing empty gallery_url values to NULL
UPDATE galleries SET gallery_url = NULL WHERE gallery_url = '';

COMMENT ON COLUMN galleries.gallery_url IS 'Legacy field - not used. Gallery URLs are generated dynamically as /gallery/{id}';
