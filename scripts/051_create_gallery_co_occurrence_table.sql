-- Create table for tracking which people appear together in galleries
CREATE TABLE IF NOT EXISTS gallery_co_occurrence (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  person_id_1 UUID NOT NULL REFERENCES people(id) ON DELETE CASCADE,
  person_id_2 UUID NOT NULL REFERENCES people(id) ON DELETE CASCADE,
  gallery_id UUID NOT NULL REFERENCES galleries(id) ON DELETE CASCADE,
  co_occurrence_count INTEGER DEFAULT 1,
  last_seen_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  
  -- Ensure person_id_1 < person_id_2 to avoid duplicates
  CONSTRAINT gallery_co_occurrence_order_check CHECK (person_id_1 < person_id_2),
  
  -- Unique constraint to prevent duplicates
  CONSTRAINT gallery_co_occurrence_unique UNIQUE (person_id_1, person_id_2, gallery_id)
);

-- Indexes for fast lookups
CREATE INDEX IF NOT EXISTS gallery_co_occurrence_person1_idx ON gallery_co_occurrence(person_id_1);
CREATE INDEX IF NOT EXISTS gallery_co_occurrence_person2_idx ON gallery_co_occurrence(person_id_2);
CREATE INDEX IF NOT EXISTS gallery_co_occurrence_gallery_idx ON gallery_co_occurrence(gallery_id);

COMMENT ON TABLE gallery_co_occurrence IS 'Tracks which people appear together in the same gallery (tournament context)';
COMMENT ON COLUMN gallery_co_occurrence.co_occurrence_count IS 'Number of photos where both people appear together in this gallery';
