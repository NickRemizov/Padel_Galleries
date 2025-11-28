-- Add foreign key constraint for organizer_id in galleries table
ALTER TABLE galleries 
DROP CONSTRAINT IF EXISTS galleries_organizer_id_fkey;

ALTER TABLE galleries 
ADD CONSTRAINT galleries_organizer_id_fkey 
FOREIGN KEY (organizer_id) 
REFERENCES organizers(id) 
ON DELETE SET NULL;

-- Create index for faster queries (if not exists)
CREATE INDEX IF NOT EXISTS idx_galleries_organizer ON galleries(organizer_id);
