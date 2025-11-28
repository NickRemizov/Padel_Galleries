-- Добавить поле has_been_processed в таблицу gallery_images
ALTER TABLE gallery_images ADD COLUMN IF NOT EXISTS has_been_processed BOOLEAN DEFAULT FALSE;

-- Миграция данных: установить TRUE для фото с лицами
UPDATE gallery_images
SET has_been_processed = TRUE
WHERE id IN (
  SELECT DISTINCT photo_id FROM photo_faces
);

-- Создать индекс для быстрого поиска необработанных фото
CREATE INDEX IF NOT EXISTS idx_gallery_images_processed ON gallery_images(has_been_processed);
