-- Миграция для удаления устаревших полей
-- Перед запуском убедитесь, что весь код переведен на использование insightface_* полей

ALTER TABLE photo_faces
DROP COLUMN IF EXISTS bounding_box,
DROP COLUMN IF EXISTS descriptor,
DROP COLUMN IF EXISTS confidence;
