-- Функция для автоматического обновления updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Триггер для обновления updated_at при изменении галереи
DROP TRIGGER IF EXISTS update_galleries_updated_at ON galleries;
CREATE TRIGGER update_galleries_updated_at
    BEFORE UPDATE ON galleries
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Функция для обновления updated_at галереи при изменении изображений
CREATE OR REPLACE FUNCTION update_gallery_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    -- Обновляем updated_at галереи при добавлении, удалении или изменении изображения
    IF TG_OP = 'DELETE' THEN
        UPDATE galleries SET updated_at = CURRENT_TIMESTAMP WHERE id = OLD.gallery_id;
        RETURN OLD;
    ELSE
        UPDATE galleries SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.gallery_id;
        RETURN NEW;
    END IF;
END;
$$ language 'plpgsql';

-- Триггеры для обновления updated_at галереи при изменении изображений
DROP TRIGGER IF EXISTS update_gallery_on_image_insert ON gallery_images;
CREATE TRIGGER update_gallery_on_image_insert
    AFTER INSERT ON gallery_images
    FOR EACH ROW
    EXECUTE FUNCTION update_gallery_updated_at();

DROP TRIGGER IF EXISTS update_gallery_on_image_update ON gallery_images;
CREATE TRIGGER update_gallery_on_image_update
    AFTER UPDATE ON gallery_images
    FOR EACH ROW
    EXECUTE FUNCTION update_gallery_updated_at();

DROP TRIGGER IF EXISTS update_gallery_on_image_delete ON gallery_images;
CREATE TRIGGER update_gallery_on_image_delete
    AFTER DELETE ON gallery_images
    FOR EACH ROW
    EXECUTE FUNCTION update_gallery_updated_at();
