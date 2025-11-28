-- Исправление URL изображений в базе данных
-- Заменяем https://s3.vlcpadel.com/ на https://api.vlcpadel.com/api/s3-proxy/
-- Этот скрипт нужно запустить ОДИН раз на сервере

-- Обновляем gallery_images
UPDATE gallery_images 
SET image_url = REPLACE(image_url, 'https://s3.vlcpadel.com/', 'https://api.vlcpadel.com/api/s3-proxy/')
WHERE image_url LIKE 'https://s3.vlcpadel.com/%';

-- Обновляем thumbnail_url если есть
UPDATE gallery_images 
SET thumbnail_url = REPLACE(thumbnail_url, 'https://s3.vlcpadel.com/', 'https://api.vlcpadel.com/api/s3-proxy/')
WHERE thumbnail_url LIKE 'https://s3.vlcpadel.com/%';

-- Обновляем photo_faces если там хранятся URL
UPDATE photo_faces 
SET photo_url = REPLACE(photo_url, 'https://s3.vlcpadel.com/', 'https://api.vlcpadel.com/api/s3-proxy/')
WHERE photo_url LIKE 'https://s3.vlcpadel.com/%';

-- Проверка результата
SELECT 'gallery_images with old URL' as check_type, COUNT(*) as count 
FROM gallery_images WHERE image_url LIKE 'https://s3.vlcpadel.com/%'
UNION ALL
SELECT 'gallery_images with new URL' as check_type, COUNT(*) as count 
FROM gallery_images WHERE image_url LIKE 'https://api.vlcpadel.com/api/s3-proxy/%';
