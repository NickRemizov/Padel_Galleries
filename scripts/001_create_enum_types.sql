-- Создать ENUM типы для категорий
CREATE TYPE person_category AS ENUM ('player', 'trainer');
CREATE TYPE face_category AS ENUM ('player', 'trainer', 'spectator', 'unknown');

-- Добавить поле category в таблицу people
ALTER TABLE people ADD COLUMN IF NOT EXISTS category person_category DEFAULT 'player';

-- Добавить поле face_category в таблицу photo_faces
ALTER TABLE photo_faces ADD COLUMN IF NOT EXISTS face_category face_category DEFAULT 'unknown';

-- Миграция данных: для всех существующих игроков
UPDATE people SET category = 'player' WHERE category IS NULL;

-- Для всех лиц с присвоенным игроком установить категорию из people
UPDATE photo_faces pf
SET face_category = CASE 
  WHEN p.category = 'player' THEN 'player'::face_category
  WHEN p.category = 'trainer' THEN 'trainer'::face_category
  ELSE 'unknown'::face_category
END
FROM people p
WHERE pf.person_id = p.id AND pf.face_category = 'unknown';

-- Создать индексы для быстрого поиска
CREATE INDEX IF NOT EXISTS idx_people_category ON people(category);
CREATE INDEX IF NOT EXISTS idx_photo_faces_category ON photo_faces(face_category);
