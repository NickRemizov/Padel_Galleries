-- Удаление дублирующих записей лиц на фото
-- Оставляем только записи с verified=true, если они есть

-- Сначала найдем и выведем информацию о дубликатах
DO $$
DECLARE
  duplicate_count INTEGER;
BEGIN
  -- Подсчитываем количество дубликатов
  SELECT COUNT(*) INTO duplicate_count
  FROM (
    SELECT photo_id, person_id, COUNT(*) as cnt
    FROM photo_faces
    WHERE person_id IS NOT NULL
    GROUP BY photo_id, person_id
    HAVING COUNT(*) > 1
  ) duplicates;
  
  RAISE NOTICE 'Найдено % групп дубликатов', duplicate_count;
END $$;

-- Удаляем дубликаты, оставляя только verified=true записи
-- Если есть несколько verified=true, оставляем самую новую
DELETE FROM photo_faces
WHERE id IN (
  SELECT pf.id
  FROM photo_faces pf
  INNER JOIN (
    -- Находим группы с дубликатами
    SELECT photo_id, person_id
    FROM photo_faces
    WHERE person_id IS NOT NULL
    GROUP BY photo_id, person_id
    HAVING COUNT(*) > 1
  ) duplicates ON pf.photo_id = duplicates.photo_id 
    AND pf.person_id = duplicates.person_id
  WHERE 
    -- Удаляем записи, которые НЕ являются "лучшими"
    pf.id NOT IN (
      SELECT DISTINCT ON (photo_id, person_id) id
      FROM photo_faces
      WHERE person_id IS NOT NULL
      ORDER BY 
        photo_id, 
        person_id,
        verified DESC,  -- Сначала verified=true
        confidence DESC, -- Потом по уверенности
        updated_at DESC  -- Потом по дате обновления
    )
);

-- Выводим результат
DO $$
DECLARE
  remaining_duplicates INTEGER;
BEGIN
  SELECT COUNT(*) INTO remaining_duplicates
  FROM (
    SELECT photo_id, person_id, COUNT(*) as cnt
    FROM photo_faces
    WHERE person_id IS NOT NULL
    GROUP BY photo_id, person_id
    HAVING COUNT(*) > 1
  ) duplicates;
  
  RAISE NOTICE 'Осталось % групп дубликатов после очистки', remaining_duplicates;
END $$;
