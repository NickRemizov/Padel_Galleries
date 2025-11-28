"""
Роутер для работы с людьми (игроками).
CRUD операции для people table.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import logging

from services.postgres_client import PostgresClient
from models.schemas import PersonCreate, PersonUpdate, PersonResponse, ClusterFace, PersonFromClusterCreate

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/people", tags=["people"])


@router.get("", response_model=List[PersonResponse])
async def get_all_people(include_stats: bool = True):
    """
    Получить всех людей.
    
    Args:
        include_stats: Включить статистику (количество фото)
    
    Returns:
        List of people with optional stats
    """
    try:
        db = PostgresClient()
        people = await db.get_all_people(include_stats=include_stats)
        
        logger.info(f"[PeopleAPI] Found {len(people)} people")
        return people
        
    except Exception as e:
        logger.error(f"[PeopleAPI] Error getting people: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{person_id}", response_model=PersonResponse)
async def get_person_by_id(person_id: str):
    """
    Получить человека по ID.
    """
    try:
        db = PostgresClient()
        person = await db.get_person_by_id(person_id)
        
        if not person:
            raise HTTPException(status_code=404, detail=f"Person {person_id} not found")
        
        logger.info(f"[PeopleAPI] Found person: {person_id}")
        return person
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[PeopleAPI] Error getting person: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("", response_model=PersonResponse)
async def create_person(data: PersonCreate):
    """
    Создать нового человека.
    """
    try:
        db = PostgresClient()
        person = await db.create_person(data.model_dump())
        
        logger.info(f"[PeopleAPI] Created person: {person['id']}")
        return person
        
    except Exception as e:
        logger.error(f"[PeopleAPI] Error creating person: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{person_id}", response_model=PersonResponse)
async def update_person(person_id: str, data: PersonUpdate):
    """
    Обновить информацию о человеке.
    """
    try:
        db = PostgresClient()
        person = await db.update_person(person_id, data.model_dump(exclude_unset=True))
        
        if not person:
            raise HTTPException(status_code=404, detail=f"Person {person_id} not found")
        
        logger.info(f"[PeopleAPI] Updated person: {person_id}")
        return person
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[PeopleAPI] Error updating person: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{person_id}")
async def delete_person(person_id: str):
    """
    Удалить человека.
    Также удаляет все связанные лица и дескрипторы.
    """
    try:
        db = PostgresClient()
        success = await db.delete_person(person_id)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Person {person_id} not found")
        
        logger.info(f"[PeopleAPI] Deleted person: {person_id}")
        return {"success": True, "message": f"Person {person_id} deleted"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[PeopleAPI] Error deleting person: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{person_id}/photos")
async def get_person_photos(person_id: str):
    """
    Получить все фото с лицами персоны.
    
    Returns:
        List of gallery images where person appears
    """
    try:
        db = PostgresClient()
        await db.connect()
        
        # Получаем все фото где есть лица этой персоны
        query = """
            SELECT DISTINCT
                gi.id,
                gi.gallery_id,
                gi.image_url,
                gi.original_url,
                gi.original_filename,
                gi.width,
                gi.height,
                gi.created_at,
                g.title as gallery_title,
                g.shoot_date as gallery_shoot_date
            FROM gallery_images gi
            INNER JOIN photo_faces pf ON pf.photo_id = gi.id
            INNER JOIN galleries g ON g.id = gi.gallery_id
            WHERE pf.person_id = $1
            ORDER BY g.shoot_date DESC, gi.created_at DESC
        """
        
        rows = await db.fetch(query, person_id)
        photos = [dict(row) for row in rows]
        
        logger.info(f"[PeopleAPI] Found {len(photos)} photos for person {person_id}")
        return {"success": True, "photos": photos, "count": len(photos)}
        
    except Exception as e:
        logger.error(f"[PeopleAPI] Error getting person photos: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{person_id}/avatar")
async def update_person_avatar(person_id: str, avatar_url: str):
    """
    Обновить аватар персоны.
    """
    try:
        db = PostgresClient()
        person = await db.update_person(person_id, {"avatar_url": avatar_url})
        
        if not person:
            raise HTTPException(status_code=404, detail=f"Person {person_id} not found")
        
        logger.info(f"[PeopleAPI] Updated avatar for person: {person_id}")
        return {"success": True, "person": person}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[PeopleAPI] Error updating avatar: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/from-cluster", response_model=PersonResponse)
async def create_person_from_cluster(data: PersonFromClusterCreate):
    """
    Создать персону из кластера лиц.
    
    Args:
        person_name: Имя нового человека
        cluster_faces: Список лиц с photo_id и descriptor
    
    Returns:
        Created person
    """
    try:
        db = PostgresClient()
        person = await db.create_person_from_cluster(
            data.person_name,
            [face.model_dump() for face in data.cluster_faces]
        )
        
        logger.info(f"[PeopleAPI] Created person from cluster: {person['id']}, faces: {len(data.cluster_faces)}")
        return person
        
    except Exception as e:
        logger.error(f"[PeopleAPI] Error creating person from cluster: {e}")
        raise HTTPException(status_code=500, detail=str(e))
