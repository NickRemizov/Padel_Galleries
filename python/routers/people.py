"""
Роутер для работы с людьми (игроками).
CRUD операции для people table.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import logging
import json

from services.postgres_client import PostgresClient
from services.face_recognition import FaceRecognitionService

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
        
        # Rebuild index after deleting person (removes their descriptors)
        face_service = FaceRecognitionService()
        await face_service.rebuild_players_index()
        logger.info(f"[PeopleAPI] Index rebuilt after deleting person {person_id}")
        
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


@router.get("/{person_id}/photos-with-details")
async def get_person_photos_with_details(person_id: str):
    """
    Получить все фото персоны с детальной информацией для админки.
    Включает: boundingBox, confidence, verified, gallery info, sort_order.
    
    Returns:
        List of photos with face details for admin gallery dialog
    """
    try:
        db = PostgresClient()
        await db.connect()
        
        query = """
            SELECT 
                gi.id,
                gi.gallery_id,
                gi.image_url,
                gi.original_url,
                gi.original_filename as filename,
                gi.width,
                gi.height,
                gi.created_at,
                pf.id as face_id,
                pf.insightface_bbox as bounding_box,
                pf.recognition_confidence as confidence,
                pf.verified,
                g.title as gallery_name,
                g.shoot_date,
                g.sort_order,
                (SELECT COUNT(*) FROM photo_faces WHERE photo_id = gi.id) as face_count
            FROM gallery_images gi
            INNER JOIN photo_faces pf ON pf.photo_id = gi.id
            INNER JOIN galleries g ON g.id = gi.gallery_id
            WHERE pf.person_id = $1
            ORDER BY g.shoot_date DESC NULLS LAST, gi.original_filename ASC
        """
        
        rows = await db.fetch(query, person_id)
        
        photos = []
        for row in rows:
            row_dict = dict(row)
            
            # Parse bounding box from JSON if needed
            bbox = row_dict.get('bounding_box')
            if bbox:
                if isinstance(bbox, str):
                    try:
                        bbox = json.loads(bbox)
                    except:
                        bbox = None
            
            photos.append({
                "id": str(row_dict['id']),
                "image_url": row_dict['image_url'],
                "gallery_id": str(row_dict['gallery_id']),
                "width": row_dict['width'] or 0,
                "height": row_dict['height'] or 0,
                "faceId": str(row_dict['face_id']),
                "confidence": float(row_dict['confidence']) if row_dict.get('confidence') else None,
                "verified": bool(row_dict.get('verified', False)),
                "boundingBox": bbox,
                "faceCount": int(row_dict.get('face_count', 1)),
                "filename": row_dict['filename'] or "unknown",
                "gallery_name": row_dict.get('gallery_name'),
                "shootDate": str(row_dict['shoot_date']) if row_dict.get('shoot_date') else None,
                "sort_order": row_dict.get('sort_order') or "filename",
                "created_at": str(row_dict['created_at']) if row_dict.get('created_at') else None,
            })
        
        logger.info(f"[PeopleAPI] Found {len(photos)} photos with details for person {person_id}")
        return {"success": True, "data": photos}
        
    except Exception as e:
        logger.error(f"[PeopleAPI] Error getting person photos with details: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{person_id}/verify-on-photo")
async def verify_person_on_photo(person_id: str, photo_id: str):
    """
    Подтвердить (верифицировать) персону на фото.
    Устанавливает verified=true и confidence=1 для соответствующей записи photo_faces.
    Перестраивает индекс распознавания.
    """
    try:
        db = PostgresClient()
        await db.connect()
        
        query = """
            UPDATE photo_faces
            SET verified = true, recognition_confidence = 1.0
            WHERE person_id = $1 AND photo_id = $2
            RETURNING id
        """
        
        result = await db.fetch(query, person_id, photo_id)
        
        if not result:
            raise HTTPException(status_code=404, detail=f"Face not found for person {person_id} on photo {photo_id}")
        
        # Rebuild index after verification (adds verified descriptor)
        face_service = FaceRecognitionService()
        await face_service.rebuild_players_index()
        logger.info(f"[PeopleAPI] Index rebuilt after verifying person {person_id} on photo {photo_id}")
        
        logger.info(f"[PeopleAPI] Verified person {person_id} on photo {photo_id}")
        return {"success": True, "data": {"verified": True}}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[PeopleAPI] Error verifying person on photo: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{person_id}/batch-verify-on-photos")
async def batch_verify_person_on_photos(person_id: str, photo_ids: List[str]):
    """
    Batch verify person on multiple photos.
    Rebuilds index once after all verifications.
    """
    try:
        db = PostgresClient()
        await db.connect()
        
        if not photo_ids:
            return {"success": True, "data": {"verified_count": 0}}
        
        # Build query with array of photo_ids
        query = """
            UPDATE photo_faces
            SET verified = true, recognition_confidence = 1.0
            WHERE person_id = $1 AND photo_id = ANY($2::uuid[])
            RETURNING id
        """
        
        result = await db.fetch(query, person_id, photo_ids)
        verified_count = len(result) if result else 0
        
        # Rebuild index once after batch verification
        if verified_count > 0:
            face_service = FaceRecognitionService()
            await face_service.rebuild_players_index()
            logger.info(f"[PeopleAPI] Index rebuilt after batch verifying {verified_count} photos for person {person_id}")
        
        logger.info(f"[PeopleAPI] Batch verified person {person_id} on {verified_count} photos")
        return {"success": True, "data": {"verified_count": verified_count}}
        
    except Exception as e:
        logger.error(f"[PeopleAPI] Error batch verifying person on photos: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{person_id}/unlink-from-photo")
async def unlink_person_from_photo(person_id: str, photo_id: str):
    """
    Отвязать персону от фото.
    Устанавливает person_id=NULL, verified=false для записи photo_faces.
    Перестраивает индекс распознавания для удаления дескриптора.
    """
    try:
        db = PostgresClient()
        await db.connect()
        
        query = """
            UPDATE photo_faces
            SET person_id = NULL, verified = false, recognition_confidence = NULL
            WHERE person_id = $1 AND photo_id = $2
            RETURNING id
        """
        
        result = await db.fetch(query, person_id, photo_id)
        unlinked_count = len(result) if result else 0
        
        # Rebuild index after unlinking (removes descriptor from index)
        if unlinked_count > 0:
            face_service = FaceRecognitionService()
            await face_service.rebuild_players_index()
            logger.info(f"[PeopleAPI] Index rebuilt after unlinking person {person_id} from photo {photo_id}")
        
        logger.info(f"[PeopleAPI] Unlinked person {person_id} from photo {photo_id}, count: {unlinked_count}")
        return {"success": True, "data": {"unlinked_count": unlinked_count}}
        
    except Exception as e:
        logger.error(f"[PeopleAPI] Error unlinking person from photo: {e}")
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
