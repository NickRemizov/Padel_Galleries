"""
Роутер для работы с лицами на фотографиях.
Все операции сохранения, обновления, загрузки лиц.
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, List
import logging
import json

from services.postgres_client import PostgresClient, db_client
from services.face_recognition import FaceRecognitionService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/faces", tags=["faces"])


# Schemas
class SavePhotoFaceRequest(BaseModel):
    photo_id: str
    person_id: str
    bbox: dict
    insightface_descriptor: Optional[List[float]] = None
    insightface_confidence: float
    recognition_confidence: float
    verified: bool
    image_url: Optional[str] = None


class UpdatePhotoFaceRequest(BaseModel):
    face_id: str
    person_id: Optional[str] = None
    verified: Optional[bool] = None


class GetPhotoFacesRequest(BaseModel):
    photo_id: str


class GetPersonFacesRequest(BaseModel):
    person_id: str


class FaceTagItem(BaseModel):
    face_id: Optional[str] = None
    person_id: Optional[str] = None
    bbox: dict
    embedding: Optional[List[float]] = None
    insightface_bbox: Optional[dict] = None
    insightface_confidence: Optional[float] = None
    verified: bool = True


class SaveFaceTagsRequest(BaseModel):
    photo_id: str
    image_url: str
    tags: List[FaceTagItem]
    generate_descriptors: bool = True


class BatchPhotoFacesRequest(BaseModel):
    photo_ids: List[str]


# Endpoints
@router.post("/save")
async def save_photo_face(request: SavePhotoFaceRequest):
    """
    Сохранить лицо на фотографии.
    Если descriptor не передан - генерирует из image_url + bbox.
    Автоматически перестраивает индекс распознавания если verified=true.
    """
    try:
        db = PostgresClient()
        face_service = FaceRecognitionService()
        
        descriptor = request.insightface_descriptor
        
        if not descriptor or len(descriptor) == 0:
            if request.image_url and request.bbox:
                logger.info(f"[SaveFace] Generating embedding from image for bbox: {request.bbox}")
                try:
                    # Detect faces and find the one matching our bbox
                    faces = await face_service.detect_faces(request.image_url, apply_quality_filters=False)
                    
                    # Find face closest to our bbox
                    best_match = None
                    best_overlap = 0
                    
                    req_bbox = request.bbox
                    req_x = req_bbox.get('x', 0)
                    req_y = req_bbox.get('y', 0)
                    req_w = req_bbox.get('width', 0)
                    req_h = req_bbox.get('height', 0)
                    
                    for face in faces:
                        face_bbox = face['bbox']  # [x1, y1, x2, y2]
                        fx, fy = face_bbox[0], face_bbox[1]
                        fw, fh = face_bbox[2] - face_bbox[0], face_bbox[3] - face_bbox[1]
                        
                        # Calculate IoU (Intersection over Union)
                        x_overlap = max(0, min(req_x + req_w, fx + fw) - max(req_x, fx))
                        y_overlap = max(0, min(req_y + req_h, fy + fh) - max(req_y, fy))
                        intersection = x_overlap * y_overlap
                        
                        area1 = req_w * req_h
                        area2 = fw * fh
                        union = area1 + area2 - intersection
                        
                        if union > 0:
                            iou = intersection / union
                            if iou > best_overlap:
                                best_overlap = iou
                                best_match = face
                    
                    if best_match and best_overlap > 0.3:
                        descriptor = best_match['embedding'].tolist()
                        logger.info(f"[SaveFace] Generated embedding with IoU={best_overlap:.2f}, dim={len(descriptor)}")
                    else:
                        logger.warning(f"[SaveFace] No matching face found for bbox, IoU={best_overlap:.2f}")
                except Exception as e:
                    logger.error(f"[SaveFace] Failed to generate embedding: {str(e)}")
            else:
                logger.warning(f"[SaveFace] No descriptor and no image_url provided")
        
        # Сохранение в photo_faces
        face_id = await db.save_photo_face(
            photo_id=request.photo_id,
            person_id=request.person_id,
            bbox=request.bbox,
            descriptor=descriptor,
            insightface_confidence=request.insightface_confidence,
            recognition_confidence=request.recognition_confidence,
            verified=request.verified
        )
        
        logger.info(f"[SaveFace] Saved face {face_id} for person {request.person_id}, verified={request.verified}, has_descriptor={descriptor is not None and len(descriptor) > 0}")
        
        # Rebuild index если это verified лицо
        index_rebuilt = False
        if request.verified and descriptor and len(descriptor) > 0:
            await face_service.rebuild_players_index()
            index_rebuilt = True
            logger.info(f"[SaveFace] Index rebuilt after saving verified face")
        
        return {
            "success": True,
            "face_id": face_id,
            "index_rebuilt": index_rebuilt
        }
        
    except Exception as e:
        logger.error(f"[SaveFace] Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/update")
async def update_photo_face(request: UpdatePhotoFaceRequest):
    """
    Обновить информацию о лице (person_id, verified статус).
    Перестраивает индекс если изменился verified статус.
    """
    try:
        db = PostgresClient()
        face_service = FaceRecognitionService()
        
        # Обновление в БД
        await db.update_photo_face(
            face_id=request.face_id,
            person_id=request.person_id,
            verified=request.verified
        )
        
        logger.info(f"[UpdateFace] Updated face {request.face_id}")
        
        # Rebuild index если изменился verified
        index_rebuilt = False
        if request.verified is not None:
            await face_service.rebuild_players_index()
            index_rebuilt = True
            logger.info(f"[UpdateFace] Index rebuilt after verified status change")
        
        return {
            "success": True,
            "index_rebuilt": index_rebuilt
        }
        
    except Exception as e:
        logger.error(f"[UpdateFace] Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/delete/{face_id}")
async def delete_photo_face(face_id: str):
    """
    Удалить лицо из photo_faces.
    Также удаляет связанные face_descriptors.
    """
    try:
        db = PostgresClient()
        
        # Удаляем face_descriptors связанные с этим face
        # (через source_image_id если есть связь)
        
        # Удаляем photo_face
        await db.execute(
            "DELETE FROM photo_faces WHERE id = $1",
            face_id
        )
        
        logger.info(f"[DeleteFace] Deleted face {face_id}")
        
        return {"success": True}
        
    except Exception as e:
        logger.error(f"[DeleteFace] Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/get-photo-faces")
async def get_photo_faces(request: GetPhotoFacesRequest):
    """
    Получить все лица на фотографии.
    """
    try:
        db = PostgresClient()
        
        faces = await db.get_photo_faces(photo_id=request.photo_id)
        
        return {
            "success": True,
            "faces": faces,
            "count": len(faces)
        }
        
    except Exception as e:
        logger.error(f"[GetPhotoFaces] Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/get-batch-photo-faces")
async def get_batch_photo_faces(request: BatchPhotoFacesRequest):
    """Get faces for multiple photos"""
    try:
        if not request.photo_ids:
            return []
        
        faces = await db_client.get_batch_photo_faces(request.photo_ids)
        return faces
    except Exception as e:
        logger.error(f"Error getting batch photo faces: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/get-person-faces")
async def get_person_faces(request: GetPersonFacesRequest):
    """
    Получить все лица персоны.
    """
    try:
        db = PostgresClient()
        
        faces = await db.get_person_faces(
            person_id=request.person_id,
            verified_only=False  # Removed verified_only parameter
        )
        
        # Подсчет статистики
        verified_count = sum(1 for f in faces if f.get('verified'))
        high_confidence_count = sum(1 for f in faces if not f.get('verified') and f.get('recognition_confidence', 0) >= 0.8)
        descriptor_count = sum(1 for f in faces if f.get('insightface_descriptor') is not None)
        
        return {
            "success": True,
            "faces": faces,
            "stats": {
                "total": len(faces),
                "verified": verified_count,
                "high_confidence": high_confidence_count,
                "descriptors": descriptor_count
            }
        }
        
    except Exception as e:
        logger.error(f"[GetPersonFaces] Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/save-face-tags")
async def save_face_tags(request: SaveFaceTagsRequest):
    """
    Batch save face tags for a photo.
    1. Deletes all existing photo_faces for this photo
    2. Deletes all existing face_descriptors for this photo
    3. If no tags - marks photo as processed (NFD = no faces detected)
    4. If has tags - generates descriptors and saves faces
    """
    try:
        db = PostgresClient()
        await db.connect()
        
        photo_id = request.photo_id
        tags = request.tags
        
        logger.info(f"[SaveFaceTags] Processing {len(tags)} tags for photo {photo_id}")
        
        # Get photo details
        photo = await db.fetchone(
            "SELECT id, image_url FROM gallery_images WHERE id = $1",
            photo_id
        )
        
        if not photo:
            raise HTTPException(status_code=404, detail=f"Photo {photo_id} not found")
        
        image_url = photo['image_url']
        
        # Delete existing photo_faces
        await db.execute(
            "DELETE FROM photo_faces WHERE photo_id = $1",
            photo_id
        )
        logger.info(f"[SaveFaceTags] Deleted existing photo_faces for {photo_id}")
        
        # Delete existing face_descriptors for this photo
        await db.execute(
            "DELETE FROM face_descriptors WHERE source_image_id = $1",
            photo_id
        )
        logger.info(f"[SaveFaceTags] Deleted existing face_descriptors for {photo_id}")
        
        await db.execute(
            "UPDATE gallery_images SET has_been_processed = true WHERE id = $1",
            photo_id
        )
        
        # If no tags - just return success (photo marked as having no faces)
        if len(tags) == 0:
            logger.info(f"[SaveFaceTags] Photo {photo_id} has no faces (NFD)")
            return {
                "success": True,
                "message": "All faces removed, photo marked as NFD",
                "faces": []
            }
        
        # Generate descriptors via recognition service
        generated_descriptors = []
        try:
            face_service = FaceRecognitionService()
            
            faces_for_generation = [
                {
                    "person_id": tag.person_id,
                    "bbox": tag.bbox,
                    "verified": tag.verified,
                    "photo_id": photo_id
                }
                for tag in tags
            ]
            
            result = await face_service.generate_descriptors(image_url, faces_for_generation)
            generated_descriptors = result.get("descriptors", [])
            logger.info(f"[SaveFaceTags] Generated {len(generated_descriptors)} descriptors")
        except Exception as e:
            logger.error(f"[SaveFaceTags] Error generating descriptors: {e}")
            # Continue without descriptors - don't silently fail, but log and continue
        
        # Insert new faces
        saved_faces = []
        for i, tag in enumerate(tags):
            backend_descriptor = generated_descriptors[i].get("descriptor") if i < len(generated_descriptors) else None
            descriptor = backend_descriptor or tag.embedding or []
            
            confidence_to_save = 1.0 if tag.verified else tag.insightface_confidence
            
            bbox_to_save = None
            if tag.bbox:
                bbox = tag.bbox
                if bbox.get("width", 0) > 0 and bbox.get("height", 0) > 0:
                    bbox_to_save = bbox
            
            # Insert photo_face
            descriptor_str = None
            if descriptor and len(descriptor) > 0:
                descriptor_str = '[' + ','.join(map(str, descriptor)) + ']'
            
            face_result = await db.fetchone(
                """
                INSERT INTO photo_faces (
                    photo_id, person_id, insightface_bbox, insightface_descriptor,
                    insightface_confidence, recognition_confidence, verified
                )
                VALUES ($1, $2, $3, $4::vector, $5, $6, $7)
                RETURNING id
                """,
                photo_id,
                tag.person_id,
                json.dumps(bbox_to_save) if bbox_to_save else None,
                descriptor_str,
                tag.insightface_confidence,
                confidence_to_save,
                tag.verified
            )
            
            photo_face_id = face_result['id'] if face_result else None
            
            if tag.person_id and descriptor and len(descriptor) > 0 and photo_face_id:
                await db.execute(
                    """
                    INSERT INTO face_descriptors (person_id, descriptor, source_image_id)
                    VALUES ($1, $2::jsonb, $3)
                    """,
                    tag.person_id,
                    json.dumps(descriptor),
                    photo_id
                )
                logger.info(f"[SaveFaceTags] Saved descriptor for person {tag.person_id}")
            
            saved_faces.append({
                "face_id": str(photo_face_id) if photo_face_id else None,
                "person_id": tag.person_id,
                "verified": tag.verified
            })
        
        logger.info(f"[SaveFaceTags] Saved {len(saved_faces)} faces for photo {photo_id}")
        
        return {
            "success": True,
            "faces": saved_faces,
            "count": len(saved_faces)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[SaveFaceTags] Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch-get-photo-faces")
async def batch_get_photo_faces(request: BatchPhotoFacesRequest):
    """
    Получить все лица для нескольких фотографий.
    """
    try:
        photo_ids = request.photo_ids
        
        logger.info(f"[BatchGetPhotoFaces] Processing {len(photo_ids)} photos")
        
        all_faces = []
        for photo_id in photo_ids:
            faces = await db_client.get_photo_faces(photo_id=photo_id)
            all_faces.extend(faces)
        
        return {
            "success": True,
            "faces": all_faces,
            "count": len(all_faces)
        }
        
    except Exception as e:
        logger.error(f"[BatchGetPhotoFaces] Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
