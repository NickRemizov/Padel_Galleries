"""
Gallery Images API endpoints
Handles CRUD operations for gallery images
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from services.postgres_client import db_client
from services.face_recognition import FaceRecognitionService

router = APIRouter(prefix="/api/galleries", tags=["galleries"])

# Schemas
class AddImageRequest(BaseModel):
    image_url: str
    original_url: str
    original_filename: str
    file_size: int
    width: int
    height: int

class AddImagesRequest(BaseModel):
    gallery_id: str
    images: List[AddImageRequest]

class DeleteImageRequest(BaseModel):
    image_id: str
    gallery_id: str

class BatchDeleteImagesRequest(BaseModel):
    image_ids: List[str]
    gallery_id: str

# Initialize services
face_recognition_service = FaceRecognitionService()

@router.get("/{gallery_id}/images")
async def get_gallery_images(gallery_id: str):
    """
    Get all images for a specific gallery
    """
    try:
        print(f"[GalleriesAPI] Loading images for gallery: {gallery_id}")
        
        images = await db_client.get_gallery_images(gallery_id)
        
        formatted_images = []
        for img in images:
            formatted_images.append({
                'id': str(img['id']),
                'gallery_id': str(img['gallery_id']),
                'image_url': img.get('image_url'),
                'original_url': img.get('original_url'),
                'display_order': img.get('display_order', 0),
                'has_been_processed': img.get('has_been_processed', False),
                'download_count': img.get('download_count', 0),
                'original_filename': img.get('original_filename'),
                'file_size': img.get('file_size'),
                'width': img.get('width'),
                'height': img.get('height'),
                'created_at': img['created_at'].isoformat() if img.get('created_at') else None
            })
        
        print(f"[GalleriesAPI] Found {len(formatted_images)} images")
        return {"success": True, "data": formatted_images}
            
    except Exception as e:
        print(f"[GalleriesAPI] Error loading gallery images: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/add-images")
async def add_gallery_images(request: AddImagesRequest):
    """
    Add multiple images to a gallery
    """
    try:
        print(f"[GalleriesAPI] Adding {len(request.images)} images to gallery: {request.gallery_id}")
        
        images_data = []
        for index, img in enumerate(request.images):
            images_data.append({
                'image_url': img.image_url,
                'original_url': img.original_url,
                'original_filename': img.original_filename,
                'file_size': img.file_size,
                'width': img.width,
                'height': img.height,
                'display_order': index
            })
        
        image_ids = await db_client.add_gallery_images(
            request.gallery_id,
            images_data
        )
        
        print(f"[GalleriesAPI] Successfully added {len(image_ids)} images")
        return {"success": True, "data": {"image_ids": image_ids}}
            
    except Exception as e:
        print(f"[GalleriesAPI] Error adding gallery images: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/delete-image")
async def delete_gallery_image(request: DeleteImageRequest):
    """
    Delete an image from gallery
    """
    try:
        print(f"[GalleriesAPI] Deleting image: {request.image_id} from gallery: {request.gallery_id}")
        
        verified_count = await db_client.fetchval(
            "SELECT COUNT(*) FROM photo_faces WHERE photo_id = $1 AND verified = true",
            request.image_id
        )
        has_verified_faces = verified_count > 0
        
        await db_client.delete_gallery_image(request.image_id, request.gallery_id)
        
        print(f"[GalleriesAPI] Successfully deleted image: {request.image_id}")
        
        # Rebuild index if had verified faces
        if has_verified_faces:
            print(f"[GalleriesAPI] Rebuilding recognition index after deleting verified faces")
            await face_recognition_service.rebuild_players_index()
        
        return {"success": True, "had_verified_faces": has_verified_faces}
            
    except Exception as e:
        print(f"[GalleriesAPI] Error deleting gallery image: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/batch-delete-images")
async def batch_delete_gallery_images(request: BatchDeleteImagesRequest):
    """
    Delete multiple images from gallery in one request.
    Rebuilds recognition index only ONCE at the end if any verified faces were deleted.
    
    Performance: O(1) index rebuilds instead of O(n)
    """
    try:
        image_ids = request.image_ids
        gallery_id = request.gallery_id
        
        if not image_ids:
            return {"success": True, "deleted_count": 0, "had_verified_faces": False}
        
        print(f"[GalleriesAPI] Batch deleting {len(image_ids)} images from gallery: {gallery_id}")
        
        # Step 1: Check if ANY of the images have verified faces (single query)
        verified_count = await db_client.fetchval(
            "SELECT COUNT(*) FROM photo_faces WHERE photo_id = ANY($1) AND verified = true",
            image_ids
        )
        has_verified_faces = verified_count > 0
        
        # Step 2: Delete all photo_faces for these images (single query)
        await db_client.execute(
            "DELETE FROM photo_faces WHERE photo_id = ANY($1)",
            image_ids
        )
        
        # Step 3: Delete all face_descriptors linked to these images (single query)
        await db_client.execute(
            "DELETE FROM face_descriptors WHERE source_image_id = ANY($1)",
            image_ids
        )
        
        # Step 4: Delete all images (single query)
        await db_client.execute(
            "DELETE FROM gallery_images WHERE id = ANY($1) AND gallery_id = $2",
            image_ids,
            gallery_id
        )
        
        deleted_count = len(image_ids)
        
        print(f"[GalleriesAPI] Batch deleted {deleted_count} images, had_verified_faces: {has_verified_faces}")
        
        # Step 5: Rebuild index ONCE if any verified faces were deleted
        if has_verified_faces:
            print(f"[GalleriesAPI] Rebuilding recognition index after batch delete")
            await face_recognition_service.rebuild_players_index()
        
        return {
            "success": True, 
            "deleted_count": deleted_count,
            "had_verified_faces": has_verified_faces,
            "index_rebuilt": has_verified_faces
        }
            
    except Exception as e:
        print(f"[GalleriesAPI] Error batch deleting gallery images: {e}")
        raise HTTPException(status_code=500, detail=str(e))
