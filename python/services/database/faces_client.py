"""Client for photo_faces and face_descriptors operations."""

from typing import List, Dict, Optional
import json
import logging
from uuid import uuid4

logger = logging.getLogger(__name__)


class FacesClient:
    """Handles operations with photo_faces and face_descriptors."""
    
    async def get_verified_faces(
        self,
        event_ids: Optional[List[str]] = None,
        person_ids: Optional[List[str]] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        min_faces_per_person: int = 3
    ) -> List[Dict]:
        """Get verified faces from PostgreSQL with filters."""
        try:
            await self.connect()
            
            query = """
                SELECT 
                    pf.id as face_id,
                    pf.person_id,
                    pf.insightface_bbox as bbox,
                    pf.photo_id,
                    p.real_name as person_name,
                    gi.image_url as photo_url,
                    gi.gallery_id,
                    g.title as gallery_name,
                    g.shoot_date as gallery_date
                FROM photo_faces pf
                JOIN people p ON pf.person_id = p.id
                JOIN gallery_images gi ON pf.photo_id = gi.id
                JOIN galleries g ON gi.gallery_id = g.id
                WHERE pf.person_id IS NOT NULL
                    AND pf.verified = TRUE
            """
            
            params = []
            param_counter = 1
            
            if event_ids:
                query += f" AND gi.gallery_id = ANY(${param_counter})"
                params.append(event_ids)
                param_counter += 1
            
            if person_ids:
                query += f" AND pf.person_id = ANY(${param_counter})"
                params.append(person_ids)
                param_counter += 1
            
            if date_from:
                query += f" AND g.shoot_date >= ${param_counter}"
                params.append(date_from)
                param_counter += 1
            
            if date_to:
                query += f" AND g.shoot_date <= ${param_counter}"
                params.append(date_to)
                param_counter += 1
            
            query += " ORDER BY p.real_name, g.shoot_date"
            
            faces = await self.fetch(query, *params)
            
            if min_faces_per_person > 0:
                person_face_counts = {}
                for face in faces:
                    pid = face['person_id']
                    person_face_counts[pid] = person_face_counts.get(pid, 0) + 1
                
                faces = [f for f in faces if person_face_counts[f['person_id']] >= min_faces_per_person]
            
            print(f"[PostgresClient] Retrieved {len(faces)} verified faces")
            return faces
            
        except Exception as e:
            logger.error(f"[PostgresClient] Error getting verified faces: {e}")
            return []
    
    async def get_verified_faces_with_descriptors(
        self,
        person_ids: Optional[List[str]] = None
    ) -> List[Dict]:
        """Get verified faces with their descriptors for training."""
        try:
            await self.connect()
            
            query = """
                SELECT 
                    pf.id as face_id,
                    pf.person_id,
                    pf.photo_id,
                    pf.insightface_descriptor,
                    pf.insightface_bbox,
                    pf.verified,
                    gi.image_url as photo_url
                FROM photo_faces pf
                JOIN gallery_images gi ON pf.photo_id = gi.id
                WHERE pf.person_id IS NOT NULL
                    AND pf.verified = TRUE
                    AND pf.insightface_descriptor IS NOT NULL
            """
            
            if person_ids:
                query += " AND pf.person_id = ANY($1)"
                faces = await self.fetch(query, person_ids)
            else:
                faces = await self.fetch(query)
            
            print(f"[PostgresClient] Retrieved {len(faces)} verified faces with descriptors")
            return faces
            
        except Exception as e:
            logger.error(f"[PostgresClient] Error getting faces with descriptors: {e}")
            return []
    
    async def save_photo_face(
        self,
        photo_id: str,
        bbox: Dict,
        person_id: Optional[str] = None,
        descriptor: Optional[List[float]] = None,
        confidence: Optional[float] = None,
        verified: bool = False
    ) -> str:
        """Save a photo face to database."""
        try:
            await self.connect()
            
            query = """
                INSERT INTO photo_faces (
                    photo_id, insightface_bbox, person_id, 
                    insightface_descriptor, recognition_confidence, verified
                )
                VALUES ($1, $2, $3, $4, $5, $6)
                RETURNING id
            """
            
            descriptor_json = json.dumps(descriptor) if descriptor else None
            bbox_json = json.dumps(bbox)
            
            face_id = await self.fetchval(
                query, 
                photo_id, bbox_json, person_id, 
                descriptor_json, confidence, verified
            )
            
            print(f"[PostgresClient] Saved photo face {face_id}")
            return str(face_id)
            
        except Exception as e:
            logger.error(f"[PostgresClient] Error saving photo face: {e}")
            raise
    
    async def get_photo_faces(self, photo_id: str) -> List[Dict]:
        """Get all faces from a photo."""
        try:
            await self.connect()
            
            query = """
                SELECT 
                    pf.id,
                    pf.photo_id,
                    pf.person_id,
                    pf.insightface_bbox,
                    pf.recognition_confidence,
                    pf.verified,
                    p.real_name as person_name,
                    p.avatar_url
                FROM photo_faces pf
                LEFT JOIN people p ON pf.person_id = p.id
                WHERE pf.photo_id = $1
                ORDER BY pf.created_at
            """
            
            rows = await self.fetch(query, photo_id)
            
            faces = []
            for row in rows:
                face = dict(row)
                # Parse insightface_bbox if it's a string
                if face.get('insightface_bbox') and isinstance(face['insightface_bbox'], str):
                    try:
                        face['insightface_bbox'] = json.loads(face['insightface_bbox'])
                    except (json.JSONDecodeError, TypeError):
                        pass
                faces.append(face)
            
            return faces
            
        except Exception as e:
            logger.error(f"[PostgresClient] Error getting photo faces: {e}")
            return []
    
    async def get_batch_photo_faces(self, photo_ids: List[str]) -> List[Dict]:
        """Get faces from multiple photos."""
        try:
            await self.connect()
            
            query = """
                SELECT 
                    pf.id,
                    pf.photo_id,
                    pf.person_id,
                    pf.insightface_bbox,
                    pf.recognition_confidence,
                    pf.verified,
                    p.real_name as person_name
                FROM photo_faces pf
                LEFT JOIN people p ON pf.person_id = p.id
                WHERE pf.photo_id = ANY($1)
                ORDER BY pf.photo_id, pf.created_at
            """
            
            rows = await self.fetch(query, photo_ids)
            
            logger.info(f"[FacesClient] Fetched {len(rows)} rows from DB")
            if rows:
                first_row = dict(rows[0])
                logger.info(f"[FacesClient] First row keys: {list(first_row.keys())}")
                logger.info(f"[FacesClient] insightface_bbox present: {'insightface_bbox' in first_row}")
                if 'insightface_bbox' in first_row:
                    bbox_value = first_row['insightface_bbox']
                    logger.info(f"[FacesClient] insightface_bbox type: {type(bbox_value)}")
                    logger.info(f"[FacesClient] insightface_bbox value: {bbox_value}")
            
            faces = []
            for row in rows:
                face = dict(row)
                
                # Parse bbox if it's a string (old data)
                if face.get('insightface_bbox') and isinstance(face['insightface_bbox'], str):
                    try:
                        face['insightface_bbox'] = json.loads(face['insightface_bbox'])
                    except (json.JSONDecodeError, TypeError):
                        pass
                
                faces.append(face)
            
            if faces:
                logger.info(f"[FacesClient] First face keys after processing: {list(faces[0].keys())}")
                logger.info(f"[FacesClient] First face bbox: {faces[0].get('insightface_bbox')}")
            
            return faces
        
        except Exception as e:
            logger.error(f"[PostgresClient] Error getting batch photo faces: {e}")
            return []
    
    async def save_face_descriptor(
        self,
        person_id: str,
        descriptor: List[float],
        photo_id: str,
        verified: bool = True
    ) -> Optional[str]:
        """Save face descriptor for a person."""
        try:
            await self.connect()
            
            query = """
                UPDATE photo_faces
                SET 
                    person_id = $1,
                    insightface_descriptor = $2,
                    verified = $3
                WHERE photo_id = $4 AND person_id IS NULL
                RETURNING id
            """
            
            descriptor_json = json.dumps(descriptor)
            face_id = await self.fetchval(query, person_id, descriptor_json, verified, photo_id)
            
            if face_id:
                print(f"[PostgresClient] Saved face descriptor for person {person_id}")
                return str(face_id)
            
            return None
            
        except Exception as e:
            logger.error(f"[PostgresClient] Error saving face descriptor: {e}")
            return None
    
    async def update_face_descriptor(
        self,
        face_id: str,
        descriptor: List[float],
        confidence: Optional[float] = None
    ) -> bool:
        """Update face descriptor and confidence."""
        try:
            await self.connect()
            
            descriptor_json = json.dumps(descriptor)
            
            if confidence is not None:
                query = """
                    UPDATE photo_faces
                    SET insightface_descriptor = $1, recognition_confidence = $2
                    WHERE id = $3
                """
                await self.execute(query, descriptor_json, confidence, face_id)
            else:
                query = """
                    UPDATE photo_faces
                    SET insightface_descriptor = $1
                    WHERE id = $2
                """
                await self.execute(query, descriptor_json, face_id)
            
            print(f"[PostgresClient] Updated face descriptor {face_id}")
            return True
            
        except Exception as e:
            logger.error(f"[PostgresClient] Error updating face descriptor: {e}")
            return False
    
    async def save_recognized_face(
        self,
        photo_id: str,
        bbox: Dict,
        person_id: str,
        confidence: float,
        descriptor: Optional[List[float]] = None,
        verified: bool = False
    ) -> str:
        """Save a recognized face."""
        return await self.save_photo_face(
            photo_id=photo_id,
            bbox=bbox,
            person_id=person_id,
            descriptor=descriptor,
            confidence=confidence,
            verified=verified
        )
    
    async def save_unknown_face(
        self,
        photo_id: str,
        bbox: Dict,
        descriptor: Optional[List[float]] = None
    ) -> str:
        """Save an unknown face (no person_id)."""
        return await self.save_photo_face(
            photo_id=photo_id,
            bbox=bbox,
            person_id=None,
            descriptor=descriptor,
            confidence=None,
            verified=False
        )
    
    async def get_unknown_faces_from_gallery(self, gallery_id: str) -> List[Dict]:
        """Get all unknown faces from a gallery."""
        try:
            await self.connect()
            
            query = """
                SELECT 
                    pf.id as face_id,
                    pf.photo_id,
                    pf.person_id,
                    pf.insightface_bbox,
                    pf.insightface_descriptor,
                    gi.image_url
                FROM photo_faces pf
                JOIN gallery_images gi ON pf.photo_id = gi.id
                WHERE gi.gallery_id = $1
                    AND pf.person_id IS NULL
                    AND pf.insightface_descriptor IS NOT NULL
                ORDER BY gi.created_at
            """
            
            faces = await self.fetch(query, gallery_id)
            print(f"[PostgresClient] Found {len(faces)} unknown faces in gallery {gallery_id}")
            return faces
            
        except Exception as e:
            logger.error(f"[PostgresClient] Error getting unknown faces: {e}")
            return []
