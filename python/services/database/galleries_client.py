from typing import List, Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)

class GalleriesClient:
    """Client for managing galleries and gallery images."""
    
    async def get_all_galleries(self, include_stats: bool = True, sort_by: str = "created_at") -> List[Dict]:
        """Get all galleries with optional statistics."""
        try:
            if include_stats:
                query = f"""
                    SELECT 
                        g.*,
                        COUNT(DISTINCT gi.id) as image_count,
                        l.name as location_name,
                        o.name as organizer_name,
                        p.name as photographer_name
                    FROM galleries g
                    LEFT JOIN gallery_images gi ON g.id = gi.gallery_id
                    LEFT JOIN locations l ON g.location_id = l.id
                    LEFT JOIN organizers o ON g.organizer_id = o.id
                    LEFT JOIN photographers p ON g.photographer_id = p.id
                    GROUP BY g.id, l.name, o.name, p.name
                    ORDER BY g.{sort_by} DESC
                """
            else:
                query = f"SELECT * FROM galleries ORDER BY {sort_by} DESC"
            
            results = await self.fetch(query)
            return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"[GalleriesClient] Error getting all galleries: {e}")
            return []
    
    async def get_gallery_by_id(self, gallery_id: str) -> Optional[Dict]:
        """Get a gallery by ID with full details."""
        try:
            query = """
                SELECT 
                    g.*,
                    COUNT(DISTINCT gi.id) as image_count,
                    l.name as location_name,
                    o.name as organizer_name,
                    p.name as photographer_name
                FROM galleries g
                LEFT JOIN gallery_images gi ON g.id = gi.gallery_id
                LEFT JOIN locations l ON g.location_id = l.id
                LEFT JOIN organizers o ON g.organizer_id = o.id
                LEFT JOIN photographers p ON g.photographer_id = p.id
                WHERE g.id = $1
                GROUP BY g.id, l.name, o.name, p.name
            """
            result = await self.fetchone(query, gallery_id)
            return dict(result) if result else None
        except Exception as e:
            logger.error(f"[GalleriesClient] Error getting gallery {gallery_id}: {e}")
            return None
    
    async def create_gallery(self, data: Dict) -> Dict:
        """Create a new gallery."""
        try:
            gallery_url = data.get('gallery_url')
            if not gallery_url or gallery_url == '':
                # Use placeholder until we can generate proper slug from title and date
                gallery_url = '/gallery/pending'
            
            query = """
                INSERT INTO galleries (
                    title, shoot_date, gallery_url, cover_image_url, 
                    cover_image_square_url, photographer_id, location_id, 
                    organizer_id, sort_order, external_gallery_url, created_at
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, NOW())
                RETURNING *
            """
            result = await self.fetchone(
                query,
                data.get('title'),
                data.get('shoot_date'),
                gallery_url,  # Use generated gallery_url instead of None
                data.get('cover_image_url'),
                data.get('cover_image_square_url'),
                data.get('photographer_id'),
                data.get('location_id'),
                data.get('organizer_id'),
                data.get('sort_order', 'filename'),
                data.get('external_gallery_url')
            )
            logger.info(f"[GalleriesClient] Created gallery: {data.get('title')}")
            return dict(result)
        except Exception as e:
            logger.error(f"[GalleriesClient] Error creating gallery: {e}")
            raise
    
    async def update_gallery(self, gallery_id: str, data: Dict) -> Optional[Dict]:
        """Update a gallery's information."""
        try:
            fields = []
            values = []
            param_count = 1
            
            if 'title' in data:
                fields.append(f"title = ${param_count}")
                values.append(data['title'])
                param_count += 1
            
            if 'shoot_date' in data:
                fields.append(f"shoot_date = ${param_count}")
                values.append(data['shoot_date'])
                param_count += 1
            
            if 'gallery_url' in data:
                fields.append(f"gallery_url = ${param_count}")
                values.append(data['gallery_url'] or None)
                param_count += 1
            
            if 'cover_image_url' in data:
                fields.append(f"cover_image_url = ${param_count}")
                values.append(data['cover_image_url'])
                param_count += 1
            
            if 'cover_image_square_url' in data:
                fields.append(f"cover_image_square_url = ${param_count}")
                values.append(data['cover_image_square_url'])
                param_count += 1
                
            if 'photographer_id' in data:
                fields.append(f"photographer_id = ${param_count}")
                values.append(data['photographer_id'])
                param_count += 1
            
            if 'location_id' in data:
                fields.append(f"location_id = ${param_count}")
                values.append(data['location_id'])
                param_count += 1
            
            if 'organizer_id' in data:
                fields.append(f"organizer_id = ${param_count}")
                values.append(data['organizer_id'])
                param_count += 1
            
            if 'sort_order' in data:
                fields.append(f"sort_order = ${param_count}")
                values.append(data['sort_order'])
                param_count += 1
            
            if 'external_gallery_url' in data:
                fields.append(f"external_gallery_url = ${param_count}")
                values.append(data['external_gallery_url'])
                param_count += 1
            
            if not fields:
                return await self.get_gallery_by_id(gallery_id)
            
            fields.append("updated_at = NOW()")
            values.append(gallery_id)
            
            query = f"""
                UPDATE galleries 
                SET {', '.join(fields)}
                WHERE id = ${param_count}
                RETURNING *
            """
            result = await self.fetchone(query, *values)
            logger.info(f"[GalleriesClient] Updated gallery: {gallery_id}")
            return dict(result)
        except Exception as e:
            logger.error(f"[GalleriesClient] Error updating gallery {gallery_id}: {e}")
            raise
    
    async def delete_gallery(self, gallery_id: str) -> bool:
        """Delete a gallery and all its images."""
        try:
            await self.execute("DELETE FROM gallery_images WHERE gallery_id = $1", gallery_id)
            
            await self.execute("DELETE FROM galleries WHERE id = $1", gallery_id)
            
            logger.info(f"[GalleriesClient] Deleted gallery: {gallery_id}")
            return True
        except Exception as e:
            logger.error(f"[GalleriesClient] Error deleting gallery {gallery_id}: {e}")
            return False
    
    async def get_gallery_images(self, gallery_id: str) -> List[Dict]:
        """Get all images in a gallery."""
        try:
            logger.info(f"[GalleriesClient] Getting images for gallery {gallery_id}")
            
            query = """
                SELECT 
                    gi.*,
                    COUNT(DISTINCT pf.id) as face_count,
                    COUNT(DISTINCT CASE WHEN pf.person_id IS NOT NULL THEN pf.id END) as recognized_count
                FROM gallery_images gi
                LEFT JOIN photo_faces pf ON gi.id = pf.photo_id
                WHERE gi.gallery_id = $1
                GROUP BY gi.id
                ORDER BY gi.display_order, gi.created_at
            """
            
            results = await self.fetch(query, gallery_id)
            logger.info(f"[GalleriesClient] Found {len(results)} images")
            return results
        except Exception as e:
            logger.error(f"[GalleriesClient] Error getting gallery images: {e}")
            return []
    
    async def add_gallery_images(
        self, 
        gallery_id: str, 
        images: List[Dict]
    ) -> List[Dict]:
        """Add multiple images to a gallery."""
        try:
            added_images = []
            for img in images:
                query = """
                    INSERT INTO gallery_images (
                        gallery_id, image_url, original_url, 
                        original_filename, display_order, file_size,
                        width, height, has_been_processed, created_at
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, false, NOW())
                    RETURNING *
                """
                result = await self.fetchone(
                    query,
                    gallery_id,
                    img.get('image_url'),
                    img.get('original_url'),
                    img.get('original_filename'),
                    img.get('display_order', 0),
                    img.get('file_size'),
                    img.get('width'),
                    img.get('height')
                )
                added_images.append(dict(result))
            
            logger.info(f"[GalleriesClient] Added {len(added_images)} images to gallery {gallery_id}")
            return added_images
        except Exception as e:
            logger.error(f"[GalleriesClient] Error adding gallery images: {e}")
            raise
    
    async def delete_gallery_image(self, image_id: str, gallery_id: str) -> bool:
        """Delete an image from a gallery."""
        try:
            query = "DELETE FROM gallery_images WHERE id = $1 AND gallery_id = $2"
            await self.execute(query, image_id, gallery_id)
            logger.info(f"[GalleriesClient] Deleted image {image_id} from gallery {gallery_id}")
            return True
        except Exception as e:
            logger.error(f"[GalleriesClient] Error deleting gallery image: {e}")
            return False
