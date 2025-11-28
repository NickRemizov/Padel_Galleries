from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class MetadataClient:
    """Client for managing photographers, locations, and organizers."""
    
    # Photographers
    async def get_all_photographers(self, include_stats: bool = True) -> List[Dict]:
        """Get all photographers with optional statistics."""
        try:
            if include_stats:
                query = """
                    SELECT 
                        p.*,
                        COUNT(DISTINCT g.id) as gallery_count
                    FROM photographers p
                    LEFT JOIN galleries g ON p.id = g.photographer_id
                    GROUP BY p.id
                    ORDER BY p.name
                """
            else:
                query = "SELECT * FROM photographers ORDER BY name"
            
            results = await self.fetch(query)
            return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"[MetadataClient] Error getting photographers: {e}")
            return []
    
    async def get_photographer_by_id(self, photographer_id: str) -> Optional[Dict]:
        """Get a photographer by ID."""
        try:
            query = "SELECT * FROM photographers WHERE id = $1"
            result = await self.fetchone(query, photographer_id)
            return dict(result) if result else None
        except Exception as e:
            logger.error(f"[MetadataClient] Error getting photographer: {e}")
            return None
    
    async def create_photographer(self, data: Dict) -> Dict:
        """Create a new photographer."""
        try:
            query = """
                INSERT INTO photographers (name, contact_info, notes, created_at)
                VALUES ($1, $2, $3, NOW())
                RETURNING *
            """
            result = await self.fetchone(
                query,
                data.get('name'),
                data.get('contact_info'),
                data.get('notes')
            )
            logger.info(f"[MetadataClient] Created photographer: {data.get('name')}")
            return dict(result)
        except Exception as e:
            logger.error(f"[MetadataClient] Error creating photographer: {e}")
            raise
    
    async def update_photographer(self, photographer_id: str, data: Dict) -> Optional[Dict]:
        """Update a photographer."""
        try:
            update_fields = []
            values = []
            param_count = 1
            
            for field in ['name', 'contact_info', 'notes']:
                if field in data:
                    update_fields.append(f"{field} = ${param_count}")
                    values.append(data[field])
                    param_count += 1
            
            if not update_fields:
                return await self.get_photographer_by_id(photographer_id)
            
            values.append(photographer_id)
            query = f"""
                UPDATE photographers
                SET {', '.join(update_fields)}, updated_at = NOW()
                WHERE id = ${param_count}
                RETURNING *
            """
            
            result = await self.fetchone(query, *values)
            return dict(result) if result else None
        except Exception as e:
            logger.error(f"[MetadataClient] Error updating photographer: {e}")
            raise
    
    async def delete_photographer(self, photographer_id: str) -> bool:
        """Delete a photographer."""
        try:
            await self.execute("DELETE FROM photographers WHERE id = $1", photographer_id)
            logger.info(f"[MetadataClient] Deleted photographer: {photographer_id}")
            return True
        except Exception as e:
            logger.error(f"[MetadataClient] Error deleting photographer: {e}")
            return False
    
    # Locations
    async def get_all_locations(self, include_stats: bool = True) -> List[Dict]:
        """Get all locations with optional statistics."""
        try:
            if include_stats:
                query = """
                    SELECT 
                        l.*,
                        COUNT(DISTINCT g.id) as gallery_count
                    FROM locations l
                    LEFT JOIN galleries g ON l.id = g.location_id
                    GROUP BY l.id
                    ORDER BY l.name
                """
            else:
                query = "SELECT * FROM locations ORDER BY name"
            
            results = await self.fetch(query)
            return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"[MetadataClient] Error getting locations: {e}")
            return []
    
    async def get_location_by_id(self, location_id: str) -> Optional[Dict]:
        """Get a location by ID."""
        try:
            query = "SELECT * FROM locations WHERE id = $1"
            result = await self.fetchone(query, location_id)
            return dict(result) if result else None
        except Exception as e:
            logger.error(f"[MetadataClient] Error getting location: {e}")
            return None
    
    async def create_location(self, data: Dict) -> Dict:
        """Create a new location."""
        try:
            query = """
                INSERT INTO locations (name, address, city, created_at)
                VALUES ($1, $2, $3, NOW())
                RETURNING *
            """
            result = await self.fetchone(
                query,
                data.get('name'),
                data.get('address'),
                data.get('city')
            )
            logger.info(f"[MetadataClient] Created location: {data.get('name')}")
            return dict(result)
        except Exception as e:
            logger.error(f"[MetadataClient] Error creating location: {e}")
            raise
    
    async def update_location(self, location_id: str, data: Dict) -> Optional[Dict]:
        """Update a location."""
        try:
            update_fields = []
            values = []
            param_count = 1
            
            for field in ['name', 'address', 'city']:
                if field in data:
                    update_fields.append(f"{field} = ${param_count}")
                    values.append(data[field])
                    param_count += 1
            
            if not update_fields:
                return await self.get_location_by_id(location_id)
            
            values.append(location_id)
            query = f"""
                UPDATE locations
                SET {', '.join(update_fields)}, updated_at = NOW()
                WHERE id = ${param_count}
                RETURNING *
            """
            
            result = await self.fetchone(query, *values)
            return dict(result) if result else None
        except Exception as e:
            logger.error(f"[MetadataClient] Error updating location: {e}")
            raise
    
    async def delete_location(self, location_id: str) -> bool:
        """Delete a location."""
        try:
            await self.execute("DELETE FROM locations WHERE id = $1", location_id)
            logger.info(f"[MetadataClient] Deleted location: {location_id}")
            return True
        except Exception as e:
            logger.error(f"[MetadataClient] Error deleting location: {e}")
            return False
    
    # Organizers
    async def get_all_organizers(self, include_stats: bool = True) -> List[Dict]:
        """Get all organizers with optional statistics."""
        try:
            if include_stats:
                query = """
                    SELECT 
                        o.*,
                        COUNT(DISTINCT g.id) as gallery_count
                    FROM organizers o
                    LEFT JOIN galleries g ON o.id = g.organizer_id
                    GROUP BY o.id
                    ORDER BY o.name
                """
            else:
                query = "SELECT * FROM organizers ORDER BY name"
            
            results = await self.fetch(query)
            return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"[MetadataClient] Error getting organizers: {e}")
            return []
    
    async def get_organizer_by_id(self, organizer_id: str) -> Optional[Dict]:
        """Get an organizer by ID."""
        try:
            query = "SELECT * FROM organizers WHERE id = $1"
            result = await self.fetchone(query, organizer_id)
            return dict(result) if result else None
        except Exception as e:
            logger.error(f"[MetadataClient] Error getting organizer: {e}")
            return None
    
    async def create_organizer(self, data: Dict) -> Dict:
        """Create a new organizer."""
        try:
            query = """
                INSERT INTO organizers (name, contact_info, notes, created_at)
                VALUES ($1, $2, $3, NOW())
                RETURNING *
            """
            result = await self.fetchone(
                query,
                data.get('name'),
                data.get('contact_info'),
                data.get('notes')
            )
            logger.info(f"[MetadataClient] Created organizer: {data.get('name')}")
            return dict(result)
        except Exception as e:
            logger.error(f"[MetadataClient] Error creating organizer: {e}")
            raise
    
    async def update_organizer(self, organizer_id: str, data: Dict) -> Optional[Dict]:
        """Update an organizer."""
        try:
            update_fields = []
            values = []
            param_count = 1
            
            for field in ['name', 'contact_info', 'notes']:
                if field in data:
                    update_fields.append(f"{field} = ${param_count}")
                    values.append(data[field])
                    param_count += 1
            
            if not update_fields:
                return await self.get_organizer_by_id(organizer_id)
            
            values.append(organizer_id)
            query = f"""
                UPDATE organizers
                SET {', '.join(update_fields)}, updated_at = NOW()
                WHERE id = ${param_count}
                RETURNING *
            """
            
            result = await self.fetchone(query, *values)
            return dict(result) if result else None
        except Exception as e:
            logger.error(f"[MetadataClient] Error updating organizer: {e}")
            raise
    
    async def delete_organizer(self, organizer_id: str) -> bool:
        """Delete an organizer."""
        try:
            await self.execute("DELETE FROM organizers WHERE id = $1", organizer_id)
            logger.info(f"[MetadataClient] Deleted organizer: {organizer_id}")
            return True
        except Exception as e:
            logger.error(f"[MetadataClient] Error deleting organizer: {e}")
            return False
