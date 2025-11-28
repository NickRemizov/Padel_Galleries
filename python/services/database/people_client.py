from typing import List, Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)

class PeopleClient:
    """Client for managing people (players) in the database."""
    
    async def get_all_people(self, include_stats: bool = True) -> List[Dict]:
        """Get all people with optional statistics."""
        try:
            if include_stats:
                query = """
                    SELECT 
                        p.*,
                        COUNT(DISTINCT pf.id) as photo_count,
                        COUNT(DISTINCT pf.photo_id) as gallery_count
                    FROM people p
                    LEFT JOIN photo_faces pf ON p.id = pf.person_id
                    GROUP BY p.id
                    ORDER BY p.real_name
                """
            else:
                query = "SELECT * FROM people ORDER BY real_name"
            
            results = await self.fetch(query)
            return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"[PeopleClient] Error getting all people: {e}")
            return []
    
    async def get_person_by_id(self, person_id: str) -> Optional[Dict]:
        """Get a person by ID with statistics."""
        try:
            query = """
                SELECT 
                    p.*,
                    COUNT(DISTINCT pf.id) as photo_count,
                    COUNT(DISTINCT pf.photo_id) as gallery_count,
                    AVG(pf.recognition_confidence) as avg_confidence
                FROM people p
                LEFT JOIN photo_faces pf ON p.id = pf.person_id
                WHERE p.id = $1
                GROUP BY p.id
            """
            result = await self.fetchone(query, person_id)
            return dict(result) if result else None
        except Exception as e:
            logger.error(f"[PeopleClient] Error getting person {person_id}: {e}")
            return None
    
    async def create_person(self, data: Dict) -> Dict:
        """Create a new person."""
        try:
            query = """
                INSERT INTO people (real_name, avatar_url, notes, created_at)
                VALUES ($1, $2, $3, NOW())
                RETURNING *
            """
            result = await self.fetchone(
                query,
                data.get('real_name') or data.get('name'),  # Support both for compatibility
                data.get('avatar_url'),
                data.get('notes')
            )
            logger.info(f"[PeopleClient] Created person: {data.get('real_name') or data.get('name')}")
            return dict(result)
        except Exception as e:
            logger.error(f"[PeopleClient] Error creating person: {e}")
            raise
    
    async def create_person_from_cluster(
        self,
        name: str,
        face_ids: List[str],
        avatar_url: Optional[str] = None,
        notes: Optional[str] = None
    ) -> Dict:
        """Create a person and assign multiple faces to them."""
        try:
            person_query = """
                INSERT INTO people (real_name, avatar_url, notes, created_at)
                VALUES ($1, $2, $3, NOW())
                RETURNING *
            """
            person = await self.fetchone(person_query, name, avatar_url, notes)
            person_id = person['id']
            
            if face_ids:
                update_query = """
                    UPDATE photo_faces
                    SET person_id = $1, verified = true, updated_at = NOW()
                    WHERE id = ANY($2::uuid[])
                """
                await self.execute(update_query, person_id, face_ids)
            
            logger.info(f"[PeopleClient] Created person from cluster: {name} with {len(face_ids)} faces")
            return dict(person)
        except Exception as e:
            logger.error(f"[PeopleClient] Error creating person from cluster: {e}")
            raise
    
    async def update_person(self, person_id: str, data: Dict) -> Optional[Dict]:
        """Update a person's information."""
        try:
            update_fields = []
            values = []
            param_count = 1
            
            if 'real_name' in data or 'name' in data:
                update_fields.append(f"real_name = ${param_count}")
                values.append(data.get('real_name') or data.get('name'))
                param_count += 1
            
            if 'avatar_url' in data:
                update_fields.append(f"avatar_url = ${param_count}")
                values.append(data['avatar_url'])
                param_count += 1
            
            if 'notes' in data:
                update_fields.append(f"notes = ${param_count}")
                values.append(data['notes'])
                param_count += 1
            
            if not update_fields:
                return await self.get_person_by_id(person_id)
            
            update_fields.append("updated_at = NOW()")
            values.append(person_id)
            
            query = f"""
                UPDATE people
                SET {', '.join(update_fields)}
                WHERE id = ${param_count}
                RETURNING *
            """
            
            result = await self.fetchone(query, *values)
            logger.info(f"[PeopleClient] Updated person: {person_id}")
            return dict(result) if result else None
        except Exception as e:
            logger.error(f"[PeopleClient] Error updating person {person_id}: {e}")
            raise
    
    async def delete_person(self, person_id: str) -> bool:
        """Delete a person and unassign their faces."""
        try:
            unassign_query = """
                UPDATE photo_faces
                SET person_id = NULL, verified = false
                WHERE person_id = $1
            """
            await self.execute(unassign_query, person_id)
            
            delete_query = "DELETE FROM people WHERE id = $1"
            await self.execute(delete_query, person_id)
            
            logger.info(f"[PeopleClient] Deleted person: {person_id}")
            return True
        except Exception as e:
            logger.error(f"[PeopleClient] Error deleting person {person_id}: {e}")
            return False
