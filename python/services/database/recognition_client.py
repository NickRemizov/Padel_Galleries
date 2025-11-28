from typing import Dict, Optional, List, Any
import logging
import json

logger = logging.getLogger(__name__)

class RecognitionClient:
    """Client for managing training configuration and recognition statistics."""
    
    async def get_recognition_config(self) -> Dict:
        """Get the recognition configuration."""
        try:
            query = "SELECT key, value FROM face_recognition_config"
            rows = await self.fetch(query)
            
            config = {}
            for row in rows:
                key = row['key']
                value = row['value']
                
                logger.info(f"[RecognitionClient] Key: {key}, Value: {value}, Type: {type(value)}")
                
                # If value is a string, try to parse it as JSON (handles old double-encoded data)
                if isinstance(value, str):
                    try:
                        value = json.loads(value)
                        logger.info(f"[RecognitionClient] Parsed string value for {key}: {value}, Type: {type(value)}")
                    except (json.JSONDecodeError, TypeError):
                        # If it's not valid JSON, keep it as string
                        pass
                
                config[key] = value
            
            # Build quality_filters from individual keys
            quality_filters = {
                "min_detection_score": config.get("min_detection_score", 0.7),
                "min_face_size": config.get("min_face_size", 80),
                "min_blur_score": config.get("min_blur_score", 100),
                "verified_threshold": config.get("verified_threshold", 0.95)
            }
            config["quality_filters"] = quality_filters
            
            logger.info(f"[RecognitionClient] Final config structure: {config}")
            
            # Set defaults if keys are missing
            return {
                "confidence_thresholds": config.get('confidence_thresholds', {
                    "low_data": 0.75,
                    "medium_data": 0.65,
                    "high_data": 0.55
                }),
                "quality_filters": quality_filters,
                "context_weight": config.get('context_weight', 0.1),
                "min_faces_per_person": config.get('min_faces_per_person', 3),
                "auto_retrain_threshold": config.get('auto_retrain_threshold', 25),
                "auto_retrain_percentage": config.get('auto_retrain_percentage', 0.1),
                "model_version": config.get('model_version', "v1.0"),
                "last_full_training": config.get('last_full_training'),
                "faces_since_last_training": config.get('faces_since_last_training', 0)
            }
            
        except Exception as e:
            logger.error(f"[RecognitionClient] Error getting recognition config: {e}")
            # Return defaults on error
            return {
                "confidence_thresholds": {
                    "low_data": 0.75,
                    "medium_data": 0.65,
                    "high_data": 0.55
                },
                "quality_filters": {
                    "min_detection_score": 0.7,
                    "min_face_size": 80,
                    "min_blur_score": 100,
                    "verified_threshold": 0.95
                },
                "context_weight": 0.1,
                "min_faces_per_person": 3,
                "auto_retrain_threshold": 25,
                "auto_retrain_percentage": 0.1,
                "model_version": "v1.0",
                "last_full_training": None,
                "faces_since_last_training": 0
            }
    
    async def update_config(self, key: str, value: Any) -> None:
        """Update a single configuration value."""
        try:
            value_json = json.dumps(value)
            
            query = """
                INSERT INTO face_recognition_config (key, value)
                VALUES ($1, $2)
                ON CONFLICT (key) DO UPDATE SET
                    value = EXCLUDED.value,
                    updated_at = NOW()
            """
            await self.execute(query, key, value_json)
            logger.info(f"[RecognitionClient] Updated config key: {key}")
            
        except Exception as e:
            logger.error(f"[RecognitionClient] Error updating config: {e}")
            raise
    
    async def update_recognition_config(self, settings: Dict):
        """Update the full recognition configuration."""
        await self.update_config("quality_filters", settings)
    
    async def get_training_statistics(self) -> Dict:
        """Get training session statistics."""
        try:
            query = """
                SELECT 
                    COUNT(*) as total_sessions,
                    AVG(total_faces) as avg_faces_per_session,
                    SUM(verified_faces) as total_verified_faces,
                    MAX(created_at) as last_session_date
                FROM training_sessions
            """
            result = await self.fetchone(query)
            return dict(result) if result else {}
        except Exception as e:
            logger.error(f"[RecognitionClient] Error getting training statistics: {e}")
            return {}
    
    async def create_training_session(self, session_data: Dict) -> str:
        """Create a new training session record."""
        try:
            query = """
                INSERT INTO training_sessions (
                    total_faces, verified_faces, training_duration_seconds, created_at
                )
                VALUES ($1, $2, $3, NOW())
                RETURNING id
            """
            result = await self.fetchone(
                query,
                session_data.get('total_faces', 0),
                session_data.get('verified_faces', 0),
                session_data.get('training_duration_seconds', 0)
            )
            session_id = str(result['id'])
            logger.info(f"[RecognitionClient] Created training session: {session_id}")
            return session_id
        except Exception as e:
            logger.error(f"[RecognitionClient] Error creating training session: {e}")
            raise
    
    async def update_training_session(self, session_id: str, updates: Dict):
        """Update a training session."""
        try:
            update_fields = []
            values = []
            param_count = 1
            
            for field in ['total_faces', 'verified_faces', 'training_duration_seconds']:
                if field in updates:
                    update_fields.append(f"{field} = ${param_count}")
                    values.append(updates[field])
                    param_count += 1
            
            if update_fields:
                values.append(session_id)
                query = f"""
                    UPDATE training_sessions
                    SET {', '.join(update_fields)}
                    WHERE id = ${param_count}
                """
                await self.execute(query, *values)
                logger.info(f"[RecognitionClient] Updated training session: {session_id}")
        except Exception as e:
            logger.error(f"[RecognitionClient] Error updating training session: {e}")
            raise
    
    async def get_recognition_stats(self) -> Dict:
        """Get overall recognition statistics."""
        try:
            query = """
                SELECT 
                    COUNT(DISTINCT pf.id) as total_faces,
                    COUNT(DISTINCT CASE WHEN pf.person_id IS NOT NULL THEN pf.id END) as recognized_faces,
                    COUNT(DISTINCT CASE WHEN pf.verified = true THEN pf.id END) as verified_faces,
                    COUNT(DISTINCT pf.person_id) as unique_people,
                    AVG(pf.recognition_confidence) as avg_confidence
                FROM photo_faces pf
            """
            result = await self.fetchone(query)
            return dict(result) if result else {}
        except Exception as e:
            logger.error(f"[RecognitionClient] Error getting recognition stats: {e}")
            return {}
    
    async def get_unverified_images(self, limit: int = 100) -> List[Dict]:
        """Get images with unverified faces."""
        try:
            query = """
                SELECT DISTINCT
                    gi.id,
                    gi.image_url,
                    gi.gallery_id,
                    g.title as gallery_title,
                    COUNT(pf.id) as unverified_count
                FROM gallery_images gi
                JOIN galleries g ON gi.gallery_id = g.id
                JOIN photo_faces pf ON gi.id = pf.photo_id
                WHERE pf.verified = false
                GROUP BY gi.id, gi.image_url, gi.gallery_id, g.title
                ORDER BY unverified_count DESC
                LIMIT $1
            """
            results = await self.fetch(query, limit)
            return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"[RecognitionClient] Error getting unverified images: {e}")
            return []
    
    async def get_unknown_faces_from_gallery(self, gallery_id: str) -> List[Dict]:
        """Get all unknown (unassigned) faces from a gallery."""
        try:
            query = """
                SELECT 
                    pf.id,
                    pf.photo_id,
                    pf.insightface_bbox,
                    pf.insightface_descriptor,
                    pf.recognition_confidence,
                    gi.image_url,
                    gi.original_url
                FROM photo_faces pf
                JOIN gallery_images gi ON pf.photo_id = gi.id
                WHERE gi.gallery_id = $1
                AND pf.person_id IS NULL
                ORDER BY gi.created_at
            """
            results = await self.fetch(query, gallery_id)
            return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"[RecognitionClient] Error getting unknown faces: {e}")
            return []
