from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
from services.postgres_client import db_client
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(tags=["config"])

class FullConfig(BaseModel):
    confidence_thresholds: Optional[Dict[str, float]] = None
    quality_filters: Optional[Dict[str, Any]] = None
    context_weight: Optional[float] = None
    min_faces_per_person: Optional[int] = None
    auto_retrain_threshold: Optional[int] = None
    auto_retrain_percentage: Optional[float] = None
    model_version: Optional[str] = None
    last_full_training: Optional[str] = None
    faces_since_last_training: Optional[int] = None

@router.get("/config")
async def get_config():
    """Get current recognition configuration"""
    logger.info("[Config] GET /config request received")
    try:
        logger.info("[Config] Fetching recognition config...")
        config = await db_client.get_recognition_config()
        logger.info(f"[Config] Config from DB: {config}")
        
        return JSONResponse(content=config)
        
    except Exception as e:
        logger.error(f"[Config] Error in get_config: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "connection_failed",
                "message": f"Internal Server Error: {str(e)}"
            }
        )

@router.put("/config")
async def update_config(config: FullConfig):
    """Update recognition configuration"""
    logger.info(f"[Config] PUT /config request received")
    try:
        config_dict = config.dict(exclude_none=True)
        logger.info(f"[Config] Fields to update: {list(config_dict.keys())}")
        
        if 'quality_filters' in config_dict:
            quality_filters = config_dict.pop('quality_filters')
            # Save each quality filter as a separate key in face_recognition_config
            for key, value in quality_filters.items():
                logger.info(f"[Config] Updating quality filter key '{key}' with value: {value}")
                await db_client.update_config(key, value)
        
        # Update remaining fields
        for key, value in config_dict.items():
            logger.info(f"[Config] Updating key '{key}' with value: {value}")
            await db_client.update_config(key, value)
        
        # Get updated config to return
        updated_config = await db_client.get_recognition_config()
        
        logger.info("[Config] Config saved successfully")
        return JSONResponse(content={"status": "ok", "config": updated_config})
        
    except Exception as e:
        logger.error(f"[Config] Error in update_config: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "connection_failed",
                "message": f"Internal Server Error: {str(e)}"
            }
        )
