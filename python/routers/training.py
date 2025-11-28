from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict
from datetime import datetime

from services.training_service import TrainingService
from services.postgres_client import db_client

router = APIRouter()
training_service = None  # Will be set by main.py

def set_training_service(service: TrainingService):
    """Set training service instance from main.py"""
    global training_service
    training_service = service

# Pydantic models for validation

class TrainingFilters(BaseModel):
    event_ids: Optional[List[str]] = None
    person_ids: Optional[List[str]] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None


class TrainingOptions(BaseModel):
    min_faces_per_person: int = 3
    include_co_occurring: bool = False
    context_weight: float = 0.1


class PrepareRequest(BaseModel):
    filters: TrainingFilters
    options: TrainingOptions


class ExecuteRequest(BaseModel):
    mode: str  # 'full' or 'incremental'
    filters: TrainingFilters
    options: TrainingOptions
    model_version: Optional[str] = 'v1.0'
    update_existing: bool = True


class ConfigUpdate(BaseModel):
    confidence_thresholds: Optional[Dict[str, float]] = None
    context_weight: Optional[float] = None
    min_faces_per_person: Optional[int] = None
    auto_retrain_threshold: Optional[int] = None
    auto_retrain_percentage: Optional[float] = None
    quality_filters: Optional[Dict[str, float]] = None



# Endpoints

@router.post("/train/prepare")
async def prepare_training(request: PrepareRequest):
    """
    Подготовка датасета для обучения (без запуска обучения).
    
    Возвращает статистику и валидацию датасета.
    """
    try:
        result = await training_service.prepare_dataset(
            filters=request.filters.dict(),
            options=request.options.dict()
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train/execute")
async def execute_training(
    request: ExecuteRequest,
    background_tasks: BackgroundTasks
):
    """
    Запуск обучения модели.
    
    Обучение выполняется в фоне. Возвращает session_id для отслеживания прогресса.
    """
    try:
        # Validate mode
        if request.mode not in ['full', 'incremental']:
            raise HTTPException(
                status_code=400,
                detail="mode must be 'full' or 'incremental'"
            )
        
        # Create session
        session_id = await training_service.execute_training(
            mode=request.mode,
            filters=request.filters.dict(),
            options={
                **request.options.dict(),
                'model_version': request.model_version,
                'update_existing': request.update_existing
            }
        )
        
        # Start training in background
        background_tasks.add_task(
            training_service._train_background,
            session_id=session_id,
            mode=request.mode,
            filters=request.filters.dict(),
            options={
                **request.options.dict(),
                'model_version': request.model_version,
                'update_existing': request.update_existing
            }
        )
        
        return {
            'session_id': session_id,
            'status': 'running',
            'message': 'Training started in background'
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/train/status/{session_id}")
async def get_training_status(session_id: str):
    """
    Получить статус обучения по session_id.
    """
    try:
        status = training_service.get_training_status(session_id)
        if 'error' in status:
            raise HTTPException(status_code=404, detail=status['error'])
        return status
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/train/history")
async def get_training_history(limit: int = 10, offset: int = 0):
    """
    Получить историю обучений.
    """
    try:
        history = training_service.get_training_history(limit, offset)
        return history
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




@router.get("/statistics")
async def get_training_statistics():
    """
    Получить статистику обученных данных.
    
    Возвращает количество людей с дескрипторами, общее количество лиц и т.д.
    """
    try:
        if training_service is None:
            return {
                "people_count": 0,
                "total_faces": 0,
                "unique_photos": 0,
                "error": "Training service not initialized"
            }
        
        await db_client.connect()
        stats = await training_service.get_training_statistics()
        return stats
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"[ERROR] Statistics endpoint failed: {str(e)}")
        print(f"[ERROR] Traceback:\n{traceback.format_exc()}")
        return {
            "people_count": 0,
            "total_faces": 0,
            "unique_photos": 0,
            "error": str(e)
        }
