from dotenv import load_dotenv
import os

load_dotenv()

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import uvicorn
from services.auth import get_current_user, get_current_user_optional, verify_google_token, create_access_token
import asyncio
import shutil
import re

from services.face_recognition import FaceRecognitionService
from services.training_service import TrainingService
from services.postgres_client import db_client  # Added db_client for connection pool initialization
from routers import training, recognition, config, faces, galleries, crud, s3_proxy, people  # Added people router

from models.schemas import (
    RecognitionResponse,
    GroupingResponse,
    PlayerGroup,
    FaceData,
    PhotoWithFaces
)

app = FastAPI(
    title="Padel Tournament Face Recognition API",
    description="API для распознавания и группировки игроков на турнирах по паделу",
    version="3.2.8"  # Updated version to 3.2.8 with metrics support (blur_score, distance_to_nearest, top_matches)
)

allowed_origins_str = os.getenv("ALLOWED_ORIGINS", "*")
allowed_origins = [origin.strip() for origin in allowed_origins_str.split(",")]

# Паттерн для Vercel preview URL (например: https://padelvalencia-abc123.vercel.app)
vercel_preview_pattern = re.compile(r"https://[a-zA-Z0-9-]+\.vercel\.app")

def is_origin_allowed(origin: str) -> bool:
    """Проверяет, разрешен ли origin"""
    # Разрешаем если origin в списке
    if origin in allowed_origins or "*" in allowed_origins:
        return True
    # Разрешаем все Vercel preview URL
    if vercel_preview_pattern.match(origin):
        return True
    return False

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Разрешаем все origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Инициализация сервиса распознавания
face_service = FaceRecognitionService()

training_service_instance = TrainingService()

training.set_training_service(training_service_instance)

@app.on_event("startup")
async def startup():
    """Initialize database connection pool on startup"""
    print("[Main] Initializing database connection pool...")
    await db_client.connect()
    print("[Main] Database connection pool ready")

@app.on_event("shutdown")
async def shutdown():
    """Close database connection pool on shutdown"""
    print("[Main] Closing database connection pool...")
    await db_client.disconnect()
    print("[Main] Database connection pool closed")

# Создаем директории для хранения данных
os.makedirs("uploads", exist_ok=True)
os.makedirs("static", exist_ok=True)

app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Простой веб-интерфейс для тестирования"""
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.post("/api/auth/google")
async def google_auth(google_token: str = Form(...)):
    """
    Аутентификация через Google OAuth
    
    - **google_token**: Google ID token от клиента
    """
    try:
        # Проверяем Google токен
        token_info = await verify_google_token(google_token)
        
        # Создаем наш JWT токен
        access_token = create_access_token(
            data={
                "sub": token_info.get("email"),
                "name": token_info.get("name"),
                "picture": token_info.get("picture")
            }
        )
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user": {
                "email": token_info.get("email"),
                "name": token_info.get("name"),
                "picture": token_info.get("picture")
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Ошибка аутентификации: {str(e)}")


@app.post("/api/players/add")
async def add_player(
    player_id: str = Form(...),
    name: str = Form(...),
    email: Optional[str] = Form(None),
    phone: Optional[str] = Form(None),
    notes: Optional[str] = Form(None),
    photos: List[UploadFile] = File(...),
    current_user: dict = Depends(get_current_user)
):
    """
    Добавление игрока в архив с фотографиями (требует аутентификации)
    
    - **player_id**: Уникальный ID игрока
    - **name**: Имя игрока
    - **email**: Email (опционально)
    - **phone**: Телефон (опционально)
    - **notes**: Заметки (опционально)
    - **photos**: Фотографии игрока для обучения
    """
    try:
        result = await face_service.add_player_to_archive(
            player_id, name, photos, email, phone, notes
        )
        return JSONResponse({
            "success": True,
            "message": f"Игрок {name} добавлен в архив",
            "data": result
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка добавления игрока: {str(e)}")


@app.get("/api/players/list")
async def list_players(current_user: dict = Depends(get_current_user)):
    """Получение списка всех игроков из архива"""
    try:
        players = face_service.db.get_all_players()
        return JSONResponse({
            "success": True,
            "players": players
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения списка: {str(e)}")


@app.post("/api/gallery/process")
async def process_gallery(
    gallery_id: str = Form(...),
    gallery_name: str = Form(...),
    event_date: Optional[str] = Form(None),
    location: Optional[str] = Form(None),
    files: List[UploadFile] = File(...),
    current_user: dict = Depends(get_current_user)
):
    """
    Массовая обработка галереи с распознаванием игроков (требует аутентификации)
    
    - **gallery_id**: ID галереи
    - **gallery_name**: Название события
    - **event_date**: Дата события (опционально)
    - **location**: Место проведения (опционально)
    - **files**: Фотографии для обработки
    """
    try:
        # Создаем галерею в базе
        face_service.db.create_gallery(
            gallery_id, gallery_name, event_date, location, len(files)
        )
        
        # Обрабатываем фотографии батчами
        result = await face_service.process_gallery_batch(gallery_id, files)
        
        return JSONResponse({
            "success": True,
            "message": f"Обработано {result['total_photos']} фото",
            "data": result
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки галереи: {str(e)}")


@app.get("/api/gallery/{gallery_id}/results")
async def get_gallery_results(
    gallery_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Получение результатов распознавания для галереи"""
    try:
        results = await face_service.get_gallery_results(gallery_id)
        return JSONResponse({
            "success": True,
            "data": results
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения результатов: {str(e)}")


@app.post("/api/upload-photos", response_model=RecognitionResponse)
async def upload_photos(
    files: List[UploadFile] = File(...),
    tournament_id: Optional[str] = Form(None),
    gallery_id: Optional[str] = Form(None),  # Added gallery_id parameter
    current_user: Optional[dict] = Depends(get_current_user_optional)
):
    """
    Загрузка фотографий турнира для распознавания лиц
    
    - **files**: Список фотографий турнира
    - **tournament_id**: ID турнира (опционально, deprecated - используйте gallery_id)
    - **gallery_id**: ID галереи (опционально)
    """
    print(f"[v0] Получен запрос на загрузку {len(files)} фото")
    
    if not files:
        print("[v0] ОШИБКА: Не загружено ни одного файла")
        raise HTTPException(status_code=400, detail="Не загружено ни одного файла")
    
    effective_id = gallery_id or tournament_id or f"tournament_{int(asyncio.get_event_loop().time() * 1000)}"
    
    tournament_dir = os.path.join("uploads", effective_id)
    os.makedirs(tournament_dir, exist_ok=True)
    
    saved_files = []
    for file in files:
        file_path = os.path.join(tournament_dir, file.filename)
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)
        saved_files.append(file_path)
        # Сбрасываем указатель файла для дальнейшей обработки
        await file.seek(0)
    
    print(f"[v0] Файлы сохранены в {tournament_dir}")
    
    try:
        print(f"[v0] Начинаем обработку {len(files)} фотографий...")
        print(f"[v0] Gallery/Tournament ID: {effective_id}")
        
        # Обрабатываем загруженные фото
        faces_data = await face_service.process_uploaded_photos(files, effective_id)
        
        print(f"[v0] Обработка завершена: найдено {len(faces_data)} лиц")
        
        return RecognitionResponse(
            success=True,
            message=f"Обработано {len(files)} фото, найдено {len(faces_data)} лиц",
            faces_count=len(faces_data),
            faces=faces_data,
            tournament_id=effective_id  # Return the effective ID used
        )
    
    except Exception as e:
        print(f"[v0] КРИТИЧЕСКАЯ ОШИБКА при обработке фото: {type(e).__name__}: {str(e)}")
        import traceback
        print(f"[v0] Traceback:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {str(e)}")


@app.post("/api/group-players", response_model=GroupingResponse)
async def group_players(
    tournament_id: Optional[str] = Form(None),
    gallery_id: Optional[str] = Form(None),  # Added gallery_id parameter
    min_cluster_size: int = Form(3),
    current_user: Optional[dict] = Depends(get_current_user_optional)
):
    """
    Автоматическая группировка игроков по лицам
    
    - **tournament_id**: ID турнира для группировки (deprecated - используйте gallery_id)
    - **gallery_id**: ID галереи для группировки
    - **min_cluster_size**: Минимальное количество фото для группы (по умолчанию 3)
    """
    effective_id = gallery_id or tournament_id
    print(f"[v0] Запрос на группировку игроков, gallery/tournament_id: {effective_id}, min_cluster_size: {min_cluster_size}")
    
    try:
        groups, ungrouped_faces = await face_service.group_faces(effective_id, min_cluster_size)
        
        for group in groups:
            for face in group.faces:
                face.player_name = group.player_name
        
        from collections import defaultdict
        
        photos_dict = defaultdict(list)
        
        # Add grouped faces
        for group in groups:
            for face in group.faces:
                photos_dict[face.image_name].append(face)
        
        # Add ungrouped faces
        for face in ungrouped_faces:
            face.player_name = "Неизвестный игрок"
            photos_dict[face.image_name].append(face)
        
        # Create PhotoWithFaces objects
        photos_with_faces = [
            PhotoWithFaces(image_name=img_name, faces=faces)
            for img_name, faces in photos_dict.items()
        ]
        
        print(f"[v0] Группировка завершена: найдено {len(groups)} групп и {len(ungrouped_faces)} одиночных лиц")
        print(f"[v0] Создано {len(photos_with_faces)} объектов PhotoWithFaces")
        
        return GroupingResponse(
            success=True,
            message=f"Найдено {len(groups)} групп игроков",
            groups_count=len(groups),
            groups=groups,
            tournament_id=effective_id,  # Return the effective ID used
            ungrouped_faces=ungrouped_faces,
            ungrouped_count=len(ungrouped_faces),
            photos_with_faces=photos_with_faces
        )
    
    except Exception as e:
        print(f"[v0] ОШИБКА группировки: {type(e).__name__}: {str(e)}")
        import traceback
        print(f"[v0] Traceback:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Ошибка группировки: {str(e)}")


@app.post("/api/feedback")
async def submit_feedback(
    face_id: str = Form(...),
    player_id: str = Form(...),
    confidence: float = Form(1.0),
    current_user: dict = Depends(get_current_user)
):
    """
    Отправка фидбека для улучшения распознавания (требует аутентификации)
    
    - **face_id**: ID лица для корректировки
    - **player_id**: Правильный ID игрока
    - **confidence**: Уверенность в корректировке (0-1)
    """
    try:
        user_email = current_user.get("sub")
        await face_service.apply_feedback(face_id, player_id, confidence, user_email)
        
        return JSONResponse({
            "success": True,
            "message": "Фидбек принят, модель обновлена"
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка применения фидбека: {str(e)}")


@app.get("/api/health")
async def health_check():
    """Проверка работоспособности сервиса"""
    return {
        "status": "healthy",
        "service": "padel-recognition",
        "model_loaded": face_service.is_ready()
    }


@app.delete("/api/clear-data")
async def clear_data(
    tournament_id: Optional[str] = None,
    current_user: Optional[dict] = Depends(get_current_user_optional)
):
    """Очистка данных турнира"""
    print(f"[v0] Запрос на очистку данных, tournament_id: {tournament_id}")
    
    try:
        await face_service.clear_tournament_data(tournament_id)
        
        if tournament_id:
            tournament_dir = os.path.join("uploads", tournament_id)
            if os.path.exists(tournament_dir):
                shutil.rmtree(tournament_dir)
                print(f"[v0] Папка {tournament_dir} удалена")
        else:
            # Удаляем все папки турниров
            if os.path.exists("uploads"):
                for item in os.listdir("uploads"):
                    item_path = os.path.join("uploads", item)
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                print("[v0] Все папки турниров удалены")
        
        print("[v0] Данные успешно очищены")
        return JSONResponse({
            "success": True,
            "message": "Данные очищены"
        })
    except Exception as e:
        print(f"[v0] ОШИБКА очистки: {type(e).__name__}: {str(e)}")
        import traceback
        print(f"[v0] Traceback:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Ошибка очистки: {str(e)}")


# Роутеры
print("[Main] Mounting routers...")
app.include_router(training.router, prefix="/api/v2", tags=["training"])
print("[Main] Mounted training router at /api/v2")
app.include_router(recognition.router, prefix="", tags=["recognition"])
print("[Main] Mounted recognition router at /")
app.include_router(config.router, prefix="/api/v2", tags=["config"])
print("[Main] Mounted config router at /api/v2")
app.include_router(faces.router, prefix="", tags=["faces"])
print("[Main] Mounted faces router at /")
app.include_router(galleries.router, prefix="", tags=["galleries"])
print("[Main] Mounted galleries router at /")
app.include_router(crud.router, prefix="/api/crud", tags=["CRUD"])
print("[Main] Mounted CRUD router at /api/crud")
app.include_router(people.router, prefix="", tags=["people"])
print("[Main] Mounted people router at /")
app.include_router(s3_proxy.router, prefix="/api/s3-proxy", tags=["S3 Proxy"])
print("[Main] Mounted S3 Proxy router at /api/s3-proxy")

if __name__ == "__main__":
    host = os.getenv("SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("SERVER_PORT", "8001"))
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True
    )
