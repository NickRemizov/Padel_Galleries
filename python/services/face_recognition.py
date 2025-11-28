import insightface
from insightface.app import FaceAnalysis
import numpy as np
import cv2
import hnswlib
import hdbscan
import json
from typing import List, Dict, Optional, Tuple
from fastapi import UploadFile
import io
from PIL import Image
import uuid
import pickle
import os
from concurrent.futures import ThreadPoolExecutor
import asyncio
import zipfile
from pathlib import Path

import models.schemas
from services.postgres_client import db_client  # Using only PostgresClient now


class FaceRecognitionService:
    def __init__(self):
        """Инициализация сервиса распознавания лиц"""
        print("[v0] Инициализация FaceRecognitionService...")
        self.app = None
        
        # Хранилище эмбеддингов и данных (для временных операций)
        self.embeddings_store: Dict[str, List[np.ndarray]] = {}
        self.faces_data_store: Dict[str, List[models.schemas.FaceData]] = {}
        self.index_store: Dict[str, hnswlib.Index] = {}
        
        self.players_index = None
        self.player_ids_map = []
        
        self.quality_filters = {
            "min_detection_score": 0.7,
            "min_face_size": 80,
            "min_blur_score": 80,
            "verified_threshold": 0.6  # Threshold for verified faces
        }
        
        print("[v0] Using PostgreSQL client for face recognition")
        
        print("[v0] Сервис распознавания лиц создан (ожидает инициализации)")
    
    def _ensure_model_unpacked(self):
        """Проверка и распаковка модели antelopev2 если нужно"""
        home_dir = Path.home()
        model_dir = home_dir / ".insightface" / "models" / "antelopev2"
        model_zip = home_dir / ".insightface" / "models" / "antelopev2.zip"
        
        print(f"[v0] Проверка модели antelopev2...")
        print(f"[v0] Путь к модели: {model_dir}")
        print(f"[v0] Путь к архиву: {model_zip}")
        
        # Проверяем существует ли распакованная модель
        if model_dir.exists():
            files = list(model_dir.glob("*.onnx"))
            print(f"[v0] Папка модели существует, найдено .onnx файлов: {len(files)}")
            if len(files) > 0:
                print(f"[v0] Модель уже распакована: {[f.name for f in files]}")
                return True
            else:
                print(f"[v0] Папка модели пуста, нужна распаковка")
        else:
            print(f"[v0] Папка модели не существует")
        
        # Проверяем существует ли архив
        if not model_zip.exists():
            print(f"[v0] ОШИБКА: Архив модели не найден: {model_zip}")
            print(f"[v0] InsightFace должен был скачать его автоматически")
            return False
        
        print(f"[v0] Архив найден, размер: {model_zip.stat().st_size} байт")
        print(f"[v0] Начинаем распаковку модели...")
        
        try:
            # Создаем директорию если не существует
            model_dir.mkdir(parents=True, exist_ok=True)
            print(f"[v0] Директория создана: {model_dir}")
            
            # Распаковываем архив
            with zipfile.ZipFile(model_zip, 'r') as zip_ref:
                file_list = zip_ref.namelist()
                print(f"[v0] Файлов в архиве: {len(file_list)}")
                print(f"[v0] Список файлов: {file_list[:10]}")  # Первые 10 файлов
                
                zip_ref.extractall(model_dir)
                print(f"[v0] Архив распакован в {model_dir}")
            
            subfolder = model_dir / "antelopev2"
            if subfolder.exists() and subfolder.is_dir():
                print(f"[v0] Найдена подпапка antelopev2/, перемещаем файлы в корень...")
                import shutil
                
                # Перемещаем все файлы из подпапки в корень
                for item in subfolder.iterdir():
                    if item.is_file():
                        dest = model_dir / item.name
                        print(f"[v0] Перемещаем {item.name} -> {dest}")
                        shutil.move(str(item), str(dest))
                
                # Удаляем пустую подпапку
                subfolder.rmdir()
                print(f"[v0] Подпапка удалена, файлы перемещены в корень")
            
            # Проверяем что файлы распаковались
            onnx_files = list(model_dir.glob("*.onnx"))
            print(f"[v0] После распаковки найдено .onnx файлов: {len(onnx_files)}")
            print(f"[v0] Файлы: {[f.name for f in onnx_files]}")
            
            if len(onnx_files) == 0:
                print(f"[v0] ОШИБКА: После распаковки не найдено .onnx файлов!")
                # Проверяем что вообще распаковалось
                all_files = list(model_dir.rglob("*"))
                print(f"[v0] Всего файлов в директории: {len(all_files)}")
                for f in all_files[:20]:  # Первые 20 файлов
                    print(f"[v0]   - {f.relative_to(model_dir)}")
                return False
            
            print(f"[v0] Модель успешно распакована!")
            return True
            
        except Exception as e:
            print(f"[v0] ОШИБКА при распаковке модели: {type(e).__name__}: {str(e)}")
            import traceback
            print(f"[v0] Traceback:\n{traceback.format_exc()}")
            return False
    
    def _ensure_initialized(self):
        """Ленивая инициализация InsightFace при первом использовании"""
        if self.app is None:
            print("[v0] ========== НАЧАЛО ИНИЦИАЛИЗАЦИИ INSIGHTFACE ==========")
            
            try:
                home_dir = Path.home()
                model_dir = home_dir / ".insightface" / "models" / "antelopev2"
                model_zip = home_dir / ".insightface" / "models" / "antelopev2.zip"
                
                print(f"[v0] Шаг 1: Проверка существования модели...")
                print(f"[v0] Путь к модели: {model_dir}")
                print(f"[v0] Путь к архиву: {model_zip}")
                
                model_ready = False
                if model_dir.exists():
                    onnx_files = list(model_dir.glob("*.onnx"))
                    if len(onnx_files) > 0:
                        print(f"[v0] ✓ Модель уже распакована: {len(onnx_files)} .onnx файлов")
                        model_ready = True
                    else:
                        print(f"[v0] Папка модели существует, но .onnx файлов нет")
                else:
                    print(f"[v0] Папка модели не существует")
                
                if not model_ready:
                    print(f"[v0] Шаг 2: Проверка архива модели...")
                    
                    # Если архив уже есть - распаковываем
                    if model_zip.exists():
                        print(f"[v0] Архив найден, распаковываем...")
                        if not self._ensure_model_unpacked():
                            raise RuntimeError("Не удалось распаковать существующий архив модели")
                        model_ready = True
                    else:
                        print(f"[v0] Архив не найден, InsightFace должен скачать его")
                        
                        print(f"[v0] Создание временного FaceAnalysis для скачивания модели...")
                        try:
                            temp_app = FaceAnalysis(
                                name='antelopev2',
                                providers=['CPUExecutionProvider']
                            )
                            print(f"[v0] Временный FaceAnalysis создан")
                            del temp_app  # Удаляем временный объект
                        except Exception as e:
                            print(f"[v0] Ошибка при создании временного FaceAnalysis: {e}")
                        
                        import time
                        max_wait = 30  # Максимум 30 секунд ожидания
                        waited = 0
                        while not model_zip.exists() and waited < max_wait:
                            print(f"[v0] Ожидание скачивания архива... ({waited}s)")
                            time.sleep(2)
                            waited += 2
                        
                        if not model_zip.exists():
                            raise RuntimeError(f"InsightFace не скачал архив модели после {max_wait}s ожидания")
                        
                        print(f"[v0] ✓ Архив скачан InsightFace")
                        
                        print(f"[v0] Распаковка скачанного архива...")
                        if not self._ensure_model_unpacked():
                            raise RuntimeError("Не удалось распаковать скачанный архив")
                        model_ready = True
                
                if not model_ready:
                    raise RuntimeError("Модель не готова после всех попыток инициализации")
                
                print(f"[v0] Шаг 3: Модель готова, создание FaceAnalysis...")
                
                self.app = FaceAnalysis(
                    name='antelopev2',
                    providers=['CPUExecutionProvider']
                )
                print(f"[v0] ✓ FaceAnalysis создан успешно")
                
                # Шаг 4: Загружаем модели в память
                print("[v0] Шаг 4: Загрузка моделей в память (prepare)...")
                self.app.prepare(ctx_id=-1, det_size=(640, 640))
                print("[v0] ✓ prepare() завершен успешно")
                
                # Шаг 5: Проверяем что модели загружены
                print("[v0] Шаг 5: Проверка загруженных моделей...")
                if hasattr(self.app, 'models'):
                    models_list = list(self.app.models.keys())
                    print(f"[v0] ✓ Загружены модели: {models_list}")
                    
                    if 'detection' not in models_list:
                        print(f"[v0] ✗ ОШИБКА: модель 'detection' не найдена!")
                        print(f"[v0] Проверяем содержимое директории модели...")
                        if model_dir.exists():
                            all_files = list(model_dir.glob("*"))
                            print(f"[v0] Файлы в {model_dir}:")
                            for f in all_files:
                                print(f"[v0]   - {f.name} ({f.stat().st_size} байт)")
                        raise RuntimeError("Модель 'detection' не загружена")
                else:
                    print(f"[v0] Предупреждение: app.models недоступен")
                
                # Шаг 6: Загружаем индекс игроков
                print("[v0] Шаг 6: Загрузка индекса игроков...")
                self._load_players_index()
                
                print("[v0] ========== INSIGHTFACE ГОТОВ К РАБОТЕ ==========")
                
            except Exception as e:
                print(f"[v0] ========== КРИТИЧЕСКАЯ ОШИБКА ИНИЦИАЛИЗАЦИИ ==========")
                print(f"[v0] Тип ошибки: {type(e).__name__}")
                print(f"[v0] Сообщение: {str(e)}")
                import traceback
                print(f"[v0] Полный traceback:\n{traceback.format_exc()}")
                print(f"[v0] =========================================================")
                raise
    
    def _load_players_index(self):
        """
        Загрузка индекса известных игроков из PostgreSQL.
        Приоритет: PostgreSQL, затем файлы на диске.
        """
        print("[v0] Загрузка индекса игроков...")
        
        try:
            print("[v0] Loading embeddings from PostgreSQL (both photo_faces and face_descriptors tables)...")
            player_ids, embeddings = db_client.get_all_player_embeddings()
            
            if len(embeddings) > 0:
                print(f"[v0] ✓ Loaded {len(embeddings)} embeddings from PostgreSQL")
                print(f"[v0] Unique people: {len(set(player_ids))}")
                print(f"[v0] Sample person IDs: {list(set(player_ids))[:5]}")
                
                # Create HNSW index
                dim = len(embeddings[0])
                self.players_index = hnswlib.Index(space='cosine', dim=dim)
                self.players_index.init_index(
                    max_elements=len(embeddings) * 2,
                    ef_construction=200,
                    M=16
                )
                
                embeddings_array = np.array(embeddings)
                self.players_index.add_items(embeddings_array, np.arange(len(embeddings)))
                self.players_index.set_ef(50)
                
                self.player_ids_map = player_ids
                print(f"[v0] ✓ HNSW index created: {len(embeddings)} embeddings for {len(set(player_ids))} unique people")
                return
            else:
                print("[v0] No embeddings found in PostgreSQL, trying fallback to disk...")
        except Exception as e:
            print(f"[v0] ERROR loading from PostgreSQL: {e}")
            import traceback
            print(f"[v0] Traceback:\n{traceback.format_exc()}")
            print("[v0] Falling back to disk...")
        
        models_dir = '/home/nickr/python/models'
        index_path = os.path.join(models_dir, 'players_index.bin')
        map_path = os.path.join(models_dir, 'player_ids_map.json')
        
        if os.path.exists(index_path) and os.path.exists(map_path):
            try:
                print(f"[v0] Loading index from disk: {index_path}")
                
                # Load player IDs map
                with open(map_path, 'r') as f:
                    self.player_ids_map = json.load(f)
                
                # Load HNSSWLIB index
                self.players_index = hnswlib.Index(space='cosine', dim=512)
                self.players_index.load_index(index_path)
                self.players_index.set_ef(50)
                
                print(f"[v0] Loaded index from disk: {len(self.player_ids_map)} embeddings for {len(set(self.player_ids_map))} players")
                return
                
            except Exception as e:
                print(f"[v0] Failed to load index from disk: {e}")
                print(f"[v0] Index will remain empty")
        
        # If we reach here, no index was loaded
        print("[v0] База игроков пуста - индекс не создан")
        self.players_index = None
        self.player_ids_map = []
    
    def is_ready(self) -> bool:
        """Проверка готовности сервиса"""
        self._ensure_initialized()
        return self.app is not None
    
    async def add_player_to_archive(
        self,
        player_id: str,
        name: str,
        photos: List[UploadFile],
        email: str = None,
        phone: str = None,
        notes: str = None
    ) -> Dict:
        """Добавление игрока в архив с фотографиями"""
        self._ensure_initialized()
        
        # Добавляем игрока в базу
        self.db.add_player(player_id, name, email, phone, notes)
        
        # Обрабатываем фотографии игрока
        embeddings_added = 0
        
        for photo in photos:
            contents = await photo.read()
            image = Image.open(io.BytesIO(contents))
            img_array = np.array(image.convert('RGB'))
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # Детектируем лица
            faces = self.app.get(img_array)
            
            # Берем самое большое лицо (предполагаем, что это игрок)
            if len(faces) > 0:
                face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                embedding = face.embedding
                
                embedding_id = str(uuid.uuid4())
                self.db.add_player_embedding(
                    embedding_id,
                    player_id,
                    embedding,
                    photo.filename,
                    float(face.det_score),
                    is_verified=True
                )
                embeddings_added += 1
        
        # Перезагружаем индекс игроков
        self._load_players_index()
        
        return {
            "player_id": player_id,
            "name": name,
            "embeddings_added": embeddings_added
        }
    
    async def process_gallery_batch(
        self,
        gallery_id: str,
        files: List[UploadFile],
        batch_size: int = 10
    ) -> Dict:
        """Пакетная обработка галереи фотографий с распознаванием игроков"""
        self._ensure_initialized()
        
        total_faces = 0
        recognized_faces = 0
        
        for i in range(0, len(files), batch_size):
            batch = files[i:i + batch_size]
            
            for file in batch:
                contents = await file.read()
                image = Image.open(io.BytesIO(contents))
                img_array = np.array(image.convert('RGB'))
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                
                # Детектируем лица
                faces = self.app.get(img_array)
                total_faces += len(faces)
                
                for face in faces:
                    face_id = str(uuid.uuid4())
                    embedding = face.embedding
                    bbox = [float(x) for x in face.bbox]
                    
                    # Ищем игрока в базе
                    player_id = None
                    confidence = 0.0
                    
                    if self.players_index is not None:
                        labels, distances = self.players_index.knn_query(
                            embedding.reshape(1, -1),
                            k=1
                        )
                        
                        similarity = 1 - distances[0][0]
                        if similarity > 0.6:  # 60% схожести
                            player_id = self.player_ids_map[labels[0][0]]
                            confidence = float(similarity)
                            recognized_faces += 1
                    
                    # Сохраняем лицо в базу
                    self.db.add_gallery_face(
                        face_id,
                        gallery_id,
                        file.filename,
                        bbox,
                        embedding,
                        player_id,
                        confidence
                    )
            
            print(f"[v0] Обработано {min(i + batch_size, len(files))}/{len(files)} фото")
        
        return {
            "total_photos": len(files),
            "total_faces": total_faces,
            "recognized_faces": recognized_faces,
            "recognition_rate": (recognized_faces / total_faces * 100) if total_faces > 0 else 0
        }
    
    async def process_uploaded_photos(
        self, 
        files: List[UploadFile], 
        tournament_id: Optional[str]
    ) -> List[models.schemas.FaceData]:
        """Обработка загруженных фотографий и извлечение лиц"""
        print(f"[v0] process_uploaded_photos вызван с {len(files)} файлами")
        
        try:
            self._ensure_initialized()
            print("[v0] InsightFace инициализирован успешно")
        except Exception as e:
            print(f"[v0] ОШИБКА инициализации InsightFace: {str(e)}")
            raise
        
        if not tournament_id:
            tournament_id = str(uuid.uuid4())
            print(f"[v0] Создан новый tournament_id: {tournament_id}")
        
        all_faces = []
        embeddings = []
        
        for idx, file in enumerate(files):
            print(f"[v0] Обработка файла {idx+1}/{len(files)}: {file.filename}")
            
            try:
                # Читаем изображение
                print(f"[v0] Чтение содержимого файла {file.filename}...")
                contents = await file.read()
                print(f"[v0] Прочитано {len(contents)} байт")
                
                print(f"[v0] Открытие изображения через PIL...")
                image = Image.open(io.BytesIO(contents))
                print(f"[v0] Изображение открыто: размер {image.size}, режим {image.mode}")
                
                print(f"[v0] Конвертация в numpy array...")
                img_array = np.array(image.convert('RGB'))
                print(f"[v0] Numpy array создан: shape {img_array.shape}")
                
                print(f"[v0] Конвертация RGB -> BGR для OpenCV...")
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                
                # Детектируем лица
                print(f"[v0] Запуск детекции лиц для {file.filename}...")
                faces = self.app.get(img_array)
                print(f"[v0] Найдено {len(faces)} лиц в {file.filename}")
                
                for face_idx, face in enumerate(faces):
                    print(f"[v0] Обработка лица {face_idx+1}/{len(faces)} из {file.filename}")
                    face_id = str(uuid.uuid4())
                    embedding = face.embedding
                    print(f"[v0] Эмбеддинг извлечен: размерность {len(embedding)}, det_score={face.det_score}")
                    
                    # Сохраняем данные лица
                    face_data = models.schemas.FaceData(
                        face_id=face_id,
                        image_name=file.filename,
                        bbox=[float(x) for x in face.bbox],
                        confidence=float(face.det_score),
                        embedding=embedding.tolist()
                    )
                    
                    all_faces.append(face_data)
                    embeddings.append(embedding)
                    print(f"[v0] Лицо {face_id} добавлено в результаты")
                
            except Exception as e:
                print(f"[v0] ОШИБКА при обработке файла {file.filename}: {type(e).__name__}: {str(e)}")
                import traceback
                print(f"[v0] Traceback:\n{traceback.format_exc()}")
                # Продолжаем обработку остальных файлов
                continue
        
        print(f"[v0] Всего обработано лиц: {len(all_faces)}")
        
        # Сохраняем в хранилище
        if tournament_id not in self.embeddings_store:
            self.embeddings_store[tournament_id] = []
            self.faces_data_store[tournament_id] = []
            print(f"[v0] Создано новое хранилище для tournament_id: {tournament_id}")
        
        self.embeddings_store[tournament_id].extend(embeddings)
        self.faces_data_store[tournament_id].extend(all_faces)
        print(f"[v0] Данные сохранены в хранилище: {len(embeddings)} эмбеддингов")
        
        # Создаем/обновляем HNSW индекс для быстрого поиска
        print(f"[v0] Построение HNSW индекса...")
        await self._build_hnsw_index(tournament_id)
        print(f"[v0] HNSW индекс построен")
        
        print(f"[v0] process_uploaded_photos завершен: возвращаем {len(all_faces)} лиц")
        return all_faces
    
    async def _build_hnsw_index(self, tournament_id: str):
        """Построение HNSW индекса для быстрого поиска похожих лиц"""
        embeddings = self.embeddings_store[tournament_id]
        
        if len(embeddings) == 0:
            return
        
        dim = len(embeddings[0])
        num_elements = len(embeddings)
        
        # Создаем индекс
        index = hnswlib.Index(space='cosine', dim=dim)
        index.init_index(max_elements=num_elements * 2, ef_construction=200, M=16)
        
        # Добавляем эмбеддинги
        embeddings_array = np.array(embeddings)
        index.add_items(embeddings_array, np.arange(num_elements))
        
        # Настраиваем параметры поиска
        index.set_ef(50)
        
        self.index_store[tournament_id] = index
        print(f"[v0] HNSW индекс построен для {num_elements} лиц")
    
    async def group_faces(
        self, 
        tournament_id: Optional[str],
        min_cluster_size: int = 3
    ) -> Tuple[List[models.schemas.PlayerGroup], List[models.schemas.FaceData]]:
        """Группировка лиц с помощью HDBSCAN"""
        print(f"[v0] group_faces вызван: tournament_id={tournament_id}, min_cluster_size={min_cluster_size}")
        
        if not tournament_id or tournament_id not in self.embeddings_store:
            print(f"[v0] ОШИБКА: Турнир не найден или нет данных")
            print(f"[v0] Доступные tournament_id: {list(self.embeddings_store.keys())}")
            raise ValueError("Турнир не найден или нет данных для группировки")
        
        embeddings = np.array(self.embeddings_store[tournament_id])
        faces_data = self.faces_data_store[tournament_id]
        print(f"[v0] Загружено {len(embeddings)} эмбеддингов для группировки")
        
        if len(embeddings) < min_cluster_size:
            print(f"[v0] ОШИБКА: Недостаточно лиц ({len(embeddings)} < {min_cluster_size})")
            raise ValueError(f"Недостаточно лиц для группировки (минимум {min_cluster_size})")
        
        print(f"[v0] Запуск HDBSCAN кластеризации...")
        
        # Применяем HDBSCAN для автоматической кластеризации
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=2,
            metric='euclidean',
            cluster_selection_epsilon=0.5
        )
        
        cluster_labels = clusterer.fit_predict(embeddings)
        print(f"[v0] HDBSCAN завершен: найдено {len(set(cluster_labels))} уникальных меток")
        
        # Группируем лица по кластерам
        groups_dict: Dict[int, List[models.schemas.FaceData]] = {}
        ungrouped_faces: List[models.schemas.FaceData] = []
        noise_count = 0
        
        for face_data, label in zip(faces_data, cluster_labels):
            if label == -1:  # Шум (не вошло ни в одну группу)
                noise_count += 1
                ungrouped_faces.append(face_data)
                continue
            
            if label not in groups_dict:
                groups_dict[label] = []
            
            groups_dict[label].append(face_data)
        
        print(f"[v0] Шум (не сгруппировано): {noise_count} лиц")
        print(f"[v0] Найдено групп: {len(groups_dict)}")
        
        # Формируем результат
        groups = []
        for cluster_id, faces in groups_dict.items():
            print(f"[v0] Группа {cluster_id}: {len(faces)} лиц")
            
            # Получаем уникальные имена изображений для превью (максимум 5)
            unique_images = list(set([f.image_name for f in faces]))[:5]
            
            group = models.schemas.PlayerGroup(
                group_id=f"player_{cluster_id}",
                player_name=f"Игрок {cluster_id + 1}",
                faces_count=len(faces),
                faces=faces,
                confidence=float(np.mean([f.confidence for f in faces])),
                sample_images=unique_images
            )
            groups.append(group)
        
        # Сохраняем несгруппированные лица для последующего извлечения
        self.faces_data_store[f"{tournament_id}_ungrouped"] = ungrouped_faces
        
        print(f"[v0] group_faces завершен: возвращаем {len(groups)} групп и {len(ungrouped_faces)} одиночных лиц")
        return groups, ungrouped_faces
    
    async def apply_feedback(
        self,
        face_id: str,
        new_player_id: str,
        confidence: float,
        user_email: str = None
    ):
        """Применение фидбека пользователя для улучшения распознавания"""
        
        success = self.db.update_face_player(face_id, new_player_id, confidence, user_email)
        
        if success:
            print(f"[v0] Фидбек применен: {face_id} -> {new_player_id}")
            
            # Например, добавить эмбеддинг в архив игрока для улучшения будущих распознаваний
        
        return success
    
    async def clear_tournament_data(self, tournament_id: Optional[str] = None):
        """Очистка данных турнира"""
        print(f"[v0] clear_tournament_data вызван: tournament_id={tournament_id}")
        
        if tournament_id:
            removed_embeddings = len(self.embeddings_store.get(tournament_id, []))
            self.embeddings_store.pop(tournament_id, None)
            self.faces_data_store.pop(tournament_id, None)
            self.index_store.pop(tournament_id, None)
            print(f"[v0] Данные турнира {tournament_id} очищены ({removed_embeddings} эмбеддингов удалено)")
        else:
            total_embeddings = sum(len(v) for v in self.embeddings_store.values())
            self.embeddings_store.clear()
            self.faces_data_store.clear()
            self.index_store.clear()
            print(f"[v0] Все данные очищены ({total_embeddings} эмбеддингов удалено)")
    
    async def get_gallery_results(self, gallery_id: str) -> Dict:
        """Получение результатов распознавания для галереи"""
        stats = self.db.get_gallery_stats(gallery_id)
        faces = self.db.get_gallery_faces(gallery_id)
        
        # Группируем по игрокам
        players_dict = {}
        for face in faces:
            player_id = face.get("player_id")
            if player_id:
                if player_id not in players_dict:
                    player_info = self.db.get_player_info(player_id)
                    players_dict[player_id] = {
                        "player_id": player_id,
                        "name": player_info.get("name") if player_info else "Unknown",
                        "faces": []
                    }
                players_dict[player_id]["faces"].append(face)
        
        return {
            "stats": stats,
            "players": list(players_dict.values()),
            "unrecognized_faces": [f for f in faces if not f.get("player_id")]
        }
    
    async def recognize_face(
        self,
        embedding: np.ndarray,
        confidence_threshold: float = 0.60
    ) -> Tuple[Optional[str], Optional[float]]:
        """
        Распознать лицо по эмбеддингу с порогом confidence.
        
        Args:
            embedding: 512-мерный эмбеддинг от InsightFace
            confidence_threshold: Минимальный порог для верифицированных лиц (0-1)
                                 Для неверифицированных используется threshold + 0.15
        
        Returns:
            Tuple of (person_id, confidence) or (None, None) if below threshold
        """
        try:
            print(f"[v0] Checking PostgreSQL for verified faces (threshold={confidence_threshold:.2f})...")
            await db_client.connect()
            verified_match = await db_client.find_verified_face_by_embedding(
                embedding, 
                similarity_threshold=confidence_threshold
            )
            
            if verified_match:
                person_id, confidence = verified_match
                print(f"[v0] ✓ Found verified face: person_id={person_id}, confidence={confidence:.3f}")
                return person_id, confidence
            else:
                print("[v0] No verified face match found, checking unverified...")
        except Exception as e:
            print(f"[v0] Error checking PostgreSQL for verified faces: {e}")
            print("[v0] Falling back to HNSWLIB index...")
        
        if self.players_index is None or len(self.player_ids_map) == 0:
            print("[v0] Players index not loaded or empty - no trained faces available")
            return None, None
        
        try:
            unverified_threshold = confidence_threshold + 0.15
            print(f"[v0] Searching in HNSWLIB index (threshold={unverified_threshold:.2f} for unverified)")
            
            # Find nearest neighbor
            labels, distances = self.players_index.knn_query(
                embedding.reshape(1, -1),
                k=1
            )
            
            # Convert cosine distance to similarity
            similarity = 1 - distances[0][0]
            confidence = float(similarity)
            
            person_id = self.player_ids_map[labels[0][0]]
            
            print(f"[v0] HNSWLIB result: person_id={person_id}, confidence={confidence:.3f}, threshold={unverified_threshold:.2f}")
            
            if confidence >= unverified_threshold:
                print(f"[v0] ✓ Confidence above unverified threshold, returning match")
                return person_id, confidence
            else:
                print(f"[v0] Confidence {confidence:.3f} below unverified threshold {unverified_threshold:.2f}, returning None")
                return None, None
                
        except Exception as e:
            print(f"[v0] Error during recognition: {type(e).__name__}: {str(e)}")
            import traceback
            print(f"[v0] Traceback:\n{traceback.format_exc()}")
            return None, None

    async def rebuild_players_index(self) -> Dict:
        """
        Rebuild the HNSWLIB index from PostgreSQL database.
        Call this after adding new face descriptors to make them available for recognition.
        
        Returns:
            Dict with rebuild statistics
        """
        print("[v3.31] Rebuilding players index...")
        
        try:
            old_count = len(self.player_ids_map) if self.player_ids_map else 0
            
            # Reload index from database
            self._load_players_index()
            
            new_count = len(self.player_ids_map) if self.player_ids_map else 0
            unique_people = len(set(self.player_ids_map)) if self.player_ids_map else 0
            
            print(f"[v3.31] ✓ Index rebuilt: {old_count} -> {new_count} descriptors for {unique_people} people")
            
            return {
                "success": True,
                "old_descriptor_count": old_count,
                "new_descriptor_count": new_count,
                "unique_people_count": unique_people
            }
            
        except Exception as e:
            print(f"[v3.31] ERROR rebuilding index: {type(e).__name__}: {str(e)}")
            import traceback
            print(f"[v3.31] Traceback:\n{traceback.format_exc()}")
            return {
                "success": False,
                "error": str(e)
            }

    async def detect_faces(self, image_url: str, apply_quality_filters: bool = True) -> List[Dict]:
        """
        Detect faces on an image from URL with optional quality filtering
        
        Args:
            image_url: URL of the image to process
            apply_quality_filters: Whether to apply quality filters (default: True)
            
        Returns:
            List of detected faces with bbox, det_score, blur_score, and embedding
        """
        self._ensure_initialized()
        
        if apply_quality_filters:
            await self.load_quality_filters()
            print(f"[v3.2.2] detect_faces called with URL: {image_url}, apply_quality_filters=True")
            print(f"[v3.2.2] Quality filters: {self.quality_filters}")
        else:
            print(f"[v3.2.2] detect_faces called with URL: {image_url}, apply_quality_filters=False (skipping filters)")
        
        try:
            # Download image from URL
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.get(image_url, timeout=30.0)
                response.raise_for_status()
                image_bytes = response.content
            
            print(f"[v3.2.2] Downloaded {len(image_bytes)} bytes from URL")
            
            # Open image
            image = Image.open(io.BytesIO(image_bytes))
            img_array = np.array(image.convert('RGB'))
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            print(f"[v3.2.2] Image converted to array: shape {img_array.shape}")
            
            # Detect faces
            faces = self.app.get(img_array)
            print(f"[v3.2.2] Detected {len(faces)} faces before filtering")
            
            filtered_results = []
            filtered_count = 0
            
            for idx, face in enumerate(faces):
                # Calculate blur score
                blur_score = self.calculate_blur_score(img_array, face.bbox)
                
                # Calculate face size
                face_width = face.bbox[2] - face.bbox[0]
                face_height = face.bbox[3] - face.bbox[1]
                face_size = min(face_width, face_height)
                
                if apply_quality_filters:
                    passes, reason = self.passes_quality_filters(
                        face.det_score,
                        face.bbox,
                        blur_score
                    )
                    
                    print(f"[v3.2.2] Face {idx+1}: det_score={face.det_score:.3f}, size={face_size:.0f}px, blur={blur_score:.1f} - {reason}")
                    
                    if not passes:
                        filtered_count += 1
                        continue
                else:
                    print(f"[v3.2.2] Face {idx+1}: det_score={face.det_score:.3f}, size={face_size:.0f}px, blur={blur_score:.1f} - KEPT (no filtering)")
                
                filtered_results.append({
                    "bbox": face.bbox,
                    "det_score": face.det_score,
                    "blur_score": blur_score,
                    "embedding": face.embedding
                })
            
            if apply_quality_filters:
                print(f"[v3.2.2] After filtering: {len(filtered_results)} faces kept, {filtered_count} filtered out")
            else:
                print(f"[v3.2.2] No filtering applied: {len(filtered_results)} faces kept")
            
            return filtered_results
            
        except Exception as e:
            print(f"[v3.2.2] ERROR in detect_faces: {type(e).__name__}: {str(e)}")
            import traceback
            print(f"[v3.2.2] Traceback:\n{traceback.format_exc()}")
            raise

    def calculate_blur_score(self, image: np.ndarray, bbox: List[float]) -> float:
        """
        Calculate blur score using Laplacian variance.
        Higher score = sharper image.
        
        Args:
            image: Full image array (BGR format)
            bbox: Face bounding box [x1, y1, x2, y2]
            
        Returns:
            Blur score (Laplacian variance). Typical range:
            - < 50: Very blurry
            - 50-100: Blurry
            - 100-200: Acceptable
            - > 200: Sharp
        """
        try:
            # Extract face region
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            
            # Add padding (10%) to include some context
            h, w = image.shape[:2]
            padding_x = int((x2 - x1) * 0.1)
            padding_y = int((y2 - y1) * 0.1)
            
            x1 = max(0, x1 - padding_x)
            y1 = max(0, y1 - padding_y)
            x2 = min(w, x2 + padding_x)
            y2 = min(h, y2 + padding_y)
            
            face_region = image[y1:y2, x1:x2]
            
            # Convert to grayscale
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            
            # Calculate Laplacian variance
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            variance = laplacian.var()
            
            return float(variance)
            
        except Exception as e:
            print(f"[v0] Error calculating blur score: {e}")
            return 0.0
    
    def passes_quality_filters(
        self, 
        det_score: float, 
        bbox: List[float], 
        blur_score: float
    ) -> Tuple[bool, str]:
        """
        Check if face passes quality filters.
        
        Args:
            det_score: Detection confidence (0-1)
            bbox: Bounding box [x1, y1, x2, y2]
            blur_score: Blur score from calculate_blur_score
            
        Returns:
            Tuple of (passes: bool, reason: str)
        """
        # Check detection score
        if det_score < self.quality_filters["min_detection_score"]:
            return False, f"det_score {det_score:.2f} < {self.quality_filters['min_detection_score']}"
        
        # Check face size
        face_width = bbox[2] - bbox[0]
        face_height = bbox[3] - bbox[1]
        face_size = min(face_width, face_height)
        
        if face_size < self.quality_filters["min_face_size"]:
            return False, f"face_size {face_size:.0f}px < {self.quality_filters['min_face_size']}px"
        
        # Check blur score
        if blur_score < self.quality_filters["min_blur_score"]:
            return False, f"blur_score {blur_score:.1f} < {self.quality_filters['min_blur_score']}"
        
        return True, "passed"
    
    async def load_quality_filters(self):
        """Load quality filters from database config"""
        try:
            print(f"[v0] Loading quality filters from PostgreSQL...")
            await db_client.connect()
            config = await db_client.get_recognition_config()
            
            if "quality_filters" in config:
                self.quality_filters = config["quality_filters"]
                print(f"[v0] Quality filters loaded from DB: {self.quality_filters}")
            else:
                print(f"[v0] No quality_filters in config, using defaults: {self.quality_filters}")
        except Exception as e:
            print(f"[v0] Error loading quality filters: {e}, using defaults: {self.quality_filters}")
            import traceback
            print(f"[v0] Traceback:\n{traceback.format_exc()}")
    
    async def update_quality_filters(self, filters: Dict):
        """Update quality filters in database and cache"""
        try:
            await db_client.connect()
            await db_client.update_recognition_config({
                "quality_filters": filters
            })
            self.quality_filters = filters
            print(f"[v0] Quality filters updated in DB: {filters}")
        except Exception as e:
            print(f"[v0] Error updating quality filters: {e}")
            raise

    async def find_person_by_embedding(self, embedding: np.ndarray, confidence_threshold: float = 0.75) -> Optional[tuple]:
        """
        Поиск персоны по embedding с учетом verified_threshold
        
        Returns:
            tuple: (person_id, confidence, is_verified) или None
        """
        if self.players_index is None or not self.player_ids_map:
            print("[v0] HNSWLIB index not initialized or empty")
            return None

        try:
            verified_threshold = self.quality_filters.get("verified_threshold", 0.6)
            unverified_threshold = confidence_threshold + 0.15
            print(f"[v0] Searching in HNSWLIB index (verified_threshold={verified_threshold:.2f}, unverified_threshold={unverified_threshold:.2f})")
            
            # Нормализуем embedding
            embedding_norm = embedding / np.linalg.norm(embedding)
            
            # Ищем ближайшего соседа
            labels, distances = self.players_index.knn_query(embedding_norm, k=1)
            
            if len(labels) == 0 or len(labels[0]) == 0:
                print("[v0] No neighbors found in HNSWLIB index")
                return None
            
            person_id = self.player_ids_map[labels[0][0]]
            distance = distances[0][0]
            confidence = 1 - distance
            
            print(f"[v0] HNSWLIB result: person_id={person_id}, confidence={confidence:.3f}, verified_threshold={verified_threshold:.2f}, unverified_threshold={unverified_threshold:.2f}")
            
            # Проверяем, является ли персона verified (есть ли у неё хотя бы один verified face)
            db = db_client
            await db.connect()
            
            is_verified_query = """
                SELECT EXISTS(
                    SELECT 1 FROM photo_faces 
                    WHERE person_id = $1 AND verified = true
                ) as is_verified
            """
            result = await db.fetchrow(is_verified_query, person_id)
            is_verified = result['is_verified'] if result else False
            
            # Используем соответствующий threshold
            threshold = verified_threshold if is_verified else unverified_threshold
            
            if confidence >= threshold:
                print(f"[v0] Match found: person_id={person_id}, confidence={confidence:.3f}, is_verified={is_verified}, threshold={threshold:.2f}")
                return (person_id, confidence, is_verified)
            else:
                print(f"[v0] Confidence {confidence:.3f} below threshold {threshold:.2f} (is_verified={is_verified}), returning None")
                return None
                
        except Exception as e:
            print(f"[v0] ERROR in find_person_by_embedding: {e}")
            import traceback
            traceback.print_exc()
            return None
