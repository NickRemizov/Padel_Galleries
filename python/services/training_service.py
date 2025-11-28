import uuid
import os
import hashlib
import json
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import numpy as np
import cv2
import hnswlib
import httpx
from sklearn.model_selection import train_test_split

from services.postgres_client import db_client
from services.face_recognition import FaceRecognitionService


class TrainingService:
    def __init__(self):
        """Инициализация сервиса обучения"""
        self.face_service = FaceRecognitionService()
        self.db_client = db_client
        self.current_session_id = None
        self.current_progress = {'current': 0, 'total': 0, 'step': ''}
        print("[TrainingService] Initialized")
    
    async def prepare_dataset(
        self,
        filters: Dict,
        options: Dict
    ) -> Dict:
        """
        Подготовка датасета для обучения (без запуска обучения).
        
        Returns:
            Dataset statistics and validation results
        """
        print("[TrainingService] Preparing dataset...")
        
        await self.db_client.connect()
        
        # Get verified faces from PostgreSQL
        faces = await self.db_client.get_verified_faces(
            event_ids=filters.get('event_ids'),
            person_ids=filters.get('person_ids'),
            date_from=filters.get('date_from'),
            date_to=filters.get('date_to'),
            min_faces_per_person=options['min_faces_per_person']
        )
        
        # If include_co_occurring, add related people
        if options.get('include_co_occurring'):
            event_ids = list(set(f['gallery_id'] for f in faces if f['gallery_id']))
            if event_ids:
                co_occurring = await self.db_client.get_co_occurring_people(event_ids)
                # Add faces of co-occurring people
                # (Implementation depends on requirements)
        
        # Group by person_id and calculate statistics
        people_faces = {}
        for face in faces:
            person_id = face['person_id']
            if person_id not in people_faces:
                people_faces[person_id] = []
            people_faces[person_id].append(face)
        
        # Calculate statistics
        face_counts = [len(faces) for faces in people_faces.values()]
        stats = {
            'total_people': len(people_faces),
            'total_faces': len(faces),
            'faces_per_person': {
                'min': min(face_counts) if face_counts else 0,
                'max': max(face_counts) if face_counts else 0,
                'avg': sum(face_counts) / len(face_counts) if face_counts else 0
            },
            'people_by_face_count': self._calculate_distribution(face_counts)
        }
        
        # Validation
        warnings = []
        errors = []
        
        if stats['total_people'] < 2:
            errors.append('Need at least 2 people for training')
        
        for person_id, person_faces in people_faces.items():
            if len(person_faces) < 5:
                person_name = person_faces[0]['person_name']
                warnings.append(f'{person_name} has only {len(person_faces)} faces')
        
        print(f"[TrainingService] Dataset prepared: {stats['total_people']} people, {stats['total_faces']} faces")
        
        return {
            'dataset_stats': stats,
            'validation': {
                'ready': len(errors) == 0,
                'warnings': warnings,
                'errors': errors
            }
        }
    
    async def execute_training(
        self,
        mode: str,
        filters: Dict,
        options: Dict
    ) -> str:
        """
        Запуск обучения модели.
        
        Returns:
            session_id for tracking progress
        """
        print(f"[TrainingService] Starting training in {mode} mode...")
        
        # Create session
        session_id = str(uuid.uuid4())
        session_data = {
            'id': session_id,
            'model_version': options.get('model_version', 'v1.0'),
            'training_mode': mode,
            'faces_count': 0,
            'people_count': 0,
            'context_weight': options['context_weight'],
            'min_faces_per_person': options['min_faces_per_person'],
            'metrics': {},
            'status': 'running'
        }
        
        await self.db_client.connect()
        await self.db_client.create_training_session(session_data)
        
        self.current_session_id = session_id
        
        print(f"[TrainingService] Training session created: {session_id}")
        
        return session_id
    
    async def _train_background(
        self,
        session_id: str,
        mode: str,
        filters: Dict,
        options: Dict
    ):
        """
        Фоновый процесс обучения с прогрессивным алгоритмом.
        Оптимизирован: переиспользует существующие эмбеддинги из БД.
        """
        print(f"[TrainingService] Background training started for session {session_id}")
        
        try:
            # Ensure InsightFace is initialized
            self.face_service._ensure_initialized()
            
            self.current_progress = {
                'current': 0,
                'total': 0,
                'step': 'Loading verified faces from database'
            }
            
            await self.db_client.connect()
            
            faces = await self.db_client.get_verified_faces_with_descriptors(
                event_ids=filters.get('event_ids'),
                person_ids=filters.get('person_ids'),
                date_from=filters.get('date_from'),
                date_to=filters.get('date_to'),
                min_faces_per_person=options['min_faces_per_person']
            )
            
            print(f"[TrainingService] Loaded {len(faces)} verified faces")
            
            faces_with_descriptors = []
            faces_without_descriptors = []
            
            for face in faces:
                if face.get('insightface_descriptor'):
                    faces_with_descriptors.append(face)
                else:
                    faces_without_descriptors.append(face)
            
            print(f"[TrainingService] Faces with descriptors: {len(faces_with_descriptors)}")
            print(f"[TrainingService] Faces without descriptors: {len(faces_without_descriptors)}")
            
            descriptors = []
            person_ids = []
            
            for face in faces_with_descriptors:
                descriptor = np.array(face['insightface_descriptor'])
                descriptors.append(descriptor)
                person_ids.append(face['person_id'])
            
            print(f"[TrainingService] Reused {len(descriptors)} existing descriptors")
            
            if faces_without_descriptors:
                self.current_progress['step'] = 'Extracting missing descriptors'
                self.current_progress['total'] = len(faces_without_descriptors)
                
                # Group by photo_id to process photos once
                photos_to_process = {}
                for face in faces_without_descriptors:
                    photo_id = face['photo_id']
                    if photo_id not in photos_to_process:
                        photos_to_process[photo_id] = []
                    photos_to_process[photo_id].append(face)
                
                processed_count = 0
                
                for photo_id, photo_faces in photos_to_process.items():
                    self.current_progress['current'] = processed_count
                    
                    try:
                        # Download photo once
                        photo_url = photo_faces[0]['photo_url']
                        image = await self._download_photo(photo_url)
                        
                        # Detect all faces in photo
                        detected_faces = self.face_service.app.get(image)
                        
                        if not detected_faces:
                            print(f"[WARNING] No faces detected in photo {photo_id}")
                            processed_count += len(photo_faces)
                            continue
                        
                        for face_data in photo_faces:
                            saved_bbox = face_data['insightface_bbox']
                            
                            saved_x1 = saved_bbox['x']
                            saved_y1 = saved_bbox['y']
                            saved_x2 = saved_bbox['x'] + saved_bbox['width']
                            saved_y2 = saved_bbox['y'] + saved_bbox['height']
                            
                            best_match = None
                            best_iou = 0
                            
                            for detected in detected_faces:
                                det_bbox = detected.bbox
                                
                                x1 = max(saved_x1, det_bbox[0])
                                y1 = max(saved_y1, det_bbox[1])
                                x2 = min(saved_x2, det_bbox[2])
                                y2 = min(saved_y2, det_bbox[3])
                                
                                intersection = max(0, x2 - x1) * max(0, y2 - y1)
                                
                                saved_area = (saved_x2 - saved_x1) * (saved_y2 - saved_y1)
                                det_area = (det_bbox[2] - det_bbox[0]) * (det_bbox[3] - det_bbox[1])
                                union = saved_area + det_area - intersection
                                
                                iou = intersection / union if union > 0 else 0
                                
                                if iou > best_iou:
                                    best_iou = iou
                                    best_match = detected
                            
                            if best_match and best_iou > 0.3:
                                descriptor = best_match.embedding
                                confidence = float(best_match.det_score)
                                
                                # Add to training data
                                descriptors.append(descriptor)
                                person_ids.append(face_data['person_id'])
                                
                                await self.db_client.store_face_descriptor(
                                    photo_face_id=face_data['face_id'],
                                    person_id=face_data['person_id'],
                                    descriptor=descriptor
                                )
                                
                                print(f"[TrainingService] Extracted descriptor for face {face_data['face_id']} (IoU: {best_iou:.2f})")
                            else:
                                print(f"[WARNING] No match for face {face_data['face_id']} (best IoU: {best_iou:.2f})")
                            
                            processed_count += 1
                        
                    except Exception as e:
                        print(f"[ERROR] Failed to process photo {photo_id}: {e}")
                        import traceback
                        print(traceback.format_exc())
                        processed_count += len(photo_faces)
                        continue
            
            print(f"[TrainingService] Total descriptors for training: {len(descriptors)} from {len(set(person_ids))} people")
            
            self.current_progress['step'] = 'Building HNSWLIB index'
            
            if mode == 'full':
                self.face_service.players_index = hnswlib.Index(space='cosine', dim=512)
                self.face_service.players_index.init_index(
                    max_elements=10000,
                    ef_construction=200,
                    M=16
                )
                self.face_service.player_ids_map = []
            
            if descriptors:
                start_idx = len(self.face_service.player_ids_map)
                self.face_service.players_index.add_items(
                    np.array(descriptors),
                    list(range(start_idx, start_idx + len(descriptors)))
                )
                self.face_service.player_ids_map.extend(person_ids)
            
            print(f"[TrainingService] Index built with {len(descriptors)} descriptors")
            
            self.current_progress['step'] = 'Saving index to disk'
            
            models_dir = '/home/nickr/python/models'
            os.makedirs(models_dir, exist_ok=True)
            
            index_path = os.path.join(models_dir, 'players_index.bin')
            map_path = os.path.join(models_dir, 'player_ids_map.json')
            
            self.face_service.players_index.save_index(index_path)
            
            with open(map_path, 'w') as f:
                json.dump(self.face_service.player_ids_map, f)
            
            print(f"[TrainingService] Index saved to {index_path}")
            print(f"[TrainingService] Player IDs map saved to {map_path}")
            
            # Calculate metrics
            self.current_progress['step'] = 'Calculating metrics'
            
            metrics = await self._calculate_metrics(descriptors, person_ids)
            
            updates = {
                'faces_count': len(descriptors),
                'people_count': len(set(person_ids)),
                'metrics': metrics,
                'status': 'completed'
            }
            
            await self.db_client.update_training_session(session_id, updates)
            
            self.current_progress['step'] = 'Completed'
            print(f"[TrainingService] Training completed successfully")
            
        except Exception as e:
            print(f"[ERROR] Training failed: {e}")
            import traceback
            print(f"[ERROR] Traceback:\n{traceback.format_exc()}")
            
            updates = {
                'status': 'failed',
                'metrics': {'error': str(e)}
            }
            await self.db_client.update_training_session(session_id, updates)
    
    async def _download_photo(self, photo_url: str) -> Optional[np.ndarray]:
        """
        Скачать фото.
        """
        try:
            async with httpx.AsyncClient() as client:
                print(f"[TrainingService] Downloading {photo_url}...")
                try:
                    response = await client.get(photo_url, timeout=10.0)
                    response.raise_for_status()
                except httpx.RequestError as e:
                    print(f"[TrainingService] Network error downloading {photo_url}: {e}")
                    return None
                except httpx.HTTPStatusError as e:
                    print(f"[TrainingService] HTTP error downloading {photo_url}: {e}")
                    return None
                
                cache_dir = 'data/cache/photos'
                os.makedirs(cache_dir, exist_ok=True)
                
                filename = hashlib.md5(photo_url.encode()).hexdigest() + '.jpg'
                local_path = os.path.join(cache_dir, filename)
                
                with open(local_path, 'wb') as f:
                    f.write(response.content)
                
                img = cv2.imread(local_path)
                if img is None:
                    print(f"[TrainingService] Failed to decode image from {local_path}")
                    return None
                    
                return img
                
        except Exception as e:
            print(f"[TrainingService] Error in _download_photo: {e}")
            return None
    
    async def _calculate_metrics(
        self,
        descriptors: List[np.ndarray],
        person_ids: List[str]
    ) -> Dict:
        """
        Рассчитать метрики качества модели.
        """
        if len(descriptors) < 10:
            return {'accuracy': 0, 'precision': 0, 'recall': 0, 'note': 'Too few samples'}
        
        indices = list(range(len(descriptors)))
        train_idx, test_idx = train_test_split(
            indices,
            test_size=0.2,
            random_state=42,
            stratify=person_ids
        )
        
        train_descriptors = [descriptors[i] for i in train_idx]
        train_person_ids = [person_ids[i] for i in train_idx]
        
        temp_index = hnswlib.Index(space='cosine', dim=512)
        temp_index.init_index(max_elements=len(train_descriptors), ef_construction=200, M=16)
        temp_index.add_items(np.array(train_descriptors), list(range(len(train_descriptors))))
        
        correct = 0
        total = len(test_idx)
        
        for i in test_idx:
            test_descriptor = descriptors[i]
            true_person_id = person_ids[i]
            
            labels, distances = temp_index.knn_query(test_descriptor.reshape(1, -1), k=1)
            predicted_person_id = train_person_ids[labels[0][0]]
            
            if predicted_person_id == true_person_id:
                correct += 1
        
        accuracy = correct / total if total > 0 else 0
        
        return {
            'accuracy': round(accuracy, 3),
            'precision': round(accuracy, 3),
            'recall': round(accuracy, 3),
            'test_samples': total,
            'correct_predictions': correct
        }
    
    def _calculate_distribution(self, face_counts: List[int]) -> Dict[str, int]:
        """Рассчитать распределение людей по количеству лиц"""
        distribution = {
            '3-4': 0,
            '5-9': 0,
            '10-14': 0,
            '15-19': 0,
            '20+': 0
        }
        
        for count in face_counts:
            if 3 <= count <= 4:
                distribution['3-4'] += 1
            elif 5 <= count <= 9:
                distribution['5-9'] += 1
            elif 10 <= count <= 14:
                distribution['10-14'] += 1
            elif 15 <= count <= 19:
                distribution['15-19'] += 1
            else:
                distribution['20+'] += 1
        
        return distribution
    
    def get_training_status(self, session_id: str) -> Dict:
        """
        Получить статус обучения.
        """
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            session = loop.run_until_complete(self._get_session_from_postgres(session_id))
        finally:
            loop.close()
        
        if not session:
            return {'error': 'Session not found'}
        
        result = {
            'session_id': session_id,
            'status': session.get('status'),
            'started_at': session.get('started_at')
        }
        
        if session_id == self.current_session_id and session.get('status') == 'running':
            result['progress'] = {
                'current': self.current_progress['current'],
                'total': self.current_progress['total'],
                'percentage': int(
                    (self.current_progress['current'] / self.current_progress['total'] * 100)
                    if self.current_progress['total'] > 0 else 0
                )
            }
            result['current_step'] = self.current_progress['step']
        
        return result
    
    async def _get_session_from_postgres(self, session_id: str) -> Optional[Dict]:
        """Helper to get training session from PostgreSQL"""
        await self.db_client.connect()
        query = "SELECT * FROM face_training_sessions WHERE id = $1"
        return await self.db_client.fetchone(query, session_id)

    async def get_training_statistics(self) -> Dict:
        """
        Получить статистику обученных данных из PostgreSQL.
        Возвращает количество людей с дескрипторами, общее количество лиц и уникальных фото.
        """
        await self.db_client.connect()
        
        # Get statistics from PostgreSQL
        stats = await self.db_client.get_training_statistics()
        
        return {
            'people_count': stats['people_count'],
            'total_faces': stats['total_faces'],
            'unique_photos': stats['unique_photos']
        }
    
    async def batch_recognize(self, limit: int = 100):
        """
        Пакетное распознавание лиц на новых фото.
        """
        print(f"[TrainingService] Starting batch recognition (limit={limit})...")
        
        try:
            await self.db_client.connect()
            print("[TrainingService] DB connected")
            
            config = await self.db_client.get_recognition_config()
            quality_filters = config.get('quality_filters', {})
            verified_threshold = quality_filters.get('verified_threshold', 0.99)
            print(f"[TrainingService] Using verified_threshold: {verified_threshold}")
            
            # Get unverified images
            print(f"[TrainingService] Fetching unverified images (limit={limit})...")
            images = await self.db_client.get_unverified_images(limit)
            print(f"[TrainingService] Found {len(images)} images to process")
            
            processed_count = 0
            recognized_count = 0
            filtered_out_count = 0
            
            for img_data in images:
                photo_id = img_data['id']
                photo_url = img_data['image_url']
                print(f"[TrainingService] Processing image {photo_id} ({photo_url})...")
                
                try:
                    # Download photo
                    print(f"[TrainingService] Downloading photo {photo_url}...")
                    image = await self._download_photo(photo_url)
                    if image is None:
                        print(f"[TrainingService] Failed to download/read image {photo_id}")
                        continue
                    print(f"[TrainingService] Photo downloaded, shape: {image.shape}")
                    
                    # Detect faces
                    print(f"[TrainingService] Detecting faces...")
                    faces = self.face_service.app.get(image)
                    print(f"[TrainingService] Detected {len(faces)} faces")
                    
                    if not faces:
                        print(f"[TrainingService] No faces found in {photo_id}")
                        continue
                        
                    for i, face in enumerate(faces):
                        print(f"[TrainingService] Processing face {i+1}/{len(faces)}...")
                        # Get embedding
                        embedding = face.embedding
                        det_score = float(face.det_score)
                        bbox = {
                            "x": float(face.bbox[0]),
                            "y": float(face.bbox[1]),
                            "width": float(face.bbox[2] - face.bbox[0]),
                            "height": float(face.bbox[3] - face.bbox[1])
                        }
                        
                        # Recognize
                        print(f"[TrainingService] Recognizing face...")
                        person_id, confidence = await self.face_service.recognize_face(embedding)
                        print(f"[TrainingService] Recognition result: person_id={person_id}, confidence={confidence}")
                        
                        verified = False
                        if confidence is not None and confidence >= verified_threshold:
                            verified = True
                            print(f"[TrainingService] Setting verified=True (confidence {confidence:.3f} >= threshold {verified_threshold:.3f})")
                        else:
                            print(f"[TrainingService] Setting verified=False (confidence {confidence if confidence else 0:.3f} < threshold {verified_threshold:.3f})")
                        
                        # Save to DB
                        print(f"[TrainingService] Saving face to DB...")
                        
                        # Convert numpy array to list for JSON serialization if needed
                        embedding_list = embedding.tolist() if hasattr(embedding, 'tolist') else embedding
                        
                        face_id = await self.db_client.save_photo_face(
                            photo_id=photo_id,
                            person_id=person_id,
                            bbox=bbox,
                            descriptor=embedding_list,
                            insightface_confidence=det_score,
                            recognition_confidence=confidence,
                            verified=verified
                        )
                        print(f"[TrainingService] Face saved successfully with face_id={face_id}, verified={verified}")
                        
                        # If recognized, save descriptor for future training
                        if person_id:
                            recognized_count += 1
                            print(f"[TrainingService] Saving descriptor for person {person_id}...")
                            await self.db_client.save_face_descriptor(
                                person_id=person_id,
                                descriptor=embedding_list,
                                source_image_id=photo_id
                            )
                            print(f"[TrainingService] Descriptor saved")
                        else:
                            filtered_out_count += 1

                    processed_count += 1
                    print(f"[TrainingService] Image {photo_id} processed successfully")
                    
                except Exception as e:
                    print(f"[ERROR] ERROR processing image {photo_id}: {e}")
                    import traceback
                    print(traceback.format_exc())
                    continue
            
            print(f"[TrainingService] Batch recognition completed. Processed {processed_count} images.")
            return {
                "processed": processed_count,
                "recognized": recognized_count,
                "filtered_out": filtered_out_count
            }
            
        except Exception as e:
            print(f"[TrainingService] CRITICAL ERROR in batch_recognize: {e}")
            import traceback
            print(traceback.format_exc())
            raise e
