from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional
from services.face_recognition import FaceRecognitionService
from services.postgres_client import db_client
import logging
import numpy as np
import hdbscan
from models.schemas import BatchRecognitionRequest

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

router = APIRouter()

face_service = FaceRecognitionService()

class DetectFacesRequest(BaseModel):
    image_url: str
    apply_quality_filters: bool = True  # Added parameter to control quality filtering


class RecognizeFaceRequest(BaseModel):
    embedding: List[float]
    confidence_threshold: float = 0.60


class GenerateDescriptorsRequest(BaseModel):
    image_url: str
    faces: List[dict]  # [{person_id, bbox, verified}]


class FaceDetectionResponse(BaseModel):
    faces: List[dict]  # [{insightface_bbox, confidence, blur_score, distance_to_nearest, top_matches}]


class FaceRecognitionResponse(BaseModel):
    person_id: Optional[str]
    confidence: Optional[float]


@router.post("/detect-faces", response_model=FaceDetectionResponse)
async def detect_faces(request: DetectFacesRequest):
    """Detect faces on an image using InsightFace"""
    try:
        logger.info("=" * 80)
        logger.info("[v3.1] ===== DETECT FACES REQUEST START =====")
        logger.info(f"[v3.1] Image URL: {request.image_url}")
        logger.info(f"[v3.1] Apply quality filters: {request.apply_quality_filters}")
        logger.info(f"[v3.1] Request timestamp: {__import__('datetime').datetime.now()}")
        
        # Detect faces
        logger.info(f"[v3.1] Calling face_service.detect_faces()...")
        faces = await face_service.detect_faces(request.image_url, apply_quality_filters=request.apply_quality_filters)
        
        logger.info(f"[v3.1] ✓ Detected {len(faces)} faces")
        
        # Format response
        faces_data = []
        for idx, face in enumerate(faces):
            logger.info(f"[v3.1] Processing face {idx + 1}/{len(faces)}")
            logger.info(f"[v3.1]   - BBox: {face['bbox']}")
            logger.info(f"[v3.1]   - Det score: {face['det_score']}")
            logger.info(f"[v3.1]   - Blur score: {face.get('blur_score', 0)}")
            logger.info(f"[v3.1]   - Embedding shape: {face['embedding'].shape}")
            
            embedding = face["embedding"]
            
            person_id, confidence = await face_service.recognize_face(embedding, confidence_threshold=0.0)
            
            # Get top 3 matches from HNSWLIB index
            top_matches = []
            distance_to_nearest = None
            
            if hasattr(face_service, 'players_index') and face_service.players_index is not None:
                try:
                    # Query index for top 3 matches
                    labels, distances = face_service.players_index.knn_query(embedding.reshape(1, -1), k=3)
                    
                    if len(distances) > 0 and len(distances[0]) > 0:
                        distance_to_nearest = float(distances[0][0])
                        
                        # Get person names for top matches
                        for i, (label_idx, distance) in enumerate(zip(labels[0], distances[0])):
                            if i >= 3:  # Only top 3
                                break
                            
                            if not hasattr(face_service, 'players_id_map') or len(face_service.players_id_map) == 0:
                                logger.warning(f"[v3.1] players_id_map is empty or not available")
                                break
                            
                            person_id_match = face_service.players_id_map[int(label_idx)]
                            
                            await db_client.connect()
                            person_data = await db_client.fetchone(
                                "SELECT real_name FROM people WHERE id = $1",
                                person_id_match
                            )
                            
                            person_name = "Unknown"
                            if person_data:
                                person_name = person_data.get("real_name", "Unknown")
                            
                            similarity = 1.0 - float(distance)  # Convert distance to similarity
                            top_matches.append({
                                "person_id": person_id_match,
                                "name": person_name,
                                "similarity": similarity
                            })
                            
                        logger.info(f"[v3.1]   - Distance to nearest: {distance_to_nearest:.4f}")
                        logger.info(f"[v3.1]   - Top matches: {len(top_matches)}")
                except Exception as e:
                    logger.warning(f"[v3.1] Could not get top matches: {str(e)}")
            
            embedding_quality = float(np.linalg.norm(embedding))
            
            face_data = {
                "insightface_bbox": {
                    "x": float(face["bbox"][0]),
                    "y": float(face["bbox"][1]),
                    "width": float(face["bbox"][2] - face["bbox"][0]),
                    "height": float(face["bbox"][3] - face["bbox"][1]),
                },
                "confidence": float(face["det_score"]),
                "blur_score": float(face.get("blur_score", 0)),
                "embedding": face["embedding"].tolist(),
                "distance_to_nearest": distance_to_nearest,
                "top_matches": top_matches,
                "embedding_quality": embedding_quality,
            }
            faces_data.append(face_data)
        
        logger.info(f"[v3.1] ✓ Returning {len(faces_data)} faces to frontend")
        logger.info("[v3.1] ===== DETECT FACES REQUEST END =====")
        logger.info("=" * 80)
        return {"faces": faces_data}
        
    except Exception as e:
        logger.error("=" * 80)
        logger.error(f"[v3.1] ❌ ERROR in detect_faces")
        logger.error(f"[v3.1] Error type: {type(e).__name__}")
        logger.error(f"[v3.1] Error message: {str(e)}")
        logger.error(f"[v3.1] Traceback:", exc_info=True)
        logger.error("=" * 80)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/recognize-face", response_model=FaceRecognitionResponse)
async def recognize_face(request: RecognizeFaceRequest):
    """Recognize a single face using the trained model"""
    try:
        logger.info("=" * 80)
        logger.info("[v3.0] ===== RECOGNIZE FACE REQUEST START =====")
        logger.info(f"[v3.0] Embedding length: {len(request.embedding)}")
        logger.info(f"[v3.0] Confidence threshold: {request.confidence_threshold}")
        logger.info(f"[v3.0] Request timestamp: {__import__('datetime').datetime.now()}")
        
        logger.info(f"[v3.0] Converting embedding to numpy array...")
        embedding = np.array(request.embedding, dtype=np.float32)
        logger.info(f"[v3.0] ✓ Embedding shape: {embedding.shape}")
        
        logger.info(f"[v3.0] Calling face_service.recognize_face()...")
        person_id, confidence = await face_service.recognize_face(
            embedding, 
            confidence_threshold=request.confidence_threshold
        )
        
        logger.info(f"[v3.0] ✓ Recognition result:")
        logger.info(f"[v3.0]   - Person ID: {person_id}")
        logger.info(f"[v3.0]   - Confidence: {confidence}")
        logger.info(f"[v3.0]   - Threshold: {request.confidence_threshold}")
        logger.info(f"[v3.0]   - Match: {'YES' if person_id else 'NO'}")
        
        logger.info("[v3.0] ===== RECOGNIZE FACE REQUEST END =====")
        logger.info("=" * 80)
        
        return {
            "person_id": person_id,
            "confidence": confidence,
        }
        
    except Exception as e:
        logger.error("=" * 80)
        logger.error(f"[v3.0] ❌ ERROR in recognize_face")
        logger.error(f"[v3.0] Error type: {type(e).__name__}")
        logger.error(f"[v3.0] Error message: {str(e)}")
        logger.error(f"[v3.0] Traceback:", exc_info=True)
        logger.error("=" * 80)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/v2/batch-recognize")
async def batch_recognize(request: BatchRecognitionRequest):
    """Batch recognize faces in galleries"""
    try:
        logger.info(f"[v3.22] ===== BATCH RECOGNIZE REQUEST =====")
        logger.info(f"[v3.22] Request data: {request}")
        
        # Use training service for batch processing
        from services.training_service import TrainingService
        service = TrainingService()
        
        # Pass limit from request
        limit = request.limit if request.limit else 100
        
        logger.info(f"[v3.22] Calling service.batch_recognize with limit={limit}")
        result = await service.batch_recognize(limit=limit)
        
        logger.info(f"[v3.22] Batch recognize result: {result}")
        return result
        
    except Exception as e:
        logger.error(f"[v3.22] ERROR in batch_recognize: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cluster-unknown-faces")
async def cluster_unknown_faces(
    gallery_id: Optional[str] = Query(None),
    min_cluster_size: int = Query(2)
):
    """
    Cluster unknown faces in a gallery by similarity
    Returns clusters sorted by size (largest first)
    
    Args:
        gallery_id: Gallery ID to cluster faces from (optional for testing)
        min_cluster_size: Minimum number of faces to form a cluster
    """
    try:
        logger.info(f"[v3.23] ===== CLUSTER UNKNOWN FACES =====")
        logger.info(f"[v3.23] Gallery ID: {gallery_id}")
        logger.info(f"[v3.23] Min cluster size: {min_cluster_size}")
        
        if not gallery_id:
            return {
                "success": True,
                "clusters": [],
                "ungrouped_faces": [],
                "message": "No gallery_id provided"
            }
        
        await db_client.connect()
        unknown_faces = await db_client.get_unknown_faces_from_gallery(gallery_id)
        
        if len(unknown_faces) < min_cluster_size:
            logger.info(f"[v3.23] Not enough faces to cluster ({len(unknown_faces)} < {min_cluster_size})")
            return {
                "clusters": [],
                "ungrouped_faces": unknown_faces
            }
        
        logger.info(f"[v3.23] Clustering {len(unknown_faces)} faces...")
        
        # Extract embeddings
        embeddings = []
        for face in unknown_faces:
            descriptor = face["insightface_descriptor"]
            if isinstance(descriptor, list):
                embeddings.append(np.array(descriptor, dtype=np.float32))
            elif isinstance(descriptor, str):
                import json
                embeddings.append(np.array(json.loads(descriptor), dtype=np.float32))
        
        embeddings_array = np.array(embeddings)
        
        # Cluster with HDBSCAN
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=1,
            metric='euclidean',
            cluster_selection_epsilon=0.5
        )
        
        cluster_labels = clusterer.fit_predict(embeddings_array)
        
        logger.info(f"[v3.23] Found {len(set(cluster_labels))} unique labels")
        
        # Group faces by cluster
        clusters_dict = {}
        ungrouped = []
        
        for face, label in zip(unknown_faces, cluster_labels):
            if label == -1:
                ungrouped.append(face)
            else:
                if label not in clusters_dict:
                    clusters_dict[label] = []
                clusters_dict[label].append(face)
        
        # Format clusters and sort by size (largest first)
        clusters = []
        for cluster_id, faces in clusters_dict.items():
            normalized_faces = []
            for face in faces:
                photo_data = await db_client.fetchone(
                    "SELECT width, height FROM gallery_images WHERE id = $1",
                    face["photo_id"]
                )
                
                if photo_data:
                    img_width = photo_data.get("width")
                    img_height = photo_data.get("height")
                    
                    # Normalize bbox if image dimensions are available
                    if img_width and img_height and face.get("insightface_bbox"):
                        bbox = face["insightface_bbox"]
                        normalized_bbox = {
                            "x": bbox["x"] / img_width,
                            "y": bbox["y"] / img_height,
                            "width": bbox["width"] / img_width,
                            "height": bbox["height"] / img_height,
                        }
                        face["bbox"] = normalized_bbox
                    else:
                        # Fallback: use original bbox (might be in pixels)
                        face["bbox"] = face.get("insightface_bbox")
                else:
                    # Fallback: use original bbox
                    face["bbox"] = face.get("insightface_bbox")
                
                normalized_faces.append(face)
            
            clusters.append({
                "cluster_id": int(cluster_id),
                "size": len(normalized_faces),
                "faces": normalized_faces
            })
        
        clusters.sort(key=lambda x: x["size"], reverse=True)
        
        logger.info(f"[v3.23] ✓ Returning {len(clusters)} clusters, {len(ungrouped)} ungrouped")
        logger.info(f"[v3.23] Cluster sizes: {[c['size'] for c in clusters]}")
        
        return {
            "clusters": clusters,
            "ungrouped_faces": ungrouped
        }
        
    except Exception as e:
        logger.error(f"[v3.23] ERROR clustering faces: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reject-face-cluster")
async def reject_face_cluster(
    gallery_id: str,
    face_ids: List[str],
    rejected_by: str,
    reason: Optional[str] = None
):
    """
    Reject a cluster of faces as not interesting
    """
    try:
        logger.info(f"[v3.23] ===== REJECT FACE CLUSTER =====")
        logger.info(f"[v3.23] Gallery ID: {gallery_id}")
        logger.info(f"[v3.23] Face IDs: {len(face_ids)}")
        logger.info(f"[v3.23] Rejected by: {rejected_by}")
        
        await db_client.connect()
        descriptors = []
        photo_ids = []
        
        for face_id in face_ids:
            face_data = await db_client.fetchone(
                "SELECT photo_id, insightface_descriptor FROM photo_faces WHERE id = $1",
                face_id
            )
            
            if face_data:
                descriptor = face_data["insightface_descriptor"]
                
                if isinstance(descriptor, list):
                    descriptors.append(np.array(descriptor, dtype=np.float32))
                elif isinstance(descriptor, str):
                    import json
                    descriptors.append(np.array(json.loads(descriptor), dtype=np.float32))
                
                photo_ids.append(face_data["photo_id"])
        
        success = await db_client.reject_face_cluster(face_ids, "system") # Assuming rejected_by is not used in DB logic
        
        if success:
            # Delete the photo_faces records
            for face_id in face_ids:
                await db_client.execute(
                    "DELETE FROM photo_faces WHERE id = $1",
                    face_id
                )
            
            logger.info(f"[v3.23] ✓ Successfully rejected and deleted {len(face_ids)} faces")
        
        return {"success": success}
        
    except Exception as e:
        logger.error(f"[v3.23] ERROR rejecting cluster: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-descriptors")
async def generate_descriptors(request: GenerateDescriptorsRequest):
    """
    Generate descriptors for manually tagged faces
    Called when admin manually assigns people to faces
    """
    try:
        logger.info(f"[v3.26] ===== GENERATE DESCRIPTORS FOR MANUAL TAGS =====")
        logger.info(f"[v3.26] Image URL: {request.image_url}")
        logger.info(f"[v3.26] Faces to process: {len(request.faces)}")
        
        # Detect all faces on the image
        detected_faces = await face_service.detect_faces(request.image_url)
        logger.info(f"[v3.26] Detected {len(detected_faces)} faces on image")
        
        generated_count = 0
        descriptors_response = []
        
        for tagged_face in request.faces:
            person_id = tagged_face["person_id"]
            tagged_bbox = tagged_face["bbox"]
            verified = tagged_face.get("verified", True)
            
            logger.info(f"[v3.26] Processing tagged face for person {person_id}")
            logger.info(f"[v3.26]   Tagged bbox: {tagged_bbox}")
            
            # Find matching detected face by IoU
            best_match = None
            best_iou = 0.0
            
            for detected_face in detected_faces:
                detected_bbox = {
                    "x": float(detected_face["bbox"][0]),
                    "y": float(detected_face["bbox"][1]),
                    "width": float(detected_face["bbox"][2] - detected_face["bbox"][0]),
                    "height": float(detected_face["bbox"][3] - detected_face["bbox"][1]),
                }
                
                iou = calculate_iou(tagged_bbox, detected_bbox)
                
                if iou > best_iou:
                    best_iou = iou
                    best_match = detected_face
            
            descriptor_data = {
                "person_id": person_id,
                "descriptor": None,
                "bbox": tagged_bbox,
                "confidence": 1.0 if verified else 0.0
            }
            
            if best_match and best_iou > 0.3:  # 30% overlap threshold
                logger.info(f"[v3.26]   Found matching detected face (IoU: {best_iou:.2f})")
                
                # Save descriptor to database
                descriptor = best_match["embedding"].tolist()
                photo_id = tagged_face.get("photo_id")
                
                descriptor_data["descriptor"] = descriptor
                
                if photo_id:
                    # using PostgreSQL to save face descriptor
                    await db_client.connect()
                    success = await db_client.save_face_descriptor(
                        person_id=person_id,
                        descriptor=descriptor,
                        source_image_id=photo_id
                    )
                    
                    if success:
                        generated_count += 1
                        logger.info(f"[v3.26]   ✓ Descriptor saved for person {person_id}")
                    else:
                        logger.error(f"[v3.26]   ✗ Failed to save descriptor")
            else:
                logger.warning(f"[v3.26]   No matching detected face found (best IoU: {best_iou:.2f})")
            
            descriptors_response.append(descriptor_data)
        
        logger.info(f"[v3.26] ✓ Generated {generated_count}/{len(request.faces)} descriptors")
        logger.info(f"[v3.26] ===== END GENERATE DESCRIPTORS =====")
        
        return {
            "success": True,
            "generated": generated_count,
            "total": len(request.faces),
            "descriptors": descriptors_response
        }
        
    except Exception as e:
        logger.error(f"[v3.26] ERROR generating descriptors: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rebuild-index")
async def rebuild_index():
    """
    Rebuild the HNSWLIB index from database.
    Call this after adding new face descriptors to make them available for recognition.
    """
    try:
        logger.info(f"[v3.31] ===== REBUILD INDEX REQUEST =====")
        
        # rebuild index using PostgreSQL client
        await db_client.connect()
        result = await face_service.rebuild_players_index()
        
        if result["success"]:
            logger.info(f"[v3.31] ✓ Index rebuilt successfully")
            logger.info(f"[v3.31]   Old count: {result['old_descriptor_count']}")
            logger.info(f"[v3.31]   New count: {result['new_descriptor_count']}")
            logger.info(f"[v3.31]   Unique people: {result['unique_people_count']}")
        else:
            logger.error(f"[v3.31] ✗ Index rebuild failed: {result.get('error')}")
        
        logger.info(f"[v3.31] ===== END REBUILD INDEX =====")
        
        return result
        
    except Exception as e:
        logger.error(f"[v3.31] ERROR rebuilding index: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/regenerate-unknown-descriptors")
async def regenerate_unknown_descriptors(gallery_id: str = Query(...)):
    """
    Regenerate insightface_descriptor for unknown faces that don't have one.
    This fixes faces that were saved without descriptors during batch recognition.
    
    Returns statistics about regeneration process.
    """
    try:
        logger.info(f"[v3.24] ===== REGENERATE UNKNOWN DESCRIPTORS =====")
        logger.info(f"[v3.24] Gallery ID: {gallery_id}")
        
        await db_client.connect()
        gallery_photos = await db_client.fetch(
            "SELECT id FROM gallery_images WHERE gallery_id = $1",
            gallery_id
        )
        
        if not gallery_photos:
            logger.info(f"[v3.24] No photos found in gallery")
            return {
                "success": True,
                "total_faces": 0,
                "regenerated": 0,
                "failed": 0,
                "already_had_descriptor": 0
            }
        
        photo_ids = [photo["id"] for photo in gallery_photos]
        logger.info(f"[v3.24] Found {len(photo_ids)} photos in gallery")
        
        faces = await db_client.fetch("""
            SELECT pf.id, pf.photo_id, pf.insightface_bbox, pf.insightface_descriptor,
                   gi.image_url
            FROM photo_faces pf
            INNER JOIN gallery_images gi ON gi.id = pf.photo_id
            WHERE pf.photo_id = ANY($1)
            AND pf.person_id IS NULL
        """, photo_ids)
        
        if not faces:
            logger.info(f"[v3.24] No unknown faces found")
            return {
                "success": True,
                "total_faces": 0,
                "regenerated": 0,
                "failed": 0,
                "already_had_descriptor": 0
            }
        
        total_faces = len(faces)
        regenerated = 0
        failed = 0
        already_had_descriptor = 0
        
        logger.info(f"[v3.24] Found {total_faces} unknown faces, checking descriptors...")
        
        for face in faces:
            face_id = face["id"]
            
            # Check if already has descriptor
            if face.get("insightface_descriptor"):
                already_had_descriptor += 1
                continue
            
            # Check if has bbox
            bbox = face.get("insightface_bbox")
            if not bbox:
                logger.warning(f"[v3.24] Face {face_id} has no bbox, skipping")
                failed += 1
                continue
            
            try:
                logger.info(f"[v3.24] Regenerating descriptor for face {face_id}")
                
                # Download and detect faces on image
                image_url = face["image_url"]
                detected_faces = await face_service.detect_faces(image_url)
                
                if not detected_faces:
                    logger.warning(f"[v3.24] No faces detected on image for face {face_id}")
                    failed += 1
                    continue
                
                # Find matching face by IoU with stored bbox
                best_match = None
                best_iou = 0.0
                
                for detected_face in detected_faces:
                    detected_bbox = {
                        "x": float(detected_face["bbox"][0]),
                        "y": float(detected_face["bbox"][1]),
                        "width": float(detected_face["bbox"][2] - detected_face["bbox"][0]),
                        "height": float(detected_face["bbox"][3] - detected_face["bbox"][1]),
                    }
                    
                    iou = calculate_iou(bbox, detected_bbox)
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_match = detected_face
                
                if best_match and best_iou > 0.3:  # 30% overlap threshold
                    # Extract descriptor
                    descriptor = best_match["embedding"]
                    
                    if len(descriptor) != 512:
                        logger.error(f"[v3.24] Invalid descriptor dimension: {len(descriptor)}, expected 512")
                        failed += 1
                        continue
                    
                    await db_client.execute("""
                        UPDATE photo_faces
                        SET insightface_descriptor = $1::vector,
                            insightface_confidence = $2
                        WHERE id = $3
                    """, descriptor.tolist(), float(best_match["det_score"]), face_id)
                    
                    regenerated += 1
                    logger.info(f"[v3.24] ✓ Descriptor regenerated for face {face_id} (IoU: {best_iou:.2f})")
                else:
                    logger.warning(f"[v3.24] No matching face found for face {face_id} (best IoU: {best_iou:.2f})")
                    failed += 1
                    
            except Exception as e:
                logger.error(f"[v3.24] Error regenerating descriptor for face {face_id}: {str(e)}")
                failed += 1
                continue
        
        logger.info(f"[v3.24] ===== REGENERATION COMPLETE =====")
        logger.info(f"[v3.24] Total faces: {total_faces}")
        logger.info(f"[v3.24] Already had descriptor: {already_had_descriptor}")
        logger.info(f"[v3.24] Regenerated: {regenerated}")
        logger.info(f"[v3.24] Failed: {failed}")
        
        return {
            "success": True,
            "total_faces": total_faces,
            "regenerated": regenerated,
            "failed": failed,
            "already_had_descriptor": already_had_descriptor
        }
        
    except Exception as e:
        logger.error(f"[v3.24] ERROR regenerating descriptors: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def calculate_iou(box1: dict, box2: dict) -> float:
    """Calculate Intersection over Union"""
    x1 = max(box1["x"], box2["x"])
    y1 = max(box1["y"], box2["y"])
    x2 = min(box1["x"] + box1["width"], box2["x"] + box2["width"])
    y2 = min(box1["y"] + box1["height"], box2["y"] + box2["height"])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = box1["width"] * box1["height"]
    area2 = box2["width"] * box2["height"]
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0
