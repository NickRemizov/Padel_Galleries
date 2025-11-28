"""
CRUD Router for all entities: galleries, photographers, locations, organizers, people.
Provides RESTful API endpoints for Create, Read, Update, Delete operations.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional
from services.postgres_client import db_client
from services.auth import verify_api_key
from models.schemas import (
    # Galleries
    GalleryCreate, GalleryUpdate, GalleryResponse,
    # Photographers
    PhotographerCreate, PhotographerUpdate, PhotographerResponse,
    # Locations
    LocationCreate, LocationUpdate, LocationResponse,
    # Organizers
    OrganizerCreate, OrganizerUpdate, OrganizerResponse,
    # People
    PersonCreate, PersonUpdate, PersonResponse,
    # PersonFromCluster
    PersonFromClusterCreate,
    # Stats
    RecognitionStatsResponse,
    # Generic
    SuccessResponse
)
import json

router = APIRouter()


# ==================== GALLERIES ====================

@router.get("/galleries", response_model=List[GalleryResponse])
async def get_galleries(sort_by: str = "created_at"):
    """Get all galleries with stats, sorted by sort_by parameter"""
    try:
        rows = await db_client.get_all_galleries(include_stats=True, sort_by=sort_by)
        return [_row_to_gallery(row) for row in rows]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/galleries/{gallery_id}", response_model=GalleryResponse)
async def get_gallery(gallery_id: str):
    """Get gallery by ID"""
    try:
        row = await db_client.get_gallery_by_id(gallery_id)
        if not row:
            raise HTTPException(status_code=404, detail="Gallery not found")
        return _row_to_gallery(row)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/galleries", response_model=GalleryResponse)
async def create_gallery(data: GalleryCreate, auth: dict = Depends(verify_api_key)):
    """Create new gallery"""
    try:
        row = await db_client.create_gallery(data.model_dump(exclude_none=True))
        return _row_to_gallery(row)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/galleries/{gallery_id}", response_model=GalleryResponse)
async def update_gallery(gallery_id: str, data: GalleryUpdate, auth: dict = Depends(verify_api_key)):
    """Update gallery"""
    try:
        row = await db_client.update_gallery(gallery_id, data.model_dump(exclude_none=True))
        if not row:
            raise HTTPException(status_code=404, detail="Gallery not found")
        return _row_to_gallery(row)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/galleries/{gallery_id}", response_model=SuccessResponse)
async def delete_gallery(gallery_id: str, auth: dict = Depends(verify_api_key)):
    """Delete gallery"""
    try:
        await db_client.delete_gallery(gallery_id)
        return SuccessResponse(message=f"Gallery {gallery_id} deleted")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _row_to_gallery(row: dict) -> GalleryResponse:
    """Convert DB row to GalleryResponse"""
    return GalleryResponse(
        id=str(row['id']),
        title=row['title'],
        shoot_date=row.get('shoot_date'),
        photographer_id=str(row['photographer_id']) if row.get('photographer_id') else None,
        location_id=str(row['location_id']) if row.get('location_id') else None,
        organizer_id=str(row['organizer_id']) if row.get('organizer_id') else None,
        cover_image_url=row.get('cover_image_url'),
        cover_image_square_url=row.get('cover_image_square_url'),
        gallery_url=row.get('gallery_url'),
        external_gallery_url=row.get('external_gallery_url'),
        sort_order=row.get('sort_order'),
        created_at=row.get('created_at'),
        updated_at=row.get('updated_at'),
        photographer_name=row.get('photographer_name'),
        location_name=row.get('location_name'),
        organizer_name=row.get('organizer_name'),
        images_count=int(row.get('images_count', 0))
    )


# ==================== PHOTOGRAPHERS ====================

@router.get("/photographers", response_model=List[PhotographerResponse])
async def get_photographers():
    """Get all photographers"""
    try:
        rows = await db_client.get_all_photographers(include_stats=True)
        return [_row_to_photographer(row) for row in rows]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/photographers/{photographer_id}", response_model=PhotographerResponse)
async def get_photographer(photographer_id: str):
    """Get photographer by ID"""
    try:
        row = await db_client.get_photographer_by_id(photographer_id)
        if not row:
            raise HTTPException(status_code=404, detail="Photographer not found")
        return _row_to_photographer(row)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/photographers", response_model=PhotographerResponse)
async def create_photographer(data: PhotographerCreate, auth: dict = Depends(verify_api_key)):
    """Create new photographer"""
    try:
        row = await db_client.create_photographer(data.model_dump(exclude_none=True))
        return _row_to_photographer(row)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/photographers/{photographer_id}", response_model=PhotographerResponse)
async def update_photographer(photographer_id: str, data: PhotographerUpdate, auth: dict = Depends(verify_api_key)):
    """Update photographer"""
    try:
        row = await db_client.update_photographer(photographer_id, data.model_dump(exclude_none=True))
        if not row:
            raise HTTPException(status_code=404, detail="Photographer not found")
        return _row_to_photographer(row)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/photographers/{photographer_id}", response_model=SuccessResponse)
async def delete_photographer(photographer_id: str, auth: dict = Depends(verify_api_key)):
    """Delete photographer"""
    try:
        await db_client.delete_photographer(photographer_id)
        return SuccessResponse(message=f"Photographer {photographer_id} deleted")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _row_to_photographer(row: dict) -> PhotographerResponse:
    """Convert DB row to PhotographerResponse"""
    return PhotographerResponse(
        id=str(row['id']),
        name=row['name'],
        created_at=row.get('created_at'),
        galleries_count=int(row.get('galleries_count', 0))
    )


# ==================== LOCATIONS ====================

@router.get("/locations", response_model=List[LocationResponse])
async def get_locations():
    """Get all locations"""
    try:
        rows = await db_client.get_all_locations(include_stats=True)
        return [_row_to_location(row) for row in rows]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/locations/{location_id}", response_model=LocationResponse)
async def get_location(location_id: str):
    """Get location by ID"""
    try:
        row = await db_client.get_location_by_id(location_id)
        if not row:
            raise HTTPException(status_code=404, detail="Location not found")
        return _row_to_location(row)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/locations", response_model=LocationResponse)
async def create_location(data: LocationCreate, auth: dict = Depends(verify_api_key)):
    """Create new location"""
    try:
        row = await db_client.create_location(data.model_dump(exclude_none=True))
        return _row_to_location(row)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/locations/{location_id}", response_model=LocationResponse)
async def update_location(location_id: str, data: LocationUpdate, auth: dict = Depends(verify_api_key)):
    """Update location"""
    try:
        row = await db_client.update_location(location_id, data.model_dump(exclude_none=True))
        if not row:
            raise HTTPException(status_code=404, detail="Location not found")
        return _row_to_location(row)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/locations/{location_id}", response_model=SuccessResponse)
async def delete_location(location_id: str, auth: dict = Depends(verify_api_key)):
    """Delete location"""
    try:
        await db_client.delete_location(location_id)
        return SuccessResponse(message=f"Location {location_id} deleted")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _row_to_location(row: dict) -> LocationResponse:
    """Convert DB row to LocationResponse"""
    return LocationResponse(
        id=str(row['id']),
        name=row['name'],
        created_at=row.get('created_at'),
        galleries_count=int(row.get('galleries_count', 0))
    )


# ==================== ORGANIZERS ====================

@router.get("/organizers", response_model=List[OrganizerResponse])
async def get_organizers():
    """Get all organizers"""
    try:
        rows = await db_client.get_all_organizers(include_stats=True)
        return [_row_to_organizer(row) for row in rows]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/organizers/{organizer_id}", response_model=OrganizerResponse)
async def get_organizer(organizer_id: str):
    """Get organizer by ID"""
    try:
        row = await db_client.get_organizer_by_id(organizer_id)
        if not row:
            raise HTTPException(status_code=404, detail="Organizer not found")
        return _row_to_organizer(row)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/organizers", response_model=OrganizerResponse)
async def create_organizer(data: OrganizerCreate, auth: dict = Depends(verify_api_key)):
    """Create new organizer"""
    try:
        row = await db_client.create_organizer(data.model_dump(exclude_none=True))
        return _row_to_organizer(row)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/organizers/{organizer_id}", response_model=OrganizerResponse)
async def update_organizer(organizer_id: str, data: OrganizerUpdate, auth: dict = Depends(verify_api_key)):
    """Update organizer"""
    try:
        row = await db_client.update_organizer(organizer_id, data.model_dump(exclude_none=True))
        if not row:
            raise HTTPException(status_code=404, detail="Organizer not found")
        return _row_to_organizer(row)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/organizers/{organizer_id}", response_model=SuccessResponse)
async def delete_organizer(organizer_id: str, auth: dict = Depends(verify_api_key)):
    """Delete organizer"""
    try:
        await db_client.delete_organizer(organizer_id)
        return SuccessResponse(message=f"Organizer {organizer_id} deleted")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _row_to_organizer(row: dict) -> OrganizerResponse:
    """Convert DB row to OrganizerResponse"""
    return OrganizerResponse(
        id=str(row['id']),
        name=row['name'],
        created_at=row.get('created_at'),
        galleries_count=int(row.get('galleries_count', 0))
    )


# ==================== PEOPLE ====================

@router.get("/people", response_model=List[PersonResponse])
async def get_people():
    """Get all people"""
    try:
        rows = await db_client.get_all_people(include_stats=True)
        return [_row_to_person(row) for row in rows]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/people/{person_id}", response_model=PersonResponse)
async def get_person(person_id: str):
    """Get person by ID"""
    try:
        row = await db_client.get_person_by_id(person_id)
        if not row:
            raise HTTPException(status_code=404, detail="Person not found")
        return _row_to_person(row)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/people", response_model=PersonResponse)
async def create_person(data: PersonCreate, auth: dict = Depends(verify_api_key)):
    """Create new person"""
    try:
        row = await db_client.create_person(data.model_dump(exclude_none=True))
        return _row_to_person(row)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/people/{person_id}", response_model=PersonResponse)
async def update_person(person_id: str, data: PersonUpdate, auth: dict = Depends(verify_api_key)):
    """Update person"""
    try:
        row = await db_client.update_person(person_id, data.model_dump(exclude_none=True))
        if not row:
            raise HTTPException(status_code=404, detail="Person not found")
        return _row_to_person(row)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/people/{person_id}", response_model=SuccessResponse)
async def delete_person(person_id: str, auth: dict = Depends(verify_api_key)):
    """Delete person"""
    try:
        await db_client.delete_person(person_id)
        return SuccessResponse(message=f"Person {person_id} deleted")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/people/from-cluster", response_model=PersonResponse)
async def create_person_from_cluster(data: PersonFromClusterCreate, auth: dict = Depends(verify_api_key)):
    """Create person from face cluster with descriptors"""
    try:
        # Convert to list of dicts for postgres_client
        cluster_faces = [
            {"photo_id": face.photo_id, "descriptor": face.descriptor}
            for face in data.cluster_faces
        ]
        
        row = await db_client.create_person_from_cluster(
            person_name=data.person_name,
            cluster_faces=cluster_faces
        )
        return _row_to_person(row)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _row_to_person(row: dict) -> PersonResponse:
    """Convert DB row to PersonResponse"""
    tournament_results = row.get('tournament_results')
    if isinstance(tournament_results, str):
        try:
            tournament_results = json.loads(tournament_results)
        except:
            tournament_results = None
    
    return PersonResponse(
        id=str(row['id']),
        real_name=row['real_name'],
        telegram_name=row.get('telegram_name'),
        telegram_nickname=row.get('telegram_nickname'),
        telegram_profile_url=row.get('telegram_profile_url'),
        facebook_profile_url=row.get('facebook_profile_url'),
        instagram_profile_url=row.get('instagram_profile_url'),
        avatar_url=row.get('avatar_url'),
        paddle_ranking=row.get('paddle_ranking'),
        category=row.get('category'),
        show_in_players_gallery=row.get('show_in_players_gallery'),
        show_photos_in_galleries=row.get('show_photos_in_galleries'),
        custom_confidence_threshold=float(row['custom_confidence_threshold']) if row.get('custom_confidence_threshold') else None,
        use_custom_confidence=row.get('use_custom_confidence'),
        tournament_results=tournament_results,
        created_at=row.get('created_at'),
        updated_at=row.get('updated_at'),
        faces_count=int(row.get('faces_count', 0))
    )


# ==================== STATS ====================

@router.get("/stats/recognition", response_model=RecognitionStatsResponse)
async def get_recognition_stats():
    """Get overall recognition statistics"""
    try:
        stats = await db_client.get_recognition_stats()
        return RecognitionStatsResponse(**stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
