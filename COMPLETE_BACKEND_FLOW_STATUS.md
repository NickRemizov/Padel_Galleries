# Complete Backend Flow Status - Full Python API Migration
**Last Updated:** 2025-11-27
**Status:** 100% Complete - All Data Operations via Python API

---

## Executive Summary

**МИГРАЦИЯ ЗАВЕРШЕНА: ВСЕ ОПЕРАЦИИ ИДУТ ЧЕРЕЗ PYTHON API**

- ✅ Все CRUD операции для people через Python API
- ✅ Все операции с photo_faces через Python API
- ✅ Все операции с face_descriptors через Python API
- ✅ Embedding генерируется ТОЛЬКО на бэкенде
- ✅ Supabase SQL (`await sql`) полностью удалён из data flow

---

## Migrated Actions Matrix

### People Operations

| Action | Old Method | New Method | Status |
|--------|------------|------------|--------|
| `getPeopleAction` | `await sql` (Supabase) | `peopleApi.getAll()` | ✅ |
| `getPeopleWithStatsAction` | `await sql` (Supabase) | `peopleApi.getAll({ include_stats: true })` | ✅ |
| `getPersonAction` | `await sql` (Supabase) | Python `/api/crud/people/{id}` | ✅ |
| `addPersonAction` | `createClient()` (Supabase) | `peopleApi.create()` | ✅ |
| `deletePersonAction` | `createClient()` (Supabase) | `peopleApi.delete()` | ✅ |
| `createPersonFromClusterAction` | `createClient()` (Supabase) | Python `/api/crud/people/from-cluster` | ✅ |

### Photo Faces Operations

| Action | Old Method | New Method | Status |
|--------|------------|------------|--------|
| `getPhotoFacesAction` | `await sql` (Supabase) | Python `/api/faces/get-photo-faces` | ✅ |
| `getBatchPhotoFacesAction` | `await sql` (Supabase) | Python `/api/faces/get-batch-photo-faces` | ✅ |
| `savePhotoFaceAction` | `createClient()` (Supabase) | Python `/api/faces/save` | ✅ |
| `saveFaceTagsAction` | Mixed | Python `/api/faces/save-face-tags` | ✅ |
| `deletePhotoFaceAction` | `deletePhotoFace()` (face-storage.ts) | Python `/api/faces/delete/{id}` | ✅ |

### Recognition Operations

| Action | Old Method | New Method | Status |
|--------|------------|------------|--------|
| `detectFacesInsightFace` | Python API | Python `/api/recognition/detect` | ✅ |
| `recognizeFaceInsightFace` | Python API | Python `/api/recognition/recognize` | ✅ |
| `rebuildRecognitionIndexAction` | Python API | Python `/api/recognition/rebuild-index` | ✅ |

---

## Python API Endpoints

### CRUD Router (`/api/crud`)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/people` | Get all people with optional stats |
| GET | `/people/{id}` | Get person by ID |
| POST | `/people` | Create new person |
| PUT | `/people/{id}` | Update person |
| DELETE | `/people/{id}` | Delete person |
| POST | `/people/from-cluster` | Create person from cluster with descriptors |

### Faces Router (`/api/faces`)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/save` | Save single photo face |
| POST | `/save-face-tags` | Batch save face tags (generates embeddings) |
| POST | `/get-photo-faces` | Get faces for single photo |
| POST | `/get-batch-photo-faces` | Get faces for multiple photos |
| DELETE | `/delete/{face_id}` | Delete photo face |

### Recognition Router (`/api/recognition`)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/detect` | Detect faces in image |
| POST | `/recognize` | Recognize face by embedding |
| POST | `/rebuild-index` | Rebuild HNSWLIB index |

---

## Key Architecture Changes

### 1. Embedding Generation on Backend Only

**Before (WRONG):**
\`\`\`
Frontend → /detect → gets embedding → sends to /save
\`\`\`

**After (CORRECT):**
\`\`\`
Frontend → /detect → gets bbox, confidence (NO embedding)
Frontend → /save-face-tags → sends image_url + bbox
Backend → generates embedding from image + bbox → saves
\`\`\`

### 2. NFD (No Faces Detected) Handling

**Correct Flow:**
\`\`\`
User removes all faces → saves empty tags array
    ↓
saveFaceTagsAction(photoId, imageUrl, [])
    ↓
Python /api/faces/save-face-tags:
    1. Delete all photo_faces for this photo
    2. Delete all face_descriptors for this photo
    3. Set has_been_processed = true
    4. Return success
    ↓
Photo shows NFD badge
\`\`\`

### 3. Data Flow Diagram

\`\`\`
┌─────────────────────────────────────────────────────────────┐
│                      FRONTEND (Vercel)                      │
│                                                             │
│  actions.ts ──────────────────────────────────────────────┐ │
│      │                                                    │ │
│      ├─ peopleApi.getAll()                               │ │
│      ├─ peopleApi.create()                               │ │
│      ├─ facesApi.saveFaceTags()                          │ │
│      ├─ facesApi.getPhotoFaces()                         │ │
│      └─ fetch(FASTAPI_URL + '/api/...')                  │ │
│                                                          │ │
└──────────────────────────┬───────────────────────────────┘ │
                           │                                  
                           ↓ HTTP                             
┌──────────────────────────────────────────────────────────────┐
│                   PYTHON API (Hetzner:8001)                  │
│                                                              │
│  routers/                                                    │
│      ├─ crud.py ────────────┐                               │
│      ├─ faces.py ───────────┼─→ postgres_client.py          │
│      └─ recognition.py ─────┘           │                   │
│                                         ↓                   │
│                              ┌─────────────────────┐        │
│                              │  PostgreSQL         │        │
│                              │  (Hetzner)          │        │
│                              │                     │        │
│                              │  - people           │        │
│                              │  - photo_faces      │        │
│                              │  - face_descriptors │        │
│                              │  - gallery_images   │        │
│                              └─────────────────────┘        │
└──────────────────────────────────────────────────────────────┘
\`\`\`

---

## Files to Keep Synced

### Python Files (copy to /home/nickr/python/)

1. `python/routers/faces.py` - Face operations endpoints
2. `python/routers/crud.py` - CRUD endpoints for people, galleries
3. `python/routers/recognition.py` - Recognition endpoints
4. `python/services/postgres_client.py` - Database client
5. `python/models/schemas.py` - Pydantic models

### After Sync Commands

\`\`\`bash
# Copy files to server
scp python/routers/*.py root@server:/home/nickr/python/routers/
scp python/services/postgres_client.py root@server:/home/nickr/python/services/
scp python/models/schemas.py root@server:/home/nickr/python/models/

# Restart Python
ssh root@server "pkill -9 -f 'uvicorn main:app' && cd /home/nickr/python && source venv/bin/activate && nohup python -m uvicorn main:app --host 0.0.0.0 --port 8001 > /tmp/fastapi.log 2>&1 &"
\`\`\`

---

## Deprecated / DO NOT USE

| File/Function | Reason |
|---------------|--------|
| `lib/face-recognition/face-storage.ts` | Uses Supabase SQL directly |
| `await sql` in actions.ts | Direct Supabase queries |
| `createClient()` for data ops | Use Python API instead |
| Embedding on frontend | Backend generates embeddings |

---

**STATUS: MIGRATION COMPLETE** ✅
