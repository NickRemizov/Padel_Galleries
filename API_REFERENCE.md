# API Reference - Galeries v0.8.2

Полное описание всех API endpoints в системе.

## Next.js API Routes

### Face Detection

#### POST /api/face-detection/detect
Детекция лиц на изображении с помощью InsightFace.

**Request Body:**
\`\`\`typescript
{
  imageUrl: string;              // URL изображения в Blob Storage
  applyQualityFilters: boolean;  // Применять ли фильтры качества
}
\`\`\`

**Response:**
\`\`\`typescript
{
  faces: Array<{
    embedding: number[];         // 512-мерный вектор
    bbox: {
      x: number;
      y: number;
      width: number;
      height: number;
    };
    det_score: number;           // Уверенность детекции (0-1)
    blur_score: number;          // Резкость лица (Laplacian)
    face_size: number;           // Размер лица в пикселях
  }>;
}
\`\`\`

**Файл:** `app/api/face-detection/detect/route.ts`

---

#### POST /api/face-detection/recognize
Распознавание лица по эмбеддингу.

**Request Body:**
\`\`\`typescript
{
  embedding: number[];           // 512-мерный вектор лица
  galleryId?: string;            // ID галереи для контекстного поиска
}
\`\`\`

**Response:**
\`\`\`typescript
{
  personId: string | null;       // ID найденного игрока или null
  confidence: number;            // Уверенность распознавания (0-1)
  matchType: 'verified' | 'unverified' | 'none';
}
\`\`\`

**Файл:** `app/api/face-detection/recognize/route.ts`

---

### Training

#### POST /api/training/cluster-unverified-faces
Кластеризация неизвестных лиц в галерее с помощью HDBSCAN.

**Request Body:**
\`\`\`typescript
{
  galleryId: string;             // ID галереи
}
\`\`\`

**Response:**
\`\`\`typescript
{
  clusters: Array<{
    cluster_id: number;          // ID кластера (-1 для шума)
    faces: Array<{
      face_id: string;
      image_id: string;
      image_url: string;
      bbox: object;
    }>;
    size: number;                // Количество лиц в кластере
  }>;
  total_faces: number;
  clustered_faces: number;
  noise_faces: number;
}
\`\`\`

**Файл:** `app/api/training/cluster-unverified-faces/route.ts`

---

#### POST /api/training/rebuild-index
Перестроение HNSWLIB индекса для быстрого поиска лиц.

**Request Body:** (пустой)

**Response:**
\`\`\`typescript
{
  success: boolean;
  message: string;
  indexed_faces: number;         // Количество проиндексированных лиц
}
\`\`\`

**Файл:** `app/api/training/rebuild-index/route.ts`

---

### Telegram

#### POST /api/telegram/webhook
Webhook для обработки сообщений от Telegram бота.

**Request Body:** (Telegram Update object)

**Response:**
\`\`\`typescript
{
  ok: boolean;
}
\`\`\`

**Файл:** `app/api/telegram/webhook/route.ts`

---

## FastAPI Endpoints

Base URL: `http://your-server-ip:8001` (или `FASTAPI_URL` env variable)

### Recognition

#### POST /detect-faces
Детекция лиц на изображении.

**Request:**
\`\`\`json
{
  "image_url": "https://blob.vercel-storage.com/...",
  "apply_quality_filters": true
}
\`\`\`

**Response:**
\`\`\`json
{
  "faces": [
    {
      "embedding": [0.123, 0.456, ...],
      "bbox": {
        "x": 100,
        "y": 200,
        "width": 150,
        "height": 180
      },
      "det_score": 0.95,
      "blur_score": 123.45,
      "face_size": 150
    }
  ]
}
\`\`\`

**Файл:** `python/routers/recognition.py`

---

#### POST /recognize-face
Распознавание лица по эмбеддингу.

**Request:**
\`\`\`json
{
  "embedding": [0.123, 0.456, ...],
  "gallery_id": "uuid" // optional
}
\`\`\`

**Response:**
\`\`\`json
{
  "person_id": "uuid",
  "confidence": 0.85,
  "match_type": "verified"
}
\`\`\`

**Файл:** `python/routers/recognition.py`

---

### Training

#### POST /api/v2/training/cluster-unverified-faces
Кластеризация неизвестных лиц.

**Request:**
\`\`\`json
{
  "gallery_id": "uuid"
}
\`\`\`

**Response:**
\`\`\`json
{
  "clusters": [
    {
      "cluster_id": 0,
      "faces": [
        {
          "face_id": "uuid",
          "image_id": "uuid",
          "image_url": "https://...",
          "bbox": {...}
        }
      ],
      "size": 5
    }
  ],
  "total_faces": 20,
  "clustered_faces": 15,
  "noise_faces": 5
}
\`\`\`

**Файл:** `python/routers/training.py`

---

#### POST /rebuild-index
Перестроение HNSWLIB индекса.

**Request:** (пустой)

**Response:**
\`\`\`json
{
  "success": true,
  "message": "Index rebuilt successfully",
  "indexed_faces": 150
}
\`\`\`

**Файл:** `python/routers/training.py`

---

## Server Actions (Next.js)

Все Server Actions находятся в `app/admin/actions.ts`.

### uploadAction
Загрузка фото в галерею.

**Parameters:**
\`\`\`typescript
formData: FormData  // Содержит файлы и galleryId
\`\`\`

**Returns:**
\`\`\`typescript
{
  success: boolean;
  message: string;
  imageIds?: string[];
}
\`\`\`

---

### createPersonAction
Создание нового игрока.

**Parameters:**
\`\`\`typescript
{
  name: string;
  telegram_username?: string;
  instagram?: string;
  vk?: string;
  rating?: number;
  avatarUrl?: string;
}
\`\`\`

**Returns:**
\`\`\`typescript
{
  success: boolean;
  person?: Person;
  error?: string;
}
\`\`\`

---

### assignClusterToPersonAction
Назначение кластера лиц игроку.

**Parameters:**
\`\`\`typescript
{
  faceIds: string[];      // ID лиц из кластера
  personId: string;       // ID игрока
  galleryId: string;      // ID галереи
}
\`\`\`

**Returns:**
\`\`\`typescript
{
  success: boolean;
  message: string;
}
\`\`\`

---

### saveFaceTagsAction
Сохранение ручной разметки лиц на фото.

**Parameters:**
\`\`\`typescript
{
  imageId: string;
  tags: Array<{
    faceId: string;
    personId: string;
  }>;
}
\`\`\`

**Returns:**
\`\`\`typescript
{
  success: boolean;
  message: string;
}
\`\`\`

---

### sendPhotosToTelegramAction
Отправка фото игроку в Telegram.

**Parameters:**
\`\`\`typescript
{
  personId: string;
  photoIds: string[];
}
\`\`\`

**Returns:**
\`\`\`typescript
{
  success: boolean;
  message: string;
  sentCount: number;
}
\`\`\`

---

## Environment Variables

### Next.js (Vercel)
\`\`\`bash
# Supabase
POSTGRES_URL="postgresql://..."
SUPABASE_URL="https://..."
SUPABASE_ANON_KEY="..."
NEXT_PUBLIC_SUPABASE_URL="https://..."
NEXT_PUBLIC_SUPABASE_ANON_KEY="..."

# Vercel Blob
BLOB_READ_WRITE_TOKEN="..."

# Telegram
TELEGRAM_BOT_TOKEN="..."
NEXT_PUBLIC_TELEGRAM_BOT_USERNAME="..."

# FastAPI Backend
FASTAPI_URL="http://your-server:8001"
NEXT_PUBLIC_FASTAPI_URL="http://your-server:8001"
\`\`\`

### FastAPI (Hetzner)
\`\`\`bash
# Database
POSTGRES_URL="postgresql://..."

# Server
SERVER_IP="0.0.0.0"
SERVER_PORT="8001"
SERVER_HOST="https://your-domain.com"

# JWT
JWT_SECRET_KEY="..."

# CORS
ALLOWED_ORIGINS="https://your-domain.com,http://localhost:3000"
\`\`\`

---

## Error Codes

### 400 Bad Request
- Невалидные параметры запроса
- Отсутствуют обязательные поля

### 401 Unauthorized
- Отсутствует или невалиден токен аутентификации

### 404 Not Found
- Запрошенный ресурс не найден

### 500 Internal Server Error
- Ошибка на сервере
- Проблемы с базой данных
- Ошибка ML модели

### 503 Service Unavailable
- FastAPI бэкенд недоступен
- Supabase недоступен
