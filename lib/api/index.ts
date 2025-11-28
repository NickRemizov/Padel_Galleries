/**
 * Типизированный API клиент для CRUD операций
 * Использует Python FastAPI бэкенд через /api/crud/* эндпоинты
 *
 * Согласно PROJECT_RULES.md - все операции идут через Python API (Вариант В)
 */

import { apiFetch } from "@/lib/apiClient"
import type { Gallery, Photographer, Location, Organizer, Person } from "@/lib/types"

// ============================================================================
// ТИПЫ ДЛЯ CREATE/UPDATE ОПЕРАЦИЙ
// ============================================================================

// Galleries
export interface GalleryCreate {
  title: string
  shoot_date: string
  gallery_url: string
  external_gallery_url?: string | null
  cover_image_url?: string | null
  cover_image_square_url?: string | null
  photographer_id?: string | null
  location_id?: string | null
  organizer_id?: string | null
  sort_order?: string | null
}

export interface GalleryUpdate {
  title?: string
  shoot_date?: string
  gallery_url?: string
  external_gallery_url?: string | null
  cover_image_url?: string | null
  cover_image_square_url?: string | null
  photographer_id?: string | null
  location_id?: string | null
  organizer_id?: string | null
  sort_order?: string | null
}

// Photographers
export interface PhotographerCreate {
  name: string
}

export interface PhotographerUpdate {
  name?: string
}

// Locations
export interface LocationCreate {
  name: string
}

export interface LocationUpdate {
  name?: string
}

// Organizers
export interface OrganizerCreate {
  name: string
}

export interface OrganizerUpdate {
  name?: string
}

// People
export interface PersonCreate {
  real_name: string
  telegram_name?: string | null
  telegram_nickname?: string | null
  telegram_profile_url?: string | null
  facebook_profile_url?: string | null
  instagram_profile_url?: string | null
  avatar_url?: string | null
  paddle_ranking?: number | null
  tournament_results?: any | null
  show_in_players_gallery?: boolean
  show_photos_in_galleries?: boolean
  custom_confidence_threshold?: number | null
  use_custom_confidence?: boolean
  category?: string | null
}

export interface PersonUpdate {
  real_name?: string
  telegram_name?: string | null
  telegram_nickname?: string | null
  telegram_profile_url?: string | null
  facebook_profile_url?: string | null
  instagram_profile_url?: string | null
  avatar_url?: string | null
  paddle_ranking?: number | null
  tournament_results?: any | null
  show_in_players_gallery?: boolean
  show_photos_in_galleries?: boolean
  custom_confidence_threshold?: number | null
  use_custom_confidence?: boolean
  category?: string | null
}

// Faces API response types
export interface FaceTag {
  personId: string | null
  insightface_bbox: {
    x: number
    y: number
    width: number
    height: number
  } | null
  insightface_confidence: number | null
  recognition_confidence: number | null
  verified: boolean
  embedding: number[] | null
  bbox?: {
    x: number
    y: number
    width: number
    height: number
  } | null
}

export interface SaveFaceTagsResult {
  success: boolean
  message?: string
  faces?: Array<{
    face_id: string | null
    person_id: string | null
    verified: boolean
  }>
  verified?: boolean
}

// ClusterFace interface for createFromCluster method
export interface ClusterFace {
  photo_id: string
  descriptor: number[]
}

// ============================================================================
// GALLERIES API
// ============================================================================

export const galleriesApi = {
  /**
   * Получить все галереи
   * @param includeStats - включить статистику (количество фото и т.д.)
   */
  async getAll(includeStats = true): Promise<Gallery[]> {
    const params = new URLSearchParams()
    if (includeStats) params.append("include_stats", "true")
    return apiFetch<Gallery[]>(`/api/crud/galleries?${params}`)
  },

  /**
   * Получить галерею по ID
   */
  async getById(id: string): Promise<Gallery> {
    return apiFetch<Gallery>(`/api/crud/galleries/${id}`)
  },

  /**
   * Создать галерею
   */
  async create(data: GalleryCreate): Promise<Gallery> {
    return apiFetch<Gallery>("/api/crud/galleries", {
      method: "POST",
      body: JSON.stringify(data),
    })
  },

  /**
   * Обновить галерею
   */
  async update(id: string, data: GalleryUpdate): Promise<Gallery> {
    return apiFetch<Gallery>(`/api/crud/galleries/${id}`, {
      method: "PUT",
      body: JSON.stringify(data),
    })
  },

  /**
   * Удалить галерею
   */
  async delete(id: string): Promise<{ success: boolean; message: string }> {
    return apiFetch(`/api/crud/galleries/${id}`, {
      method: "DELETE",
    })
  },
}

// ============================================================================
// PHOTOGRAPHERS API
// ============================================================================

export const photographersApi = {
  /**
   * Получить всех фотографов
   */
  async getAll(): Promise<Photographer[]> {
    return apiFetch<Photographer[]>("/api/crud/photographers")
  },

  /**
   * Получить фотографа по ID
   */
  async getById(id: string): Promise<Photographer> {
    return apiFetch<Photographer>(`/api/crud/photographers/${id}`)
  },

  /**
   * Создать фотографа
   */
  async create(data: PhotographerCreate): Promise<Photographer> {
    return apiFetch<Photographer>("/api/crud/photographers", {
      method: "POST",
      body: JSON.stringify(data),
    })
  },

  /**
   * Обновить фотографа
   */
  async update(id: string, data: PhotographerUpdate): Promise<Photographer> {
    return apiFetch<Photographer>(`/api/crud/photographers/${id}`, {
      method: "PUT",
      body: JSON.stringify(data),
    })
  },

  /**
   * Удалить фотографа
   */
  async delete(id: string): Promise<{ success: boolean; message: string }> {
    return apiFetch(`/api/crud/photographers/${id}`, {
      method: "DELETE",
    })
  },
}

// ============================================================================
// LOCATIONS API
// ============================================================================

export const locationsApi = {
  /**
   * Получить все локации
   */
  async getAll(): Promise<Location[]> {
    return apiFetch<Location[]>("/api/crud/locations")
  },

  /**
   * Получить локацию по ID
   */
  async getById(id: string): Promise<Location> {
    return apiFetch<Location>(`/api/crud/locations/${id}`)
  },

  /**
   * Создать локацию
   */
  async create(data: LocationCreate): Promise<Location> {
    return apiFetch<Location>("/api/crud/locations", {
      method: "POST",
      body: JSON.stringify(data),
    })
  },

  /**
   * Обновить локацию
   */
  async update(id: string, data: LocationUpdate): Promise<Location> {
    return apiFetch<Location>(`/api/crud/locations/${id}`, {
      method: "PUT",
      body: JSON.stringify(data),
    })
  },

  /**
   * Удалить локацию
   */
  async delete(id: string): Promise<{ success: boolean; message: string }> {
    return apiFetch(`/api/crud/locations/${id}`, {
      method: "DELETE",
    })
  },
}

// ============================================================================
// ORGANIZERS API
// ============================================================================

export const organizersApi = {
  /**
   * Получить всех организаторов
   */
  async getAll(): Promise<Organizer[]> {
    return apiFetch<Organizer[]>("/api/crud/organizers")
  },

  /**
   * Получить организатора по ID
   */
  async getById(id: string): Promise<Organizer> {
    return apiFetch<Organizer>(`/api/crud/organizers/${id}`)
  },

  /**
   * Создать организатора
   */
  async create(data: OrganizerCreate): Promise<Organizer> {
    return apiFetch<Organizer>("/api/crud/organizers", {
      method: "POST",
      body: JSON.stringify(data),
    })
  },

  /**
   * Обновить организатора
   */
  async update(id: string, data: OrganizerUpdate): Promise<Organizer> {
    return apiFetch<Organizer>(`/api/crud/organizers/${id}`, {
      method: "PUT",
      body: JSON.stringify(data),
    })
  },

  /**
   * Удалить организатора
   */
  async delete(id: string): Promise<{ success: boolean; message: string }> {
    return apiFetch(`/api/crud/organizers/${id}`, {
      method: "DELETE",
    })
  },
}

// ============================================================================
// PEOPLE API
// ============================================================================

export const peopleApi = {
  /**
   * Получить всех людей
   * @param includeStats - включить статистику (количество фото)
   */
  async getAll(includeStats = true): Promise<Person[]> {
    const params = new URLSearchParams()
    if (includeStats) params.append("include_stats", "true")
    return apiFetch<Person[]>(`/api/crud/people?${params}`)
  },

  /**
   * Получить человека по ID
   */
  async getById(id: string): Promise<Person> {
    return apiFetch<Person>(`/api/crud/people/${id}`)
  },

  /**
   * Создать человека
   */
  async create(data: PersonCreate): Promise<Person> {
    return apiFetch<Person>("/api/crud/people", {
      method: "POST",
      body: JSON.stringify(data),
    })
  },

  /**
   * Обновить человека
   */
  async update(id: string, data: PersonUpdate): Promise<Person> {
    return apiFetch<Person>(`/api/crud/people/${id}`, {
      method: "PUT",
      body: JSON.stringify(data),
    })
  },

  /**
   * Удалить человека
   */
  async delete(id: string): Promise<{ success: boolean; message: string }> {
    return apiFetch(`/api/crud/people/${id}`, {
      method: "DELETE",
    })
  },

  /**
   * Создать человека из кластера лиц
   * @param personName - имя человека
   * @param clusterFaces - массив лиц с photo_id и descriptor
   */
  async createFromCluster(personName: string, clusterFaces: ClusterFace[]): Promise<Person> {
    return apiFetch<Person>("/api/crud/people/from-cluster", {
      method: "POST",
      body: JSON.stringify({
        person_name: personName,
        cluster_faces: clusterFaces,
      }),
    })
  },

  /**
   * Получить все фото персоны
   * @param personId - ID персоны
   */
  async getPhotos(personId: string): Promise<{ success: boolean; photos: any[]; count: number }> {
    return apiFetch(`/api/people/${personId}/photos`)
  },

  /**
   * Обновить аватар персоны
   * @param personId - ID персоны
   * @param avatarUrl - URL нового аватара
   */
  async updateAvatar(personId: string, avatarUrl: string): Promise<{ success: boolean; person: Person }> {
    return apiFetch(`/api/people/${personId}/avatar`, {
      method: "PUT",
      body: JSON.stringify({ avatar_url: avatarUrl }),
    })
  },
}

// ============================================================================
// STATS API
// ============================================================================

export const statsApi = {
  /**
   * Получить статистику распознавания лиц
   */
  async getRecognition(): Promise<any> {
    return apiFetch<any>("/api/crud/stats/recognition")
  },
}

// ============================================================================
// FACES API
// ============================================================================

export const facesApi = {
  /**
   * Batch save face tags for a photo
   * Deletes existing faces and saves new ones
   */
  async saveFaceTags(photoId: string, imageUrl: string, tags: FaceTag[]): Promise<SaveFaceTagsResult> {
    return apiFetch<SaveFaceTagsResult>("/api/faces/save-face-tags", {
      method: "POST",
      body: JSON.stringify({
        photo_id: photoId,
        image_url: imageUrl,
        tags: tags.map((tag) => ({
          person_id: tag.personId,
          bbox: tag.bbox || tag.insightface_bbox,
          insightface_bbox: tag.insightface_bbox,
          embedding: tag.embedding,
          verified: tag.verified ?? true,
        })),
      }),
    })
  },

  /**
   * Get all faces for a photo
   */
  async getPhotoFaces(photoId: string): Promise<{ success: boolean; faces: any[]; count: number }> {
    return apiFetch("/api/faces/get-photo-faces", {
      method: "POST",
      body: JSON.stringify({ photo_id: photoId }),
    })
  },

  /**
   * Get all faces for a person
   */
  async getPersonFaces(
    personId: string,
    verifiedOnly = false,
  ): Promise<{ success: boolean; faces: any[]; stats: any }> {
    return apiFetch("/api/faces/get-person-faces", {
      method: "POST",
      body: JSON.stringify({ person_id: personId, verified_only: verifiedOnly }),
    })
  },
}
