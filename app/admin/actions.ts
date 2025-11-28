"use server"

import { createClient } from "@/lib/supabase/server"
import { revalidatePath } from "next/cache"
import { redirect } from "next/navigation"
import { apiFetch } from "@/lib/apiClient"
import { sql } from "@/lib/db"
import { photographersApi, locationsApi, organizersApi, galleriesApi, peopleApi, facesApi } from "@/lib/api"

// Define types for clarity
type PhotoFaceRow = {
  id: string
  photo_id: string
  person_id: string | null
  verified: boolean
  insightface_bbox: any // This will be parsed or null
  insightface_confidence: number | null
  recognition_confidence: number | null // Added for recognition confidence
  blur_score: number | null
  created_at: string
  real_name: string | null
  avatar_url: string | null
}

type FaceTag = {
  personId: string | null
  insightface_bbox: { x: number; y: number; width: number; height: number }
  embedding?: number[]
  insightface_confidence: number | null
  recognition_confidence: number | null // Added for recognition confidence
  verified: boolean
}

type BoundingBox = { x: number; y: number; width: number; height: number }

export async function savePhotoFaceAction(
  photoId: string,
  personId: string | null,
  insightfaceBbox: { x: number; y: number; width: number; height: number } | null,
  insightfaceConfidence: number | null, // Detection confidence from InsightFace detector
  recognitionConfidence: number | null, // Recognition confidence from HNSWLIB matching
  verified: boolean,
  imageUrl?: string, // Added imageUrl, removed insightfaceDescriptor - backend generates embedding now
) {
  console.log("[v0] ===== SAVE PHOTO FACE ACTION STARTED =====")
  console.log("[v0] savePhotoFaceAction called:", {
    photoId,
    personId,
    verified,
    insightfaceConfidence,
    recognitionConfidence,
    hasImageUrl: !!imageUrl,
  })

  try {
    const response = await fetch(`${process.env.FASTAPI_URL}/api/faces/save`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        photo_id: photoId,
        person_id: personId,
        bbox: insightfaceBbox,
        insightface_confidence: insightfaceConfidence,
        recognition_confidence: recognitionConfidence,
        verified: verified,
        image_url: imageUrl, // Pass image_url for backend embedding generation
      }),
    })

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: "Unknown error" }))
      throw new Error(errorData.detail || `HTTP error! status: ${response.status}`)
    }

    const result = await response.json()

    if (!result.success) {
      console.error("[v0] Error saving photo face:", result.error)
      return {
        success: false,
        error: result.error || "Failed to save photo face",
      }
    }

    console.log("[v0] Photo face saved successfully:", {
      face_id: result.face_id,
      index_rebuilt: result.index_rebuilt,
    })

    if (result.index_rebuilt) {
      console.log("[v4.2] ✅ Recognition index was rebuilt")
    }

    return {
      success: true,
      data: { id: result.face_id },
      indexRebuilt: result.index_rebuilt,
    }
  } catch (error: any) {
    console.error("[v0] Error saving photo face:", error)
    return {
      success: false,
      error: error.message || "Failed to save photo face",
    }
  }
}

export async function signInAction(formData: FormData) {
  const email = formData.get("email") as string
  const password = formData.get("password") as string

  console.log("[v0] Sign in attempt for:", email)

  const supabase = createClient()

  const { error } = await supabase.auth.signInWithPassword({
    email,
    password,
  })

  if (error) {
    console.log("[v0] Sign in error:", error.message)
    return { error: error.message }
  }

  console.log("[v0] Sign in successful")
  revalidatePath("/admin", "layout")
  redirect("/admin")
}

export async function signUpAction(formData: FormData) {
  const email = formData.get("email") as string
  const password = formData.get("password") as string

  console.log("[v0] Sign up attempt for:", email)

  const supabase = createClient()

  const { error } = await supabase.auth.signUp({
    email,
    password,
    options: {
      emailRedirectTo:
        process.env.NEXT_PUBLIC_DEV_SUPABASE_REDIRECT_URL || `${process.env.NEXT_PUBLIC_SUPABASE_URL}/admin`,
      data: {
        email_confirm: false,
      },
    },
  })

  if (error) {
    console.log("[v0] Sign up error:", error.message)
    return { error: error.message }
  }

  console.log("[v0] Sign up successful")
  return { success: true }
}

export async function signOutAction() {
  const supabase = createClient()
  await supabase.auth.signOut()
  redirect("/admin/login")
}

export async function addGalleryAction(formData: FormData) {
  console.log("[v0] ===== ADD GALLERY ACTION STARTED =====")

  const title = formData.get("title") as string
  const shootDate = formData.get("shoot_date") as string
  const galleryUrl = (formData.get("gallery_url") as string) || "/gallery/pending"
  const externalGalleryUrl = (formData.get("external_gallery_url") as string) || null
  const coverImageUrl = formData.get("cover_image_url") as string
  const coverImageSquareUrl = (formData.get("cover_image_square_url") as string) || null
  const photographerId = formData.get("photographer_id") as string
  const locationId = formData.get("location_id") as string
  const organizerId = formData.get("organizer_id") as string

  console.log("[v0] Gallery data:", { title, shootDate, photographerId, locationId, organizerId })

  try {
    console.log("[v0] Calling galleriesApi.create...")
    const result = await galleriesApi.create({
      title,
      shoot_date: shootDate,
      gallery_url: galleryUrl,
      external_gallery_url: externalGalleryUrl || null,
      cover_image_url: coverImageUrl || null,
      cover_image_square_url: coverImageSquareUrl || null,
      photographer_id: photographerId && photographerId !== "none" ? photographerId : null,
      location_id: locationId && locationId !== "none" ? locationId : null,
      organizer_id: organizerId && organizerId !== "none" ? organizerId : null,
    })
    console.log("[v0] galleriesApi.create SUCCESS:", result)

    revalidatePath("/")
    revalidatePath("/admin")
    return { success: true }
  } catch (error: any) {
    console.error("[v0] ERROR in addGalleryAction:", error)
    console.error("[v0] Error stack:", error.stack)
    return { error: error.message || String(error) }
  }
}

export async function updateGalleryAction(id: string, formData: FormData) {
  console.log("[v0] ===== UPDATE GALLERY ACTION STARTED =====")
  console.log("[v0] Gallery ID:", id)

  const title = formData.get("title") as string
  const shootDate = formData.get("shoot_date") as string
  const galleryUrl = (formData.get("gallery_url") as string) || ""
  const externalGalleryUrl = (formData.get("external_gallery_url") as string) || null
  const coverImageUrl = formData.get("cover_image_url") as string
  const coverImageSquareUrl = (formData.get("cover_image_square_url") as string) || null
  const photographerId = formData.get("photographer_id") as string
  const locationId = formData.get("location_id") as string
  const organizerId = formData.get("organizer_id") as string

  try {
    console.log("[v0] Calling galleriesApi.update...")
    await galleriesApi.update(id, {
      title,
      shoot_date: shootDate,
      gallery_url: galleryUrl,
      external_gallery_url: externalGalleryUrl || null,
      cover_image_url: coverImageUrl || null,
      cover_image_square_url: coverImageSquareUrl || null,
      photographer_id: photographerId === "none" ? null : photographerId || null,
      location_id: locationId === "none" ? null : locationId || null,
      organizer_id: organizerId === "none" ? null : organizerId || null,
    })
    console.log("[v0] galleriesApi.update SUCCESS")

    revalidatePath("/")
    revalidatePath("/admin")
    return { success: true }
  } catch (error: any) {
    console.error("[v0] ERROR in updateGalleryAction:", error)
    return { error: error.message || String(error) }
  }
}

export async function deleteGalleryAction(id: string) {
  console.log("[v0] ===== DELETE GALLERY ACTION STARTED =====")
  console.log("[v0] Gallery ID:", id)

  try {
    console.log("[v0] Calling galleriesApi.delete...")
    await galleriesApi.delete(id)
    console.log("[v0] galleriesApi.delete SUCCESS")

    revalidatePath("/")
    revalidatePath("/admin")
    return { success: true }
  } catch (error: any) {
    console.error("[v0] ERROR in deleteGalleryAction:", error)
    return { error: error.message || String(error) }
  }
}

export async function addPhotographerAction(formData: FormData) {
  console.log("[v0] ===== ADD PHOTOGRAPHER ACTION STARTED =====")

  const name = formData.get("name") as string
  console.log("[v0] Photographer name:", name)

  try {
    console.log("[v0] Calling photographersApi.create...")
    const result = await photographersApi.create({ name })
    console.log("[v0] photographersApi.create SUCCESS:", result)

    revalidatePath("/admin")
    return { success: true }
  } catch (error: any) {
    console.error("[v0] ERROR in addPhotographerAction:", error)
    console.error("[v0] Error stack:", error.stack)
    return { error: error.message || String(error) }
  }
}

export async function updatePhotographerAction(id: string, formData: FormData) {
  console.log("[v0] ===== UPDATE PHOTOGRAPHER ACTION STARTED =====")
  console.log("[v0] Photographer ID:", id)

  const name = formData.get("name") as string

  try {
    console.log("[v0] Calling photographersApi.update...")
    await photographersApi.update(id, { name })
    console.log("[v0] photographersApi.update SUCCESS")

    revalidatePath("/admin")
    return { success: true }
  } catch (error: any) {
    console.error("[v0] ERROR in updatePhotographerAction:", error)
    return { error: error.message || String(error) }
  }
}

export async function deletePhotographerAction(id: string) {
  console.log("[v0] ===== DELETE PHOTOGRAPHER ACTION STARTED =====")
  console.log("[v0] Photographer ID:", id)

  try {
    console.log("[v0] Calling photographersApi.delete...")
    await photographersApi.delete(id)
    console.log("[v0] photographersApi.delete SUCCESS")

    revalidatePath("/admin")
    return { success: true }
  } catch (error: any) {
    console.error("[v0] ERROR in deletePhotographerAction:", error)
    return { error: error.message || String(error) }
  }
}

export async function addLocationAction(formData: FormData) {
  console.log("[v0] ===== ADD LOCATION ACTION STARTED =====")

  const name = formData.get("name") as string
  console.log("[v0] Location name:", name)

  try {
    console.log("[v0] Calling locationsApi.create...")
    const result = await locationsApi.create({ name })
    console.log("[v0] locationsApi.create SUCCESS:", result)

    revalidatePath("/admin")
    return { success: true }
  } catch (error: any) {
    console.error("[v0] ERROR in addLocationAction:", error)
    return { error: error.message || String(error) }
  }
}

export async function updateLocationAction(id: string, formData: FormData) {
  console.log("[v0] ===== UPDATE LOCATION ACTION STARTED =====")
  console.log("[v0] Location ID:", id)

  const name = formData.get("name") as string

  try {
    console.log("[v0] Calling locationsApi.update...")
    await locationsApi.update(id, { name })
    console.log("[v0] locationsApi.update SUCCESS")

    revalidatePath("/admin")
    return { success: true }
  } catch (error: any) {
    console.error("[v0] ERROR in updateLocationAction:", error)
    return { error: error.message || String(error) }
  }
}

export async function deleteLocationAction(id: string) {
  console.log("[v0] ===== DELETE LOCATION ACTION STARTED =====")
  console.log("[v0] Location ID:", id)

  try {
    console.log("[v0] Calling locationsApi.delete...")
    await locationsApi.delete(id)
    console.log("[v0] locationsApi.delete SUCCESS")

    revalidatePath("/admin")
    return { success: true }
  } catch (error: any) {
    console.error("[v0] ERROR in deleteLocationAction:", error)
    return { error: error.message || String(error) }
  }
}

export async function addOrganizerAction(formData: FormData) {
  console.log("[v0] ===== ADD ORGANIZER ACTION STARTED =====")

  const name = formData.get("name") as string
  console.log("[v0] Organizer name:", name)

  try {
    console.log("[v0] Calling organizersApi.create...")
    const result = await organizersApi.create({ name })
    console.log("[v0] organizersApi.create SUCCESS:", result)

    revalidatePath("/admin")
    return { success: true }
  } catch (error: any) {
    console.error("[v0] ERROR in addOrganizerAction:", error)
    return { error: error.message || String(error) }
  }
}

export async function updateOrganizerAction(id: string, formData: FormData) {
  console.log("[v0] ===== UPDATE ORGANIZER ACTION STARTED =====")
  console.log("[v0] Organizer ID:", id)

  const name = formData.get("name") as string

  try {
    console.log("[v0] Calling organizersApi.update...")
    await organizersApi.update(id, { name })
    console.log("[v0] organizersApi.update SUCCESS")

    revalidatePath("/admin")
    return { success: true }
  } catch (error: any) {
    console.error("[v0] ERROR in updateOrganizerAction:", error)
    return { error: error.message || String(error) }
  }
}

export async function deleteOrganizerAction(id: string) {
  console.log("[v0] ===== DELETE ORGANIZER ACTION STARTED =====")
  console.log("[v0] Organizer ID:", id)

  try {
    console.log("[v0] Calling organizersApi.delete...")
    await organizersApi.delete(id)
    console.log("[v0] organizersApi.delete SUCCESS")

    revalidatePath("/admin")
    return { success: true }
  } catch (error: any) {
    console.error("[v0] ERROR in deleteOrganizerAction:", error)
    return { error: error.message || String(error) }
  }
}

export async function addGalleryImagesAction(
  galleryId: string,
  imageUrls: {
    imageUrl: string
    originalUrl: string
    originalFilename: string
    fileSize: number
    width: number
    height: number
  }[],
) {
  try {
    const response = await fetch(`${process.env.FASTAPI_URL}/api/galleries/add-images`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        gallery_id: galleryId,
        images: imageUrls.map((img, index) => ({
          image_url: img.imageUrl,
          original_url: img.originalUrl,
          original_filename: img.originalFilename,
          file_size: img.fileSize,
          width: img.width,
          height: img.height,
        })),
      }),
    })

    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.detail || "Failed to add gallery images")
    }

    const result = await response.json()

    revalidatePath(`/gallery/${galleryId}`)
    revalidatePath("/admin")
    return { success: true }
  } catch (error) {
    console.error("[v0] Error adding gallery images:", error)
    return { error: error instanceof Error ? error.message : "Failed to add gallery images" }
  }
}

export async function getGalleryImagesAction(galleryId: string) {
  try {
    const response = await fetch(`${process.env.FASTAPI_URL}/api/galleries/${galleryId}/images`)

    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.detail || "Failed to load gallery images")
    }

    const result = await response.json()
    return result
  } catch (error) {
    console.error("[v0] Error loading gallery images:", error)
    return { success: false, error: error instanceof Error ? error.message : "Failed to load gallery images" }
  }
}

export async function deleteGalleryImageAction(imageId: string, galleryId: string) {
  try {
    const response = await fetch(`${process.env.FASTAPI_URL}/api/galleries/delete-image`, {
      method: "DELETE",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        image_id: imageId,
        gallery_id: galleryId,
      }),
    })

    if (!response.ok) {
      const error = await response.json()
      throw new Error(error.detail || "Failed to delete gallery image")
    }

    const result = await response.json()

    revalidatePath(`/gallery/${galleryId}`)
    revalidatePath("/admin")
    return { success: true }
  } catch (error: any) {
    console.error("[v0] Error deleting gallery image:", error)
    return { error: error.message }
  }
}

export async function deleteAllGalleryImagesAction(galleryId: string) {
  try {
    // Get all images for this gallery via Python API
    const imagesResponse = await apiFetch<{ success: boolean; data: any[] }>(`/api/galleries/${galleryId}/images`)

    if (!imagesResponse.success || !imagesResponse.data) {
      return { success: true } // No images to delete
    }

    const images = imagesResponse.data

    // Delete each image via Python API (which handles face cleanup)
    for (const image of images) {
      await apiFetch(`/api/galleries/delete-image`, {
        method: "DELETE",
        body: JSON.stringify({
          image_id: image.id,
          gallery_id: galleryId,
        }),
      })
    }

    console.log("[v0] Deleted all gallery images for gallery:", galleryId)

    revalidatePath(`/gallery/${galleryId}`)
    revalidatePath("/admin")
    return { success: true }
  } catch (error: any) {
    console.error("[v0] Error deleting all gallery images:", error)
    return { error: error.message }
  }
}

export async function updateGalleryImageOrderAction(galleryId: string, imageOrders: { id: string; order: number }[]) {
  try {
    // TODO: Create Python API endpoint for batch image order update
    // For now, update individually
    for (const item of imageOrders) {
      await apiFetch(`/api/galleries/images/${item.id}/order`, {
        method: "PUT",
        body: JSON.stringify({ display_order: item.order }),
      })
    }

    revalidatePath(`/gallery/${galleryId}`)
    revalidatePath("/admin")
    return { success: true }
  } catch (error: any) {
    console.error("[v0] Error updating image order:", error)
    return { error: error.message }
  }
}

export async function updateGallerySortOrderAction(galleryId: string, sortOrder: string) {
  try {
    await galleriesApi.update(galleryId, { sort_order: sortOrder })

    revalidatePath(`/gallery/${galleryId}`)
    revalidatePath("/admin")
    return { success: true }
  } catch (error: any) {
    console.error("[v0] Error updating gallery sort order:", error)
    return { error: error.message }
  }
}

export async function addPersonAction(formData: FormData) {
  const realName = formData.get("real_name") as string
  const telegramName = (formData.get("telegram_name") as string) || null
  const telegramNickname = (formData.get("telegram_nickname") as string) || null
  const telegramProfileUrl = (formData.get("telegram_profile_url") as string) || null
  const facebookProfileUrl = (formData.get("facebook_profile_url") as string) || null
  const instagramProfileUrl = (formData.get("instagram_profile_url") as string) || null
  const paddleRanking = formData.get("paddle_ranking")
    ? Number.parseInt(formData.get("paddle_ranking") as string)
    : null
  const avatarUrl = (formData.get("avatar_url") as string) || null

  console.log("[v0] addPersonAction: Starting person creation with data:", {
    realName,
    telegramName,
    telegramNickname,
    paddleRanking,
    avatarUrl,
  })

  try {
    console.log("[v0] addPersonAction: Calling peopleApi.create()...")

    const person = await peopleApi.create({
      real_name: realName,
      telegram_name: telegramName,
      telegram_nickname: telegramNickname,
      telegram_profile_url: telegramProfileUrl,
      facebook_profile_url: facebookProfileUrl,
      instagram_profile_url: instagramProfileUrl,
      paddle_ranking: paddleRanking,
      avatar_url: avatarUrl,
    })

    console.log("[v0] addPersonAction: CREATE successful, result:", person)

    revalidatePath("/admin")
    return { success: true, data: person }
  } catch (error: any) {
    console.error("[v0] addPersonAction ERROR:", {
      message: error.message,
      code: error.code,
      status: error.status,
    })
    return { success: false, error: error.message || String(error) }
  }
}

export async function updatePersonAction(id: string, formData: FormData) {
  const realName = formData.get("real_name") as string
  const telegramName = (formData.get("telegram_name") as string) || null
  const telegramNickname = (formData.get("telegram_nickname") as string) || null
  const telegramProfileUrl = (formData.get("telegram_profile_url") as string) || null
  const facebookProfileUrl = (formData.get("facebook_profile_url") as string) || null
  const instagramProfileUrl = (formData.get("instagram_profile_url") as string) || null
  const paddleRanking = formData.get("paddle_ranking")
    ? Number.parseInt(formData.get("paddle_ranking") as string)
    : null
  const tournamentResultsStr = (formData.get("tournament_results") as string) || "[]"

  let tournamentResults = []
  try {
    tournamentResults = JSON.parse(tournamentResultsStr)
  } catch (e) {
    console.error("[v0] Error parsing tournament results:", e)
    return { error: "Неверный формат JSON для результатов турниров" }
  }

  try {
    await peopleApi.update(id, {
      real_name: realName,
      telegram_name: telegramName,
      telegram_nickname: telegramNickname,
      telegram_profile_url: telegramProfileUrl,
      facebook_profile_url: facebookProfileUrl,
      instagram_profile_url: instagramProfileUrl,
      paddle_ranking: paddleRanking,
      tournament_results: tournamentResults,
    })

    revalidatePath("/admin")
    return { success: true }
  } catch (error: any) {
    console.error("[v0] Error updating person:", error)
    return { error: error.message || String(error) }
  }
}

export async function deletePersonAction(id: string) {
  // Python API автоматически сбрасывает photo_faces и удаляет face_descriptors
  try {
    await peopleApi.delete(id)
    revalidatePath("/admin")
    return { success: true }
  } catch (error: any) {
    console.error("[v0] Error deleting person:", error)
    return { error: error.message || String(error) }
  }
}

export async function saveFaceTagsAction(photoId: string, imageUrl: string, tags: FaceTag[]) {
  console.log("[v0] saveFaceTagsAction called with", tags.length, "tags for photo", photoId)

  try {
    const result = await facesApi.saveFaceTags(
      photoId,
      imageUrl,
      tags.map((tag) => ({
        personId: tag.personId,
        insightface_bbox: tag.insightface_bbox,
        insightface_confidence: tag.insightface_confidence,
        recognition_confidence: tag.recognition_confidence,
        verified: tag.verified,
        embedding: tag.embedding || null,
      })),
    )

    if (!result.success) {
      console.error("[v0] saveFaceTagsAction failed:", result)
      return { error: result.message || "Failed to save face tags" }
    }

    console.log("[v0] saveFaceTagsAction success:", result)
    revalidatePath("/admin")
    return { success: true }
  } catch (error: any) {
    console.error("[v0] Error saving face tags:", error)
    return { error: error.message }
  }
}

export async function getAllFaceDescriptorsAction() {
  try {
    // Python API should provide this endpoint if needed
    // For now, return empty as this might not be needed anymore
    console.log("[v0] getAllFaceDescriptorsAction - deprecated, face descriptors managed by Python")
    return { success: true, data: [] }
  } catch (error: any) {
    console.error("[v0] Error getting face descriptors:", error)
    return { error: error.message || "Failed to get face descriptors" }
  }
}

export async function saveFaceDescriptorAction(
  personId: string,
  insightfaceDescriptor: number[],
  sourceImageId: string,
) {
  console.log("[v0] saveFaceDescriptorAction - deprecated, descriptors saved automatically by Python API")
  try {
    // This is now handled automatically by Python API when saving faces
    // No need for separate descriptor save
    return { success: true, data: { message: "Handled by Python API" } }
  } catch (error: any) {
    console.error("[v0] Error saving face descriptor:", error)
    return { error: error.message || "Failed to save face descriptor" }
  }
}

export async function getPhotoFacesAction(photoId: string) {
  console.log("[v0] getPhotoFacesAction called for:", photoId)

  try {
    const response = await fetch(`${process.env.FASTAPI_URL}/api/faces/get-photo-faces`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        photo_id: photoId,
      }),
    })

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    const result = await response.json()

    if (!result.success) {
      throw new Error(result.error || "Failed to get photo faces")
    }

    console.log(`[v0] getPhotoFacesAction: Found ${result.faces?.length || 0} faces for photo ${photoId}`)

    const formattedFaces = result.faces.map((face: any) => ({
      id: face.id,
      photo_id: face.photo_id,
      person_id: face.person_id,
      verified: face.verified,
      insightface_bbox: face.insightface_bbox,
      insightface_confidence: face.insightface_confidence,
      recognition_confidence: face.recognition_confidence,
      blur_score: face.blur_score,
      created_at: face.created_at,
      people: face.person_name
        ? {
            real_name: face.person_name,
            avatar_url: null,
          }
        : null,
    }))

    return { success: true, data: formattedFaces }
  } catch (error: any) {
    console.error("[v0] Error loading photo faces:", error)
    return { success: false, error: error.message || "Failed to load photo faces" }
  }
}

export async function getBatchPhotoFacesAction(photoIds: string[]) {
  console.log("[v0] getBatchPhotoFacesAction called for:", photoIds.length, "photos")

  if (photoIds.length === 0) {
    return { success: true, data: [] }
  }

  try {
    const response = await fetch(`${process.env.FASTAPI_URL}/api/faces/get-batch-photo-faces`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-API-Key": process.env.API_SECRET_KEY || "",
      },
      body: JSON.stringify({ photo_ids: photoIds }),
    })

    if (!response.ok) {
      const error = await response.text()
      console.error("[v0] getBatchPhotoFacesAction error:", error)
      return { success: false, error: `API error: ${response.status}` }
    }

    const faces = await response.json()
    console.log(`[v0] getBatchPhotoFacesAction: Found ${faces?.length || 0} faces for ${photoIds.length} photos`)

    return {
      success: true,
      data: faces,
    }
  } catch (error: any) {
    console.error("[v0] Error getting batch photo faces:", error)
    return { success: false, error: error.message || "Failed to get batch photo faces" }
  }
}

export async function deletePhotoFaceAction(faceId: string) {
  try {
    const response = await fetch(`${process.env.FASTAPI_URL}/api/faces/delete/${faceId}`, {
      method: "DELETE",
      headers: { "Content-Type": "application/json" },
    })

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: "Unknown error" }))
      throw new Error(errorData.detail || `HTTP error! status: ${response.status}`)
    }

    const result = await response.json()

    if (!result.success) {
      throw new Error(result.error || "Failed to delete photo face")
    }

    revalidatePath("/admin")
    return { success: true }
  } catch (error: any) {
    console.error("[v0] Error deleting photo face:", error)
    return { success: false, error: error.message || "Failed to delete photo face" }
  }
}

export async function updatePhotoFaceAction(
  faceId: string,
  updates: {
    person_id?: string | null
    insightface_confidence?: number | null // Detection confidence
    recognition_confidence?: number | null // Recognition/matching confidence
    verified?: boolean
  },
) {
  try {
    const response = await fetch(`${process.env.FASTAPI_URL}/api/faces/update`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        face_id: faceId,
        person_id: updates.person_id,
        verified: updates.verified,
      }),
    })

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    const result = await response.json()

    if (!result.success) {
      throw new Error(result.error || "Failed to update photo face")
    }

    if (result.index_rebuilt) {
      console.log("[v4.2] ✅ Recognition index was rebuilt after update")
    }

    revalidatePath("/admin")
    return { success: true }
  } catch (error: any) {
    console.error("[v0] Error updating photo face:", error)
    return { error: error.message || "Failed to update photo face" }
  }
}

export async function getGalleryFaceRecognitionStatsAction(galleryId: string) {
  try {
    console.log("[v0] Getting face recognition stats for gallery:", galleryId)

    // Get all images for the gallery via Python API
    const imagesResponse = await apiFetch<{ success: boolean; data: any[] }>(`/api/galleries/${galleryId}/images`)

    if (!imagesResponse.success || !imagesResponse.data) {
      return { success: true, data: {} }
    }

    const images = imagesResponse.data
    console.log("[v0] Found images:", images.length)

    if (images.length === 0) {
      return { success: true, data: {} }
    }

    const imageIds = images.map((img) => img.id)

    // Get all photo faces for these images via Python API
    const facesResponse = await apiFetch<any[]>(`/api/faces/batch-get-photo-faces`, {
      method: "POST",
      body: JSON.stringify({ photo_ids: imageIds }),
    })

    const faces = facesResponse || []
    console.log("[v0] Found photo faces:", faces.length)

    // Calculate stats for each image
    const stats: Record<string, { total: number; recognized: number; fullyRecognized: boolean }> = {}

    for (const image of images) {
      const imageFaces = faces.filter((f: any) => f.photo_id === image.id)
      const recognizedFaces = imageFaces.filter((f: any) => f.person_id && f.verified)

      stats[image.id] = {
        total: imageFaces.length,
        recognized: recognizedFaces.length,
        fullyRecognized: imageFaces.length > 0 && imageFaces.length === recognizedFaces.length,
      }

      if (imageFaces.length > 0) {
        console.log(
          `[v0] Image ${image.id}: ${recognizedFaces.length}/${imageFaces.length} faces verified, fully: ${stats[image.id].fullyRecognized}`,
        )
      }
    }

    console.log("[v0] Stats calculated for", Object.keys(stats).length, "images")
    return { success: true, data: stats }
  } catch (error: any) {
    console.error("[v0] Error getting face recognition stats:", error)
    return { error: error.message || "Failed to get face recognition stats" }
  }
}

export async function getPersonPhotosAction(personId: string) {
  try {
    const response = await peopleApi.getPhotos(personId)
    return { success: true, data: response.photos || [] }
  } catch (error: any) {
    console.error("[v0] Error getting person photos:", error)
    return { error: error.message || "Failed to get person photos" }
  }
}

export async function updatePersonAvatarAction(personId: string, avatarUrl: string) {
  try {
    await peopleApi.updateAvatar(personId, avatarUrl)

    revalidatePath("/admin")
    return { success: true }
  } catch (error: any) {
    console.error("[v0] Error updating person avatar:", error)
    return { error: error.message || "Failed to update person avatar" }
  }
}

export async function getRecognitionStatsAction() {
  const supabase = createClient()

  try {
    console.log("[v0] [getRecognitionStatsAction] Starting...")

    const { count: peopleCount, error: peopleError } = await supabase
      .from("people")
      .select("*", { count: "exact", head: true })

    if (peopleError) throw peopleError
    console.log("[v0] [getRecognitionStatsAction] Total people:", peopleCount)

    let allPhotoFaces: any[] = []
    let hasMore = true
    let offset = 0
    const pageSize = 1000

    while (hasMore) {
      const { data: batch, error: batchError } = await supabase
        .from("photo_faces")
        .select("photo_id, person_id, verified, confidence, recognition_confidence") // Added recognition_confidence
        .range(offset, offset + pageSize - 1)

      if (batchError) throw batchError

      if (batch && batch.length > 0) {
        allPhotoFaces = allPhotoFaces.concat(batch)
        offset += pageSize
        hasMore = batch.length === pageSize
      } else {
        hasMore = false
      }
    }

    console.log("[v0] [getRecognitionStatsAction] Loaded", allPhotoFaces.length, "photo_faces records total")

    const verifiedCount = allPhotoFaces.filter((pf) => pf.verified).length
    const highConfidenceCount = allPhotoFaces.filter(
      (pf) =>
        (pf.confidence && pf.confidence >= 0.6 && !pf.verified) ||
        (pf.recognition_confidence && pf.recognition_confidence >= 0.6 && !pf.verified),
    ).length

    console.log("[v0] [getRecognitionStatsAction] Total verified faces:", verifiedCount)
    console.log("[v0] [getRecognitionStatsAction] Total high confidence (not verified) faces:", highConfidenceCount)

    const { count: descriptorsCount, error: descriptorsError } = await supabase
      .from("face_descriptors")
      .select("*", { count: "exact", head: true })

    if (descriptorsError) throw descriptorsError
    console.log("[v0] [getRecognitionStatsAction] Total descriptors:", descriptorsCount)

    const { data: peopleData, error: peopleDataError } = await supabase
      .from("people")
      .select("id, real_name, telegram_name")
      .order("real_name")

    if (peopleDataError) throw peopleDataError
    console.log("[v0] [getRecognitionStatsAction] Processing", peopleData?.length, "people")

    const facesByPerson = new Map<string, any[]>()
    for (const face of allPhotoFaces) {
      if (!face.person_id) continue // Skip if person_id is null or undefined
      if (!facesByPerson.has(face.person_id)) {
        facesByPerson.set(face.person_id, [])
      }
      facesByPerson.get(face.person_id)!.push(face)
    }

    const peopleStats = []
    for (const person of peopleData || []) {
      const photoFaces = facesByPerson.get(person.id) || []

      const verifiedPhotoIds = new Set(photoFaces.filter((pf) => pf.verified).map((pf) => pf.photo_id))

      const highConfPhotoIds = new Set(
        photoFaces
          .filter(
            (pf) =>
              (pf.confidence && pf.confidence >= 0.6 && !pf.verified) ||
              (pf.recognition_confidence && pf.recognition_confidence >= 0.6 && !pf.verified),
          )
          .map((pf) => pf.photo_id),
      )

      const totalConfirmed = verifiedPhotoIds.size + highConfPhotoIds.size

      const { count: personDescCount, error: descError } = await supabase
        .from("face_descriptors")
        .select("*", { count: "exact", head: true })
        .eq("person_id", person.id)

      // Debug logging for Сергей Цырульник
      if (person.real_name === "Сергей Цырульник") {
        console.log("[v0] [DEBUG] Сергей Цырульник descriptor query:")
        console.log("[v0] [DEBUG]   Person ID:", person.id)
        console.log("[v0] [DEBUG]   Count result:", personDescCount)
        console.log("[v0] [DEBUG]   Error:", descError)

        // Also fetch actual descriptors to see what's there
        const { data: actualDescriptors, error: fetchError } = await supabase
          .from("face_descriptors")
          .select("id, person_id, created_at")
          .eq("person_id", person.id)

        console.log("[v0] [DEBUG]   Actual descriptors:", actualDescriptors)
        console.log("[v0] [DEBUG]   Fetch error:", fetchError)
      }

      peopleStats.push({
        id: person.id,
        name: person.real_name,
        telegramName: person.telegram_name,
        verifiedPhotos: verifiedPhotoIds.size,
        highConfidencePhotos: highConfPhotoIds.size,
        descriptors: personDescCount || 0,
        totalConfirmed: totalConfirmed,
      })
    }

    peopleStats.sort((a, b) => b.totalConfirmed - a.totalConfirmed)

    console.log("[v0] [getRecognitionStatsAction] Completed successfully")

    return {
      success: true,
      data: {
        summary: {
          totalPeople: peopleCount || 0,
          totalVerifiedFaces: verifiedCount,
          totalHighConfidenceFaces: highConfidenceCount,
          totalDescriptors: descriptorsCount || 0,
        },
        peopleStats: peopleStats,
      },
    }
  } catch (error: any) {
    console.error("[v0] Error getting recognition stats:", error)
    return { error: error.message || "Failed to get recognition stats" }
  }
}

export async function getPersonPhotosWithDetailsAction(personId: string) {
  try {
    console.log("[v0] ===== getPersonPhotosWithDetailsAction START =====")
    console.log("[v0] Querying photos for person_id:", personId)

    // Use Python API to get person photos with details
    const response = await peopleApi.getPhotos(personId)

    if (!response.success) {
      throw new Error("Failed to get person photos")
    }

    const photos = response.photos || []
    console.log("[v0] Found", photos.length, "photos for person")

    return { success: true, data: photos }
  } catch (error: any) {
    console.error("[v0] Error getting person photos with details:", error)
    return { error: error.message || "Failed to get person photos" }
  }
}

export async function deleteFaceDescriptorsForPhotoAction(photoId: string, personId: string) {
  const supabase = createClient()

  try {
    const { error: descError } = await supabase
      .from("face_descriptors")
      .delete()
      .eq("source_image_id", photoId)
      .eq("person_id", personId)

    if (descError) throw descError

    const { error: faceError } = await supabase
      .from("photo_faces")
      .delete()
      .eq("photo_id", photoId)
      .eq("person_id", personId)

    if (faceError) throw faceError

    revalidatePath("/admin")
    return { success: true }
  } catch (error: any) {
    console.error("[v0] Error deleting face descriptors:", error)
    return { error: error.message || "Failed to delete face descriptors" }
  }
}

export async function verifyPersonOnPhotoAction(photoId: string, personId: string) {
  const supabase = createClient()

  try {
    const { error } = await supabase
      .from("photo_faces")
      .update({
        verified: true,
        confidence: 1.0,
        recognition_confidence: 1.0, // Set recognition confidence to 1.0 as well
      })
      .eq("photo_id", photoId)
      .eq("person_id", personId) // Added this to update specific person on photo

    if (error) throw error

    revalidatePath("/admin")
    return { success: true }
  } catch (error: any) {
    console.error("[v0] Error verifying person on photo:", error)
    return { error: error.message || "Failed to verify person" }
  }
}

export async function updatePersonVisibilityAction(
  personId: string,
  field: "show_in_players_gallery" | "show_photos_in_galleries",
  value: boolean,
) {
  try {
    // Safe to use field directly since it's typed and validated by TypeScript
    await sql.unsafe(`
      UPDATE people
      SET ${field} = ${value}
      WHERE id = '${personId}'
    `)

    revalidatePath("/admin")
    return { success: true }
  } catch (error: any) {
    console.error("[v0] Error updating person visibility:", error)
    return { error: error.message || "Failed to update person visibility" }
  }
}

export async function debugPersonPhotosAction(personRealName: string) {
  const supabase = createClient()

  try {
    const { data: person, error: personError } = await supabase
      .from("people")
      .select("id, real_name")
      .eq("real_name", personRealName)
      .single()

    if (personError || !person) {
      return { error: `Person "${personRealName}" not found` }
    }

    console.log(`[v0] Debug: Found person ${person.real_name} (${person.id})`)

    const { data: faceDescriptors, error: descriptorsError } = await supabase
      .from("face_descriptors")
      .select("id, person_id, source_image_id, created_at")
      .eq("person_id", person.id)
      .order("created_at", { ascending: true })

    if (descriptorsError) throw descriptorsError

    console.log(`[v0] Debug: Found ${faceDescriptors?.length || 0} face_descriptors for ${person.real_name}`)

    const { data: photoFaces, error: facesError } = await supabase
      .from("photo_faces")
      .select(
        "id, photo_id, person_id, confidence, recognition_confidence, verified, gallery_images(id, original_filename, gallery_id)",
      ) // Added recognition_confidence
      .eq("person_id", person.id)

    if (facesError) throw facesError

    console.log(`[v0] Debug: Found ${photoFaces?.length || 0} photo_faces records for ${person.real_name}`)

    const photoIds = [...new Set(photoFaces?.map((pf) => pf.photo_id) || [])]
    console.log(`[v0] Debug: Unique photos: ${photoIds.length}`)

    const descriptorsByPhoto = new Map<string, any[]>()
    for (const desc of faceDescriptors || []) {
      if (!desc.source_image_id) continue
      if (!descriptorsByPhoto.has(desc.source_image_id)) {
        descriptorsByPhoto.set(desc.source_image_id, [])
      }
      descriptorsByPhoto.get(desc.source_image_id)!.push(desc)
    }

    const photoDetails = []
    for (const photoId of photoIds) {
      const { data: allFacesOnPhoto, error: allFacesError } = await supabase
        .from("photo_faces")
        .select("id, person_id, confidence, recognition_confidence, verified, people(real_name, telegram_name)") // Added recognition_confidence
        .eq("photo_id", photoId)

      if (allFacesError) {
        console.error(`[v0] Error getting faces for photo ${photoId}:`, allFacesError)
        continue
      }

      const photoFace = photoFaces?.find((pf) => pf.photo_id === photoId)
      const filename = photoFace?.gallery_images?.original_filename || "Unknown"

      const facesInfo =
        allFacesOnPhoto?.map((face) => ({
          personName: face.people?.real_name || face.people?.telegram_name || "Unknown",
          personId: face.person_id,
          verified: face.verified,
          confidence: face.confidence,
          recognition_confidence: face.recognition_confidence, // Added recognition_confidence
        })) || []

      const descriptorsForPhoto = descriptorsByPhoto.get(photoId) || []

      photoDetails.push({
        photoId,
        filename,
        faces: facesInfo,
        descriptors: descriptorsForPhoto,
      })
    }

    photoDetails.sort((a, b) => a.filename.localeCompare(b.filename))

    console.log(`\n[v0] ========== DEBUG: All photos and descriptors for ${person.real_name} ==========`)
    console.log(`[v0] Total face_descriptors: ${faceDescriptors?.length || 0}`)
    console.log(`[v0] Total photo_faces: ${photoFaces?.length || 0}`)
    console.log(`[v0] Unique photos: ${photoIds.length}`)

    photoDetails.forEach((photo, index) => {
      console.log(`[v0] Photo ${index + 1}: ${photo.filename}`)
      console.log(`[v0]   Photo ID: ${photo.photoId}`)
      console.log(`[v0]   Total faces on photo: ${photo.faces.length}`)
      console.log(`[v0]   Descriptors for this photo: ${photo.descriptors.length}`)

      if (photo.descriptors.length > 0) {
        console.log(`[v0]   Descriptor IDs:`)
        photo.descriptors.forEach((desc, descIndex) => {
          console.log(`[v0]     ${descIndex + 1}. ID: ${desc.id}, Created: ${desc.created_at}`)
        })
      }

      photo.faces.forEach((face, faceIndex) => {
        console.log(`[v0]   Face ${faceIndex + 1}:`)
        console.log(`[v0]     Person: ${face.personName} (ID: ${face.personId})`)
        console.log(`[v0]     Verified: ${face.verified}`)
        console.log(`[v0]     Confidence: ${face.confidence}`)
        console.log(`[v0]     Recognition Confidence: ${face.recognition_confidence}`)
      })
    })

    const orphanedDescriptors = (faceDescriptors || []).filter((desc) => !photoIds.includes(desc.source_image_id || ""))

    if (orphanedDescriptors.length > 0) {
      console.log(`\n[v0] WARNING: Found ${orphanedDescriptors.length} orphaned descriptors (no matching photo_faces):`)
      orphanedDescriptors.forEach((desc, index) => {
        console.log(`[v0]   Orphaned descriptor ${index + 1}:`)
        console.log(`[v0]     ID: ${desc.id}`)
        console.log(`[v0]     Source Image ID: ${desc.source_image_id || "NULL"}`)
        console.log(`[v0]     Created: ${desc.created_at}`)
      })
    }

    console.log(`[v0] ========== END DEBUG ==========\n`)

    return {
      success: true,
      data: {
        personName: person.real_name,
        totalDescriptors: faceDescriptors?.length || 0,
        totalPhotoFacesRecords: photoFaces?.length || 0,
        uniquePhotos: photoIds.length,
        orphanedDescriptors: orphanedDescriptors.length,
        photos: photoDetails,
      },
    }
  } catch (error: any) {
    console.error("[v0] Error in debug action:", error)
    return { error: error.message || "Failed to debug person photos" }
  }
}

export async function syncVerifiedAndConfidenceAction() {
  const supabase = createClient()

  try {
    console.log("[v0] Starting sync of verified and confidence fields...")

    const { data: verifiedRecords, error: verifiedError } = await supabase
      .from("photo_faces")
      .update({ confidence: 1.0, recognition_confidence: 1.0 }) // Updated to include recognition_confidence
      .eq("verified", true)
      .select("id")

    if (verifiedError) throw verifiedError

    const verifiedCount = verifiedRecords?.length || 0
    console.log(
      `[v0] Updated ${verifiedCount} records: set confidence=1 and recognition_confidence=1 where verified=true`,
    )

    const { data: confidenceRecords, error: confidenceError } = await supabase
      .from("photo_faces")
      .update({ verified: true })
      .or("confidence.eq.1.0,recognition_confidence.eq.1.0") // Updated to check both confidences
      .select("id")

    if (confidenceError) throw confidenceError

    const confidenceCount = confidenceRecords?.length || 0
    console.log(
      `[v0] Updated ${confidenceCount} records: set verified=true where confidence=1 or recognition_confidence=1`,
    )

    const { count: totalVerified } = await supabase
      .from("photo_faces")
      .select("*", { count: "exact", head: true })
      .eq("verified", true)

    const { count: totalConfidence1 } = await supabase
      .from("photo_faces")
      .select("*", { count: "exact", head: true })
      .eq("confidence", 1.0)

    const { count: totalRecognitionConfidence1 } = await supabase
      .from("photo_faces")
      .select("*", { count: "exact", head: true })
      .eq("recognition_confidence", 1.0)

    console.log(
      `[v0] Final stats: ${totalVerified} verified records, ${totalConfidence1} confidence=1 records, ${totalRecognitionConfidence1} recognition_confidence=1 records`,
    )

    revalidatePath("/admin")
    return {
      success: true,
      data: {
        updatedVerified: verifiedCount,
        updatedConfidence: confidenceCount,
        totalVerified: totalVerified || 0,
        totalConfidence1: totalConfidence1 || 0,
        totalRecognitionConfidence1: totalRecognitionConfidence1 || 0,
      },
    }
  } catch (error: any) {
    console.error("[v0] Error syncing verified and confidence:", error)
    return { error: error.message || "Failed to sync verified and confidence" }
  }
}

export async function debugPhotoFacesAction(filename: string) {
  const supabase = createClient()

  try {
    const { data: photo, error: photoError } = await supabase
      .from("gallery_images")
      .select("id, original_filename, gallery_id")
      .eq("original_filename", filename)
      .single()

    if (photoError || !photo) {
      return { error: `Photo "${filename}" not found` }
    }

    console.log(`[v0] Debug: Found photo ${photo.original_filename} (${photo.id})`)

    const { data: photoFaces, error: facesError } = await supabase
      .from("photo_faces")
      .select(
        "id, photo_id, person_id, confidence, recognition_confidence, verified, insightface_bbox, people(real_name, telegram_name)",
      ) // Added recognition_confidence
      .eq("photo_id", photo.id)

    if (facesError) throw facesError

    console.log(`[v0] Debug: Found ${photoFaces?.length || 0} faces on photo ${photo.original_filename}`)

    console.log(`\n[v0] ========== DEBUG: Faces on photo "${photo.original_filename}" ==========`)
    console.log(`[v0] Photo ID: ${photo.id}`)
    console.log(`[v0] Gallery ID: ${photo.gallery_id}`)
    console.log(`[v0] Total faces detected: ${photoFaces?.length || 0}`)

    photoFaces?.forEach((face, index) => {
      const personName = face.people?.real_name || face.people?.telegram_name || "Unknown"
      console.log(`\n[v0] Face ${index + 1}:`)
      console.log(`[v0]   Person: ${personName}`)
      console.log(`[v0]   Person ID: ${face.person_id}`)
      console.log(`[v0]   Verified: ${face.verified}`)
      console.log(`[v0]   Confidence: ${face.confidence}`)
      console.log(`[v0]   Recognition Confidence: ${face.recognition_confidence}`) // Added recognition_confidence
      console.log(`[v0]   Bounding Box:`, face.insightface_bbox) // Use insightface_bbox
    })
    console.log(`[v0] ========== END DEBUG ==========\n`)

    return {
      success: true,
      data: {
        photoId: photo.id,
        filename: photo.original_filename,
        galleryId: photo.gallery_id,
        totalFaces: photoFaces?.length || 0,
        faces:
          photoFaces?.map((face) => ({
            personName: face.people?.real_name || face.people?.telegram_name || "Unknown",
            personId: face.person_id,
            verified: face.verified,
            confidence: face.confidence,
            recognition_confidence: face.recognition_confidence, // Added recognition_confidence
            boundingBox: face.insightface_bbox, // Use insightface_bbox
          })) || [],
      },
    }
  } catch (error: any) {
    console.error("[v0] Error in debug photo action:", error)
    return { error: error.message || "Failed to debug photo faces" }
  }
}

export async function cleanupUnverifiedFacesAction() {
  const supabase = createClient()

  try {
    console.log("[v0] Starting cleanup of unverified face descriptors...")

    const { count: totalBefore } = await supabase.from("photo_faces").select("*", { count: "exact", head: true })

    const { count: verifiedBefore } = await supabase
      .from("photo_faces")
      .select("*", { count: "exact", head: true })
      .eq("verified", true)

    console.log(`[v0] Before cleanup: ${totalBefore} total records, ${verifiedBefore} verified`)

    const { data: inconsistentRecords, error: inconsistentError } = await supabase
      .from("photo_faces")
      .select(
        "id, photo_id, person_id, verified, confidence, recognition_confidence, people(real_name, telegram_name), gallery_images(original_filename)", // Added recognition_confidence
      )
      .eq("verified", true)
      .or("confidence.neq.1.0,recognition_confidence.neq.1.0") // Updated condition

    if (inconsistentError) throw inconsistentError

    if (inconsistentRecords && inconsistentRecords.length > 0) {
      console.log(
        `[v0] WARNING: Found ${inconsistentRecords.length} records with verified=true but confidence/recognition_confidence != 1:`,
      )
      inconsistentRecords.forEach((record) => {
        const personName = record.people?.real_name || record.people?.telegram_name || "Unknown"
        const filename = record.gallery_images?.original_filename || "Unknown"
        console.log(
          `[v0]   - Photo: ${filename}, Person: ${personName}, Confidence: ${record.confidence}, Recognition Confidence: ${record.recognition_confidence}`,
        )
      })
    }

    const { data: unverifiedFaces, error: fetchError } = await supabase
      .from("photo_faces")
      .select("photo_id, person_id")
      .eq("verified", false)

    if (fetchError) throw fetchError

    console.log(`[v0] Found ${unverifiedFaces?.length || 0} unverified photo_faces to delete`)

    if (unverifiedFaces && unverifiedFaces.length > 0) {
      // Group by photo_id and person_id to delete matching descriptors
      const descriptorsToDelete = unverifiedFaces.map((face) => ({
        source_image_id: face.photo_id,
        person_id: face.person_id,
      }))

      let deletedDescriptorsCount = 0
      // Delete descriptors in batches to avoid query size limits
      for (const { source_image_id, person_id } of descriptorsToDelete) {
        if (!person_id) continue // Skip if no person_id

        const { data: deletedDescs, error: descError } = await supabase
          .from("face_descriptors")
          .delete()
          .eq("source_image_id", source_image_id)
          .eq("person_id", person_id)
          .select("id")

        if (descError) {
          console.error(`[v0] Error deleting descriptors for image ${source_image_id}:`, descError)
        } else {
          deletedDescriptorsCount += deletedDescs?.length || 0
        }
      }

      console.log(`[v0] Deleted ${deletedDescriptorsCount} corresponding face_descriptors`)
    }

    const { data: deletedRecords, error: deleteError } = await supabase
      .from("photo_faces")
      .delete()
      .eq("verified", false)
      .select("id")

    if (deleteError) throw deleteError

    const deletedCount = deletedRecords?.length || 0
    console.log(`[v0] Deleted ${deletedCount} unverified photo_faces records`)

    const { count: totalAfter } = await supabase.from("photo_faces").select("*", { count: "exact", head: true })

    const { count: verifiedAfter } = await supabase
      .from("photo_faces")
      .select("*", { count: "exact", head: true })
      .eq("verified", true)

    console.log(`[v0] After cleanup: ${totalAfter} total records, ${verifiedAfter} verified`)
    console.log(`[v0] Cleanup complete!`)

    revalidatePath("/admin")
    return {
      success: true,
      data: {
        before: {
          total: totalBefore || 0,
          verified: verifiedBefore || 0,
        },
        after: {
          total: totalAfter || 0,
          verified: verifiedAfter || 0,
        },
        deleted: deletedCount,
        inconsistentRecords:
          inconsistentRecords?.map((record) => ({
            photoFilename: record.gallery_images?.original_filename || "Unknown",
            personName: record.people?.real_name || record.people?.telegram_name || "Unknown",
            confidence: record.confidence,
            recognition_confidence: record.recognition_confidence,
          })) || [],
      },
    }
  } catch (error: any) {
    console.error("[v0] Error cleaning up unverified faces:", error)
    return { error: error.message || "Failed to cleanup unverified faces" }
  }
}

export async function cleanupDuplicateFacesAction() {
  const supabase = createClient()

  try {
    console.log("[v0] Starting cleanup of duplicate face descriptors...")

    const { count: totalBefore } = await supabase.from("photo_faces").select("*", { count: "exact", head: true })

    console.log(`[v0] Before cleanup: ${totalBefore} total photo_faces records`)

    let allRecords: any[] = []
    let hasMore = true
    let offset = 0
    const pageSize = 1000

    while (hasMore) {
      const { data: batch, error: batchError } = await supabase
        .from("photo_faces")
        .select("id, photo_id, person_id, verified, confidence, recognition_confidence, created_at") // Added recognition_confidence
        .order("created_at", { ascending: true })
        .range(offset, offset + pageSize - 1)

      if (batchError) throw batchError

      if (batch && batch.length > 0) {
        allRecords = allRecords.concat(batch)
        offset += pageSize
        hasMore = batch.length === pageSize
      } else {
        hasMore = false
      }
    }

    console.log(`[v0] Loaded ${allRecords.length} records for analysis`)

    const groupedRecords = new Map<string, any[]>()
    for (const record of allRecords) {
      const key = `${record.person_id}_${record.photo_id}`
      if (!groupedRecords.has(key)) {
        groupedRecords.set(key, [])
      }
      groupedRecords.get(key)!.push(record)
    }

    const duplicateGroups = Array.from(groupedRecords.entries()).filter(([_, records]) => records.length > 1)

    console.log(`[v0] Found ${duplicateGroups.length} groups with duplicates`)

    let totalDeleted = 0
    const idsToDelete: string[] = []

    for (const [key, records] of duplicateGroups) {
      records.sort((a, b) => {
        if (a.verified !== b.verified) return b.verified ? 1 : -1
        // Prioritize higher confidence, then recognition_confidence, then creation date
        if (a.recognition_confidence !== b.recognition_confidence)
          return (b.recognition_confidence || 0) - (a.recognition_confidence || 0)
        if (a.confidence !== b.confidence) return (b.confidence || 0) - (a.confidence || 0)
        return new Date(a.created_at).getTime() - new Date(b.created_at).getTime()
      })

      const toKeep = records[0]
      const toDelete = records.slice(1)

      console.log(
        `[v0] Group ${key}: keeping record ${toKeep.id} (verified=${toKeep.verified}, confidence=${toKeep.confidence}, recognition_confidence=${toKeep.recognition_confidence}, created_at=${toKeep.created_at}), deleting ${toDelete.length} duplicates`,
      )

      idsToDelete.push(...toDelete.map((r) => r.id))
      totalDeleted += toDelete.length
    }

    const batchSize = 100

    if (idsToDelete.length > 0) {
      for (let i = 0; i < idsToDelete.length; i += batchSize) {
        const batch = idsToDelete.slice(i, i + batchSize)
        const { error: deleteError } = await supabase.from("photo_faces").delete().in("id", batch)

        if (deleteError) {
          console.error(`[v0] Error deleting descriptor batch ${i / batchSize + 1}:`, deleteError)
          throw deleteError
        }
      }
    }

    console.log(`[v0] Deleted ${totalDeleted} duplicate photo_faces records`)

    console.log("[v0] Cleaning up orphaned face_descriptors...")

    const validCombinations = new Set(
      allRecords.filter((r) => !idsToDelete.includes(r.id)).map((r) => `${r.person_id}_${r.photo_id}`),
    )

    let allDescriptors: any[] = []
    let descOffset = 0
    let descHasMore = true
    const descriptorPageSize = 1000

    while (descHasMore) {
      const { data: descBatch, error: descError } = await supabase
        .from("face_descriptors")
        .select("id, person_id, source_image_id")
        .range(descOffset, descOffset + descriptorPageSize - 1)

      if (descError) throw descError

      if (descBatch && descBatch.length > 0) {
        allDescriptors = allDescriptors.concat(descBatch)
        descOffset += descriptorPageSize
        descHasMore = descBatch.length === pageSize
      } else {
        hasMore = false
      }
    }

    console.log(`[v0] Loaded ${allDescriptors.length} face_descriptors for analysis`)

    const orphanedDescriptorIds = allDescriptors
      .filter((desc) => {
        const key = `${desc.person_id}_${desc.source_image_id}`
        return !validCombinations.has(key)
      })
      .map((desc) => desc.id)

    console.log(`[v0] Found ${orphanedDescriptorIds.length} orphaned face_descriptors`)

    let deletedDescriptors = 0
    if (orphanedDescriptorIds.length > 0) {
      const descriptorBatchSize = 100
      for (let i = 0; i < orphanedDescriptorIds.length; i += descriptorBatchSize) {
        const batch = orphanedDescriptorIds.slice(i, i + descriptorBatchSize)
        const { error: deleteDescError } = await supabase.from("face_descriptors").delete().in("id", batch)

        if (deleteDescError) {
          console.error(`[v0] Error deleting descriptor batch ${i / batchSize + 1}:`, deleteDescError)
          throw deleteDescError
        }
        deletedDescriptors += batch.length
      }
    }

    console.log(`[v0] Deleted ${deletedDescriptors} orphaned face_descriptors`)

    const { count: totalAfter } = await supabase.from("photo_faces").select("*", { count: "exact", head: true })

    const { count: verifiedAfter } = await supabase
      .from("photo_faces")
      .select("*", { count: "exact", head: true })
      .eq("verified", true)

    const { count: descriptorsAfter } = await supabase
      .from("face_descriptors")
      .select("*", { count: "exact", head: true })

    console.log(
      `[v0] After cleanup: ${totalAfter} photo_faces records, ${verifiedAfter} verified, ${descriptorsAfter} face_descriptors`,
    )
    console.log(`[v0] Cleanup complete!`)

    revalidatePath("/admin")
    return {
      success: true,
      data: {
        before: {
          total: totalBefore || 0,
        },
        after: {
          total: totalAfter || 0,
          verified: verifiedAfter || 0,
          descriptors: descriptorsAfter || 0,
        },
        deleted: totalDeleted,
        deletedDescriptors: deletedDescriptors,
        duplicateGroups: duplicateGroups.length,
      },
    }
  } catch (error: any) {
    console.error("[v0] Error cleaning up duplicate faces:", error)
    return { error: error.message || "Failed to cleanup duplicate faces" }
  }
}

export async function cleanupPersonDescriptorsAction(personRealName: string) {
  const supabase = createClient()

  try {
    console.log("[v0] Starting cleanup for person:", personRealName)

    const { data: person, error: personError } = await supabase
      .from("people")
      .select("id, real_name")
      .eq("real_name", personRealName)
      .single()

    if (personError || !person) {
      return { error: `Person not found: ${personRealName}` }
    }

    console.log("[v0] Found person:", person.real_name, person.id)

    const { count: descriptorsBefore } = await supabase
      .from("face_descriptors")
      .select("id", { count: "exact", head: true })
      .eq("person_id", person.id)

    console.log("[v0] Before cleanup:", descriptorsBefore, "descriptors")

    const { data: photoFaces, error: pfError } = await supabase
      .from("photo_faces")
      .select("photo_id, person_id, verified, confidence, recognition_confidence") // Added recognition_confidence
      .eq("person_id", person.id)

    if (pfError) throw pfError

    console.log("[v0] Loaded", photoFaces?.length || 0, "photo_faces records")

    const verifiedMap = new Map<string, boolean>()
    for (const pf of photoFaces || []) {
      const key = `${pf.person_id}_${pf.photo_id}`
      verifiedMap.set(key, pf.verified)
    }

    const { data: descriptors, error: descError } = await supabase
      .from("face_descriptors")
      .select("id, person_id, source_image_id, created_at")
      .eq("person_id", person.id)
      .order("created_at", { ascending: true })

    if (descError) throw descError

    console.log("[v0] Loaded", descriptors?.length || 0, "descriptors")

    const groupedDescriptors = new Map<string, any[]>()
    for (const descriptor of descriptors || []) {
      if (!descriptor.source_image_id) continue

      const photoId = descriptor.source_image_id
      if (!groupedDescriptors.has(photoId)) {
        groupedDescriptors.set(photoId, [])
      }
      groupedDescriptors.get(photoId)!.push(descriptor)
    }

    const duplicateGroups = Array.from(groupedDescriptors.entries()).filter(([_, descs]) => descs.length > 1)

    console.log("[v0] Found", duplicateGroups.length, "photos with duplicates")

    let totalDeleted = 0
    let verifiedGroupsCount = 0
    let unverifiedGroupsCount = 0
    const idsToDelete: string[] = []

    for (const [photoId, descs] of duplicateGroups) {
      const key = `${person.id}_${photoId}`
      const isVerified = verifiedMap.get(key) || false

      if (isVerified) {
        // Keep OLDEST for verified
        descs.sort((a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime())
        verifiedGroupsCount++
      } else {
        // Keep NEWEST for unverified
        descs.sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime())
        unverifiedGroupsCount++
      }

      const toDelete = descs.slice(1)
      idsToDelete.push(...toDelete.map((d) => d.id))
      totalDeleted += toDelete.length
    }

    console.log("[v0] Verified groups:", verifiedGroupsCount, "Unverified groups:", unverifiedGroupsCount)

    if (idsToDelete.length > 0) {
      const { error: deleteError } = await supabase.from("face_descriptors").delete().in("id", idsToDelete)

      if (deleteError) throw deleteError
    }

    console.log("[v0] Deleted", totalDeleted, "duplicates")

    const { count: descriptorsAfter } = await supabase
      .from("face_descriptors")
      .select("id", { count: "exact", head: true })
      .eq("person_id", person.id)

    const { count: photoFacesCount } = await supabase
      .from("photo_faces")
      .select("id", { count: "exact", head: true })
      .eq("person_id", person.id)

    console.log("[v0] After cleanup:", descriptorsAfter, "descriptors,", photoFacesCount, "photo_faces")

    revalidatePath("/admin")
    return {
      success: true,
      data: {
        personName: person.real_name,
        before: {
          descriptors: descriptorsBefore || 0,
        },
        after: {
          descriptors: descriptorsAfter || 0,
          photoFaces: photoFacesCount || 0,
        },
        deletedDescriptors: totalDeleted,
        duplicateGroups: duplicateGroups.length,
        verifiedGroups: verifiedGroupsCount,
        unverifiedGroups: unverifiedGroupsCount,
      },
    }
  } catch (error: any) {
    console.error("[v0] Error cleaning up person descriptors:", error)
    return { error: error.message || "Failed to cleanup person descriptors" }
  }
}

export async function regeneratePersonDescriptorsAction(personId: string) {
  const supabase = createClient()

  try {
    console.log("[v0] Starting descriptor regeneration for person ID:", personId)

    const { data: person, error: personError } = await supabase
      .from("people")
      .select("id, real_name")
      .eq("id", personId)
      .single()

    if (personError || !person) {
      return { error: `Person not found with ID: ${personId}` }
    }

    console.log("[v0] Found person:", person.real_name, person.id)

    const { error: deleteDescError } = await supabase.from("face_descriptors").delete().eq("person_id", person.id)

    if (deleteDescError) throw deleteDescError

    console.log("[v0] Deleted all existing descriptors for", person.real_name)

    const { data: verifiedFaces, error: facesError } = await supabase
      .from("photo_faces")
      .select("id, photo_id, insightface_bbox, gallery_images(id, image_url, original_filename)")
      .eq("person_id", person.id)
      .eq("verified", true)

    if (facesError) throw facesError

    console.log("[v0] Found", verifiedFaces?.length || 0, "verified faces for regeneration")

    // The client will need to:
    // 1. Load each image
    // 2. Extract face region using bounding_box
    // 3. Generate descriptor using face-api.js
    // 4. Save descriptor back to database

    return {
      success: true,
      data: {
        personId: person.id,
        personName: person.real_name,
        facesToProcess:
          verifiedFaces?.map((face: any) => ({
            faceId: face.id,
            photoId: face.photo_id,
            boundingBox: face.insightface_bbox, // Use insightface_bbox
            imageUrl: face.gallery_images?.image_url,
            filename: face.gallery_images?.original_filename,
          })) || [],
      },
    }
  } catch (error: any) {
    console.error("[v0] Error regenerating person descriptors:", error)
    return { error: error.message || "Failed to regenerate person descriptors" }
  }
}

export async function cleanupDuplicateDescriptorsAction() {
  const supabase = createClient()

  try {
    console.log("[v0] Starting cleanup of duplicate face descriptors...")

    const { count: descriptorsBefore } = await supabase
      .from("face_descriptors")
      .select("*", { count: "exact", head: true })

    console.log(`[v0] Before cleanup: ${descriptorsBefore} total face_descriptors`)

    // Get all verified photo_faces with person info and image details
    let allPhotoFaces: any[] = []
    let pfHasMore = true
    let pfOffset = 0
    const pfPageSize = 1000

    while (pfHasMore) {
      const { data: pfBatch, error: pfError } = await supabase
        .from("photo_faces")
        .select(
          `
          photo_id,
          person_id,
          verified
        `,
        )
        .range(pfOffset, pfOffset + pfPageSize - 1)

      if (pfError) throw pfError

      if (pfBatch && pfBatch.length > 0) {
        allPhotoFaces = allPhotoFaces.concat(pfBatch)
        pfOffset += pfPageSize
        pfHasMore = pfBatch.length === pfPageSize
      } else {
        pfHasMore = false
      }
    }

    console.log(`[v0] Loaded ${allPhotoFaces.length} photo_faces records`)

    const verifiedMap = new Map<string, boolean>()
    for (const pf of allPhotoFaces) {
      const key = `${pf.person_id}_${pf.photo_id}`
      verifiedMap.set(key, pf.verified)
    }

    // Load all descriptors, ordered by creation date
    let allDescriptors: any[] = []
    let hasMore = true
    let offset = 0
    const pageSize = 1000

    while (hasMore) {
      const { data: batch, error: batchError } = await supabase
        .from("face_descriptors")
        .select("id, person_id, source_image_id, created_at")
        .order("created_at", { ascending: true })
        .range(offset, offset + pageSize - 1)

      if (batchError) throw batchError

      if (batch && batch.length > 0) {
        allDescriptors = allDescriptors.concat(batch)
        offset += pageSize
        hasMore = batch.length === pageSize
      } else {
        hasMore = false
      }
    }

    console.log(`[v0] Loaded ${allDescriptors.length} face_descriptors for analysis`)

    // Group descriptors by person_id and source_image_id
    const groupedDescriptors = new Map<string, any[]>()
    for (const descriptor of allDescriptors) {
      if (!descriptor.source_image_id) continue

      const key = `${descriptor.person_id}_${descriptor.source_image_id}`
      if (!groupedDescriptors.has(key)) {
        groupedDescriptors.set(key, [])
      }
      groupedDescriptors.get(key)!.push(descriptor)
    }

    // Identify groups with more than one descriptor
    const duplicateGroups = Array.from(groupedDescriptors.entries()).filter(
      ([_, descriptors]) => descriptors.length > 1,
    )

    console.log(`[v0] Found ${duplicateGroups.length} groups with duplicate descriptors`)

    let totalDeleted = 0
    let verifiedGroupsCount = 0
    let unverifiedGroupsCount = 0
    const idsToDelete: string[] = []

    for (const [key, descriptors] of duplicateGroups) {
      const isVerified = verifiedMap.get(key) || false // Use the verified status from photo_faces

      if (isVerified) {
        // For verified entries, keep the OLDEST descriptor
        descriptors.sort((a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime())
        verifiedGroupsCount++
        console.log(
          `[v0] Group ${key} (VERIFIED): keeping OLDEST descriptor ${descriptors[0].id} (created ${descriptors[0].created_at})`,
        )
      } else {
        // For unverified entries, keep the NEWEST descriptor
        descriptors.sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime())
        unverifiedGroupsCount++
        console.log(
          `[v0] Group ${key} (unverified): keeping NEWEST descriptor ${descriptors[0].id} (created ${descriptors[0].created_at})`,
        )
      }

      // Mark all descriptors in the group except the one to keep for deletion
      const toDelete = descriptors.slice(1)
      idsToDelete.push(...toDelete.map((d) => d.id))
      totalDeleted += toDelete.length
    }

    console.log(`[v0] Processed ${verifiedGroupsCount} verified groups and ${unverifiedGroupsCount} unverified groups`)

    // Delete the identified duplicate descriptors in batches
    if (idsToDelete.length > 0) {
      const batchSize = 100
      for (let i = 0; i < idsToDelete.length; i += batchSize) {
        const batch = idsToDelete.slice(i, i + batchSize)
        const { error: deleteError } = await supabase.from("face_descriptors").delete().in("id", batch)

        if (deleteError) {
          console.error(`[v0] Error deleting descriptor batch ${i / batchSize + 1}:`, deleteError)
          throw deleteError
        }
      }
    }

    console.log(`[v0] Deleted ${totalDeleted} duplicate face_descriptors`)

    // Get final counts
    const { count: descriptorsAfter } = await supabase
      .from("face_descriptors")
      .select("*", { count: "exact", head: true })

    const { count: photoFacesCount } = await supabase.from("photo_faces").select("*", { count: "exact", head: true })

    console.log(`[v0] After cleanup: ${descriptorsAfter} face_descriptors, ${photoFacesCount} photo_faces records`)
    console.log(`[v0] Cleanup complete!`)

    revalidatePath("/admin")
    return {
      success: true,
      data: {
        before: {
          descriptors: descriptorsBefore || 0,
        },
        after: {
          descriptors: descriptorsAfter || 0,
          photoFaces: photoFacesCount || 0,
        },
        deletedDescriptors: totalDeleted,
        duplicateGroups: duplicateGroups.length,
        verifiedGroups: verifiedGroupsCount,
        unverifiedGroups: unverifiedGroupsCount,
      },
    }
  } catch (error: any) {
    console.error("[v0] Error cleaning up duplicate descriptors:", error)
    return { error: error.message || "Failed to cleanup duplicate descriptors" }
  }
}

export async function getPhotosWithExcessDescriptorsAction() {
  const supabase = createClient()

  try {
    console.log("[v0] Finding photos with excess descriptors...")

    // Get all verified photo_faces with person info and image details
    let allPhotoFaces: any[] = []
    let hasMore = true
    let offset = 0
    const pageSize = 1000

    while (hasMore) {
      const { data: batch, error: batchError } = await supabase
        .from("photo_faces")
        .select(
          `
          photo_id,
          person_id,
          verified,
          insightface_bbox,
          gallery_images(id, image_url, original_filename)
        `,
        )
        .eq("verified", true)
        .range(offset, offset + pageSize - 1)

      if (batchError) throw batchError

      if (batch && batch.length > 0) {
        allPhotoFaces = allPhotoFaces.concat(batch)
        offset += pageSize
        hasMore = batch.length === pageSize
      } else {
        hasMore = false
      }
    }

    console.log(`[v0] Loaded ${allPhotoFaces.length} verified photo_faces`)

    // Group by photo_id to count faces per photo
    const facesByPhoto = new Map<string, any[]>()
    for (const face of allPhotoFaces) {
      if (!facesByPhoto.has(face.photo_id)) {
        facesByPhoto.set(face.photo_id, [])
      }
      facesByPhoto.get(face.photo_id)!.push(face)
    }

    // Get all descriptors
    let allDescriptors: any[] = []
    let descHasMore = true
    let descOffset = 0

    while (descHasMore) {
      const { data: descBatch, error: descError } = await supabase
        .from("face_descriptors")
        .select("id, person_id, source_image_id")
        .range(descOffset, descOffset + pageSize - 1)

      if (descError) throw descError

      if (descBatch && descBatch.length > 0) {
        allDescriptors = allDescriptors.concat(descBatch)
        descOffset += pageSize
        descHasMore = descBatch.length === pageSize
      } else {
        hasMore = false
      }
    }

    console.log(`[v0] Loaded ${allDescriptors.length} descriptors`)

    // Group descriptors by photo_id and person_id
    const descriptorsByPhotoAndPerson = new Map<string, number>()
    for (const desc of allDescriptors) {
      if (!desc.source_image_id) continue
      const key = `${desc.source_image_id}_${desc.person_id}`
      descriptorsByPhotoAndPerson.set(key, (descriptorsByPhotoAndPerson.get(key) || 0) + 1)
    }

    // Find photos where descriptor count > face count
    const photosToProcess: any[] = []
    const peopleStats = new Map<string, { personId: string; photoCount: number }>()

    for (const [photoId, faces] of facesByPhoto.entries()) {
      const uniquePersonIds = new Set(faces.map((f) => f.person_id))
      const faceCount = uniquePersonIds.size

      // Count total descriptors for this photo
      let descriptorCount = 0
      for (const personId of uniquePersonIds) {
        const key = `${photoId}_${personId}`
        descriptorCount += descriptorsByPhotoAndPerson.get(key) || 0
      }

      if (descriptorCount > faceCount) {
        console.log(`[v0] Photo ${photoId}: ${descriptorCount} descriptors > ${faceCount} faces`)

        // Add to processing list
        for (const face of faces) {
          photosToProcess.push({
            photoId: face.photo_id,
            personId: face.person_id,
            boundingBox: face.insightface_bbox, // Use insightface_bbox
            imageUrl: face.gallery_images?.image_url,
            filename: face.gallery_images?.original_filename,
          })

          // Update people stats
          if (!peopleStats.has(face.person_id)) {
            peopleStats.set(face.person_id, { personId: face.person_id, photoCount: 0 })
          }
          const stats = peopleStats.get(face.person_id)!
          stats.photoCount++
        }
      }
    }

    // Get person names for stats
    const personIds = Array.from(peopleStats.keys())
    const { data: people, error: peopleError } = await supabase
      .from("people")
      .select("id, real_name, telegram_name")
      .in("id", personIds)

    if (peopleError) throw peopleError

    const peopleMap = new Map(people?.map((p) => [p.id, p.real_name || p.telegram_name]) || [])

    const statsArray = Array.from(peopleStats.values()).map((stat) => ({
      personId: stat.personId,
      personName: peopleMap.get(stat.personId) || "Unknown",
      photoCount: stat.photoCount,
    }))

    console.log(`[v0] Found ${photosToProcess.length} faces to process across ${statsArray.length} people`)

    return {
      success: true,
      data: {
        facesToProcess: photosToProcess,
        peopleStats: statsArray,
        totalFaces: photosToProcess.length,
      },
    }
  } catch (error: any) {
    console.error("[v0] Error getting photos with excess descriptors:", error)
    return { error: error.message || "Failed to get photos with excess descriptors" }
  }
}

export async function getVerifiedPeopleForPhotoAction(photoId: string) {
  const supabase = createClient()

  try {
    const { data: faces, error } = await supabase
      .from("photo_faces")
      .select("person_id, people(id, real_name, telegram_name)")
      .eq("photo_id", photoId)
      .eq("verified", true)

    if (error) {
      console.error("[v0] Error fetching verified people for photo:", error)
      return { success: false, error: error.message }
    }

    const people = (faces || [])
      .filter((face) => face.people)
      .map((face) => ({
        id: face.people!.id,
        name: face.people!.real_name || face.people!.telegram_name || "Неизвестный",
      }))

    return { success: true, data: people }
  } catch (error: any) {
    console.error("[v0] Error getting verified people for photo:", error)
    return { success: false, error: error.message || "Failed to get verified people" }
  }
}

export async function prepareTrainingDatasetAction() {
  const supabase = createClient()

  try {
    console.log("[v0] Starting dataset preparation...")

    // Get all verified photo_faces with person info and image details
    let allPhotoFaces: any[] = []
    let hasMore = true
    let offset = 0
    const pageSize = 1000

    while (hasMore) {
      const { data: batch, error: batchError } = await supabase
        .from("photo_faces")
        .select(
          `
          photo_id,
          person_id,
          verified,
          insightface_bbox,
          gallery_images(id, image_url, original_filename)
        `,
        )
        .eq("verified", true)
        .range(offset, offset + pageSize - 1)

      if (batchError) throw batchError

      if (batch && batch.length > 0) {
        allPhotoFaces = allPhotoFaces.concat(batch)
        offset += pageSize
        hasMore = batch.length === pageSize
      } else {
        hasMore = false
      }
    }

    console.log(`[v0] Loaded ${allPhotoFaces.length} verified photo_faces`)

    // Group by photo_id to count faces per photo
    const facesByPhoto = new Map<string, any[]>()
    for (const face of allPhotoFaces) {
      if (!facesByPhoto.has(face.photo_id)) {
        facesByPhoto.set(face.photo_id, [])
      }
      facesByPhoto.get(face.photo_id)!.push(face)
    }

    // Get all descriptors
    let allDescriptors: any[] = []
    let descHasMore = true
    let descOffset = 0

    while (descHasMore) {
      const { data: descBatch, error: descError } = await supabase
        .from("face_descriptors")
        .select("id, person_id, source_image_id")
        .range(descOffset, descOffset + pageSize - 1)

      if (descError) throw descError

      if (descBatch && descBatch.length > 0) {
        allDescriptors = allDescriptors.concat(descBatch)
        descOffset += pageSize
        descHasMore = descBatch.length === pageSize
      } else {
        hasMore = false
      }
    }

    console.log(`[v0] Loaded ${allDescriptors.length} descriptors`)

    // Group descriptors by photo_id and person_id
    const descriptorsByPhotoAndPerson = new Map<string, number>()
    for (const desc of allDescriptors) {
      if (!desc.source_image_id) continue
      const key = `${desc.source_image_id}_${desc.person_id}`
      descriptorsByPhotoAndPerson.set(key, (descriptorsByPhotoAndPerson.get(key) || 0) + 1)
    }

    // Prepare training data
    const trainingData: Array<{ imageUrl: string; label: string }> = []
    const personImageCount = new Map<string, number>()

    for (const [photoId, faces] of facesByPhoto.entries()) {
      for (const face of faces) {
        const personId = face.person_id
        const imageUrl = face.gallery_images?.image_url
        const filename = face.gallery_images?.original_filename

        if (!personId || !imageUrl || !filename) continue

        // Ensure we don't exceed a reasonable number of images per person for training
        const currentCount = personImageCount.get(personId) || 0
        if (currentCount >= 100) continue // Limit to 100 images per person

        // Check if there are corresponding descriptors
        const descriptorCount = descriptorsByPhotoAndPerson.get(`${photoId}_${personId}`) || 0
        if (descriptorCount === 0) continue

        trainingData.push({
          imageUrl: imageUrl,
          label: personId, // Use person_id as the label
        })

        personImageCount.set(personId, currentCount + 1)

        console.log(
          `[v0] Added to training data: PhotoID=${photoId}, PersonID=${personId}, Filename=${filename}, ImageURL=${imageUrl}`,
        )
      }
    }

    console.log(`[v0] Prepared ${trainingData.length} entries for training dataset.`)

    // Optional: Save training data to a file or return it
    // For now, just log and return success. In a real app, you might want to save this.
    // Example: await supabase.storage.from('datasets').upload(`training_data_${Date.now()}.json`, JSON.stringify(trainingData));

    return {
      success: true,
      data: {
        datasetSize: trainingData.length,
        trainingData: trainingData, // Returning data for potential further processing or download
      },
    }
  } catch (error: any) {
    console.error("[v0] Error preparing training dataset:", error)
    return { error: error.message || "Failed to prepare training dataset" }
  }
}

export async function savePhotoFaceTagsAction(
  photoId: string,
  imageUrl: string,
  tags: {
    personId: string
    insightface_bbox: { x: number; y: number; width: number; height: number }
    embedding?: number[]
    insightface_confidence: number | null
    recognition_confidence: number | null // Added recognition_confidence
    verified: boolean
  }[],
) {
  console.log("[v0] savePhotoFaceTagsAction called for photo:", photoId, "with", tags.length, "tags")

  try {
    // Fetch photo URL using direct SQL
    const photoResult = await sql`
      SELECT image_url
      FROM gallery_images
      WHERE id = ${photoId}
    `

    if (photoResult.length === 0) {
      console.error("[v0] Photo not found:", photoId)
      return { error: "Photo not found" }
    }

    const photo = photoResult[0]
    console.log("[v0] Photo found:", photo.image_url)

    // Delete existing tags and descriptors for this photo using direct SQL
    await sql`DELETE FROM photo_faces WHERE photo_id = ${photoId}`
    await sql`DELETE FROM face_descriptors WHERE source_image_id = ${photoId}`

    if (tags.length > 0) {
      console.log("[v0] Calling backend to generate descriptors")

      try {
        const result = await apiFetch("/api/recognition/generate-descriptors", {
          method: "POST",
          body: JSON.stringify({
            image_url: photo.image_url,
            faces: tags.map((tag) => ({
              person_id: tag.personId,
              bbox: tag.insightface_bbox,
              verified: tag.verified,
              photo_id: photoId,
            })),
          }),
        })

        console.log("[v0] Descriptors generated:", result)
      } catch (fetchError) {
        console.error("[v0] Error calling backend for descriptors:", fetchError)
        // Continue anyway - save the tags even if descriptor generation fails
      }
    }

    // Insert new tags using direct SQL
    if (tags.length > 0) {
      const values = tags.map(
        (tag) =>
          sql`(${photoId}, ${tag.personId}, ${JSON.stringify(tag.insightface_bbox)}::jsonb, ${tag.insightface_confidence}, ${tag.recognition_confidence}, ${tag.verified})`, // Added recognition_confidence
      )

      console.log("[v0] Inserting", tags.length, "tags")

      await sql`
        INSERT INTO photo_faces (
          photo_id,
          person_id,
          insightface_bbox,
          insightface_confidence,
          recognition_confidence,
          verified
        )
        VALUES ${sql.join(values, sql`, `)}
      `
    }

    console.log("[v0] Tags saved successfully")

    revalidatePath("/admin")
    return { success: true }
  } catch (error: any) {
    console.error("[v0] Error in savePhotoFaceTagsAction:", error)
    return { error: error.message || "Failed to save tags" }
  }
}

// NEW FUNCTION: getUnknownFaceClustersAction
export async function getUnknownFaceClustersAction(galleryId: string) {
  try {
    const data = await apiFetch(`/cluster-unknown-faces?gallery_id=${galleryId}`, {
      method: "POST",
    })

    return { success: true, data: data.clusters }
  } catch (error: any) {
    console.error("[v0] Error clustering unknown faces:", error)
    return { error: error.message || "Failed to cluster unknown faces" }
  }
}

// NEW FUNCTION: regenerateUnknownDescriptorsAction
export async function regenerateUnknownDescriptorsAction(galleryId: string) {
  try {
    console.log(`[v0] Calling regenerate-unknown-descriptors for gallery ${galleryId}`)

    const data = await apiFetch(`/regenerate-unknown-descriptors?gallery_id=${galleryId}`, {
      method: "POST",
    })

    return {
      success: true,
      data: {
        totalFaces: data.total_faces,
        regenerated: data.regenerated,
        failed: data.failed,
        alreadyHadDescriptor: data.already_had_descriptor,
      },
    }
  } catch (error: any) {
    console.error("[v0] Error regenerating descriptors:", error)
    return { error: error.message || "Failed to regenerate descriptors" }
  }
}

export async function rejectFaceClusterAction(clusterFaces: { photo_id: string; descriptor: number[] }[]) {
  try {
    await apiFetch("/api/recognition/reject-faces", {
      method: "POST",
      body: JSON.stringify({ faces: clusterFaces }),
    })

    return { success: true }
  } catch (error: any) {
    console.error("[v0] Error rejecting faces:", error)
    return { error: error.message || "Failed to reject faces" }
  }
}

export async function createPersonFromClusterAction(
  personName: string,
  clusterFaces: { photo_id: string; descriptor: number[] }[],
) {
  try {
    const person = await peopleApi.createFromCluster(personName, clusterFaces)

    // Rebuild recognition index after creating person with descriptors
    await rebuildRecognitionIndexAction()

    revalidatePath("/admin")
    return { success: true, data: person }
  } catch (error: any) {
    console.error("[createPersonFromClusterAction] Error:", error)
    return { error: error.message || "Failed to create person from cluster" }
  }
}

export async function generateMissingDescriptorsAction() {
  const supabase = createClient()

  try {
    const { data: facesWithoutDescriptors, error: queryError } = await supabase.rpc(
      "get_verified_faces_without_descriptors",
    )

    if (queryError) {
      console.error("[v0] Error fetching faces without descriptors:", queryError)
      return { error: queryError.message }
    }

    if (!facesWithoutDescriptors || facesWithoutDescriptors.length === 0) {
      return { success: true, processed: 0, message: "No faces without descriptors found" }
    }

    const photoMap = new Map<
      string,
      Array<{
        person_id: string
        bbox: { x: number; y: number; width: number; height: number }
        verified: boolean
      }>
    >()

    const photoUrls = new Map<string, string>()

    for (const face of facesWithoutDescriptors) {
      if (!face.photo_id) continue // Skip if photo_id is missing
      if (!photoMap.has(face.photo_id)) {
        photoMap.set(face.photo_id, [])
        photoUrls.set(face.photo_id, face.image_url)
      }

      photoMap.get(face.photo_id)!.push({
        person_id: face.person_id,
        bbox: face.insightface_bbox, // Use insightface_bbox
        verified: face.verified,
      })
    }

    let successCount = 0
    let errorCount = 0
    const errors: string[] = []

    // Process each photo
    for (const [photoId, faces] of photoMap.entries()) {
      const imageUrl = photoUrls.get(photoId)!

      try {
        const result = await apiFetch("/api/recognition/generate-descriptors", {
          method: "POST",
          body: JSON.stringify({
            image_url: imageUrl,
            faces: faces,
          }),
        })

        console.log(`[v0] Generated ${result.generated} descriptors for photo ${photoId}`)
        successCount += result.generated || 0
      } catch (fetchError: any) {
        console.error(`[v0] Error processing photo ${photoId}:`, fetchError)
        errors.push(`Photo ${photoId}: ${fetchError.message}`)
        errorCount++
      }
    }

    if (successCount > 0) {
      console.log("[v3.31] Rebuilding HNSWLIB index after generating descriptors...")
      try {
        const rebuildResult = await apiFetch("/api/recognition/rebuild-index", {
          method: "POST",
        })

        console.log("[v3.31] ✓ Index rebuilt successfully:", rebuildResult)
      } catch (rebuildError: any) {
        console.error("[v3.31] ✗ Error rebuilding index:", rebuildError)
      }
    }

    revalidatePath("/admin")

    return {
      success: true,
      processed: successCount,
      errors: errorCount,
      errorDetails: errors.length > 0 ? errors : undefined,
      message: `Generated ${successCount} descriptors. ${errorCount > 0 ? `${errorCount} errors occurred.` : ""}`,
    }
  } catch (error: any) {
    console.error("[v0] Error in generateMissingDescriptorsAction:", error)
    return { error: error.message || "Failed to generate descriptors" }
  }
}

export async function rebuildRecognitionIndexAction() {
  try {
    console.log("[v3.31] Calling rebuild-index endpoint...")

    const result = await apiFetch("/api/recognition/rebuild-index", {
      method: "POST",
    })

    console.log("[v3.31] ✓ Index rebuilt:", result)

    return {
      success: result.success,
      oldCount: result.old_descriptor_count,
      newCount: result.new_descriptor_count,
      uniquePeople: result.unique_people_count,
      message: `Index rebuilt: ${result.new_descriptor_count} descriptors for ${result.unique_people_count} people`,
    }
  } catch (error: any) {
    console.error("[v3.31] Error rebuilding index:", error)
    return { error: error.message || "Failed to rebuild index" }
  }
}

export async function assignClusterToPersonAction(
  personId: string,
  clusterFaces: { photo_id: string; descriptor: number[] }[],
) {
  console.log("[v0] assignClusterToPersonAction started")
  console.log("[v0] personId:", personId)
  console.log("[v0] clusterFaces count:", clusterFaces.length)
  console.log(
    "[v0] clusterFaces details:",
    clusterFaces.map((f) => ({
      photo_id: f.photo_id,
      has_descriptor: !!f.descriptor,
      descriptor_length: f.descriptor?.length || 0,
    })),
  )

  const supabase = createClient()

  try {
    // Insert descriptors for the new person
    const descriptorsToInsert = clusterFaces.map((face) => ({
      person_id: personId,
      source_image_id: face.photo_id,
      descriptor: `[${face.descriptor.join(",")}]`,
    }))

    const { error: insertDescriptorsError } = await supabase.from("face_descriptors").insert(descriptorsToInsert)

    if (insertDescriptorsError) {
      console.error("[v0] Error inserting descriptors:", insertDescriptorsError)
      return { error: "Failed to insert descriptors" }
    }

    // Update photo_faces to link to the person
    for (const face of clusterFaces) {
      const { error: updateFaceError } = await supabase
        .from("photo_faces")
        .update({
          person_id: personId,
          verified: false, // Mark as unverified initially, will be re-verified later
          recognition_confidence: null, // Reset recognition confidence
        })
        .eq("photo_id", face.photo_id)
        .is("person_id", null) // Only update if person_id is currently null

      if (updateFaceError) {
        console.error(`[v0] Error updating photo_face for photo ${face.photo_id}:`, updateFaceError)
        // Continue processing other faces even if one fails
      }
    }

    // Rebuild the recognition index
    console.log("[v0] Rebuilding recognition index after assigning cluster...")
    await rebuildRecognitionIndexAction()

    revalidatePath("/admin")
    console.log("[v0] assignClusterToPersonAction completed successfully")
    return { success: true }
  } catch (error: any) {
    console.error("[v0] Error in assignClusterToPersonAction:", error)
    return { error: error.message || "Failed to assign cluster to person" }
  }
}

export async function getPersonAction(personId: string) {
  "use server"

  try {
    const response = await fetch(`${process.env.FASTAPI_URL}/api/crud/people/${personId}`, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
      },
    })

    if (!response.ok) {
      if (response.status === 404) {
        console.error("[SERVER] Person not found:", personId)
        return {
          success: false,
          error: "Person not found",
        }
      }
      const error = await response.text()
      throw new Error(error)
    }

    const person = await response.json()

    return {
      success: true,
      data: person,
    }
  } catch (error) {
    console.error("[SERVER] Error in getPersonAction:", error)
    return {
      success: false,
      error: error instanceof Error ? error.message : "Failed to fetch person",
    }
  }
}

export async function getPeopleAction() {
  "use server"

  try {
    const people = await peopleApi.getAll(false)

    return {
      success: true,
      data:
        people.map((p) => ({
          id: p.id,
          real_name: p.real_name,
          telegram_name: p.telegram_name,
          telegram_nickname: p.telegram_nickname,
          avatar_url: p.avatar_url,
        })) || [],
    }
  } catch (error) {
    console.error("[SERVER] Error fetching people:", error)
    return {
      success: false,
      error: error instanceof Error ? error.message : "Failed to fetch people",
    }
  }
}

export async function unlinkPersonFromPhotoAction(photoId: string, personId: string) {
  "use server"

  try {
    console.log(`[v0] unlinkPersonFromPhotoAction: Removing person ${personId} from photo ${photoId}`)

    // Delete the photo_face record for this person on this photo
    await sql`
      DELETE FROM photo_faces
      WHERE photo_id = ${photoId}
        AND person_id = ${personId}
    `
    // Also delete corresponding face descriptors
    await sql`
      DELETE FROM face_descriptors
      WHERE source_image_id = ${photoId}
        AND person_id = ${personId}
    `

    console.log(`[v0] unlinkPersonFromPhotoAction: Successfully removed person from photo and descriptors`)

    revalidatePath("/admin")
    return { success: true }
  } catch (error: any) {
    console.error("[v0] unlinkPersonFromPhotoAction: Error removing person from photo:", error)
    return { success: false, error: error.message }
  }
}

// NEW FUNCTION: getPeopleWithStatsAction
export async function getPeopleWithStatsAction() {
  try {
    // Get all people with stats from Python API
    const people = await peopleApi.getAll(true)

    // Python API already returns faces_count, so we just map to expected format
    const peopleWithStats = people.map((person) => ({
      ...person,
      photos_count: person.faces_count || 0,
      verified_count: 0, // TODO: Add to Python API if needed
      unverified_count: 0, // TODO: Add to Python API if needed
      avg_confidence: null, // TODO: Add to Python API if needed
      descriptor_count: 0, // TODO: Add to Python API if needed
    }))

    return {
      success: true,
      data: peopleWithStats,
    }
  } catch (error) {
    console.error("[SERVER] Error fetching people with stats:", error)
    return {
      success: false,
      error: error instanceof Error ? error.message : "Failed to fetch people with stats",
    }
  }
}
