import { sql } from "@/lib/db"

async function retryWithBackoff<T>(fn: () => Promise<T>, maxRetries = 5, initialDelay = 2000): Promise<T> {
  let lastError: Error | null = null

  for (let i = 0; i < maxRetries; i++) {
    try {
      return await fn()
    } catch (error: any) {
      lastError = error

      // Check if it's a rate limit error (including SyntaxError from parsing "Too Many Requests")
      const isRateLimit =
        error?.message?.includes("Too Many") ||
        error?.message?.includes("SyntaxError") ||
        error?.code === "429" ||
        error?.status === 429 ||
        (error instanceof SyntaxError && error.message.includes("Too Many"))

      if (!isRateLimit || i === maxRetries - 1) {
        throw error
      }

      // More aggressive exponential backoff: 2s, 4s, 8s, 16s, 32s
      const delay = initialDelay * Math.pow(2, i)
      console.log(`[v0] Rate limited, retrying in ${delay}ms... (attempt ${i + 1}/${maxRetries})`)
      await new Promise((resolve) => setTimeout(resolve, delay))
    }
  }

  throw lastError
}

async function safeSupabaseCall<T>(fn: () => Promise<T>): Promise<T> {
  try {
    return await fn()
  } catch (error: any) {
    // If it's a SyntaxError with "Too Many" in message, it's a rate limit error
    if (error instanceof SyntaxError && error.message.includes("Too Many")) {
      const rateLimitError = new Error("Too Many Requests")
      ;(rateLimitError as any).status = 429
      throw rateLimitError
    }
    throw error
  }
}

export async function saveFaceDescriptor(
  personId: string,
  descriptor: number[], // changed from Float32Array to number[] for easier handling
  sourceImageId: string | null,
) {
  return retryWithBackoff(async () => {
    try {
      if (sourceImageId) {
        await sql`
          DELETE FROM face_descriptors 
          WHERE person_id = ${personId} AND source_image_id = ${sourceImageId}
        `
      }

      const embeddingString = `[${descriptor.join(",")}]`
      const [data] = await sql`
        INSERT INTO face_descriptors (person_id, descriptor, source_image_id)
        VALUES (${personId}, ${embeddingString}, ${sourceImageId})
        RETURNING *
      `
      return data
    } catch (error) {
      console.error("[v0] Error saving face descriptor:", error)
      throw error
    }
  })
}

export async function getAllFaceDescriptors() {
  return retryWithBackoff(async () => {
    try {
      const descriptors = await sql`
        SELECT fd.id, fd.person_id, fd.descriptor, fd.source_image_id, p.real_name
        FROM face_descriptors fd
        LEFT JOIN people p ON fd.person_id = p.id
        ORDER BY fd.created_at DESC
      `

      return descriptors.map((item) => ({
        id: item.id,
        personId: item.person_id,
        personName: item.real_name || "Unknown",
        descriptor: typeof item.descriptor === "string" ? JSON.parse(item.descriptor) : item.descriptor,
        sourceImageId: item.source_image_id || null,
      }))
    } catch (error) {
      console.error("[v0] Error fetching face descriptors:", error)
      throw error
    }
  })
}

export async function savePhotoFace(
  photoId: string,
  personId: string | null,
  insightface_bbox: any,
  confidence: number | null, // legacy param kept for signature compatibility, but mapped to insightface_confidence
  verified = false,
  recognition_confidence?: number | null,
) {
  return retryWithBackoff(async () => {
    try {
      const [data] = await sql`
        INSERT INTO photo_faces (
            photo_id, 
            person_id, 
            insightface_bbox, 
            insightface_confidence, 
            recognition_confidence, 
            verified
        )
        VALUES (
            ${photoId}, 
            ${personId}, 
            ${sql.json(insightface_bbox)}, 
            ${confidence}, 
            ${recognition_confidence || null}, 
            ${verified}
        )
        RETURNING *
      `
      return data
    } catch (error) {
      console.error("[v0] Error saving photo face:", error)
      throw error
    }
  })
}

export async function updatePhotoFace(
  faceId: string,
  updates: {
    person_id?: string | null
    verified?: boolean
  },
) {
  return retryWithBackoff(async () => {
    try {
      const [data] = await sql`
        UPDATE photo_faces
        SET ${sql(updates)}
        WHERE id = ${faceId}
        RETURNING *
      `
      return data
    } catch (error) {
      console.error("[v0] Error updating photo face:", error)
      throw error
    }
  })
}

export async function getPhotoFaces(photoId: string) {
  return retryWithBackoff(async () => {
    try {
      const faces = await sql`
        SELECT pf.*, p.real_name, p.id as person_pk
        FROM photo_faces pf
        LEFT JOIN people p ON pf.person_id = p.id
        WHERE pf.photo_id = ${photoId}
      `

      return faces.map((face) => ({
        id: face.id,
        person_id: face.person_id,
        insightface_bbox: face.insightface_bbox,
        confidence: face.insightface_confidence, // mapping for compatibility
        recognition_confidence: face.recognition_confidence,
        blur_score: face.blur_score,
        verified: face.verified,
        people: face.person_id ? { id: face.person_pk, real_name: face.real_name } : null,
      }))
    } catch (error) {
      console.error("[v0] Error fetching photo faces:", error)
      throw error
    }
  })
}

export async function deletePhotoFace(faceId: string) {
  return retryWithBackoff(async () => {
    try {
      await sql`DELETE FROM photo_faces WHERE id = ${faceId}`
    } catch (error) {
      console.error("[v0] Error deleting photo face:", error)
      throw error
    }
  })
}
