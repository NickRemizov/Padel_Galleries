import { createClient } from "@/lib/supabase/server"
import { notFound } from "next/navigation"
import { GalleryView } from "@/components/gallery-view"
import type { Gallery } from "@/lib/types"

export const revalidate = 60

export default async function GalleryPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params
  const supabase = await createClient()

  const { data: gallery, error } = await supabase
    .from("galleries")
    .select(
      `
      *,
      photographers (*),
      locations (*),
      organizers (*),
      gallery_images (
        id,
        gallery_id,
        image_url,
        original_url,
        original_filename,
        file_size,
        width,
        height,
        display_order,
        download_count,
        created_at
      )
    `,
    )
    .eq("id", id)
    .order("display_order", { referencedTable: "gallery_images", ascending: true })
    .single()

  if (error || !gallery) {
    notFound()
  }

  if (gallery.gallery_images && gallery.gallery_images.length > 0) {
    const imageIds = gallery.gallery_images.map((img: any) => img.id)

    // Get photo faces for filtering (high confidence or verified)
    const { data: photoFaces } = await supabase
      .from("photo_faces")
      .select(`
        photo_id,
        person_id,
        people!inner(show_photos_in_galleries)
      `)
      .in("photo_id", imageIds)
      .or("verified.eq.true,confidence.gte.0.8")

    // Find photo IDs that should be hidden (have people with show_photos_in_galleries = false)
    const hiddenPhotoIds = new Set(
      photoFaces
        ?.filter((face: any) => face.people?.show_photos_in_galleries === false)
        .map((face: any) => face.photo_id) || [],
    )

    // Filter out hidden photos
    gallery.gallery_images = gallery.gallery_images.filter((img: any) => !hiddenPhotoIds.has(img.id))

    const { data: verifiedFaces } = await supabase
      .from("photo_faces")
      .select(`
        photo_id,
        people!inner(id, real_name, telegram_name)
      `)
      .in(
        "photo_id",
        gallery.gallery_images.map((img: any) => img.id),
      )
      .eq("verified", true)

    // Group people by photo_id
    const peopleByPhoto = new Map<string, Array<{ id: string; name: string }>>()
    verifiedFaces?.forEach((face: any) => {
      if (!peopleByPhoto.has(face.photo_id)) {
        peopleByPhoto.set(face.photo_id, [])
      }
      peopleByPhoto.get(face.photo_id)?.push({
        id: face.people.id,
        name: face.people.real_name || face.people.telegram_name || "Unknown",
      })
    })

    // Add people data to each image
    gallery.gallery_images = gallery.gallery_images.map((img: any) => ({
      ...img,
      people: peopleByPhoto.get(img.id) || [],
    }))
  }

  return <GalleryView gallery={gallery as Gallery} />
}
