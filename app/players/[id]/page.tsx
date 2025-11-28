import { createClient } from "@/lib/supabase/server"
import { notFound } from "next/navigation"
import { PlayerGalleryView } from "@/components/player-gallery-view"
import type { Person } from "@/lib/types"

interface PlayerGalleryPageProps {
  params: Promise<{ id: string }>
}

export const revalidate = 300

export default async function PlayerGalleryPage({ params }: PlayerGalleryPageProps) {
  const { id } = await params
  const supabase = await createClient()

  const { data: player, error: playerError } = await supabase.from("people").select("*").eq("id", id).single()

  if (playerError || !player) {
    notFound()
  }

  const { data: photoFaces, error: photosError } = await supabase
    .from("photo_faces")
    .select(
      `
      id,
      confidence,
      verified,
      gallery_images!inner (
        id,
        image_url,
        original_url,
        original_filename,
        file_size,
        width,
        height,
        gallery_id,
        galleries!inner (
          id,
          title,
          shoot_date
        )
      )
    `,
    )
    .eq("person_id", id)
    .or("verified.eq.true,confidence.gte.80")

  if (photosError) {
    console.error("[v0] Error fetching player photos:", photosError)
  }

  const images: any[] =
    photoFaces?.map((face: any) => ({
      ...face.gallery_images,
      gallery: face.gallery_images.galleries,
    })) || []

  images.sort((a, b) => {
    const dateA = new Date(a.gallery?.shoot_date || 0).getTime()
    const dateB = new Date(b.gallery?.shoot_date || 0).getTime()

    if (dateA !== dateB) {
      return dateB - dateA
    }

    return (b.original_filename || "").localeCompare(a.original_filename || "")
  })

  return <PlayerGalleryView player={player as Person} images={images} />
}
