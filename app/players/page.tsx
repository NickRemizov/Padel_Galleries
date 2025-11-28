import { createClient } from "@/lib/supabase/server"
import { PlayersGrid } from "@/components/players-grid"
import { AuthButton } from "@/components/auth-button"
import { MainNav } from "@/components/main-nav"
import Link from "next/link"
import type { Person } from "@/lib/types"

export const revalidate = 300

export default async function PlayersPage() {
  const supabase = await createClient()

  const { data: players, error } = await supabase
    .from("people")
    .select("*")
    .eq("show_in_players_gallery", true)
    .not("avatar_url", "is", null)

  if (error) {
    console.error("[v0] Error fetching players:", error)
  }

  // Fetch photo_faces with gallery information for all players
  const { data: photoFaces, error: photoFacesError } = await supabase
    .from("photo_faces")
    .select(
      `
      person_id,
      photo_id,
      gallery_images!inner (
        gallery_id,
        galleries!inner (
          shoot_date
        )
      )
    `,
    )
    .in("person_id", players?.map((p) => p.id) || [])

  if (photoFacesError) {
    console.error("[v0] Error fetching photo faces:", photoFacesError)
  }

  // Process players to get photo count and most recent gallery date
  const playersWithData = players?.map((player: any) => {
    // Get all photo faces for this player
    const playerPhotoFaces = photoFaces?.filter((pf: any) => pf.person_id === player.id) || []
    const photoCount = playerPhotoFaces.length

    // Find the most recent gallery date for this player
    let mostRecentDate: string | null = null
    if (playerPhotoFaces.length > 0) {
      const dates = playerPhotoFaces
        .map((pf: any) => pf.gallery_images?.galleries?.shoot_date)
        .filter((date: any) => date != null)

      if (dates.length > 0) {
        mostRecentDate = dates.sort((a: string, b: string) => b.localeCompare(a))[0]
      }
    }

    return {
      ...player,
      _count: {
        photo_faces: photoCount,
      },
      _mostRecentGalleryDate: mostRecentDate,
    }
  })

  // Sort by most recent gallery date (descending), then by name
  const sortedPlayers = playersWithData?.sort((a: any, b: any) => {
    // Players with no gallery dates go to the end
    if (!a._mostRecentGalleryDate && !b._mostRecentGalleryDate) {
      return a.real_name.localeCompare(b.real_name)
    }
    if (!a._mostRecentGalleryDate) return 1
    if (!b._mostRecentGalleryDate) return -1

    // Sort by date descending (most recent first)
    const dateCompare = b._mostRecentGalleryDate.localeCompare(a._mostRecentGalleryDate)
    if (dateCompare !== 0) return dateCompare

    // If same date, sort by name
    return a.real_name.localeCompare(b.real_name)
  })

  return (
    <main className="min-h-screen bg-background border-background">
      <div className="mx-auto py-12 shadow-none">
        <header className="mb-12 text-center relative px-4">
          <div className="absolute right-0 top-0">
            <AuthButton />
          </div>
          <Link href="/" className="inline-block mb-4 hover:opacity-80 transition-opacity">
            <h1 className="font-serif font-bold tracking-tight text-foreground text-8xl">Padel in Valencia</h1>
          </Link>
          <MainNav />
        </header>

        <PlayersGrid players={(sortedPlayers as Person[]) || []} />
      </div>
    </main>
  )
}
