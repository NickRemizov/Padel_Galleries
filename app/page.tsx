import { GalleryGrid } from "@/components/gallery-grid"
import { AuthButton } from "@/components/auth-button"
import { MainNav } from "@/components/main-nav"
import type { Gallery } from "@/lib/types"
import { AlertCircle } from "lucide-react"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import sql from "@/lib/db"

export const revalidate = 60

export default async function HomePage() {
  let galleries: any[] = []
  let fetchError = null

  try {
    galleries = await sql`
      SELECT 
        g.*,
        (SELECT count(*) FROM gallery_images WHERE gallery_id = g.id) as image_count,
        row_to_json(p.*) as photographer,
        row_to_json(l.*) as location,
        row_to_json(o.*) as organizer
      FROM galleries g
      LEFT JOIN photographers p ON g.photographer_id = p.id
      LEFT JOIN locations l ON g.location_id = l.id
      LEFT JOIN organizers o ON g.organizer_id = o.id
      ORDER BY g.shoot_date DESC
    `
  } catch (e: any) {
    fetchError = e
    console.warn("[v0] Warning: Could not connect to Database:", e.message)
  }

  const galleriesWithCount = galleries?.map((gallery: any) => ({
    ...gallery,
    _count: {
      gallery_images: gallery.image_count || 0,
    },
    // Map singular joined fields to plural names expected by Gallery interface
    photographers: gallery.photographer,
    locations: gallery.location,
    organizers: gallery.organizer,
  }))

  return (
    <main className="min-h-screen bg-background border-background">
      <div className="mx-auto py-12 shadow-none">
        <header className="mb-12 text-center relative px-4">
          <div className="absolute right-0 top-0">
            <AuthButton />
          </div>
          <h1 className="mb-4 font-serif font-bold tracking-tight text-foreground text-8xl">Padel in Valencia</h1>
          <MainNav />
        </header>

        {fetchError ? (
          <div className="max-w-2xl mx-auto px-4">
            <Alert variant="warning" className="bg-yellow-50 border-yellow-200 text-yellow-800">
              <AlertCircle className="h-4 w-4 text-yellow-600" />
              <AlertTitle>Connection Issue</AlertTitle>
              <AlertDescription>
                Could not load galleries.
                <br />
                <span className="text-xs opacity-70 mt-2 block font-mono">
                  {fetchError.message || JSON.stringify(fetchError)}
                </span>
              </AlertDescription>
            </Alert>
          </div>
        ) : (
          <GalleryGrid galleries={(galleriesWithCount as Gallery[]) || []} />
        )}
      </div>
    </main>
  )
}
