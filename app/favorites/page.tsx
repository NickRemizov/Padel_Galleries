import { createClient } from "@/lib/supabase/server"
import { cookies } from "next/headers"
import { redirect } from "next/navigation"
import Link from "next/link"
import { AuthButton } from "@/components/auth-button"
import { MainNav } from "@/components/main-nav"
import { FavoritesGrid } from "@/components/favorites-grid"
import { Button } from "@/components/ui/button"

export const dynamic = "force-dynamic"

export default async function FavoritesPage() {
  const cookieStore = await cookies()
  const userId = cookieStore.get("user_id")?.value

  if (!userId) {
    redirect("/")
  }

  const supabase = await createClient()

  const { data: favorites, error } = await supabase
    .from("favorites")
    .select(
      `
      *,
      gallery_images (
        id,
        gallery_id,
        image_url,
        original_url,
        original_filename,
        file_size,
        width,
        height,
        created_at
      )
    `,
    )
    .eq("user_id", userId)
    .order("created_at", { ascending: false })

  if (error) {
    console.error("[v0] Error fetching favorites:", error)
  }

  return (
    <main className="min-h-screen bg-background">
      <div className="container mx-auto px-4 py-12">
        <header className="mb-12 relative">
          <div className="absolute right-0 top-0">
            <AuthButton />
          </div>
          <Link href="/" className="inline-block mb-4 hover:opacity-80 transition-opacity">
            <h1 className="text-center font-serif font-bold tracking-tight text-foreground text-6xl">
              Padel in Valencia
            </h1>
          </Link>
          <MainNav />
        </header>

        {!favorites || favorites.length === 0 ? (
          <div className="text-center py-12">
            <p className="text-muted-foreground text-lg mb-4">У вас пока нет избранных фотографий</p>
            <Link href="/">
              <Button>Перейти к галереям</Button>
            </Link>
          </div>
        ) : (
          <FavoritesGrid favorites={favorites} />
        )}
      </div>
    </main>
  )
}
