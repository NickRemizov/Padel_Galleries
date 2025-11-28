import { Spinner } from "@/components/ui/spinner"
import { AuthButton } from "@/components/auth-button"
import { MainNav } from "@/components/main-nav"
import Link from "next/link"

export default function FavoritesLoading() {
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

        <div className="flex items-center justify-center py-20">
          <div className="flex flex-col items-center gap-4">
            <Spinner className="size-8" />
            <p className="text-sm text-muted-foreground">Загрузка избранного...</p>
          </div>
        </div>
      </div>
    </main>
  )
}
