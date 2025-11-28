import { Spinner } from "@/components/ui/spinner"

export default function GalleryLoading() {
  return (
    <div className="flex min-h-screen items-center justify-center bg-background">
      <div className="flex flex-col items-center gap-4">
        <Spinner className="size-8" />
        <p className="text-sm text-muted-foreground">Загрузка галереи...</p>
      </div>
    </div>
  )
}
