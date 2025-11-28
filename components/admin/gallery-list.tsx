"use client"

import { useState, useEffect } from "react"
import Image from "next/image"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Trash2, ExternalLink } from "lucide-react"
import { deleteGalleryAction } from "@/app/admin/actions"
import { EditGalleryDialog } from "./edit-gallery-dialog"
import { GalleryImagesManager } from "./gallery-images-manager"
import type { Gallery, Photographer, Location, Organizer } from "@/lib/types"
import { getGalleryFaceRecognitionStatsAction } from "@/app/admin/actions"

interface GalleryListProps {
  galleries: Gallery[]
  photographers: Photographer[]
  locations: Location[]
  organizers: Organizer[]
  onDelete?: () => void
}

function formatDateTime(dateString: string): string {
  const date = new Date(dateString)
  const day = String(date.getDate()).padStart(2, "0")
  const month = String(date.getMonth() + 1).padStart(2, "0")
  const year = String(date.getFullYear()).slice(-2)
  const hours = String(date.getHours()).padStart(2, "0")
  const minutes = String(date.getMinutes()).padStart(2, "0")
  return `${day}.${month}.${year} ${hours}:${minutes}`
}

export function GalleryList({ galleries, photographers, locations, organizers, onDelete }: GalleryListProps) {
  const [deletingId, setDeletingId] = useState<string | null>(null)
  const [galleryStats, setGalleryStats] = useState<
    Map<string, { isFullyVerified: boolean; verifiedCount: number; totalCount: number }>
  >(new Map())

  useEffect(() => {
    async function loadVerificationStatus() {
      const statsMap = new Map<string, { isFullyVerified: boolean; verifiedCount: number; totalCount: number }>()

      for (const gallery of galleries) {
        const result = await getGalleryFaceRecognitionStatsAction(gallery.id)
        if (result.success && result.data) {
          const stats = Object.values(result.data)
          const hasImages = stats.length > 0
          const allVerified = stats.every((stat) => stat.fullyRecognized)
          const verifiedCount = stats.filter((stat) => stat.fullyRecognized).length
          const totalCount = stats.length

          statsMap.set(gallery.id, {
            isFullyVerified: hasImages && allVerified,
            verifiedCount,
            totalCount,
          })
        }
      }

      setGalleryStats(statsMap)
    }

    loadVerificationStatus()
  }, [galleries])

  async function handleDelete(id: string) {
    if (!confirm("Вы уверены, что хотите удалить эту галерею?")) return

    setDeletingId(id)
    await deleteGalleryAction(id)
    setDeletingId(null)
    onDelete?.()
  }

  if (galleries.length === 0) {
    return (
      <Card>
        <CardContent className="flex min-h-[200px] items-center justify-center">
          <p className="text-muted-foreground">Нет галерей</p>
        </CardContent>
      </Card>
    )
  }

  return (
    <div className="grid gap-4">
      {galleries.map((gallery) => {
        const imagesCount = gallery.gallery_images?.length ?? 0
        const stats = galleryStats.get(gallery.id)
        const isFullyVerified = stats?.isFullyVerified ?? false
        const verifiedCount = stats?.verifiedCount ?? 0
        const totalCount = stats?.totalCount ?? imagesCount
        const verificationPercentage = totalCount > 0 ? Math.round((verifiedCount / totalCount) * 100) : 0

        return (
          <Card key={gallery.id}>
            <CardContent className="flex items-center gap-4 p-4">
              <div
                className="relative h-32 w-24 flex-shrink-0 overflow-hidden rounded-md bg-muted cursor-pointer hover:opacity-80 transition-opacity"
                onClick={() => {
                  const galleryButton = document.querySelector(`[data-gallery-id="${gallery.id}"]`) as HTMLButtonElement
                  if (galleryButton) galleryButton.click()
                }}
              >
                <Image
                  src={gallery.cover_image_url || "/placeholder.svg"}
                  alt={gallery.title}
                  fill
                  className="object-cover"
                  sizes="96px"
                />
              </div>

              <div className="flex-1">
                <h3 className="font-semibold">{gallery.title}</h3>
                <p className="text-sm text-muted-foreground">
                  {new Date(gallery.shoot_date).toLocaleDateString("ru-RU")}
                </p>
                <div className="mt-1 flex gap-4 text-xs text-muted-foreground">
                  {gallery.organizers && <span>Организатор: {gallery.organizers.name}</span>}
                  {gallery.locations && <span>Место съемки: {gallery.locations.name}</span>}
                  {gallery.photographers && <span>Фотограф: {gallery.photographers.name}</span>}
                  {gallery.external_gallery_url && (
                    <a
                      href={gallery.external_gallery_url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center gap-1 hover:underline"
                      onClick={(e) => e.stopPropagation()}
                    >
                      Источник
                      <ExternalLink className="h-3 w-3" />
                    </a>
                  )}
                </div>
                <div className="mt-1 flex gap-4 text-xs text-muted-foreground">
                  <span>Количество фотографий в галерее: {imagesCount}</span>
                  {totalCount > 0 && (
                    <span>
                      Подтверждено фото: {verifiedCount} ({verificationPercentage}%)
                    </span>
                  )}
                </div>
                <div className="mt-1 flex gap-4 text-xs text-muted-foreground">
                  <span>Создание галереи: {formatDateTime(gallery.created_at)}</span>
                  <span>Внесение последних изменений: {formatDateTime(gallery.updated_at)}</span>
                </div>
              </div>

              <div className="flex gap-2">
                <GalleryImagesManager
                  galleryId={gallery.id}
                  galleryTitle={gallery.title}
                  initialSortOrder={gallery.sort_order || undefined}
                  isFullyVerified={isFullyVerified}
                  data-gallery-id={gallery.id}
                />
                <EditGalleryDialog
                  gallery={gallery}
                  photographers={photographers}
                  locations={locations}
                  organizers={organizers}
                />
                <Button variant="outline" size="sm" asChild>
                  <a href={`/gallery/${gallery.id}`} target="_blank" rel="noopener noreferrer">
                    <ExternalLink className="h-4 w-4" />
                  </a>
                </Button>
                <Button
                  variant="destructive"
                  size="sm"
                  onClick={() => handleDelete(gallery.id)}
                  disabled={deletingId === gallery.id}
                >
                  <Trash2 className="h-4 w-4" />
                </Button>
              </div>
            </CardContent>
          </Card>
        )
      })}
    </div>
  )
}
