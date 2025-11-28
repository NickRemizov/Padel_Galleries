"use client"

import Image from "next/image"
import { Calendar, MapPin, Users, Camera, ExternalLink } from "lucide-react"
import type { Gallery } from "@/lib/types"

interface GalleryInfoCardProps {
  gallery: Gallery
}

export function GalleryInfoCard({ gallery }: GalleryInfoCardProps) {
  const formattedDate = new Date(gallery.shoot_date).toLocaleDateString("ru-RU", {
    day: "2-digit",
    month: "2-digit",
    year: "numeric",
  })

  const shortDate = new Date(gallery.shoot_date).toLocaleDateString("ru-RU", {
    day: "2-digit",
    month: "2-digit",
  })

  const imageCount = gallery.gallery_images?.length || 0

  return (
    <div className="flex h-full bg-card rounded-lg overflow-hidden shadow-lg">
      <div className="relative w-2/3 aspect-square">
        <Image
          src={gallery.cover_image_square_url || gallery.cover_image_url || "/placeholder.svg?height=600&width=600"}
          alt={gallery.title}
          fill
          className="object-cover object-top"
          sizes="(max-width: 640px) 66vw, (max-width: 1024px) 50vw, 33vw"
          priority
        />
      </div>

      {/* Right side - Info (1/3 width) */}
      <div className="w-1/3 p-4 sm:p-6 flex flex-col justify-between bg-card">
        <div>
          <h2 className="font-bold text-lg sm:text-xl lg:text-2xl mb-4 text-balance leading-tight">
            {gallery.title} {shortDate}
          </h2>

          <div className="space-y-3 text-sm sm:text-base text-muted-foreground">
            <div className="flex items-start gap-2">
              <Calendar className="h-4 w-4 sm:h-5 sm:w-5 mt-0.5 flex-shrink-0" />
              <span className="break-words">{formattedDate}</span>
            </div>

            {gallery.organizers && (
              <div className="flex items-start gap-2">
                <Users className="h-4 w-4 sm:h-5 sm:w-5 mt-0.5 flex-shrink-0" />
                <span className="break-words">{gallery.organizers.name}</span>
              </div>
            )}

            {gallery.locations && (
              <div className="flex items-start gap-2">
                <MapPin className="h-4 w-4 sm:h-5 sm:w-5 mt-0.5 flex-shrink-0" />
                <span className="break-words">{gallery.locations.name}</span>
              </div>
            )}

            {gallery.photographers && (
              <div className="flex items-start gap-2">
                <Camera className="h-4 w-4 sm:h-5 sm:w-5 mt-0.5 flex-shrink-0" />
                <span className="break-words">{gallery.photographers.name}</span>
              </div>
            )}
          </div>
        </div>

        <div className="mt-4 pt-4 border-t border-border">
          <div className="flex items-center justify-between text-sm text-muted-foreground">
            <div className="flex items-center gap-1">
              <Camera className="h-4 w-4" />
              <span>{imageCount} фото</span>
            </div>
            {gallery.photographers?.nickname && (
              <div className="flex items-center gap-1">
                <Users className="h-4 w-4" />
                <span>{gallery.photographers.nickname}</span>
              </div>
            )}
          </div>

          {gallery.external_gallery_url && (
            <a
              href={gallery.external_gallery_url}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 mt-3 text-sm hover:text-foreground transition-colors underline cursor-pointer"
              onClick={(e) => e.stopPropagation()}
            >
              <ExternalLink className="h-4 w-4" />
              <span>Источник</span>
            </a>
          )}
        </div>
      </div>
    </div>
  )
}
