"use client"

import type React from "react"

import Image from "next/image"
import Link from "next/link"
import { Calendar, MapPin, Camera, Users, ImageIcon, ExternalLink } from "lucide-react"
import { Card, CardContent } from "@/components/ui/card"
import type { Gallery } from "@/lib/types"

interface GalleryCardProps {
  gallery: Gallery
  onOrganizerClick?: (organizerName: string) => void
  onLocationClick?: (locationName: string) => void
}

export function GalleryCard({ gallery, onOrganizerClick, onLocationClick }: GalleryCardProps) {
  const formattedDate = new Date(gallery.shoot_date).toLocaleDateString("ru-RU", {
    year: "numeric",
    month: "long",
    day: "numeric",
  })

  const shortDate = new Date(gallery.shoot_date).toLocaleDateString("ru-RU", {
    day: "2-digit",
    month: "2-digit",
  })

  const handleOrganizerClick = (e: React.MouseEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (gallery.organizers?.name && onOrganizerClick) {
      onOrganizerClick(gallery.organizers.name)
    }
  }

  const handleLocationClick = (e: React.MouseEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (gallery.locations?.name && onLocationClick) {
      onLocationClick(gallery.locations.name)
    }
  }

  const handleExternalLinkClick = (e: React.MouseEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (gallery.external_gallery_url) {
      window.open(gallery.external_gallery_url, "_blank", "noopener,noreferrer")
    }
  }

  const hasImages = (gallery._count?.gallery_images || 0) > 0
  const hasExternalUrl = !!gallery.external_gallery_url
  const isLinkActive = hasImages || hasExternalUrl
  const linkHref = hasImages ? `/gallery/${gallery.id}` : gallery.external_gallery_url || "#"
  const linkTarget = hasImages ? undefined : "_blank"
  const linkRel = hasImages ? undefined : "noopener noreferrer"
  const imageCount = gallery._count?.gallery_images || 0

  const CardWrapper = isLinkActive ? Link : "div"
  const cardWrapperProps = isLinkActive
    ? {
        href: linkHref,
        target: linkTarget,
        rel: linkRel,
        className: "group",
      }
    : {
        className: "group cursor-not-allowed opacity-60",
      }

  return (
    <CardWrapper {...(cardWrapperProps as any)}>
      <Card className="overflow-hidden transition-all duration-300 hover:shadow-xl mx-0 my-0 bg-slate-200 p-0 leading-6">
        <div className="flex sm:hidden h-full">
          <div className="relative w-2/3 aspect-square">
            <Image
              src={gallery.cover_image_square_url || gallery.cover_image_url || "/placeholder.svg?height=600&width=600"}
              alt={gallery.title}
              fill
              className="object-cover object-top transition-transform duration-500 group-hover:scale-105"
              sizes="66vw"
            />
          </div>

          <div className="w-1/3 p-2 sm:p-3 flex flex-col justify-between bg-card">
            <div>
              <h2 className="font-semibold text-sm sm:text-base mb-2 text-balance leading-tight">
                {gallery.title} {shortDate}
              </h2>

              <div className="space-y-1.5 text-xs text-muted-foreground">
                <div className="flex items-start gap-1">
                  <Calendar className="h-3 w-3 mt-0.5 flex-shrink-0" />
                  <span className="break-words text-[10px] sm:text-xs">{formattedDate}</span>
                </div>

                {gallery.organizers && (
                  <div
                    className="flex items-start gap-1 cursor-pointer hover:text-foreground transition-colors"
                    onClick={handleOrganizerClick}
                  >
                    <Users className="h-3 w-3 mt-0.5 flex-shrink-0" />
                    <span className="break-words underline decoration-dotted text-[10px] sm:text-xs">
                      {gallery.organizers.name}
                    </span>
                  </div>
                )}

                {gallery.locations && (
                  <div
                    className="flex items-start gap-1 cursor-pointer hover:text-foreground transition-colors"
                    onClick={handleLocationClick}
                  >
                    <MapPin className="h-3 w-3 mt-0.5 flex-shrink-0" />
                    <span className="break-words underline decoration-dotted text-[10px] sm:text-xs">
                      {gallery.locations.name}
                    </span>
                  </div>
                )}

                {gallery.photographers && (
                  <div className="flex items-start gap-1">
                    <Camera className="h-3 w-3 mt-0.5 flex-shrink-0" />
                    <span className="break-words text-[10px] sm:text-xs">{gallery.photographers.name}</span>
                  </div>
                )}
              </div>
            </div>

            <div className="mt-2 pt-2 border-t border-border">
              {hasImages && (
                <div className="flex items-center gap-1 text-[10px] sm:text-xs text-muted-foreground mb-1">
                  <ImageIcon className="h-3 w-3" />
                  <span>{imageCount} фото</span>
                </div>
              )}

              {gallery.external_gallery_url && (
                <span
                  role="button"
                  tabIndex={0}
                  className="flex items-center gap-1 text-[10px] sm:text-xs hover:text-foreground transition-colors underline cursor-pointer"
                  onClick={handleExternalLinkClick}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      handleExternalLinkClick(e as any)
                    }
                  }}
                >
                  <ExternalLink className="h-3 w-3" />
                  <span>Источник</span>
                </span>
              )}
            </div>
          </div>
        </div>

        <div className="hidden sm:block">
          <div className="relative aspect-[3/4] overflow-hidden bg-muted">
            <Image
              src={gallery.cover_image_url || "/placeholder.svg?height=400&width=600"}
              alt={gallery.title}
              fill
              className="object-cover transition-transform duration-500 group-hover:scale-105 border-0 mx-0 my-0"
              sizes="(max-width: 1024px) 33vw, (max-width: 1280px) 25vw, 20vw"
            />
            <div className="absolute bottom-1 left-1 right-1 sm:bottom-2 sm:left-2 sm:right-2 bg-black/40 text-white px-2 py-1 sm:px-3 sm:py-2 rounded-md backdrop-blur-sm">
              <h2 className="font-semibold text-balance leading-tight text-xl sm:text-3xl lg:text-2xl xl:text-4xl">
                {gallery.title} {shortDate}
              </h2>
            </div>
            {hasImages && (
              <div className="absolute top-1 left-1 sm:top-2 sm:left-2 flex items-center gap-1 sm:gap-1.5 bg-black/30 text-white px-1.5 py-0.5 sm:px-2 sm:py-1 rounded-md backdrop-blur-sm">
                <ImageIcon className="h-2.5 w-2.5 sm:h-3 sm:w-3" />
                <span className="text-[10px] sm:text-xs font-medium">{imageCount} фото</span>
              </div>
            )}
            {gallery.photographers && (
              <div className="absolute top-1 right-1 sm:top-2 sm:right-2 flex items-center gap-1 sm:gap-1.5 bg-black/30 text-white px-1.5 py-0.5 sm:px-2 sm:py-1 rounded-md backdrop-blur-sm">
                <Camera className="h-2.5 w-2.5 sm:h-3 sm:w-3" />
                <span className="text-[10px] sm:text-xs font-medium">{gallery.photographers.name}</span>
              </div>
            )}
          </div>

          <CardContent className="px-2 sm:px-4 pt-0 pb-1 sm:pb-2">
            <div className="space-y-0.5 sm:space-y-1.5 text-xs sm:text-sm text-muted-foreground">
              <div className="flex items-center gap-1.5 sm:gap-2 text-sm sm:text-base">
                <Calendar className="w-3 h-3 sm:w-4 sm:h-4" />
                <span>{formattedDate}</span>
              </div>

              <div
                className={`flex items-center gap-2 text-base ${
                  gallery.organizers?.name ? "cursor-pointer hover:text-foreground transition-colors" : "invisible"
                }`}
                onClick={gallery.organizers?.name ? handleOrganizerClick : undefined}
                role={gallery.organizers?.name ? "button" : undefined}
                tabIndex={gallery.organizers?.name ? 0 : undefined}
                onKeyDown={
                  gallery.organizers?.name
                    ? (e) => {
                        if (e.key === "Enter" || e.key === " ") {
                          handleOrganizerClick(e as any)
                        }
                      }
                    : undefined
                }
              >
                <Users className="w-4 h-4" />
                <span className={gallery.organizers?.name ? "underline decoration-dotted" : ""}>
                  {gallery.organizers?.name || "\u00A0"}
                </span>
              </div>

              <div
                className={`flex items-center gap-2 text-base ${
                  gallery.locations?.name ? "cursor-pointer hover:text-foreground transition-colors" : "invisible"
                }`}
                onClick={gallery.locations?.name ? handleLocationClick : undefined}
                role={gallery.locations?.name ? "button" : undefined}
                tabIndex={gallery.locations?.name ? 0 : undefined}
                onKeyDown={
                  gallery.locations?.name
                    ? (e) => {
                        if (e.key === "Enter" || e.key === " ") {
                          handleLocationClick(e as any)
                        }
                      }
                    : undefined
                }
              >
                <MapPin className="w-4 h-4" />
                <span className={gallery.locations?.name ? "underline decoration-dotted" : ""}>
                  {gallery.locations?.name || "\u00A0"}
                </span>
              </div>
            </div>
          </CardContent>
        </div>
      </Card>
    </CardWrapper>
  )
}
