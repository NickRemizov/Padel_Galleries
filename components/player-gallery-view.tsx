"use client"

import { useState, useEffect, useRef } from "react"
import { ArrowLeft } from "lucide-react"
import type { Person, GalleryImage } from "@/lib/types"
import { Button } from "@/components/ui/button"
import Link from "next/link"
import { ImageLightbox } from "@/components/image-lightbox"
import { usePathname, useSearchParams } from "next/navigation"

interface PlayerGalleryViewProps {
  player: Person
  images: (GalleryImage & { gallery?: any })[]
}

export function PlayerGalleryView({ player, images }: PlayerGalleryViewProps) {
  const [lightboxOpen, setLightboxOpen] = useState(false)
  const [currentIndex, setCurrentIndex] = useState(0)
  const gridRef = useRef<HTMLDivElement>(null)
  const isotopeRef = useRef<any>(null)
  const pathname = usePathname()
  const searchParams = useSearchParams()

  useEffect(() => {
    setLightboxOpen(false)
  }, [pathname])

  const lightboxImages = images.map((img, index) => ({
    id: img.id,
    url: img.image_url,
    originalUrl: img.original_url,
    alt: `${player.real_name} - изображение ${index + 1}`,
    filename: img.original_filename || `image-${index + 1}.jpg`,
    fileSize: img.file_size || undefined,
    width: img.width || undefined,
    height: img.height || undefined,
    galleryTitle: img.gallery?.title,
    galleryDate: img.gallery?.shoot_date,
  }))

  const handleImageClick = (index: number) => {
    setCurrentIndex(index)
    setLightboxOpen(true)
  }

  const formatShortDate = (dateString: string | null | undefined) => {
    if (!dateString) return ""
    const date = new Date(dateString)
    const day = String(date.getDate()).padStart(2, "0")
    const month = String(date.getMonth() + 1).padStart(2, "0")
    return `${day}.${month}`
  }

  useEffect(() => {
    const initIsotope = async () => {
      if (typeof window === "undefined" || !gridRef.current || images.length === 0) return

      try {
        const Isotope = (await import("isotope-layout")).default
        const imagesLoaded = (await import("imagesloaded")).default

        if (isotopeRef.current) {
          console.log("[v0] Destroying previous Isotope instance")
          isotopeRef.current.destroy()
          isotopeRef.current = null
        }

        if (!gridRef.current) return

        const items = gridRef.current.querySelectorAll(".gallery-image-item")
        console.log("[v0] Isotope init for player", player.id, "- found", items.length, "items")

        imagesLoaded(gridRef.current, () => {
          if (!gridRef.current) return

          console.log("[v0] Images loaded, initializing Isotope for player", player.id)

          isotopeRef.current = new Isotope(gridRef.current, {
            itemSelector: ".gallery-image-item",
            layoutMode: "masonry",
            masonry: {
              columnWidth: ".gallery-image-item",
              gutter: 8,
            },
            percentPosition: true,
            transitionDuration: "0.3s",
          })

          setTimeout(() => {
            if (isotopeRef.current) {
              isotopeRef.current.layout()
              console.log("[v0] Isotope layout refreshed (1st)")
            }
          }, 100)

          setTimeout(() => {
            if (isotopeRef.current) {
              isotopeRef.current.layout()
              console.log("[v0] Isotope layout refreshed (2nd)")
            }
          }, 300)
        })
      } catch (error) {
        console.error("[v0] Failed to initialize Isotope:", error)
      }
    }

    initIsotope()

    return () => {
      if (isotopeRef.current) {
        console.log("[v0] Cleaning up Isotope instance")
        isotopeRef.current.destroy()
        isotopeRef.current = null
      }
    }
  }, [player.id, images.length])

  useEffect(() => {
    const photoParam = searchParams.get("photo")
    if (photoParam) {
      const photoIndex = images.findIndex((img) => img.id === photoParam)

      if (photoIndex !== -1) {
        setCurrentIndex(photoIndex)
        setLightboxOpen(true)
      }
    }
  }, [searchParams, images])

  if (images.length === 0) {
    return (
      <div className="min-h-screen bg-background flex flex-col items-center justify-center">
        <h1 className="text-3xl font-semibold mb-4">{player.real_name}</h1>
        <p className="text-muted-foreground mb-6">У этого игрока пока нет фотографий</p>
        <Link href="/players">
          <Button variant="outline">
            <ArrowLeft className="mr-2 h-4 w-4" />
            Вернуться к списку игроков
          </Button>
        </Link>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-background">
      <div className="max-w-7xl pl-4 pr-12 py-2">
        <Link href="/players">
          <Button variant="ghost" className="mb-2">
            <ArrowLeft className="mr-2 h-4 w-4" />
            Назад к игрокам
          </Button>
        </Link>
        <h1 className="text-3xl font-bold mb-2">{player.real_name}</h1>
        {player.telegram_nickname && (
          <a
            href={player.telegram_profile_url || `https://t.me/${player.telegram_nickname.replace("@", "")}`}
            target="_blank"
            rel="noopener noreferrer"
            className="text-muted-foreground hover:text-foreground transition-colors underline decoration-dotted"
          >
            {player.telegram_nickname}
          </a>
        )}
      </div>

      <div className="px-1 sm:px-2 pb-8">
        <div ref={gridRef} className="isotope-gallery-grid">
          {images.map((image, index) => (
            <div
              key={image.id}
              className="gallery-image-item cursor-pointer group relative overflow-hidden rounded-lg"
              onClick={() => handleImageClick(index)}
            >
              <img
                src={image.image_url || "/placeholder.svg"}
                alt={`${player.real_name} - изображение ${index + 1}`}
                className="w-full h-auto transition-transform duration-300 group-hover:scale-105"
                loading={index === 0 ? "eager" : "lazy"}
              />
              <div className="absolute inset-0 bg-black/0 group-hover:bg-black/20 transition-colors duration-300" />
              {image.gallery && (
                <div className="absolute bottom-0 left-0 right-0 p-2 bg-gradient-to-t from-black/70 to-transparent">
                  <p className="text-white text-xs leading-tight">
                    {image.gallery.title} {formatShortDate(image.gallery.shoot_date)}
                  </p>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      <ImageLightbox
        images={lightboxImages}
        currentIndex={currentIndex}
        isOpen={lightboxOpen}
        onClose={() => setLightboxOpen(false)}
        onNavigate={setCurrentIndex}
        currentPlayerId={player.id}
      />
    </div>
  )
}
