"use client"

import { useSearchParams } from "next/navigation"

import { useState, useMemo, useEffect, useRef } from "react"
import { ArrowLeft } from "lucide-react"
import type { Gallery } from "@/lib/types"
import { Button } from "@/components/ui/button"
import Link from "next/link"
import { ImageLightbox } from "@/components/image-lightbox"
import { GalleryInfoCard } from "@/components/gallery-info-card"

interface GalleryViewProps {
  gallery: Gallery
}

export function GalleryView({ gallery }: GalleryViewProps) {
  const [lightboxOpen, setLightboxOpen] = useState(false)
  const [currentIndex, setCurrentIndex] = useState(0)
  const searchParams = useSearchParams()
  const gridRef = useRef<HTMLDivElement>(null)
  const isotopeRef = useRef<any>(null)

  const shortDate = new Date(gallery.shoot_date).toLocaleDateString("ru-RU", {
    day: "2-digit",
    month: "2-digit",
  })

  const images = gallery.gallery_images || []

  const sortedImages = useMemo(() => {
    const imagesCopy = [...images]
    const sortOrder = gallery.sort_order || "filename"

    switch (sortOrder) {
      case "filename":
        return imagesCopy.sort((a, b) => (a.original_filename || "").localeCompare(b.original_filename || ""))
      case "created":
      case "added":
        return imagesCopy.sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime())
      default:
        return imagesCopy
    }
  }, [images, gallery.sort_order])

  const lightboxImages = sortedImages.map((img, index) => ({
    id: img.id,
    url: img.image_url,
    originalUrl: img.original_url,
    alt: `${gallery.title} - изображение ${index + 1}`,
    filename: img.original_filename || `image-${index + 1}.jpg`,
    fileSize: img.file_size,
    width: img.width,
    height: img.height,
  }))

  const handleImageClick = (index: number) => {
    setCurrentIndex(index)
    setLightboxOpen(true)
  }

  useEffect(() => {
    const initIsotope = async () => {
      if (typeof window === "undefined" || !gridRef.current || sortedImages.length === 0) return

      try {
        const Isotope = (await import("isotope-layout")).default
        const imagesLoaded = (await import("imagesloaded")).default

        imagesLoaded(gridRef.current, () => {
          if (!gridRef.current) return

          const items = gridRef.current.querySelectorAll(".gallery-image-item")
          if (items.length === 0) {
            console.error("[v0] No gallery items found for Isotope")
            return
          }

          const regularItem = gridRef.current.querySelector(".gallery-image-item:not(.gallery-image-item--width2)")

          if (isotopeRef.current) {
            isotopeRef.current.destroy()
          }

          const isotopeOptions: any = {
            itemSelector: ".gallery-image-item",
            layoutMode: "masonry",
            masonry: {
              gutter: 8,
            },
            percentPosition: true,
            transitionDuration: "0.3s",
          }

          if (regularItem) {
            isotopeOptions.masonry.columnWidth = ".gallery-image-item:not(.gallery-image-item--width2)"
          }

          isotopeRef.current = new Isotope(gridRef.current, isotopeOptions)
        })
      } catch (error) {
        console.error("[v0] Failed to initialize Isotope:", error)
      }
    }

    initIsotope()

    return () => {
      if (isotopeRef.current) {
        isotopeRef.current.destroy()
        isotopeRef.current = null
      }
    }
  }, [sortedImages.length])

  useEffect(() => {
    const photoParam = searchParams.get("photo")
    if (photoParam) {
      // Try to find photo by ID first
      const photoIndex = sortedImages.findIndex((img) => img.id === photoParam)

      if (photoIndex !== -1) {
        // Found by ID
        setCurrentIndex(photoIndex)
        setLightboxOpen(true)
      } else {
        // Fallback: try as numeric index for backwards compatibility
        const numericIndex = Number.parseInt(photoParam, 10)
        if (!isNaN(numericIndex) && numericIndex >= 0 && numericIndex < sortedImages.length) {
          setCurrentIndex(numericIndex)
          setLightboxOpen(true)
        }
      }
    }
  }, [searchParams, sortedImages])

  return (
    <div className="min-h-screen bg-background">
      <div className="max-w-7xl pl-4 pr-12 py-2">
        <Link href="/">
          <Button variant="ghost" className="mb-2">
            <ArrowLeft className="mr-2 h-4 w-4" />
            Назад к галереям
          </Button>
        </Link>
      </div>

      <div className="px-1 sm:px-2 pb-8">
        {images.length === 0 ? (
          <div className="text-center py-12 text-muted-foreground">
            <p>В этой галерее пока нет изображений</p>
          </div>
        ) : (
          <div ref={gridRef} className="isotope-gallery-grid">
            <div className="gallery-image-item gallery-image-item--width2">
              <GalleryInfoCard gallery={gallery} />
            </div>

            {sortedImages.map((image, index) => {
              const imagePeople = (image as any).people || []

              return (
                <div
                  key={image.id}
                  className="gallery-image-item cursor-pointer group relative overflow-hidden rounded-lg"
                  onClick={() => handleImageClick(index)}
                >
                  <img
                    src={image.image_url || "/placeholder.svg"}
                    alt={`${gallery.title} - изображение ${index + 1}`}
                    className="w-full h-auto transition-transform duration-300 group-hover:scale-105"
                    loading={index === 0 ? "eager" : "lazy"}
                  />
                  <div className="absolute inset-0 bg-black/0 group-hover:bg-black/20 transition-colors duration-300" />

                  {imagePeople.length > 0 && (
                    <div className="absolute bottom-0 left-0 right-0 p-2 bg-gradient-to-t from-black/70 to-transparent">
                      <p className="text-white text-xs font-medium leading-tight">
                        {imagePeople.map((person: any) => person.name).join(", ")}
                      </p>
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        )}
      </div>

      <ImageLightbox
        images={lightboxImages}
        currentIndex={currentIndex}
        isOpen={lightboxOpen}
        onClose={() => setLightboxOpen(false)}
        onNavigate={setCurrentIndex}
        galleryId={gallery.id}
      />
    </div>
  )
}
