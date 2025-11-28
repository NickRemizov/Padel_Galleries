"use client"

import Link from "next/link"
import Masonry from "react-masonry-css"
import type { Favorite } from "@/lib/types"

interface FavoritesGridProps {
  favorites: (Favorite & {
    gallery_images?: {
      id: string
      gallery_id: string
      image_url: string
      original_url: string
      original_filename: string
      file_size: number | null
      width: number | null
      height: number | null
      created_at: string
    }
  })[]
}

export function FavoritesGrid({ favorites }: FavoritesGridProps) {
  const breakpointColumns = {
    default: 4,
    1536: 3,
    1024: 2,
    640: 1,
  }

  return (
    <Masonry breakpointCols={breakpointColumns} className="flex -ml-4 w-auto" columnClassName="pl-4 bg-clip-padding">
      {favorites.map((favorite) => {
        const image = favorite.gallery_images
        if (!image) return null

        return (
          <Link
            key={favorite.id}
            href={`/gallery/${image.gallery_id}?photo=${image.id}`}
            className="block mb-4 group relative overflow-hidden rounded-lg"
          >
            <img
              src={image.image_url || "/placeholder.svg"}
              alt={image.original_filename}
              className="w-full h-auto transition-transform duration-300 group-hover:scale-105"
            />
            <div className="absolute inset-0 bg-black/0 group-hover:bg-black/20 transition-colors" />
          </Link>
        )
      })}
    </Masonry>
  )
}
