"use client"

import { useState, useEffect } from "react"
import { Star } from "lucide-react"
import { Button } from "./ui/button"
import { useAuth } from "@/lib/auth-context"
import { cn } from "@/lib/utils"

interface FavoriteButtonProps {
  imageId: string
  className?: string
}

export function FavoriteButton({ imageId, className }: FavoriteButtonProps) {
  const { user } = useAuth()
  const [isFavorited, setIsFavorited] = useState(false)
  const [isLoading, setIsLoading] = useState(false)

  useEffect(() => {
    if (user) {
      fetchFavoriteStatus()
    }
  }, [imageId, user])

  async function fetchFavoriteStatus() {
    try {
      const response = await fetch(`/api/favorites/${imageId}`)
      const data = await response.json()
      setIsFavorited(data.isFavorited)
    } catch (error) {
      console.error("[v0] Error fetching favorite status:", error)
    }
  }

  async function toggleFavorite() {
    if (!user) {
      alert("Войдите через Telegram, чтобы добавлять фото в избранное")
      return
    }

    setIsLoading(true)
    try {
      const response = await fetch(`/api/favorites/${imageId}`, {
        method: "POST",
      })

      if (response.ok) {
        const data = await response.json()
        setIsFavorited(data.isFavorited)
      }
    } catch (error) {
      console.error("[v0] Error toggling favorite:", error)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <Button
      variant="ghost"
      size="icon"
      onClick={toggleFavorite}
      disabled={isLoading}
      className={className}
      title={isFavorited ? "Убрать из избранного" : "Добавить в избранное"}
    >
      <Star className={cn("h-6 w-6 transition-colors", isFavorited ? "fill-yellow-500 text-yellow-500" : "")} />
    </Button>
  )
}
