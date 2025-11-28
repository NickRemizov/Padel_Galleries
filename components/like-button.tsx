"use client"

import { useState, useEffect } from "react"
import { Heart } from "lucide-react"
import { Button } from "./ui/button"
import { useAuth } from "@/lib/auth-context"
import { cn } from "@/lib/utils"

interface LikeButtonProps {
  imageId: string
  className?: string
  showCount?: boolean
}

export function LikeButton({ imageId, className, showCount = true }: LikeButtonProps) {
  const { user } = useAuth()
  const [isLiked, setIsLiked] = useState(false)
  const [likesCount, setLikesCount] = useState(0)
  const [isLoading, setIsLoading] = useState(false)

  useEffect(() => {
    fetchLikes()
  }, [imageId])

  async function fetchLikes() {
    try {
      const response = await fetch(`/api/likes/${imageId}`)
      const data = await response.json()
      setIsLiked(data.isLiked)
      setLikesCount(data.count)
    } catch (error) {
      console.error("[v0] Error fetching likes:", error)
    }
  }

  async function toggleLike() {
    if (!user) {
      // Show login dialog or message
      alert("Войдите через Telegram, чтобы ставить лайки")
      return
    }

    setIsLoading(true)
    try {
      const response = await fetch(`/api/likes/${imageId}`, {
        method: "POST",
      })

      if (response.ok) {
        const data = await response.json()
        setIsLiked(data.isLiked)
        setLikesCount(data.count)
      }
    } catch (error) {
      console.error("[v0] Error toggling like:", error)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <Button variant="ghost" size="icon" onClick={toggleLike} disabled={isLoading} className={className}>
      <Heart className={cn("h-6 w-6 transition-colors", isLiked ? "fill-red-500 text-red-500" : "")} />
      {showCount && <span className="text-sm">{likesCount}</span>}
    </Button>
  )
}
