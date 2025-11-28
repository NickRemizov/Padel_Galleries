"use client"

import type React from "react"

import { useState, useEffect } from "react"
import { X, ChevronLeft, ChevronRight, Download, LinkIcon, MessageCircle } from "lucide-react"
import { Button } from "@/components/ui/button"
import { LikeButton } from "@/components/like-button"
import { FavoriteButton } from "@/components/favorite-button"
import { CommentsSection } from "@/components/comments-section"
import { cn } from "@/lib/utils"
import Link from "next/link"

interface ImageLightboxProps {
  images: Array<{
    id?: string
    url: string
    originalUrl: string
    alt: string
    filename?: string
    fileSize?: number
    width?: number
    height?: number
    galleryTitle?: string
    galleryDate?: string
  }>
  currentIndex: number
  isOpen: boolean
  onClose: () => void
  onNavigate: (index: number) => void
  galleryId?: string
  currentPlayerId?: string
}

export function ImageLightbox({
  images,
  currentIndex,
  isOpen,
  onClose,
  onNavigate,
  galleryId,
  currentPlayerId,
}: ImageLightboxProps) {
  const currentImage = images[currentIndex]
  const [showCopied, setShowCopied] = useState(false)
  const [showComments, setShowComments] = useState(false)
  const [touchStart, setTouchStart] = useState<number | null>(null)
  const [touchEnd, setTouchEnd] = useState<number | null>(null)
  const [verifiedPeople, setVerifiedPeople] = useState<Array<{ id: string; name: string }>>([])

  const minSwipeDistance = 50

  useEffect(() => {
    const fetchVerifiedPeople = async () => {
      if (!currentImage?.id) {
        setVerifiedPeople([])
        return
      }

      try {
        const response = await fetch(`/api/images/${currentImage.id}/people`)
        if (response.ok) {
          const people = await response.json()
          setVerifiedPeople(people)
        } else {
          setVerifiedPeople([])
        }
      } catch (error) {
        console.error("[v0] Error fetching verified people:", error)
        setVerifiedPeople([])
      }
    }

    fetchVerifiedPeople()
  }, [currentImage?.id])

  useEffect(() => {
    if (!isOpen) return

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "ArrowLeft") {
        e.preventDefault()
        handlePrev()
      } else if (e.key === "ArrowRight") {
        e.preventDefault()
        handleNext()
      } else if (e.key === "Escape") {
        e.preventDefault()
        onClose()
      }
    }

    window.addEventListener("keydown", handleKeyDown)
    return () => window.removeEventListener("keydown", handleKeyDown)
  }, [isOpen, currentIndex, images.length])

  const formatShortDate = (dateString?: string) => {
    if (!dateString) return ""

    const date = new Date(dateString)
    const months = ["янв", "фев", "мар", "апр", "май", "июн", "июл", "авг", "сен", "окт", "ноя", "дек"]

    const day = date.getDate()
    const month = months[date.getMonth()]
    const year = date.getFullYear()

    return `${day} ${month} ${year}`
  }

  const formatDateDDMM = (dateString?: string) => {
    if (!dateString) return ""

    const date = new Date(dateString)
    const day = String(date.getDate()).padStart(2, "0")
    const month = String(date.getMonth() + 1).padStart(2, "0")

    return `${day}.${month}`
  }

  const onTouchStart = (e: React.TouchEvent) => {
    setTouchEnd(null)
    setTouchStart(e.targetTouches[0].clientX)
  }

  const onTouchMove = (e: React.TouchEvent) => {
    setTouchEnd(e.targetTouches[0].clientX)
  }

  const onTouchEnd = () => {
    if (!touchStart || !touchEnd) return

    const distance = touchStart - touchEnd
    const isLeftSwipe = distance > minSwipeDistance
    const isRightSwipe = distance < -minSwipeDistance

    if (isLeftSwipe) {
      handleNext()
    } else if (isRightSwipe) {
      handlePrev()
    }

    setTouchStart(null)
    setTouchEnd(null)
  }

  const handleDownload = async () => {
    try {
      if (currentImage?.id) {
        fetch(`/api/downloads/${currentImage.id}`, {
          method: "POST",
        }).catch((error) => {
          console.error("[v0] Error tracking download:", error)
        })
      }

      const response = await fetch(currentImage.originalUrl)
      const blob = await response.blob()
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement("a")
      a.href = url
      a.download = currentImage.filename || `image-${currentIndex + 1}.jpg`
      document.body.appendChild(a)
      a.click()
      window.URL.revokeObjectURL(url)
      document.body.removeChild(a)
    } catch (error) {
      console.error("Error downloading image:", error)
    }
  }

  const formatFileSize = (bytes?: number) => {
    if (!bytes) return "N/A"
    return `${(bytes / 1024).toFixed(0)} KB`
  }

  const formatDimensions = (width?: number, height?: number) => {
    if (!width || !height) return ""
    return `${width} × ${height} px`
  }

  const handlePrev = () => {
    const prevIndex = currentIndex === 0 ? images.length - 1 : currentIndex - 1
    onNavigate(prevIndex)
  }

  const handleNext = () => {
    const nextIndex = currentIndex === images.length - 1 ? 0 : currentIndex + 1
    onNavigate(nextIndex)
  }

  const handleShare = async () => {
    if (!currentImage?.id) {
      alert("Невозможно создать ссылку на это фото")
      return
    }

    let url: string
    if (galleryId) {
      url = `${window.location.origin}/gallery/${galleryId}?photo=${currentImage.id}`
    } else if (currentPlayerId) {
      url = `${window.location.origin}/players/${currentPlayerId}?photo=${currentImage.id}`
    } else {
      url = `${window.location.origin}${window.location.pathname}?photo=${currentImage.id}`
    }

    try {
      await navigator.clipboard.writeText(url)
      setShowCopied(true)
      setTimeout(() => setShowCopied(false), 2000)
    } catch (error) {
      console.error("Error copying to clipboard:", error)
      alert("Не удалось скопировать ссылку")
    }
  }

  if (!isOpen || !currentImage) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Overlay */}
      <div className="absolute inset-0 bg-black/90" onClick={onClose} />

      {/* Content */}
      <div
        className="relative z-10 flex items-center justify-center w-full h-full"
        onTouchStart={onTouchStart}
        onTouchMove={onTouchMove}
        onTouchEnd={onTouchEnd}
      >
        {verifiedPeople.length > 0 && (
          <div className="absolute top-4 left-4 flex flex-col gap-1 z-20">
            {verifiedPeople.map((person) => {
              const isCurrentPlayer = person.id === currentPlayerId
              return isCurrentPlayer ? (
                <div
                  key={person.id}
                  className="bg-black/70 text-white/50 px-3 py-1.5 rounded-full text-sm cursor-default"
                >
                  {person.name}
                </div>
              ) : (
                <Link
                  key={person.id}
                  href={`/players/${person.id}`}
                  className="bg-black/70 hover:bg-black/80 text-white px-3 py-1.5 rounded-full text-sm transition-colors"
                >
                  {person.name}
                </Link>
              )
            })}
          </div>
        )}

        <div className="absolute top-4 left-1/2 -translate-x-1/2 md:block hidden bg-black/70 text-white px-4 py-2 rounded-full text-sm">
          {currentIndex + 1} / {images.length}
        </div>
        <div className="absolute bottom-4 left-4 md:hidden bg-black/70 text-white px-4 py-2 rounded-full text-sm">
          {currentIndex + 1} / {images.length}
        </div>

        {showCopied && (
          <div className="absolute top-20 left-1/2 -translate-x-1/2 bg-green-600 text-white px-4 py-2 rounded-full text-sm">
            Ссылка скопирована в буфер обмена
          </div>
        )}

        {/* Close button */}
        <Button
          variant="ghost"
          size="icon"
          className="absolute top-4 right-4 text-white hover:bg-white/20 bg-black/50"
          onClick={onClose}
        >
          <X className="h-6 w-6" />
        </Button>

        {/* Share button */}
        {currentImage?.id && (
          <Button
            variant="ghost"
            size="icon"
            className="absolute top-4 right-16 text-white hover:bg-white/20 bg-black/50"
            onClick={handleShare}
            title="Скопировать ссылку на фото"
          >
            <LinkIcon className="h-6 w-6" />
          </Button>
        )}

        {/* Download button */}
        <Button
          variant="ghost"
          size="icon"
          className="absolute top-4 right-28 text-white hover:bg-white/20 bg-black/50"
          onClick={handleDownload}
        >
          <Download className="h-6 w-6" />
        </Button>

        {/* Comments button */}
        {currentImage?.id && (
          <Button
            variant="ghost"
            size="icon"
            className={cn(
              "absolute top-4 right-40 text-white hover:bg-white/20 bg-black/50",
              showComments && "bg-white/30",
            )}
            onClick={() => setShowComments(!showComments)}
            title="Комментарии"
          >
            <MessageCircle className="h-6 w-6" />
          </Button>
        )}

        {/* Like button */}
        {currentImage?.id && (
          <LikeButton
            imageId={currentImage.id}
            className="absolute top-4 right-52 bg-black/50 text-white hover:bg-black/60"
          />
        )}

        {/* Favorite button */}
        {currentImage?.id && (
          <FavoriteButton
            imageId={currentImage.id}
            className="absolute top-4 right-64 bg-black/50 text-white hover:bg-black/60"
          />
        )}

        {/* Image */}
        <img
          src={currentImage.url || "/placeholder.svg"}
          alt={currentImage.alt}
          className="max-w-[95vw] max-h-[95vh] w-auto h-auto object-contain"
        />

        <div className="absolute bottom-4 left-1/2 -translate-x-1/2 md:block hidden bg-black/70 text-white px-4 py-2 rounded-full text-sm">
          {currentImage.filename || `image-${currentIndex + 1}.jpg`}
          {currentPlayerId && currentImage.galleryTitle && (
            <>
              {" | "}
              {currentImage.galleryTitle}
              {currentImage.galleryDate && ` ${formatDateDDMM(currentImage.galleryDate)}`}
            </>
          )}
          {formatDimensions(currentImage.width, currentImage.height) && (
            <> | {formatDimensions(currentImage.width, currentImage.height)}</>
          )}
          {" | "}
          {formatFileSize(currentImage.fileSize)}
        </div>
        <div className="absolute bottom-4 right-4 md:hidden bg-black/70 text-white px-4 py-2 rounded-full text-sm">
          {currentImage.filename || `image-${currentIndex + 1}.jpg`}
        </div>

        {/* Previous button */}
        {images.length > 1 && (
          <Button
            variant="ghost"
            size="icon"
            className="absolute left-4 top-1/2 -translate-y-1/2 text-white hover:bg-white/20 bg-black/50"
            onClick={handlePrev}
          >
            <ChevronLeft className="h-8 w-8" />
          </Button>
        )}

        {/* Next button */}
        {images.length > 1 && (
          <Button
            variant="ghost"
            size="icon"
            className="absolute right-4 top-1/2 -translate-y-1/2 text-white hover:bg-white/20 bg-black/50"
            onClick={handleNext}
          >
            <ChevronRight className="h-8 w-8" />
          </Button>
        )}

        {/* Comments panel */}
        {showComments && currentImage?.id && (
          <div className="absolute right-4 top-20 bottom-20 w-96 bg-background rounded-lg shadow-xl overflow-hidden flex flex-col">
            <div className="flex-1 overflow-y-auto p-4">
              <CommentsSection imageId={currentImage.id} />
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
