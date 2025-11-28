"use client"

import type React from "react"

import { useState, useEffect, useMemo, memo, useRef } from "react"
import Image from "next/image"
import { Button } from "@/components/ui/button"
import { Checkbox } from "@/components/ui/checkbox"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Card, CardContent } from "@/components/ui/card"
import { Images, Trash2, Upload, UserPlus, Scan, Download } from "lucide-react"

import {
  addGalleryImagesAction,
  deleteGalleryImageAction,
  deleteAllGalleryImagesAction,
  updateGallerySortOrderAction,
  getGalleryFaceRecognitionStatsAction,
  getBatchPhotoFacesAction,
  getGalleryImagesAction,
} from "@/app/admin/actions"
import type { GalleryImage } from "@/lib/types"
import { FaceTaggingDialog } from "./face-tagging-dialog"
import { AutoRecognitionDialog } from "./auto-recognition-dialog"
import { UnknownFacesReviewDialog } from "./unknown-faces-review-dialog"

interface GalleryImagesManagerProps {
  galleryId: string
  galleryTitle: string
  initialSortOrder?: string
  isFullyVerified?: boolean
}

function formatFileSize(bytes: number | null): string {
  if (!bytes) return "N/A"
  const kb = bytes / 1024
  return `${kb.toFixed(1)} KB`
}

interface PhotoFace {
  verified: boolean
  confidence: number
  person_id: string | null
}

const PhotoCard = memo(function PhotoCard({
  image,
  photoFacesMap,
  recognitionStats,
  onTag,
  onDelete,
  isSelected,
  onToggleSelect,
}: {
  image: GalleryImage
  photoFacesMap: Record<string, PhotoFace[]>
  recognitionStats: Record<string, { fullyRecognized: boolean }>
  onTag: (id: string, url: string) => void
  onDelete: (id: string) => void
  isSelected: boolean
  onToggleSelect: (id: string) => void
}) {
  const faces = photoFacesMap[image.id]
  const hasDetected = faces && faces.length > 0
  const hasVerified = faces && faces.length > 0 ? faces.every((face) => face.verified === true) : false
  const hasUnknown = faces?.some((face) => face.person_id === null) || false
  const isFullyRecognized = recognitionStats[image.id]?.fullyRecognized || false
  const hasBeenProcessed = image.has_been_processed || false

  const nonVerifiedFaces = faces?.filter((face) => !face.verified && face.person_id !== null) || []
  const confidence =
    nonVerifiedFaces.length > 0
      ? Math.round(
          (nonVerifiedFaces.reduce((sum, face) => sum + (face.confidence || 0), 0) / nonVerifiedFaces.length) * 100,
        )
      : null

  const unknownCount = faces?.filter((f) => f.person_id === null).length || 0
  const recognizedCount = faces?.filter((f) => f.person_id !== null).length || 0
  const totalCount = faces?.length || 0

  let badgeType: "verified" | "recognized" | "partial" | "nfd" | "none" = "none"

  if (isFullyRecognized || hasVerified) {
    badgeType = "verified" // Highest priority: all faces verified
  } else if (hasDetected && hasUnknown && !isFullyRecognized) {
    badgeType = "partial" // Some faces recognized, some unknown
  } else if (confidence !== null && !hasUnknown && !isFullyRecognized) {
    badgeType = "recognized" // All recognized but not verified
  } else if (hasBeenProcessed && !hasDetected) {
    badgeType = "nfd" // Processed but no faces found
  }

  return (
    <div
      className="group relative overflow-hidden rounded-lg border cursor-pointer"
      onClick={() => onTag(image.id, image.image_url)}
    >
      <div className="relative aspect-square">
        <Image
          src={image.image_url || "/placeholder.svg"}
          alt={image.original_filename}
          fill
          className="object-cover"
          sizes="250px"
        />
        <div
          className="absolute left-2 top-2 z-20"
          onClick={(e) => {
            e.stopPropagation()
            onToggleSelect(image.id)
          }}
        >
          <Checkbox
            checked={isSelected}
            className="bg-white border-2 border-gray-400 data-[state=checked]:bg-blue-500 data-[state=checked]:border-blue-500"
          />
        </div>
        <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none">
          <Button
            variant="secondary"
            size="icon"
            className="absolute right-2 bottom-2 pointer-events-auto"
            onClick={(e) => {
              e.stopPropagation()
              onTag(image.id, image.image_url)
            }}
          >
            <UserPlus className="h-4 w-4" />
          </Button>
          <Button
            variant="destructive"
            size="icon"
            className="absolute right-2 top-2 pointer-events-auto"
            onClick={(e) => {
              e.stopPropagation()
              onDelete(image.id)
            }}
          >
            <Trash2 className="h-4 w-4" />
          </Button>
        </div>
        {image.download_count > 0 && (
          <div className="absolute left-2 top-10 bg-black/70 text-white rounded px-2 py-1 text-xs flex items-center gap-1 shadow-lg z-10">
            <Download className="h-3 w-3" />
            {image.download_count}
          </div>
        )}
        {badgeType === "nfd" && (
          <div className="absolute left-2 bottom-2 bg-gray-500 text-white rounded px-2 py-1 text-xs font-semibold shadow-lg z-10">
            NFD
          </div>
        )}
        {badgeType === "partial" && (
          <div className="absolute left-2 bottom-2 bg-orange-500 text-white rounded px-2 py-1 text-xs font-semibold shadow-lg z-10">
            {recognizedCount}/{unknownCount}/{totalCount}
          </div>
        )}
        {badgeType === "recognized" && (
          <div className="absolute left-2 bottom-2 bg-blue-500 text-white rounded px-2 py-1 text-xs font-semibold shadow-lg z-10">
            {confidence}%
          </div>
        )}
        {badgeType === "verified" && (
          <div className="absolute left-2 bottom-2 bg-green-500 text-white rounded-full w-6 h-6 flex items-center justify-center text-sm shadow-lg z-10">
            ✓
          </div>
        )}
      </div>
      <div className="bg-background p-2 border-t">
        <p className="text-xs font-medium truncate" title={image.original_filename}>
          {image.original_filename}
        </p>
        <div className="flex items-center justify-between mt-1">
          <p className="text-xs text-muted-foreground">{new Date(image.created_at).toLocaleDateString("ru-RU")}</p>
          {image.width && image.height && (
            <p className="text-xs text-muted-foreground">
              {image.width}×{image.height}
            </p>
          )}
        </div>
      </div>
    </div>
  )
})

export function GalleryImagesManager({
  galleryId,
  galleryTitle,
  initialSortOrder,
  isFullyVerified,
  ...props
}: GalleryImagesManagerProps & React.ComponentProps<typeof Button>) {
  const [open, setOpen] = useState(false)
  const [images, setImages] = useState<GalleryImage[]>([])
  const [loading, setLoading] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState("")
  const [sortBy, setSortBy] = useState<SortOption>((initialSortOrder as SortOption) || "filename")
  const [taggingImage, setTaggingImage] = useState<{ id: string; url: string } | null>(null)
  const [autoRecognitionMode, setAutoRecognitionMode] = useState<"all" | "remaining" | null>(null)
  const [showUnknownFaces, setShowUnknownFaces] = useState(false)
  const [recognitionStats, setRecognitionStats] = useState<Record<string, { fullyRecognized: boolean }>>({})
  const [photoFacesMap, setPhotoFacesMap] = useState<Record<string, PhotoFace[]>>({})
  const [isDragging, setIsDragging] = useState(false)
  const dragCounter = useRef(0)
  const [selectedPhotos, setSelectedPhotos] = useState<Set<string>>(new Set())
  const [confirmDialog, setConfirmDialog] = useState<{
    open: boolean
    action: "delete" | null
    count: number
  }>({ open: false, action: null, count: 0 })
  const [singleDeleteDialog, setSingleDeleteDialog] = useState<{
    open: boolean
    imageId: string | null
    filename: string | null
  }>({ open: false, imageId: null, filename: null })

  useEffect(() => {
    if (open) {
      loadImages()
      loadRecognitionStats()
    }
  }, [open])

  useEffect(() => {
    if (images.length > 0) {
      loadPhotoFaces()
    }
  }, [images])

  async function loadImages() {
    setLoading(true)
    try {
      console.log("[v0] Loading images for gallery:", galleryId)
      const result = await getGalleryImagesAction(galleryId)
      console.log("[v0] getGalleryImagesAction result:", result)
      console.log("[v0] result.success:", result.success)
      console.log("[v0] result.data:", result.data)
      console.log("[v0] result.data length:", result.data?.length)

      if (result.success && result.data) {
        console.log("[v0] Setting images:", result.data.length, "items")
        setImages(result.data)
      } else {
        console.error("[v0] Error loading images:", result.error)
      }
    } catch (error) {
      console.error("[v0] Error loading images:", error)
    }
    setLoading(false)
  }

  async function loadRecognitionStats() {
    const result = await getGalleryFaceRecognitionStatsAction(galleryId)
    if (result.success && result.data) {
      setRecognitionStats(result.data)
    }
  }

  async function loadPhotoFaces() {
    const photoIds = images.map((img) => img.id)

    console.log("[v4.6] GalleryImagesManager: Calling getBatchPhotoFacesAction with photoIds:", photoIds)

    const result = await getBatchPhotoFacesAction(photoIds)

    console.log("[v4.6] GalleryImagesManager: getBatchPhotoFacesAction result:", {
      success: result.success,
      successType: typeof result.success,
      dataLength: result.data?.length,
      error: result.error,
      fullResult: result,
    })

    if (result.success && result.data) {
      const facesMap: Record<string, PhotoFace[]> = {}

      for (const face of result.data) {
        console.log("[v4.6] GalleryImagesManager: Face from DB:", {
          photo_id: face.photo_id,
          person_id: face.person_id,
          recognition_confidence: face.recognition_confidence,
          verified: face.verified,
        })

        if (!facesMap[face.photo_id]) {
          facesMap[face.photo_id] = []
        }
        facesMap[face.photo_id].push({
          verified: face.verified,
          confidence: face.recognition_confidence || 0,
          person_id: face.person_id || null,
        })
      }

      console.log("[v4.6] GalleryImagesManager: Loaded photo faces map:", {
        photosCount: Object.keys(facesMap).length,
        photoIds: Object.keys(facesMap),
      })

      setPhotoFacesMap(facesMap)
    } else {
      console.log("[v4.6] GalleryImagesManager: No faces data or error:", {
        success: result.success,
        error: result.error,
        hasData: !!result.data,
      })
    }
  }

  function getPhotoConfidence(imageId: string): number | null {
    const faces = photoFacesMap[imageId]

    if (!faces || faces.length === 0) return null

    const nonVerifiedFaces = faces.filter((face) => !face.verified && face.person_id !== null)

    if (nonVerifiedFaces.length === 0) return null

    const avgConfidence =
      nonVerifiedFaces.reduce((sum, face) => sum + (face.confidence || 0), 0) / nonVerifiedFaces.length
    return Math.round(avgConfidence * 100)
  }

  function hasVerifiedFaces(imageId: string): boolean {
    const faces = photoFacesMap[imageId]
    if (!faces || faces.length === 0) return false

    const hasVerified = faces.every((face) => face.verified === true)
    return hasVerified
  }

  function hasDetectedFaces(imageId: string): boolean {
    const faces = photoFacesMap[imageId]
    return faces && faces.length > 0
  }

  function hasUnknownFaces(imageId: string): boolean {
    const faces = photoFacesMap[imageId]
    if (!faces || faces.length === 0) return false

    return faces.some((face) => face.person_id === null)
  }

  const sortedImages = useMemo(() => {
    const imagesCopy = [...images]

    switch (sortBy) {
      case "filename":
        return imagesCopy.sort((a, b) => (a.original_filename || "").localeCompare(b.original_filename || ""))
      case "created":
      case "added":
        return imagesCopy.sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime())
      default:
        return imagesCopy
    }
  }, [images, sortBy])

  function handleDragEnter(e: React.DragEvent) {
    e.preventDefault()
    e.stopPropagation()
    dragCounter.current++
    if (dragCounter.current === 1) {
      setIsDragging(true)
    }
  }

  function handleDragLeave(e: React.DragEvent) {
    e.preventDefault()
    e.stopPropagation()
    dragCounter.current--
    if (dragCounter.current === 0) {
      setIsDragging(false)
    }
  }

  function handleDragOver(e: React.DragEvent) {
    e.preventDefault()
    e.stopPropagation()
  }

  async function handleDrop(e: React.DragEvent) {
    e.preventDefault()
    e.stopPropagation()
    dragCounter.current = 0
    setIsDragging(false)

    const files = Array.from(e.dataTransfer.files).filter((file) => file.type.startsWith("image/"))

    if (files.length === 0) return

    await processFiles(files)
  }

  async function processFiles(files: File[]) {
    setUploading(true)
    setUploadProgress(`Загрузка 0 из ${files.length}...`)

    try {
      const uploadedImages: {
        imageUrl: string
        originalUrl: string
        originalFilename: string
        fileSize: number
        width: number
        height: number
      }[] = []

      for (let i = 0; i < files.length; i++) {
        const file = files[i]
        setUploadProgress(`Загрузка ${i + 1} из ${files.length}...`)

        const dimensions = await getImageDimensions(file)

        const formData = new FormData()
        formData.append("file", file)

        const response = await fetch("/api/upload", {
          method: "POST",
          body: formData,
        })

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}))

          if (errorData.code === "QUOTA_EXCEEDED") {
            throw new Error(errorData.error || "Квота хранилища исчерпана")
          }

          throw new Error(errorData.error || `Ошибка загрузки ${file.name}`)
        }

        const data = await response.json()
        uploadedImages.push({
          imageUrl: data.url,
          originalUrl: data.url,
          originalFilename: file.name,
          fileSize: file.size,
          width: dimensions.width,
          height: dimensions.height,
        })
      }

      const result = await addGalleryImagesAction(galleryId, uploadedImages)

      if (result.success) {
        await loadImages()
        await loadRecognitionStats()
        await loadPhotoFaces()
        setUploadProgress("")
      } else {
        alert("Ошибка при сохранении изображений")
      }
    } catch (error) {
      console.error("Error uploading images:", error)
      const errorMessage = error instanceof Error ? error.message : "Ошибка загрузки изображений"
      alert(errorMessage)
    } finally {
      setUploading(false)
      setUploadProgress("")
    }
  }

  async function handleImageUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const files = e.target.files
    if (!files || files.length === 0) return

    await processFiles(Array.from(files))
    e.target.value = ""
  }

  async function handleDelete(imageId: string) {
    const image = images.find((img) => img.id === imageId)
    if (!image) return

    setSingleDeleteDialog({
      open: true,
      imageId,
      filename: image.original_filename,
    })
  }

  async function handleDeleteAll() {
    setConfirmDialog({
      open: true,
      action: "delete",
      count: images.length,
    })
  }

  async function handleDeleteSelected() {
    if (selectedPhotos.size > 0) {
      for (const photoId of selectedPhotos) {
        await deleteGalleryImageAction(photoId, galleryId)
      }
      setSelectedPhotos(new Set())
    }
    await loadImages()
    await loadPhotoFaces()
  }

  async function confirmBatchDelete() {
    if (selectedPhotos.size > 0) {
      for (const photoId of selectedPhotos) {
        await deleteGalleryImageAction(photoId, galleryId)
      }
      setSelectedPhotos(new Set())
    } else {
      await deleteAllGalleryImagesAction(galleryId)
    }
    setConfirmDialog({ open: false, action: null, count: 0 })
    await loadImages()
    await loadPhotoFaces()
  }

  function handleClearSelection() {
    setSelectedPhotos(new Set())
  }

  function handleCancelConfirmation(e: React.MouseEvent) {
    e.stopPropagation()
    setConfirmDialog({ open: false, action: null, count: 0 })
  }

  async function handleConfirmAction(e: React.MouseEvent) {
    e.stopPropagation()
    await confirmBatchDelete()
  }

  async function handleSortChange(value: string) {
    setSortBy(value as SortOption)
    await updateGallerySortOrderAction(galleryId, value)
  }

  function getImageDimensions(file: File): Promise<{ width: number; height: number }> {
    return new Promise((resolve, reject) => {
      const img = new window.Image()
      img.onload = () => {
        resolve({ width: img.width, height: img.height })
      }
      img.onerror = reject
      img.src = URL.createObjectURL(file)
    })
  }

  function togglePhotoSelection(photoId: string) {
    setSelectedPhotos((prev) => {
      const newSet = new Set(prev)
      if (newSet.has(photoId)) {
        newSet.delete(photoId)
      } else {
        newSet.add(photoId)
      }
      return newSet
    })
  }

  const allPhotosVerified = useMemo(() => {
    if (images.length === 0) return false

    return images.every((image) => {
      const stats = recognitionStats[image.id]
      const hasVerified = hasVerifiedFaces(image.id)
      return stats?.fullyRecognized || hasVerified
    })
  }, [images, recognitionStats, photoFacesMap])

  async function confirmSingleDelete() {
    if (!singleDeleteDialog.imageId) return

    const result = await deleteGalleryImageAction(singleDeleteDialog.imageId, galleryId)
    if (result.success) {
      await loadImages()
      await loadPhotoFaces()
    }

    setSingleDeleteDialog({ open: false, imageId: null, filename: null })
  }

  return (
    <>
      <Dialog
        open={open}
        onOpenChange={(next) => {
          if ((confirmDialog.open || singleDeleteDialog.open) && !next) return
          setOpen(next)
        }}
      >
        <DialogTrigger asChild>
          <Button
            variant={isFullyVerified ? "default" : "outline"}
            size="sm"
            className={isFullyVerified ? "bg-green-600 hover:bg-green-700 text-white" : ""}
            {...props}
          >
            <Images className="h-4 w-4" />
          </Button>
        </DialogTrigger>
        <DialogContent
          className="sm:max-w-[1400px] max-h-[90vh] flex flex-col"
          onDragEnter={handleDragEnter}
          onDragLeave={handleDragLeave}
          onDragOver={handleDragOver}
          onDrop={handleDrop}
          onInteractOutside={(e) => {
            if (confirmDialog.open || singleDeleteDialog.open) e.preventDefault()
          }}
          onEscapeKeyDown={(e) => {
            if (confirmDialog.open || singleDeleteDialog.open) e.preventDefault()
          }}
        >
          <div className="sticky top-0 z-10 pb-4 border-b">
            <DialogHeader>
              <DialogTitle>Управление фотографиями</DialogTitle>
              <DialogDescription>{galleryTitle}</DialogDescription>
            </DialogHeader>

            {isDragging && (
              <div className="absolute inset-0 z-50 flex items-center justify-center backdrop-blur-sm border-4 border-dashed border-primary rounded-lg bg-white/80 dark:bg-black/80">
                <div className="text-center">
                  <Upload className="h-16 w-16 mx-auto mb-4 text-primary" />
                  <p className="text-xl font-semibold">Перетащите фото сюда</p>
                  <p className="text-sm text-muted-foreground mt-2">Отпустите, чтобы загрузить</p>
                </div>
              </div>
            )}

            <div className="flex items-center gap-4 flex-wrap mt-4">
              <label htmlFor="image-upload" className="cursor-pointer">
                <div className="flex items-center gap-2 rounded-md border border-input bg-background px-4 py-2 hover:bg-accent hover:text-accent-foreground">
                  <Upload className="h-4 w-4" />
                  <span>Загрузить фото</span>
                </div>
                <input
                  id="image-upload"
                  type="file"
                  accept="image/*"
                  multiple
                  onChange={handleImageUpload}
                  disabled={uploading}
                  className="hidden"
                />
              </label>
              {images.length > 0 && (
                <Button
                  variant="secondary"
                  onClick={() => setAutoRecognitionMode("remaining")}
                  disabled={uploading || allPhotosVerified}
                >
                  <Scan className="h-4 w-4 mr-2" />
                  Распознать фото
                </Button>
              )}
              {images.length > 0 && (
                <Button variant="secondary" onClick={() => setShowUnknownFaces(true)} disabled={uploading}>
                  <UserPlus className="h-4 w-4 mr-2" />
                  Неизвестные лица
                </Button>
              )}
              {images.length > 0 && (
                <Button
                  variant="destructive"
                  onClick={handleDeleteAll}
                  disabled={uploading}
                  className="min-w-[180px] justify-start"
                >
                  <Trash2 className="h-4 w-4 mr-2" />
                  {selectedPhotos.size > 0 ? `Удалить ${selectedPhotos.size} фото` : "Удалить все фото"}
                </Button>
              )}
              {images.length > 0 && (
                <Select value={sortBy} onValueChange={handleSortChange}>
                  <SelectTrigger className="w-[220px]">
                    <SelectValue placeholder="Сортировка" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="filename">По названию файла</SelectItem>
                    <SelectItem value="created">По времени создания</SelectItem>
                    <SelectItem value="added">По времени добавления</SelectItem>
                  </SelectContent>
                </Select>
              )}
              {uploading && <span className="text-sm text-muted-foreground">{uploadProgress}</span>}
            </div>
          </div>

          <div className="flex-1 overflow-y-auto space-y-4">
            {loading ? (
              <div className="flex items-center justify-center py-8">
                <p className="text-muted-foreground">Загрузка...</p>
              </div>
            ) : images.length === 0 ? (
              <Card>
                <CardContent className="flex min-h-[200px] items-center justify-center">
                  <p className="text-muted-foreground">Нет фотографий. Загрузите первые фото!</p>
                </CardContent>
              </Card>
            ) : (
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
                {sortedImages.map((image) => (
                  <PhotoCard
                    key={image.id}
                    image={image}
                    photoFacesMap={photoFacesMap}
                    recognitionStats={recognitionStats}
                    onTag={(id, url) => setTaggingImage({ id, url })}
                    onDelete={handleDelete}
                    isSelected={selectedPhotos.has(image.id)}
                    onToggleSelect={togglePhotoSelection}
                  />
                ))}
              </div>
            )}

            <div className="flex items-center justify-between">
              {selectedPhotos.size > 0 && (
                <div className="flex items-center gap-2">
                  <Button variant="outline" size="sm" onClick={handleClearSelection}>
                    Снять выделение
                  </Button>
                  <Button variant="destructive" size="sm" onClick={handleDeleteSelected}>
                    Удалить выбранные ({selectedPhotos.size})
                  </Button>
                </div>
              )}

              <p className="text-sm text-muted-foreground">Всего фотографий: {images.length}</p>
            </div>
          </div>
        </DialogContent>
      </Dialog>

      <AlertDialog
        open={singleDeleteDialog.open}
        onOpenChange={(next) => setSingleDeleteDialog((s) => ({ ...s, open: next }))}
      >
        <AlertDialogContent
          onPointerDownOutside={(e) => e.preventDefault()}
          onEscapeKeyDown={(e) => e.preventDefault()}
        >
          <AlertDialogHeader>
            <AlertDialogTitle>Подтверждение удаления</AlertDialogTitle>
            <AlertDialogDescription>
              Вы действительно хотите удалить <span className="font-semibold">{singleDeleteDialog.filename}</span> из
              галереи <span className="font-semibold">{galleryTitle}</span>? Это действие невозможно отменить!
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Отмена</AlertDialogCancel>
            <AlertDialogAction onClick={confirmSingleDelete} className="bg-destructive hover:bg-destructive/90">
              Удалить
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      <AlertDialog open={confirmDialog.open} onOpenChange={(next) => setConfirmDialog((s) => ({ ...s, open: next }))}>
        <AlertDialogContent
          onPointerDownOutside={(e) => e.preventDefault()}
          onEscapeKeyDown={(e) => e.preventDefault()}
        >
          <AlertDialogHeader>
            <AlertDialogTitle>Подтвердите удаление</AlertDialogTitle>
            <AlertDialogDescription>
              {selectedPhotos.size > 0 ? (
                <>
                  Вы действительно хотите удалить выбранные фотографии из галереи{" "}
                  <span className="font-semibold">{galleryTitle}</span>? Это действие невозможно отменить!
                </>
              ) : (
                <>
                  Вы действительно хотите удалить все фотографии (
                  <span className="font-semibold">{confirmDialog.count} шт.</span>) из галереи{" "}
                  <span className="font-semibold">{galleryTitle}</span>? Это действие невозможно отменить!
                </>
              )}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel onClick={handleCancelConfirmation}>Отмена</AlertDialogCancel>
            <AlertDialogAction onClick={handleConfirmAction}>Подтвердить</AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      {taggingImage && (
        <FaceTaggingDialog
          imageId={taggingImage.id}
          imageUrl={taggingImage.url}
          hasBeenProcessed={images.find((img) => img.id === taggingImage.id)?.has_been_processed || false}
          open={!!taggingImage}
          onOpenChange={async (open) => {
            if (!open) {
              console.log("[v0] GalleryImagesManager: FaceTaggingDialog closed, keeping gallery open without reload")
              setTaggingImage(null)
            }
          }}
          onSave={async () => {
            console.log("[v0] GalleryImagesManager: FaceTaggingDialog onSave called")
            await loadRecognitionStats()
            await loadPhotoFaces()
            console.log("[v0] GalleryImagesManager: Data reloaded after face save")
          }}
        />
      )}

      {autoRecognitionMode && (
        <AutoRecognitionDialog
          images={sortedImages}
          open={!!autoRecognitionMode}
          mode={autoRecognitionMode}
          onOpenChange={(open) => {
            if (!open) {
              setAutoRecognitionMode(null)
              loadRecognitionStats()
              loadPhotoFaces()
            }
          }}
        />
      )}

      {showUnknownFaces && (
        <UnknownFacesReviewDialog
          galleryId={galleryId}
          open={showUnknownFaces}
          onOpenChange={setShowUnknownFaces}
          onComplete={() => {
            loadImages()
            loadRecognitionStats()
            loadPhotoFaces()
          }}
        />
      )}
    </>
  )
}

type SortOption = "filename" | "created" | "added"
