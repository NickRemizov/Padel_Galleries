"use client"

import { useState, useEffect, useCallback } from "react"
import { useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from "@/components/ui/dialog"
import { Slider } from "@/components/ui/slider"
import { Loader2, ImageIcon, ZoomIn } from "lucide-react"
import Cropper from "react-easy-crop"
import type { Area } from "react-easy-crop"
import { getPersonPhotosAction, updatePersonAvatarAction } from "@/app/admin/actions" // Import actions here

interface AvatarSelectorProps {
  personId: string
  personName?: string
  currentAvatar?: string | null
  onAvatarUpdated?: () => void
  open?: boolean
  onOpenChange?: (open: boolean) => void
  onAvatarSelected?: () => void
  preselectedPhotoId?: string | null
}

export function AvatarSelector({
  personId,
  personName,
  currentAvatar,
  onAvatarUpdated,
  open: controlledOpen,
  onOpenChange: controlledOnOpenChange,
  onAvatarSelected,
  preselectedPhotoId,
}: AvatarSelectorProps) {
  const router = useRouter()

  const [internalOpen, setInternalOpen] = useState(false)
  const open = controlledOpen !== undefined ? controlledOpen : internalOpen
  const setOpen = controlledOnOpenChange || setInternalOpen

  const [photos, setPhotos] = useState<any[]>([])
  const [loading, setLoading] = useState(false)
  const [selectedPhoto, setSelectedPhoto] = useState<any>(null)
  const [cropping, setCropping] = useState(false)

  const [crop, setCrop] = useState({ x: 0, y: 0 })
  const [zoom, setZoom] = useState(1)
  const [croppedAreaPixels, setCroppedAreaPixels] = useState<Area | null>(null)

  useEffect(() => {
    if (open) {
      loadPhotos()
    }
  }, [open])

  useEffect(() => {
    if (preselectedPhotoId && photos.length > 0) {
      const photo = photos.find((p) => p.id === preselectedPhotoId)
      if (photo) {
        handlePhotoSelect(photo)
      }
    }
  }, [preselectedPhotoId, photos])

  async function loadPhotos() {
    setLoading(true)
    const result = await getPersonPhotosAction(personId)
    setLoading(false)

    if (result.success && result.data) {
      setPhotos(result.data)
    }
  }

  function handlePhotoSelect(photo: any) {
    setSelectedPhoto(photo)
    setCropping(true)
    setCrop({ x: 0, y: 0 })
    setZoom(1)
    setCroppedAreaPixels(null)
  }

  const onCropComplete = useCallback((croppedArea: Area, croppedAreaPixels: Area) => {
    setCroppedAreaPixels(croppedAreaPixels)
  }, [])

  async function getCroppedImg(imageSrc: string, pixelCrop: Area): Promise<Blob | null> {
    const image = new Image()
    image.crossOrigin = "anonymous"

    return new Promise((resolve, reject) => {
      image.onload = () => {
        const canvas = document.createElement("canvas")
        const ctx = canvas.getContext("2d")

        if (!ctx) {
          reject(new Error("Could not get canvas context"))
          return
        }

        canvas.width = 420
        canvas.height = 560

        // Draw cropped and resized image
        ctx.drawImage(image, pixelCrop.x, pixelCrop.y, pixelCrop.width, pixelCrop.height, 0, 0, 420, 560)

        canvas.toBlob(
          (blob) => {
            if (blob) {
              resolve(blob)
            } else {
              reject(new Error("Failed to create blob"))
            }
          },
          "image/jpeg",
          0.9,
        )
      }

      image.onerror = () => {
        reject(new Error("Failed to load image"))
      }

      image.src = imageSrc
    })
  }

  async function handleCropComplete() {
    if (!selectedPhoto || !croppedAreaPixels) {
      console.error("[v0] Missing selected photo or crop data")
      return
    }

    try {
      setLoading(true)
      console.log("[v0] Creating cropped image...")

      const blob = await getCroppedImg(selectedPhoto.image_url, croppedAreaPixels)

      if (!blob) {
        throw new Error("Failed to create cropped image")
      }

      console.log("[v0] Blob created, uploading...")

      const formData = new FormData()
      const filename = `avatar-${personId}-${Date.now()}.jpg`
      formData.append("file", blob, filename)

      const response = await fetch("/api/upload", {
        method: "POST",
        body: formData,
      })

      if (!response.ok) {
        const error = await response.json()
        throw new Error(error.error || "Upload failed")
      }

      const result = await response.json()
      console.log("[v0] Upload successful:", result.url)

      // Update person avatar
      await updatePersonAvatarAction(personId, result.url)

      setCropping(false)
      setSelectedPhoto(null)
      setCrop({ x: 0, y: 0 })
      setZoom(1)
      setCroppedAreaPixels(null)

      router.refresh()

      onAvatarUpdated?.()
      onAvatarSelected?.()
      setOpen(false)
    } catch (error) {
      console.error("[v0] Error uploading avatar:", error)
      alert("Ошибка при загрузке аватара. Попробуйте еще раз.")
    } finally {
      setLoading(false)
    }
  }

  const triggerButton =
    controlledOpen === undefined ? (
      <Button type="button" variant="outline" size="sm" onClick={() => setOpen(true)}>
        <ImageIcon className="mr-2 h-4 w-4" />
        Выбрать из галереи
      </Button>
    ) : null

  return (
    <>
      {triggerButton}

      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent className="max-h-[90vh] overflow-y-auto sm:max-w-[900px]">
          <DialogHeader>
            <DialogTitle>Выбрать аватар{personName ? ` для ${personName}` : ""}</DialogTitle>
          </DialogHeader>

          {!cropping ? (
            <div className="py-4">
              {loading ? (
                <div className="flex items-center justify-center py-12">
                  <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
                </div>
              ) : photos.length === 0 ? (
                <div className="py-12 text-center text-muted-foreground">
                  <p>Нет фотографий, где человек распознан</p>
                  <p className="mt-2 text-sm">Сначала отметьте человека на фотографиях в галереях</p>
                </div>
              ) : (
                <div className="grid grid-cols-2 gap-4 md:grid-cols-3 lg:grid-cols-4">
                  {photos.map((photo) => (
                    <button
                      key={photo.id}
                      onClick={() => handlePhotoSelect(photo)}
                      className="group relative aspect-square overflow-hidden rounded-lg border-2 border-transparent transition-all hover:border-primary"
                    >
                      <img
                        src={photo.image_url || "/placeholder.svg"}
                        alt=""
                        className="h-full w-full object-cover transition-transform group-hover:scale-105"
                      />
                    </button>
                  ))}
                </div>
              )}
            </div>
          ) : (
            <div className="py-4">
              <div className="mb-4">
                <p className="text-sm text-muted-foreground">
                  Выберите область для аватара (формат 3:4, 420×560 пикселей). Перетащите и масштабируйте изображение.
                </p>
              </div>

              <div className="relative mx-auto h-[500px] w-full max-w-2xl bg-muted">
                <Cropper
                  image={selectedPhoto.image_url || "/placeholder.svg"}
                  crop={crop}
                  zoom={zoom}
                  aspect={3 / 4}
                  onCropChange={setCrop}
                  onZoomChange={setZoom}
                  onCropComplete={onCropComplete}
                  objectFit="contain"
                />
              </div>

              <div className="mx-auto mt-6 flex max-w-md items-center gap-4">
                <ZoomIn className="h-5 w-5 text-muted-foreground" />
                <Slider
                  value={[zoom]}
                  onValueChange={(value) => setZoom(value[0])}
                  min={1}
                  max={3}
                  step={0.1}
                  className="flex-1"
                />
                <span className="text-sm text-muted-foreground">{Math.round(zoom * 100)}%</span>
              </div>

              <DialogFooter className="mt-6">
                <Button
                  variant="outline"
                  onClick={() => {
                    setCropping(false)
                    setSelectedPhoto(null)
                    setCrop({ x: 0, y: 0 })
                    setZoom(1)
                    setCroppedAreaPixels(null)
                  }}
                  disabled={loading}
                >
                  Назад
                </Button>
                <Button onClick={handleCropComplete} disabled={loading || !croppedAreaPixels}>
                  {loading ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Сохранение...
                    </>
                  ) : (
                    "Сохранить аватар"
                  )}
                </Button>
              </DialogFooter>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </>
  )
}
