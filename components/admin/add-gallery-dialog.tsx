"use client"

import type React from "react"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Plus, X, Camera } from "lucide-react"
import { addGalleryAction } from "@/app/admin/actions"
import type { Photographer, Location, Organizer } from "@/lib/types"
import Image from "next/image"
import { ImageCropper } from "./image-cropper"

interface AddGalleryDialogProps {
  photographers: Photographer[]
  locations: Location[]
  organizers: Organizer[]
  onSuccess?: () => void
}

export function AddGalleryDialog({ photographers, locations, organizers, onSuccess }: AddGalleryDialogProps) {
  const [open, setOpen] = useState(false)
  const [loading, setLoading] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [coverImageUrl, setCoverImageUrl] = useState("")
  const [coverImageSquareUrl, setCoverImageSquareUrl] = useState("")
  const [previewUrl, setPreviewUrl] = useState("")
  const [previewSquareUrl, setPreviewSquareUrl] = useState("")
  const [title, setTitle] = useState("")
  const [shootDate, setShootDate] = useState("")
  const [photographerId, setPhotographerId] = useState("")
  const [cropperOpen, setCropperOpen] = useState(false)
  const [cropperSquareOpen, setCropperSquareOpen] = useState(false)
  const [originalImage, setOriginalImage] = useState("")
  const [externalGalleryUrl, setExternalGalleryUrl] = useState("")
  const [error, setError] = useState<string | null>(null)

  async function handleImageSelect(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (!file) return

    const imageUrl = URL.createObjectURL(file)
    setOriginalImage(imageUrl)
    setCropperOpen(true)
  }

  async function handleSquareImageSelect(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (!file) return

    const imageUrl = URL.createObjectURL(file)
    setOriginalImage(imageUrl)
    setCropperSquareOpen(true)
  }

  async function handleCropComplete(croppedBlob: Blob) {
    setUploading(true)
    setCropperOpen(false)

    try {
      const formData = new FormData()
      formData.append("file", croppedBlob, "cropped-image.jpg")

      const response = await fetch("/api/upload", {
        method: "POST",
        body: formData,
      })

      if (!response.ok) throw new Error("Upload failed")

      const data = await response.json()
      setCoverImageUrl(data.url)
      setPreviewUrl(URL.createObjectURL(croppedBlob))
    } catch (error) {
      console.error("Error uploading image:", error)
      alert("Ошибка загрузки изображения")
    } finally {
      setUploading(false)
    }
  }

  async function handleSquareCropComplete(croppedBlob: Blob) {
    setUploading(true)
    setCropperSquareOpen(false)

    try {
      const formData = new FormData()
      formData.append("file", croppedBlob, "cropped-square-image.jpg")

      const response = await fetch("/api/upload", {
        method: "POST",
        body: formData,
      })

      if (!response.ok) throw new Error("Upload failed")

      const data = await response.json()
      setCoverImageSquareUrl(data.url)
      setPreviewSquareUrl(URL.createObjectURL(croppedBlob))
    } catch (error) {
      console.error("Error uploading image:", error)
      alert("Ошибка загрузки изображения")
    } finally {
      setUploading(false)
    }
  }

  function handleCropCancel() {
    setCropperOpen(false)
    setOriginalImage("")
  }

  function handleSquareCropCancel() {
    setCropperSquareOpen(false)
    setOriginalImage("")
  }

  function clearImage() {
    setCoverImageUrl("")
    setPreviewUrl("")
  }

  function clearSquareImage() {
    setCoverImageSquareUrl("")
    setPreviewSquareUrl("")
  }

  async function handleSubmit(formData: FormData) {
    console.log("[v0] AddGalleryDialog: handleSubmit called")

    if (coverImageUrl) {
      formData.set("cover_image_url", coverImageUrl)
    }
    if (coverImageSquareUrl) {
      formData.set("cover_image_square_url", coverImageSquareUrl)
    }
    if (externalGalleryUrl) {
      formData.set("external_gallery_url", externalGalleryUrl)
    }

    setLoading(true)
    setError(null)

    try {
      const result = await addGalleryAction(formData)
      console.log("[v0] AddGalleryDialog: result =", result)

      if (result.success) {
        setOpen(false)
        setCoverImageUrl("")
        setCoverImageSquareUrl("")
        setPreviewUrl("")
        setPreviewSquareUrl("")
        setTitle("")
        setShootDate("")
        setPhotographerId("")
        setExternalGalleryUrl("")
        onSuccess?.()
      } else if (result.error) {
        console.error("[v0] AddGalleryDialog: error =", result.error)
        setError(result.error)
      }
    } catch (err: any) {
      console.error("[v0] AddGalleryDialog: exception =", err)
      setError(err.message || "Неизвестная ошибка")
    } finally {
      setLoading(false)
    }
  }

  const selectedPhotographer = photographers.find((p) => p.id === photographerId)
  const shortDate = shootDate
    ? new Date(shootDate).toLocaleDateString("ru-RU", {
        day: "2-digit",
        month: "2-digit",
      })
    : ""

  return (
    <>
      <Dialog open={open} onOpenChange={setOpen}>
        <DialogTrigger asChild>
          <Button>
            <Plus className="mr-2 h-4 w-4" />
            Добавить галерею
          </Button>
        </DialogTrigger>
        <DialogContent className="sm:max-w-[525px] max-h-[90vh] overflow-y-auto">
          <form action={handleSubmit}>
            <DialogHeader>
              <DialogTitle>Добавить новую галерею</DialogTitle>
              <DialogDescription>Заполните информацию о галерее</DialogDescription>
            </DialogHeader>

            <div className="grid gap-4 py-4">
              {error && (
                <div className="text-sm text-red-500 bg-red-50 p-3 rounded border border-red-200">Ошибка: {error}</div>
              )}

              <div className="grid gap-2">
                <Label htmlFor="title">Название съемки</Label>
                <Input id="title" name="title" value={title} onChange={(e) => setTitle(e.target.value)} required />
              </div>

              <div className="grid gap-2">
                <Label htmlFor="shoot_date">Дата съемки</Label>
                <Input
                  id="shoot_date"
                  name="shoot_date"
                  type="date"
                  value={shootDate}
                  onChange={(e) => setShootDate(e.target.value)}
                  required
                />
              </div>

              <div className="grid gap-2">
                <Label htmlFor="cover_image">Заголовочное фото (3:4)</Label>
                {!previewUrl ? (
                  <div className="flex items-center gap-2">
                    <Input
                      id="cover_image"
                      type="file"
                      accept="image/*"
                      onChange={handleImageSelect}
                      disabled={uploading}
                      required
                    />
                    {uploading && <span className="text-sm text-muted-foreground">Загрузка...</span>}
                  </div>
                ) : (
                  <div className="relative">
                    <div className="relative aspect-[3/4] w-full overflow-hidden rounded-lg border">
                      <Image src={previewUrl || "/placeholder.svg"} alt="Preview" fill className="object-cover" />

                      {/* Title and date overlay in bottom left */}
                      {title && (
                        <div className="absolute bottom-3 left-3 right-3 bg-black/40 text-white px-4 py-3 rounded-md backdrop-blur-sm">
                          <h2 className="font-semibold text-balance leading-tight text-5xl">
                            {title} {shortDate}
                          </h2>
                        </div>
                      )}

                      {/* Photographer overlay in top right */}
                      {selectedPhotographer && (
                        <div className="absolute top-3 right-3 flex items-center gap-2 bg-black/40 text-white px-3 py-1.5 rounded-md backdrop-blur-sm">
                          <Camera className="h-4 w-4" />
                          <span className="text-sm font-medium">{selectedPhotographer.name}</span>
                        </div>
                      )}
                    </div>
                    <Button
                      type="button"
                      variant="destructive"
                      size="icon"
                      className="absolute -right-2 -top-2"
                      onClick={clearImage}
                    >
                      <X className="h-4 w-4" />
                    </Button>
                  </div>
                )}
              </div>

              <div className="grid gap-2">
                <Label htmlFor="cover_image_square">Заголовочное фото (квадрат)</Label>
                <p className="text-xs text-muted-foreground">Используется в карточках на мобильных устройствах</p>
                {!previewSquareUrl ? (
                  <div className="flex items-center gap-2">
                    <Input
                      id="cover_image_square"
                      type="file"
                      accept="image/*"
                      onChange={handleSquareImageSelect}
                      disabled={uploading}
                      required
                    />
                    {uploading && <span className="text-sm text-muted-foreground">Загрузка...</span>}
                  </div>
                ) : (
                  <div className="relative">
                    <div className="relative aspect-square w-full overflow-hidden rounded-lg border">
                      <Image
                        src={previewSquareUrl || "/placeholder.svg"}
                        alt="Square Preview"
                        fill
                        className="object-cover object-top"
                      />
                    </div>
                    <Button
                      type="button"
                      variant="destructive"
                      size="icon"
                      className="absolute -right-2 -top-2"
                      onClick={clearSquareImage}
                    >
                      <X className="h-4 w-4" />
                    </Button>
                  </div>
                )}
              </div>

              <div className="grid gap-2">
                <Label htmlFor="organizer_id">Организатор</Label>
                <Select name="organizer_id">
                  <SelectTrigger>
                    <SelectValue placeholder="Выберите организатора" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="none">Не указан</SelectItem>
                    {organizers.map((organizer) => (
                      <SelectItem key={organizer.id} value={organizer.id}>
                        {organizer.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="grid gap-2">
                <Label htmlFor="location_id">Место съемки</Label>
                <Select name="location_id">
                  <SelectTrigger>
                    <SelectValue placeholder="Выберите место" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="none">Не указано</SelectItem>
                    {locations.map((location) => (
                      <SelectItem key={location.id} value={location.id}>
                        {location.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="grid gap-2">
                <Label htmlFor="photographer_id">Фотограф</Label>
                <Select name="photographer_id" value={photographerId} onValueChange={setPhotographerId}>
                  <SelectTrigger>
                    <SelectValue placeholder="Выберите фотографа" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="none">Не указан</SelectItem>
                    {photographers.map((photographer) => (
                      <SelectItem key={photographer.id} value={photographer.id}>
                        {photographer.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="grid gap-2">
                <Label htmlFor="external_gallery_url">Ссылка на галерею (необязательно)</Label>
                <Input
                  id="external_gallery_url"
                  name="external_gallery_url"
                  type="url"
                  placeholder="https://example.com/gallery"
                  value={externalGalleryUrl}
                  onChange={(e) => setExternalGalleryUrl(e.target.value)}
                />
                <p className="text-xs text-muted-foreground">Используется, если нет загруженных фотографий</p>
              </div>
            </div>

            <DialogFooter>
              <Button
                type="submit"
                disabled={loading || uploading || (!coverImageUrl && !externalGalleryUrl) || !coverImageSquareUrl}
              >
                {loading ? "Добавление..." : "Добавить"}
              </Button>
            </DialogFooter>
          </form>
        </DialogContent>
      </Dialog>

      <ImageCropper
        image={originalImage}
        open={cropperOpen}
        onCropComplete={handleCropComplete}
        onCancel={handleCropCancel}
        aspectRatio={3 / 4}
        title="Кадрирование обложки (3:4)"
      />

      <ImageCropper
        image={originalImage}
        open={cropperSquareOpen}
        onCropComplete={handleSquareCropComplete}
        onCancel={handleSquareCropCancel}
        aspectRatio={1}
        title="Кадрирование обложки (квадрат)"
      />
    </>
  )
}
