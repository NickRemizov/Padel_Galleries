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
import { Pencil, X, Camera } from "lucide-react"
import { updateGalleryAction } from "@/app/admin/actions"
import type { Gallery, Photographer, Location, Organizer } from "@/lib/types"
import Image from "next/image"
import { ImageCropper } from "./image-cropper"

interface EditGalleryDialogProps {
  gallery: Gallery
  photographers: Photographer[]
  locations: Location[]
  organizers: Organizer[]
}

export function EditGalleryDialog({ gallery, photographers, locations, organizers }: EditGalleryDialogProps) {
  const [open, setOpen] = useState(false)
  const [loading, setLoading] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [coverImageUrl, setCoverImageUrl] = useState(gallery.cover_image_url || "")
  const [coverImageSquareUrl, setCoverImageSquareUrl] = useState(gallery.cover_image_square_url || "")
  const [previewUrl, setPreviewUrl] = useState(gallery.cover_image_url || "")
  const [previewSquareUrl, setPreviewSquareUrl] = useState(gallery.cover_image_square_url || "")
  const [title, setTitle] = useState(gallery.title)
  const [shootDate, setShootDate] = useState(new Date(gallery.shoot_date).toISOString().split("T")[0])
  const [photographerId, setPhotographerId] = useState(gallery.photographer_id || "")
  const [cropperOpen, setCropperOpen] = useState(false)
  const [cropperSquareOpen, setCropperSquareOpen] = useState(false)
  const [originalImage, setOriginalImage] = useState("")
  const [externalGalleryUrl, setExternalGalleryUrl] = useState(gallery.external_gallery_url || "")

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
    const result = await updateGalleryAction(gallery.id, formData)
    setLoading(false)

    if (result.success) {
      setOpen(false)
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
          <Button variant="outline" size="sm">
            <Pencil className="h-4 w-4" />
          </Button>
        </DialogTrigger>
        <DialogContent className="sm:max-w-[525px] max-h-[90vh] overflow-y-auto">
          <form action={handleSubmit}>
            <DialogHeader>
              <DialogTitle>Редактировать галерею</DialogTitle>
              <DialogDescription>Измените информацию о галерее</DialogDescription>
            </DialogHeader>

            <div className="grid gap-4 py-4">
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
                <Select name="organizer_id" defaultValue={gallery.organizer_id || "none"}>
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
                <Select name="location_id" defaultValue={gallery.location_id || "none"}>
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
              <Button type="submit" disabled={loading || uploading || !title || !shootDate}>
                {loading ? "Сохранение..." : "Сохранить"}
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
