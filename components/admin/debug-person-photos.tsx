"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { debugPersonPhotosAction, debugPhotoFacesAction } from "@/app/admin/actions"
import { Bug } from "lucide-react"

export function DebugPersonPhotos() {
  const [personName, setPersonName] = useState("Дарья Михеева")
  const [photoFilename, setPhotoFilename] = useState("Tournament 05-10-25-131.jpg")
  const [loading, setLoading] = useState(false)

  const handleDebugPerson = async () => {
    setLoading(true)
    try {
      const result = await debugPersonPhotosAction(personName)
      if (result.error) {
        console.error("[v0] Debug error:", result.error)
      } else {
        console.log("[v0] Debug completed successfully. Check console for details.")
      }
    } catch (error) {
      console.error("[v0] Debug failed:", error)
    } finally {
      setLoading(false)
    }
  }

  const handleDebugPhoto = async () => {
    setLoading(true)
    try {
      const result = await debugPhotoFacesAction(photoFilename)
      if (result.error) {
        console.error("[v0] Debug error:", result.error)
      } else {
        console.log("[v0] Debug completed successfully. Check console for details.")
      }
    } catch (error) {
      console.error("[v0] Debug failed:", error)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex items-center gap-4">
      <div className="flex items-center gap-2">
        <Input
          type="text"
          value={personName}
          onChange={(e) => setPersonName(e.target.value)}
          placeholder="Имя человека"
          className="w-48"
        />
        <Button onClick={handleDebugPerson} disabled={loading} variant="outline" size="sm">
          <Bug className="mr-2 h-4 w-4" />
          {loading ? "Загрузка..." : "Debug Person"}
        </Button>
      </div>

      <div className="flex items-center gap-2">
        <Input
          type="text"
          value={photoFilename}
          onChange={(e) => setPhotoFilename(e.target.value)}
          placeholder="Имя файла фото"
          className="w-64"
        />
        <Button onClick={handleDebugPhoto} disabled={loading} variant="outline" size="sm">
          <Bug className="mr-2 h-4 w-4" />
          {loading ? "Загрузка..." : "Debug Photo"}
        </Button>
      </div>
    </div>
  )
}
