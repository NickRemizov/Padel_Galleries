"use client"

import { DialogDescription, DialogTitle } from "@/components/ui/dialog"
import { FaceRecognitionDetailsDialog, type DetailedFace } from "./face-recognition-details-dialog"
import type { TaggedFace } from "@/lib/types" // Declare the TaggedFace variable here
import { useState, useEffect, useRef } from "react"
import { Dialog, DialogContent, DialogHeader } from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Badge } from "@/components/ui/badge"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import { Loader2, Save, X, Plus, Maximize2, Minimize2, Scan, Check } from "lucide-react"
import {
  savePhotoFaceAction,
  getPhotoFacesAction,
  deletePhotoFaceAction,
  getPeopleAction,
  getPersonAction,
  saveFaceTagsAction,
} from "@/app/admin/actions"
import type { Person } from "@/lib/types"
import { AddPersonDialog } from "./add-person-dialog"
import { toast } from "@/components/ui/use-toast"

const FASTAPI_URL = process.env.NEXT_PUBLIC_FASTAPI_URL!

interface FaceTaggingDialogProps {
  imageId: string
  imageUrl: string
  open: boolean
  onOpenChange: (open: boolean) => void
  onSave?: () => Promise<void>
  hasBeenProcessed?: boolean
}

export function FaceTaggingDialog({
  imageId,
  imageUrl,
  open,
  onOpenChange,
  onSave,
  hasBeenProcessed,
}: FaceTaggingDialogProps) {
  const [people, setPeople] = useState<Person[]>([])
  const [taggedFaces, setTaggedFaces] = useState<TaggedFace[]>([])
  const [selectedFaceIndex, setSelectedFaceIndex] = useState<number | null>(null)
  const [saving, setSaving] = useState(false)
  const [detecting, setDetecting] = useState(false)
  const [showAddPerson, setShowAddPerson] = useState(false)
  const [imageFitMode, setImageFitMode] = useState<"contain" | "cover">("contain")
  const [redetecting, setRedetecting] = useState(false)
  const [showDetailsDialog, setShowDetailsDialog] = useState(false)
  const [detailedFaces, setDetailedFaces] = useState<DetailedFace[]>([])
  const [hasRedetectedData, setHasRedetectedData] = useState(false)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const imageRef = useRef<HTMLImageElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)

  const getDisplayFileName = () => {
    try {
      const rawFileName = imageUrl.split("/").pop()?.split("?")[0] || "unknown"
      const decodedFileName = decodeURIComponent(rawFileName)
      const cleanedFileName = decodedFileName.replace(/-[a-zA-Z0-9]{20,}(\.[^.]+)$/, "$1")
      const maxLength = 50
      if (cleanedFileName.length > maxLength) {
        return cleanedFileName.substring(0, maxLength) + "..."
      }
      return cleanedFileName
    } catch (error) {
      return imageUrl.split("/").pop()?.split("?")[0] || "unknown"
    }
  }

  const displayFileName = getDisplayFileName()
  const fullFileName = (() => {
    try {
      const rawFileName = imageUrl.split("/").pop()?.split("?")[0] || "unknown"
      const decodedFileName = decodeURIComponent(rawFileName)
      return decodedFileName.replace(/-[a-zA-Z0-9]{20,}(\.[^.]+)$/, "$1")
    } catch {
      return imageUrl.split("/").pop()?.split("?")[0] || "unknown"
    }
  })()

  useEffect(() => {
    if (open) {
      loadPeopleAndExistingFaces()
    }
  }, [open]) // Removed imageId from dependencies - it should NOT trigger reload

  useEffect(() => {
    if (!showAddPerson && open) {
      loadPeople()
    }
  }, [showAddPerson, open])

  async function loadPeople() {
    const result = await getPeopleAction()
    if (result.success && result.data) {
      setPeople(result.data as Person[])
    }
  }

  async function loadPeopleAndExistingFaces() {
    await loadPeople()

    const existingResult = await getPhotoFacesAction(imageId)
    const existingFaces = existingResult.success && existingResult.data ? existingResult.data : []

    if (existingFaces.length > 0) {
      const tagged: TaggedFace[] = existingFaces.map((existing) => {
        return {
          face: {
            insightface_bbox: existing.insightface_bbox || { x: 0, y: 0, width: 0, height: 0 },
            confidence: existing.recognition_confidence || 0,
            blur_score: undefined,
            distance_to_nearest: undefined,
            top_matches: undefined,
          },
          id: existing.id,
          person_id: existing.person_id || null,
          person_real_name: existing.person_real_name || "",
          recognition_source: existing.recognition_source || "manual",
          confidence_score: existing.confidence_score || null,
          verified: existing.verified !== undefined ? existing.verified : false,
        }
      })

      setTaggedFaces(tagged)
      drawFaces(tagged)
    } else {
      if (!hasBeenProcessed) {
        console.log("[v0] No existing faces and photo not processed, triggering auto-detection")
        await detectAndRecognizeFaces()
      } else {
        console.log("[v0] No existing faces but photo already processed, skipping auto-detection")
      }
    }
  }

  async function detectAndRecognizeFaces() {
    try {
      setDetecting(true)
      const faces = await detectFacesInsightFace(imageUrl)
      const tagged: TaggedFace[] = await Promise.all(
        faces.map(async (face) => {
          const recognition = await recognizeFaceInsightFace(face.embedding)
          console.log("[v4.0] FaceTaggingDialog: Recognition result for face:", {
            face_bbox: face.insightface_bbox,
            detection_confidence: face.confidence,
            recognition_confidence: recognition?.confidence ?? null,
            person_name: recognition?.person_name,
          })
          return {
            face,
            person_id: recognition?.person_id || null,
            person_real_name: recognition?.person_name || "",
            recognition_source: "insightface",
            confidence_score: recognition?.confidence || null,
            verified: false,
          }
        }),
      )

      console.log(
        "[v4.0] FaceTaggingDialog: All faces after detection:",
        tagged.map((t) => ({
          person_id: t.person_id,
          person_real_name: t.person_real_name,
          recognition_source: t.recognition_source,
          confidence_score: t.confidence_score,
          verified: t.verified,
        })),
      )

      setTaggedFaces(tagged)
      const detailed: DetailedFace[] = tagged.map((t) => ({
        insightface_bbox: t.face.insightface_bbox,
        size: t.face.insightface_bbox.width,
        blur_score: t.face.blur_score,
        detection_score: t.face.confidence || 0,
        insightface_confidence: t.confidence_score || undefined,
        person_name: t.person_real_name || undefined,
        verified: t.verified,
        distance_to_nearest: t.face.distance_to_nearest,
        top_matches: t.face.top_matches,
        embedding_quality: (t.face as any).embedding_quality,
      }))
      setDetailedFaces(detailed)
      setHasRedetectedData(true)
      drawFaces(tagged)
    } catch (error) {
      console.error("Error detecting faces:", error)
    } finally {
      setDetecting(false)
    }
  }

  async function detectFacesInsightFace(imageUrl: string, applyFilters = true): Promise<any[]> {
    const apiUrl = `/api/face-detection/detect`
    const response = await fetch(apiUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_url: imageUrl, apply_quality_filters: applyFilters }),
    })

    if (!response.ok) {
      const error = await response.text()
      console.error("Detect faces error:", error)
      throw new Error("Failed to detect faces")
    }

    const data = await response.json()

    return data.faces.map((face: any) => ({
      insightface_bbox: face.insightface_bbox,
      confidence: face.confidence,
      blur_score: face.blur_score,
      embedding: face.embedding,
      distance_to_nearest: face.distance_to_nearest,
      top_matches: face.top_matches,
      embedding_quality: face.embedding_quality,
    }))
  }

  async function recognizeFaceInsightFace(embedding: number[]) {
    const apiUrl = `/api/face-detection/recognize`
    const response = await fetch(apiUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        embedding: embedding,
      }),
    })

    if (!response.ok) {
      return null
    }

    const data = await response.json()

    console.log("[v4.0] FaceTaggingDialog: Recognition API response:", data)

    if (!data.person_id) {
      return null
    }

    let person = people.find((p) => p.id === data.person_id)

    if (!person) {
      console.log("[v4.0] FaceTaggingDialog: Person not in cache, fetching from DB:", data.person_id)
      const result = await getPersonAction(data.person_id)
      if (result.success && result.data) {
        person = result.data as Person
        setPeople((prev) => [...prev, person!])
      }
    }

    return {
      person_id: data.person_id,
      person_name: person?.real_name || "Unknown",
      confidence: data.confidence,
    }
  }

  function calculateIoU(box1: any, box2: any): number {
    const x1 = Math.max(box1.x, box2.x)
    const y1 = Math.max(box1.y, box2.y)
    const x2 = Math.min(box1.x + box1.width, box2.x + box2.width)
    const y2 = Math.min(box1.y + box1.height, box2.y + box2.height)

    const intersectionWidth = Math.max(0, x2 - x1)
    const intersectionHeight = Math.max(0, y2 - y1)
    const intersectionArea = intersectionWidth * intersectionHeight

    const box1Area = box1.width * box1.height
    const box2Area = box2.width * box2.height
    const unionArea = box1Area + box2Area - intersectionArea

    return unionArea > 0 ? intersectionArea / unionArea : 0
  }

  function getFaceColor(index: number): string {
    const colors = [
      "#ef4444", // red
      "#3b82f6", // blue
      "#22c55e", // green
      "#f59e0b", // orange
      "#8b5cf6", // purple
      "#ec4899", // pink
      "#14b8a6", // teal
      "#f97316", // orange-red
    ]
    return colors[index % colors.length]
  }

  function drawFaces(faces: TaggedFace[]) {
    const canvas = canvasRef.current
    const image = imageRef.current
    if (!canvas || !image || !image.complete) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    canvas.width = image.naturalWidth
    canvas.height = image.naturalHeight

    ctx.clearRect(0, 0, canvas.width, canvas.height)
    ctx.drawImage(image, 0, 0)

    console.log("[v0] drawFaces: Total faces to draw:", faces.length)

    faces.forEach((taggedFace, index) => {
      const { insightface_bbox } = taggedFace.face

      console.log(`[v0] drawFaces: Face ${index}:`, {
        has_bbox: !!insightface_bbox,
        bbox_value: insightface_bbox,
        bbox_type: typeof insightface_bbox,
        person_real_name: taggedFace.person_real_name,
        verified: taggedFace.verified,
      })

      if (!insightface_bbox) {
        console.log(`[v0] drawFaces: Skipping face ${index} - no bbox`)
        return
      }

      if (insightface_bbox.width === 0 || insightface_bbox.height === 0) {
        console.log(`[v0] drawFaces: Skipping face ${index} - zero-sized bbox:`, insightface_bbox)
        return
      }

      const isSelected = index === selectedFaceIndex

      const faceColor = getFaceColor(index)
      ctx.strokeStyle = isSelected ? "#3b82f6" : faceColor
      ctx.lineWidth = isSelected ? 8 : 4

      console.log(`[v0] drawFaces: Drawing bbox for face ${index}:`, {
        x: insightface_bbox.x,
        y: insightface_bbox.y,
        width: insightface_bbox.width,
        height: insightface_bbox.height,
        color: ctx.strokeStyle,
      })

      ctx.strokeRect(insightface_bbox.x, insightface_bbox.y, insightface_bbox.width, insightface_bbox.height)

      if (taggedFace.person_real_name || taggedFace.confidence_score) {
        const confidenceText = taggedFace.verified
          ? " (100%)"
          : taggedFace.confidence_score
            ? ` (${Math.round(taggedFace.confidence_score * 100)}%)`
            : ""

        const nameText = taggedFace.person_real_name || "Unknown"
        const label = `${nameText}${confidenceText}`

        ctx.font = "bold 20px sans-serif"
        const textWidth = ctx.measureText(label).width
        const padding = 10
        const labelHeight = 32

        const labelX = insightface_bbox.x
        const labelY = insightface_bbox.y - labelHeight - 5

        ctx.fillStyle = isSelected ? "#3b82f6" : faceColor
        ctx.fillRect(labelX, labelY, textWidth + padding * 2, labelHeight)

        ctx.fillStyle = "#ffffff"
        ctx.fillText(label, labelX + padding, labelY + labelHeight - 8)
      }
    })
  }

  function handleFaceClick(index: number) {
    setSelectedFaceIndex(index === selectedFaceIndex ? null : index)
  }

  function handlePersonSelect(personId: string) {
    if (selectedFaceIndex === null) return

    const person = people.find((p) => p.id === personId)
    if (!person) return

    const updated = [...taggedFaces]
    updated[selectedFaceIndex] = {
      ...updated[selectedFaceIndex],
      person_id: person.id,
      person_real_name: person.real_name,
      verified: true,
    }
    setTaggedFaces(updated)
    drawFaces(updated)
  }

  function handleRemoveFace(index: number) {
    const updated = taggedFaces.filter((_, i) => i !== index)
    setTaggedFaces(updated)
    setSelectedFaceIndex(null)
    drawFaces(updated)
  }

  async function handleSaveWithoutClose() {
    setSaving(true)
    try {
      const existingResult = await getPhotoFacesAction(imageId)
      const existingFaces = existingResult.success && existingResult.data ? existingResult.data : []

      console.log("[v4.1] handleSaveWithoutClose: Starting save process")
      console.log(
        "[v4.1] Existing faces from DB:",
        existingFaces.map((f) => ({ id: f.id, person_id: f.person_id, verified: f.verified })),
      )
      console.log(
        "[v4.1] Current taggedFaces in state:",
        taggedFaces.map((f) => ({
          person_id: f.person_id,
          person_real_name: f.person_real_name,
          verified: f.verified,
        })),
      )

      for (const face of existingFaces) {
        console.log("[v4.1] Deleting face:", face.id)
        await deletePhotoFaceAction(face.id)
      }

      console.log("[v4.1] All existing faces deleted, now saving new faces")

      if (taggedFaces.length === 0) {
        console.log("[v4.1] No tagged faces to save, completing")
        console.log("[v4.1] Save process completed successfully, closing dialog")
        console.log("[v4.1] Calling onSave callback to reload gallery data")
        onSave()
        onOpenChange(false)
        return
      } else {
        for (const taggedFace of taggedFaces) {
          if (!taggedFace.person_id) continue

          const isVerified = true
          const insightfaceConfidenceToSave = taggedFace.confidence_score ?? 1.0
          const recognitionConfidenceToSave = 1.0

          console.log("[v4.1] Saving face for person:", {
            person_id: taggedFace.person_id,
            person_real_name: taggedFace.person_real_name,
            insightfaceConfidence: insightfaceConfidenceToSave,
            recognitionConfidence: recognitionConfidenceToSave,
          })

          const result = await savePhotoFaceAction(
            imageId,
            taggedFace.person_id,
            taggedFace.face.insightface_bbox,
            insightfaceConfidenceToSave,
            recognitionConfidenceToSave,
            isVerified,
            imageUrl,
          )

          if (!result.success) {
            console.error("Save failed:", result.error)
            alert(`Ошибка сохранения: ${result.error}`)
            setSaving(false)
            return
          }

          console.log("[v4.1] Face saved successfully")
        }
      }

      console.log("[v4.1] Save process completed successfully")

      if (onSave) {
        console.log("[v4.1] Calling onSave callback to reload gallery data")
        await onSave()
        console.log("[v4.1] onSave completed, data reloaded")
      }
    } catch (error) {
      console.error("Error saving face tags:", error)
      alert("Ошибка при сохранении тегов")
    } finally {
      setSaving(false)
    }
  }

  async function handleSave() {
    setSaving(true)
    try {
      console.log("[v5.0] handleSave: Starting save process with saveFaceTagsAction")
      console.log(
        "[v5.0] Current taggedFaces:",
        taggedFaces.map((f) => ({
          person_id: f.person_id,
          person_real_name: f.person_real_name,
          verified: f.verified,
        })),
      )

      // Build tags array for batch save
      const tags = taggedFaces
        .filter((tf) => tf.person_id) // Only faces with assigned person
        .map((tf) => ({
          person_id: tf.person_id!,
          insightface_bbox: tf.face.insightface_bbox,
          bbox: tf.face.insightface_bbox, // Added bbox for Python endpoint
          insightface_confidence: tf.confidence_score ?? 1.0,
          recognition_confidence: 1.0,
          verified: true,
          embedding: null, // Embedding generated on backend
        }))

      console.log("[v5.0] Sending", tags.length, "tags to saveFaceTagsAction")

      const result = await saveFaceTagsAction(imageId, imageUrl, tags)

      if (result.error) {
        const errorMessage = typeof result.error === "string" ? result.error : JSON.stringify(result.error)
        console.error("[v5.0] Save failed:", errorMessage)
        alert(`Ошибка сохранения: ${errorMessage}`)
        setSaving(false)
        return
      }

      console.log("[v5.0] Save completed successfully")

      if (onSave) {
        console.log("[v5.0] Calling onSave callback to reload gallery data")
        await onSave()
      }
      onOpenChange(false)
    } catch (error: any) {
      const errorMessage = error?.message || JSON.stringify(error) || "Unknown error"
      console.error("[v5.0] Error in handleSave:", errorMessage)
      alert(`Ошибка: ${errorMessage}`)
    } finally {
      setSaving(false)
    }
  }

  async function handleRedetect() {
    if (!imageUrl) return

    try {
      setDetecting(true)
      const faces = await detectFacesInsightFace(imageUrl, true)
      const tagged: TaggedFace[] = await Promise.all(
        faces.map(async (face) => {
          const recognition = await recognizeFaceInsightFace(face.embedding)
          return {
            face,
            person_id: recognition?.person_id || null,
            person_real_name: recognition?.person_name || "",
            recognition_source: "insightface",
            confidence_score: recognition?.confidence || null,
            verified: false,
          }
        }),
      )

      setTaggedFaces(tagged)
      const detailed: DetailedFace[] = tagged.map((t) => ({
        insightface_bbox: t.face.insightface_bbox,
        size: t.face.insightface_bbox.width,
        blur_score: t.face.blur_score,
        detection_score: t.face.confidence || 0,
        insightface_confidence: t.confidence_score || undefined,
        person_name: t.person_real_name || undefined,
        verified: t.verified,
        distance_to_nearest: t.face.distance_to_nearest,
        top_matches: t.face.top_matches,
        embedding_quality: (t.face as any).embedding_quality,
      }))
      setDetailedFaces(detailed)
      setHasRedetectedData(true)
    } catch (error) {
      console.error("Error redetecting faces:", error)
    } finally {
      setDetecting(false)
    }
  }

  async function handleRedetectWithoutFilters() {
    if (!imageUrl) return

    try {
      setDetecting(true)
      const faces = await detectFacesInsightFace(imageUrl, false)
      const recognized = await Promise.all(
        faces.map(async (face) => {
          const recognition = await recognizeFaceInsightFace(face.embedding)
          return {
            face,
            person_id: recognition?.person_id || null,
            person_real_name: recognition?.person_name || "",
            recognition_source: "insightface",
            confidence_score: recognition?.confidence || null,
            verified: false,
          }
        }),
      )

      const newTagged = mergeFaces(taggedFaces, recognized)
      setTaggedFaces(newTagged)
      const detailed: DetailedFace[] = newTagged.map((t) => ({
        insightface_bbox: t.face.insightface_bbox,
        size: t.face.insightface_bbox.width,
        blur_score: t.face.blur_score,
        detection_score: t.face.confidence || 0,
        insightface_confidence: t.confidence_score || undefined,
        person_name: t.person_real_name || undefined,
        verified: t.verified,
        distance_to_nearest: t.face.distance_to_nearest,
        top_matches: t.face.top_matches,
        embedding_quality: (t.face as any).embedding_quality,
      }))
      setDetailedFaces(detailed)
      setHasRedetectedData(true)
    } catch (error) {
      console.error("Error redetecting without filters:", error)
      toast({
        title: "Ошибка",
        description: "Не удалось обнаружить лица без фильтров",
        variant: "destructive",
      })
    } finally {
      setDetecting(false)
    }
  }

  useEffect(() => {
    if (taggedFaces.length > 0 && imageRef.current?.complete) {
      drawFaces(taggedFaces)
    }
  }, [imageFitMode])

  const hasUnassignedFaces = taggedFaces.some((face) => !face.person_id)
  const canSave = !saving && !hasUnassignedFaces

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-[90vw] h-[90vh] overflow-hidden flex flex-col">
        <DialogHeader>
          <DialogTitle>Тегирование лиц v3.10</DialogTitle>
          <DialogDescription title={fullFileName}>
            Файл: {displayFileName} | Обнаружено лиц: {taggedFaces.length}. Кликните на лицо, чтобы назначить человека.
          </DialogDescription>
          <div className="absolute top-4 right-12 flex gap-2">
            <Tooltip>
              <TooltipTrigger asChild>
                <span>
                  <Button
                    size="sm"
                    onClick={handleSaveWithoutClose}
                    disabled={!canSave}
                    className="h-8 w-8 p-0 bg-green-600 hover:bg-green-700 text-white"
                  >
                    <Check className="h-4 w-4" />
                  </Button>
                </span>
              </TooltipTrigger>
              {hasUnassignedFaces ? (
                <TooltipContent>
                  <p>Назначьте всех людей или удалите неизвестные лица перед сохранением</p>
                </TooltipContent>
              ) : (
                <TooltipContent>
                  <p>Сохранить без закрытия окна</p>
                </TooltipContent>
              )}
            </Tooltip>
            <Button
              variant={imageFitMode === "contain" ? "default" : "outline"}
              size="sm"
              onClick={() => setImageFitMode("contain")}
              title="Вписать в окно"
            >
              <Minimize2 className="h-4 w-4" />
            </Button>
            <Button
              variant={imageFitMode === "cover" ? "default" : "outline"}
              size="sm"
              onClick={() => setImageFitMode("cover")}
              title="Масштаб по длинной стороне"
            >
              <Maximize2 className="h-4 w-4" />
            </Button>
          </div>
        </DialogHeader>

        <div className="flex-1 flex flex-col gap-4 min-h-0">
          {detecting ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="h-8 w-8 animate-spin" />
              <span className="ml-2">Обнаружение лиц с помощью InsightFace...</span>
            </div>
          ) : (
            <>
              <div
                ref={containerRef}
                className={`relative w-full border rounded-lg bg-black flex-1 min-h-0 overflow-auto`}
              >
                <img
                  ref={imageRef}
                  src={imageUrl || "/placeholder.svg"}
                  alt="Фото для тегирования"
                  className="hidden"
                  crossOrigin="anonymous"
                  onLoad={() => {
                    drawFaces(taggedFaces)
                  }}
                />
                <canvas
                  ref={canvasRef}
                  className={`w-full cursor-pointer ${
                    imageFitMode === "contain" ? "object-contain h-full" : "object-cover"
                  }`}
                  onClick={(e) => {
                    const canvas = canvasRef.current
                    if (!canvas) return

                    const rect = canvas.getBoundingClientRect()

                    const { renderedWidth, renderedHeight, offsetX, offsetY } = getRenderedImageDimensions(
                      canvas,
                      imageFitMode,
                    )

                    const clickX = e.clientX - rect.left
                    const clickY = e.clientY - rect.top

                    if (
                      clickX < offsetX ||
                      clickX > offsetX + renderedWidth ||
                      clickY < offsetY ||
                      clickY > offsetY + renderedHeight
                    ) {
                      return
                    }

                    const imageX = ((clickX - offsetX) / renderedWidth) * canvas.width
                    const imageY = ((clickY - offsetY) / renderedHeight) * canvas.height

                    const clickedIndex = taggedFaces.findIndex((taggedFace) => {
                      const { insightface_bbox } = taggedFace.face
                      if (!insightface_bbox) return false
                      return (
                        imageX >= insightface_bbox.x &&
                        imageX <= insightface_bbox.x + insightface_bbox.width &&
                        imageY >= insightface_bbox.y &&
                        imageY <= insightface_bbox.y + insightface_bbox.height
                      )
                    })

                    if (clickedIndex !== -1) {
                      handleFaceClick(clickedIndex)
                    }
                  }}
                />
              </div>

              <div className="flex items-center gap-3 px-1 min-h-[52px]">
                {taggedFaces.some((face) => face.person_real_name) && (
                  <div className="flex flex-wrap gap-2">
                    {taggedFaces.map((taggedFace, index) => {
                      if (!taggedFace.person_real_name) return null
                      const faceColor = getFaceColor(index)
                      const isSelected = index === selectedFaceIndex
                      const confidenceText = taggedFace.verified
                        ? " (100%)"
                        : taggedFace.confidence_score
                          ? ` (${Math.round(taggedFace.confidence_score * 100)}%)`
                          : ""
                      return (
                        <Badge
                          key={index}
                          style={{ backgroundColor: faceColor, color: "#ffffff" }}
                          className={`text-sm px-3 py-1.5 cursor-pointer transition-all ${
                            isSelected ? "ring-2 ring-offset-2 ring-blue-500 scale-110" : "hover:scale-105"
                          }`}
                          onClick={() => handleFaceClick(index)}
                        >
                          {taggedFace.person_real_name}
                          {confidenceText}
                        </Badge>
                      )
                    })}
                  </div>
                )}

                {selectedFaceIndex !== null && (
                  <div className="flex items-center gap-2 ml-auto">
                    <Select value={taggedFaces[selectedFaceIndex].person_id || ""} onValueChange={handlePersonSelect}>
                      <SelectTrigger className="w-[250px]">
                        <SelectValue placeholder="Выберите человека" />
                      </SelectTrigger>
                      <SelectContent>
                        {people.map((person) => (
                          <SelectItem key={person.id} value={person.id}>
                            {person.real_name}
                            {person.telegram_name && ` (${person.telegram_name})`}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                    {taggedFaces[selectedFaceIndex].person_id &&
                      (taggedFaces[selectedFaceIndex].verified ||
                        (taggedFaces[selectedFaceIndex].confidence_score !== null &&
                          !isNaN(taggedFaces[selectedFaceIndex].confidence_score!) &&
                          taggedFaces[selectedFaceIndex].confidence_score! > 0)) && (
                        <Badge variant="secondary" className="whitespace-nowrap">
                          {taggedFaces[selectedFaceIndex].verified
                            ? "100%"
                            : `${Math.round(taggedFaces[selectedFaceIndex].confidence_score! * 100)}%`}
                        </Badge>
                      )}
                    <Button variant="destructive" size="icon" onClick={() => handleRemoveFace(selectedFaceIndex)}>
                      <X className="h-4 w-4" />
                    </Button>
                  </div>
                )}
              </div>
            </>
          )}
        </div>

        <div className="flex-shrink-0">
          <div className="flex items-center justify-between gap-4">
            <div className="flex gap-2">
              <Button variant="outline" size="sm" onClick={() => setShowAddPerson(true)}>
                <Plus className="mr-2 h-4 w-4" />
                Добавить человека
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={handleRedetectWithoutFilters}
                disabled={redetecting || detecting}
              >
                {redetecting ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Повторное распознавание...
                  </>
                ) : (
                  <>
                    <Scan className="mr-2 h-4 w-4" />
                    Распознать без фильтров
                  </>
                )}
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setShowDetailsDialog(true)}
                disabled={!hasRedetectedData}
              >
                Показать метрики
              </Button>
            </div>

            <div className="flex gap-2 ml-auto">
              <Button variant="outline" onClick={() => onOpenChange(false)}>
                Отмена
              </Button>
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <span>
                      <Button onClick={handleSave} disabled={!canSave}>
                        {saving ? (
                          <>
                            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                            Сохранение...
                          </>
                        ) : (
                          <>
                            <Save className="mr-2 h-4 w-4" />
                            Сохранить
                          </>
                        )}
                      </Button>
                    </span>
                  </TooltipTrigger>
                  {hasUnassignedFaces && (
                    <TooltipContent>
                      <p>Назначьте всех людей или удалите неизвестные лица перед сохранением</p>
                    </TooltipContent>
                  )}
                </Tooltip>
              </TooltipProvider>
            </div>
          </div>
        </div>
      </DialogContent>
      <AddPersonDialog open={showAddPerson} onOpenChange={setShowAddPerson} />
      <FaceRecognitionDetailsDialog
        open={showDetailsDialog}
        onOpenChange={setShowDetailsDialog}
        faces={detailedFaces}
        imageUrl={imageUrl} // Pass imageUrl for face preview
      />
    </Dialog>
  )
}

function getRenderedImageDimensions(canvas: HTMLCanvasElement, mode: "contain" | "cover") {
  const rect = canvas.getBoundingClientRect()
  const canvasAspect = canvas.width / canvas.height
  const displayAspect = rect.width / rect.height

  let renderedWidth: number
  let renderedHeight: number
  let offsetX: number
  let offsetY: number

  if (mode === "contain") {
    if (canvasAspect > displayAspect) {
      renderedWidth = rect.width
      renderedHeight = rect.width / canvasAspect
      offsetX = 0
      offsetY = (rect.height - renderedHeight) / 2
    } else {
      renderedHeight = rect.height
      renderedWidth = rect.height * canvasAspect
      offsetX = (rect.width - renderedWidth) / 2
      offsetY = 0
    }
  } else {
    if (canvasAspect > displayAspect) {
      renderedHeight = rect.height
      renderedWidth = rect.height * canvasAspect
      offsetX = (rect.width - renderedWidth) / 2
      offsetY = 0
    } else {
      renderedWidth = rect.width
      renderedHeight = rect.width / canvasAspect
      offsetX = 0
      offsetY = (rect.height - renderedHeight) / 2
    }
  }

  return { renderedWidth, renderedHeight, offsetX, offsetY }
}

function mergeFaces(existingFaces: TaggedFace[], newFaces: TaggedFace[]): TaggedFace[] {
  // Implement the logic to merge faces here
  // This is a placeholder implementation
  return newFaces.map((newFace) => {
    const existingFace = existingFaces.find((face) => face.id === newFace.id)
    return existingFace ? { ...existingFace, ...newFace } : newFace
  })
}
