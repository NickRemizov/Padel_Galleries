"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Progress } from "@/components/ui/progress"
import { Badge } from "@/components/ui/badge"
import { Loader2, CheckCircle2, XCircle } from "lucide-react"
import { Checkbox } from "@/components/ui/checkbox"
import {
  saveFaceDescriptorAction,
  savePhotoFaceAction,
  updatePhotoFaceAction,
  getPhotoFacesAction,
  getBatchPhotoFacesAction,
  deletePhotoFaceAction,
  markPhotoAsProcessedAction, // Added import for marking photo as processed
} from "@/app/admin/actions"
import { calculateIoU } from "@/lib/face-recognition/utils"
import type { GalleryImage } from "@/lib/types"

const VERSION = "v3.15" // Updated version to track has_been_processed marking

interface AutoRecognitionDialogProps {
  images: GalleryImage[]
  open: boolean
  onOpenChange: (open: boolean) => void
  mode: "all" | "remaining"
}

interface ProcessingResult {
  imageId: string
  filename: string
  facesFound: number
  facesRecognized: number
  status: "pending" | "processing" | "success" | "error"
  error?: string
}

export function AutoRecognitionDialog({ images, open, onOpenChange, mode }: AutoRecognitionDialogProps) {
  const [processing, setProcessing] = useState(false)
  const [results, setResults] = useState<ProcessingResult[]>([])
  const [currentIndex, setCurrentIndex] = useState(0)
  const [applyQualityFilters, setApplyQualityFilters] = useState(true)

  async function detectFacesInsightFace(imageUrl: string) {
    const apiUrl = `/api/face-detection/detect` // Fixed endpoint back to /api/face-detection/detect (Next.js proxy that calls FastAPI /detect-faces)
    console.log(`[${VERSION}] Calling detect-faces proxy API:`, apiUrl)
    console.log(`[${VERSION}] Apply quality filters:`, applyQualityFilters)

    const response = await fetch(apiUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image_url: imageUrl,
        apply_quality_filters: applyQualityFilters,
      }),
    })

    if (!response.ok) {
      const error = await response.text()
      console.error(`[${VERSION}] Detect faces error:`, error)
      throw new Error("Failed to detect faces")
    }

    const data = await response.json()
    console.log(`[${VERSION}] Detected`, data.faces.length, "faces")

    return data.faces
      .filter((face: any) => face.insightface_bbox && face.insightface_bbox.x !== undefined)
      .map((face: any) => ({
        insightface_bbox: face.insightface_bbox, // Use insightface_bbox from backend
        insightface_confidence: face.confidence, // Map from API response
        insightface_descriptor: face.embedding, // Map from API response
      }))
  }

  async function recognizeFaceInsightFace(embedding: number[]) {
    const apiUrl = `/api/face-detection/recognize`
    console.log(`[${VERSION}] Calling recognize-face proxy API:`, apiUrl)

    const response = await fetch(apiUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        embedding: embedding,
      }),
    })

    if (!response.ok) {
      console.log(`[${VERSION}] Recognize face failed (no match)`)
      return null
    }

    const data = await response.json()
    console.log(`[${VERSION}] Recognition result:`, data)

    return {
      personId: data.person_id,
      confidence: data.confidence,
    }
  }

  async function startProcessing() {
    setProcessing(true)
    setCurrentIndex(-1)

    let imagesToProcess = images

    if (mode === "remaining") {
      const batchResult = await getBatchPhotoFacesAction(images.map((img) => img.id))

      if (batchResult.success && batchResult.data) {
        const facesMap = new Map<string, any[]>()
        for (const face of batchResult.data) {
          if (!facesMap.has(face.photo_id)) {
            facesMap.set(face.photo_id, [])
          }
          facesMap.get(face.photo_id)!.push(face)
        }

        imagesToProcess = images.filter((image) => {
          const faces = facesMap.get(image.id) || []
          const hasUnverifiedFaces = faces.some((face) => !face.verified)
          return hasUnverifiedFaces
        })
      }
    }

    const initialResults: ProcessingResult[] = imagesToProcess.map((img) => ({
      imageId: img.id,
      filename: img.original_filename,
      facesFound: 0,
      facesRecognized: 0,
      status: "pending",
    }))
    setResults(initialResults)

    const batchSize = 2

    for (let i = 0; i < imagesToProcess.length; i++) {
      const image = imagesToProcess[i]
      setCurrentIndex(i)

      setResults((prev) => prev.map((r, idx) => (idx === i ? { ...r, status: "processing" as const } : r)))

      try {
        const existingFacesResult = await getPhotoFacesAction(image.id)
        const existingFaces = existingFacesResult.success ? existingFacesResult.data || [] : []

        console.log(`[${VERSION}] Image ${image.original_filename}: Found ${existingFaces.length} existing faces`)

        const faces = await detectFacesInsightFace(image.image_url)

        const matchedFaceIds = new Set<string>()

        if (faces.length === 0) {
          if (applyQualityFilters) {
            console.log(`[${VERSION}] No faces detected with quality filters, deleting all unverified faces`)
            for (const existingFace of existingFaces) {
              if (!existingFace.verified) {
                await deletePhotoFaceAction(existingFace.id)
                console.log(`[${VERSION}] Deleted unverified face ${existingFace.id}`)
              }
            }
          }

          await markPhotoAsProcessedAction(image.id)
          console.log(`[${VERSION}] Marked photo ${image.id} as processed (no faces detected)`)

          setResults((prev) =>
            prev.map((r, idx) =>
              idx === i
                ? {
                    ...r,
                    facesFound: 0,
                    facesRecognized: 0,
                    status: "success" as const,
                  }
                : r,
            ),
          )
          continue
        }

        let recognizedCount = 0
        let skippedCount = 0
        let allFacesFullConfidence = true

        for (let j = 0; j < faces.length; j++) {
          const face = faces[j]

          if (!face.insightface_bbox || face.insightface_bbox.x === undefined) {
            console.warn(`[${VERSION}] Skipping face with invalid bounding box`)
            continue
          }

          const match = await recognizeFaceInsightFace(face.insightface_descriptor)

          const overlappingFace = existingFaces.find((existing) => {
            if (!existing.insightface_bbox) return false
            const iou = calculateIoU(face.insightface_bbox, existing.insightface_bbox)
            return iou > 0.5
          })

          if (overlappingFace) {
            matchedFaceIds.add(overlappingFace.id)

            if (overlappingFace.verified) {
              console.log(
                `[${VERSION}] Skipping face - already verified as ${overlappingFace.people?.real_name || "unknown"}`,
              )
              skippedCount++
              continue
            }

            console.log(`[${VERSION}] Updating unverified face with new recognition results`)
            if (match && match.personId) {
              if (match.confidence < 1.0) {
                allFacesFullConfidence = false
              }
              await saveFaceDescriptorAction(match.personId, face.insightface_descriptor, image.id)
              await updatePhotoFaceAction(overlappingFace.id, {
                person_id: match.personId,
                insightface_confidence: face.insightface_confidence, // Detection confidence from detector
                recognition_confidence: match.confidence, // Recognition confidence from matching
                verified: false,
              })
              recognizedCount++
            } else {
              allFacesFullConfidence = false
              await updatePhotoFaceAction(overlappingFace.id, {
                person_id: null,
                recognition_confidence: null, // No recognition confidence when not matched
                verified: false,
              })
            }
            continue
          }

          if (match && match.personId) {
            if (match.confidence < 1.0) {
              allFacesFullConfidence = false
            }
            await saveFaceDescriptorAction(match.personId, face.insightface_descriptor, image.id)
            await savePhotoFaceAction(
              image.id,
              match.personId,
              face.insightface_bbox,
              face.insightface_descriptor,
              face.insightface_confidence, // Detection confidence
              match.confidence, // Recognition confidence
              false,
            )
            recognizedCount++
          } else {
            allFacesFullConfidence = false
            await savePhotoFaceAction(
              image.id,
              null,
              face.insightface_bbox,
              face.insightface_descriptor,
              face.insightface_confidence,
              0, // Recognition confidence is 0 when not recognized
              false,
            )
          }
        }

        if (applyQualityFilters) {
          for (const existingFace of existingFaces) {
            if (!existingFace.verified && !matchedFaceIds.has(existingFace.id)) {
              await deletePhotoFaceAction(existingFace.id)
              console.log(
                `[${VERSION}] Deleted unmatched unverified face ${existingFace.id} (filtered out by quality settings)`,
              )
            }
          }
        }

        if (mode === "all" && !allFacesFullConfidence) {
          for (const existingFace of existingFaces) {
            if (existingFace.verified) {
              await updatePhotoFaceAction(existingFace.id, {
                verified: false,
              })
            }
          }
        }

        console.log(
          `[${VERSION}] Processed ${faces.length} faces, recognized ${recognizedCount}, skipped ${skippedCount} verified faces`,
        )

        await markPhotoAsProcessedAction(image.id)
        console.log(`[${VERSION}] Marked photo ${image.id} as processed`)

        setResults((prev) =>
          prev.map((r, idx) =>
            idx === i
              ? {
                  ...r,
                  facesFound: faces.length,
                  facesRecognized: recognizedCount,
                  status: "success" as const,
                }
              : r,
          ),
        )
      } catch (error) {
        console.error(`[${VERSION}] Error processing image:`, error)
        setResults((prev) =>
          prev.map((r, idx) =>
            idx === i
              ? {
                  ...r,
                  status: "error" as const,
                  error: error instanceof Error ? error.message : "Unknown error",
                }
              : r,
          ),
        )
      }
    }

    setProcessing(false)
  }

  const totalImages = results.length > 0 ? results.length : images.length
  const processedImages = processing ? currentIndex + 1 : results.filter((r) => r.status !== "pending").length
  const progress = totalImages > 0 ? (processedImages / totalImages) * 100 : 0

  const totalFaces = results.reduce((sum, r) => sum + r.facesFound, 0)
  const totalRecognized = results.reduce((sum, r) => sum + r.facesRecognized, 0)

  const imagesToProcess = mode === "remaining" ? results.length : images.length

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[600px] max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>
            {mode === "all" ? "Автоматическое распознавание лиц" : "Распознавание оставшихся фото"}
          </DialogTitle>
          <DialogDescription>
            {mode === "all"
              ? `Обработка ${imagesToProcess} фотографий. Это может занять несколько минут.`
              : `Обработка ${imagesToProcess} фотографий без ручной верификации. Это может занять несколько минут.`}
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          {!processing && results.length === 0 ? (
            <div className="space-y-4">
              <p className="text-sm text-muted-foreground">
                {mode === "all"
                  ? "Система автоматически обнаружит и распознает лица на всех фотографиях галереи используя InsightFace."
                  : "Система обработает только фотографии без ручной верификации используя InsightFace."}
              </p>

              <div className="flex items-center space-x-2">
                <Checkbox
                  id="apply-quality-filters-auto"
                  checked={applyQualityFilters}
                  onCheckedChange={(checked) => setApplyQualityFilters(checked as boolean)}
                />
                <label
                  htmlFor="apply-quality-filters-auto"
                  className="text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
                >
                  Применять настройки качества
                </label>
              </div>
              <p className="text-xs text-muted-foreground">
                Если выключено, будут распознаны все лица без фильтрации по качеству
              </p>

              <Button onClick={startProcessing} className="w-full">
                Начать обработку
              </Button>
            </div>
          ) : (
            <>
              <div className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <span>Прогресс</span>
                  <span>
                    {processedImages} / {totalImages}
                  </span>
                </div>
                <Progress value={progress} />
              </div>

              {processing && (
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  <span>Обработка фотографий...</span>
                </div>
              )}

              {!processing && (
                <div className="space-y-2 rounded-lg border p-4">
                  <h3 className="font-semibold">Результаты</h3>
                  <div className="grid grid-cols-2 gap-2 text-sm">
                    <div>
                      <span className="text-muted-foreground">Обработано фото:</span>
                      <span className="ml-2 font-medium">{results.filter((r) => r.status === "success").length}</span>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Ошибок:</span>
                      <span className="ml-2 font-medium">{results.filter((r) => r.status === "error").length}</span>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Найдено лиц:</span>
                      <span className="ml-2 font-medium">{totalFaces}</span>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Распознано:</span>
                      <span className="ml-2 font-medium">{totalRecognized}</span>
                    </div>
                  </div>
                </div>
              )}

              <div className="max-h-[300px] space-y-2 overflow-y-auto">
                {results.map((result, index) => (
                  <div key={result.imageId} className="flex items-center justify-between rounded-lg border p-3 text-sm">
                    <div className="flex-1 truncate">
                      <p className="truncate font-medium" title={result.filename}>
                        {result.filename}
                      </p>
                      {result.status === "success" && (
                        <p className="text-xs text-muted-foreground">
                          Лиц: {result.facesFound}, распознано: {result.facesRecognized}
                        </p>
                      )}
                      {result.status === "error" && <p className="text-xs text-destructive">{result.error}</p>}
                    </div>
                    <div>
                      {result.status === "pending" && <Badge variant="secondary">Ожидание</Badge>}
                      {result.status === "processing" && (
                        <Badge variant="secondary">
                          <Loader2 className="mr-1 h-3 w-3 animate-spin" />
                          Обработка
                        </Badge>
                      )}
                      {result.status === "success" && (
                        <Badge variant="default">
                          <CheckCircle2 className="mr-1 h-3 w-3" />
                          Готово
                        </Badge>
                      )}
                      {result.status === "error" && (
                        <Badge variant="destructive">
                          <XCircle className="mr-1 h-3 w-3" />
                          Ошибка
                        </Badge>
                      )}
                    </div>
                  </div>
                ))}
              </div>

              {!processing && (
                <Button onClick={() => onOpenChange(false)} className="w-full">
                  Закрыть
                </Button>
              )}
            </>
          )}
        </div>
      </DialogContent>
    </Dialog>
  )
}
