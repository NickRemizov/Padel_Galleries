"use client"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Checkbox } from "@/components/ui/checkbox"
import { Progress } from "@/components/ui/progress"
import { Badge } from "@/components/ui/badge"
import { Loader2, Scan, CheckCircle2, XCircle } from "lucide-react"
import { saveBatchRecognitionResultsAction } from "@/app/admin/actions"
import type { Gallery } from "@/lib/types"

interface ProcessingResult {
  imageId: string
  filename: string
  galleryTitle: string
  facesFound: number
  facesRecognized: number
  status: "pending" | "processing" | "success" | "error"
  error?: string
}

function formatShortDate(dateString: string): string {
  const date = new Date(dateString)
  const day = String(date.getDate()).padStart(2, "0")
  const month = String(date.getMonth() + 1).padStart(2, "0")
  return `${day}.${month}`
}

export function BatchRecognitionManager() {
  const [galleries, setGalleries] = useState<Gallery[]>([])
  const [selectedGalleries, setSelectedGalleries] = useState<string[]>([])
  const [processing, setProcessing] = useState(false)
  const [results, setResults] = useState<ProcessingResult[]>([])
  const [currentIndex, setCurrentIndex] = useState(0)
  const [totalImages, setTotalImages] = useState(0)
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.6)

  useEffect(() => {
    loadGalleries()
  }, [])

  async function loadGalleries() {
    try {
      const { getGalleriesAction } = require("@/app/admin/actions") // Dynamic import to avoid build error if not there yet
      const result = await getGalleriesAction()
      if (result.success) {
        setGalleries(result.data)
      }
    } catch (e) {
      console.error("Failed to load galleries", e)
    }
  }

  function toggleGallery(galleryId: string) {
    setSelectedGalleries((prev) =>
      prev.includes(galleryId) ? prev.filter((id) => id !== galleryId) : [...prev, galleryId],
    )
  }

  function selectAll() {
    setGalleries(galleries.map((g) => g.id))
  }

  function deselectAll() {
    setSelectedGalleries([])
  }

  async function startBatchProcessing() {
    if (selectedGalleries.length === 0) {
      alert("Выберите хотя бы одну галерею")
      return
    }

    setProcessing(true)
    setResults([])
    setCurrentIndex(0)

    try {
      console.log("[v2.20] Starting batch processing with proxy API...")
      console.log("[v2.20] Selected galleries:", selectedGalleries)
      console.log("[v2.20] Confidence threshold:", confidenceThreshold)

      const response = await fetch("/api/batch-face-recognition", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ galleryIds: selectedGalleries }),
      })

      if (!response.ok) {
        throw new Error(`Failed to fetch images: ${response.statusText}`)
      }

      const { images } = await response.json()
      console.log("[v2.20] Fetched images:", images.length)
      setTotalImages(images.length)

      const initialResults: ProcessingResult[] = images.map((img: any) => ({
        imageId: img.id,
        filename: img.original_filename,
        galleryTitle: galleries.find((g) => g.id === img.gallery_id)?.title || "Unknown",
        facesFound: 0,
        facesRecognized: 0,
        status: "pending",
      }))
      setResults(initialResults)

      for (let i = 0; i < images.length; i++) {
        const image = images[i]
        setCurrentIndex(i)

        setResults((prev) => prev.map((r, idx) => (idx === i ? { ...r, status: "processing" as const } : r)))

        try {
          console.log(`[v2.20] Processing image ${i + 1}/${images.length}: ${image.original_filename}`)

          const detectResponse = await fetch(`/api/face-detection/detect`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              image_url: image.image_url,
              apply_quality_filters: true,
            }),
          })

          if (!detectResponse.ok) {
            throw new Error(`Proxy detect-faces failed: ${detectResponse.statusText}`)
          }

          const { faces } = await detectResponse.json()
          console.log(`[v2.20] Detected ${faces.length} faces in ${image.original_filename}`)

          if (faces.length === 0) {
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
          const faceDataArray: any[] = []

          for (const face of faces) {
            try {
              const recognizeResponse = await fetch(`/api/face-detection/recognize`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                  embedding: face.embedding,
                  confidence_threshold: confidenceThreshold,
                }),
              })

              if (!recognizeResponse.ok) {
                console.error(`[v2.20] Proxy recognize-face failed: ${recognizeResponse.statusText}`)
                continue
              }

              const recognition = await recognizeResponse.json()
              console.log(`[v2.20] Recognition result:`, recognition)

              const faceData = {
                person_id:
                  recognition.person_id && recognition.confidence >= confidenceThreshold ? recognition.person_id : null,
                insightface_bbox: face.insightface_bbox,
                insightface_descriptor: `[${face.embedding.join(",")}]`,
                insightface_confidence: face.confidence,
                recognition_confidence: recognition.confidence,
                embedding: face.embedding,
              }

              faceDataArray.push(faceData)

              if (recognition.person_id && recognition.confidence >= confidenceThreshold) {
                recognizedCount++
                console.log(`[v2.20] Saved recognized face: ${recognition.person_id} (${recognition.confidence})`)
              } else {
                console.log(`[v2.20] Saved unknown face (confidence: ${recognition.confidence})`)
              }
            } catch (faceError) {
              console.error("[v2.20] Error recognizing face:", faceError)
            }
          }

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

          // Save batch recognition results using server action
          await saveBatchRecognitionResultsAction(image.id, faceDataArray)
        } catch (error) {
          console.error("[v2.20] Error processing image:", error)
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

      console.log("[v2.20] Batch processing completed")
    } catch (error) {
      console.error("[v2.20] Batch processing error:", error)
      alert(`Ошибка при пакетной обработке: ${error instanceof Error ? error.message : "Unknown error"}`)
    } finally {
      setProcessing(false)
    }
  }

  const progress = totalImages > 0 ? ((currentIndex + 1) / totalImages) * 100 : 0
  const totalFaces = results.reduce((sum, r) => sum + r.facesFound, 0)
  const totalRecognized = results.reduce((sum, r) => sum + r.facesRecognized, 0)

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold">Пакетная обработка фото [v2.20-SUPABASE]</h2>
        <p className="text-muted-foreground">Автоматическое распознавание лиц на всех фотографиях</p>
      </div>

      {!processing && results.length === 0 ? (
        <div className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Выберите галереи для обработки</CardTitle>
              <CardDescription>Система обработает все фотографии в выбранных галереях</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <label className="text-sm font-medium">
                  Порог уверенности: {(confidenceThreshold * 100).toFixed(0)}%
                </label>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={confidenceThreshold * 100}
                  onChange={(e) => setConfidenceThreshold(Number(e.target.value) / 100)}
                  className="w-full"
                />
                <p className="text-xs text-muted-foreground">
                  Лица с уверенностью ниже этого порога будут сохранены как неопознанные
                </p>
              </div>

              <div className="max-h-[400px] space-y-2 overflow-y-auto">
                {galleries.map((gallery) => (
                  <div key={gallery.id} className="flex items-center space-x-2 rounded-lg border p-3">
                    <Checkbox
                      id={gallery.id}
                      checked={selectedGalleries.includes(gallery.id)}
                      onCheckedChange={() => toggleGallery(gallery.id)}
                    />
                    <label htmlFor={gallery.id} className="flex-1 cursor-pointer text-sm font-medium leading-none">
                      {gallery.title} {gallery.shoot_date && `(${formatShortDate(gallery.shoot_date)})`}
                      <span className="ml-2 text-muted-foreground">({gallery._count?.gallery_images || 0} фото)</span>
                    </label>
                  </div>
                ))}
              </div>

              <div className="flex gap-2">
                <Button variant="outline" size="sm" onClick={selectAll}>
                  Выбрать все
                </Button>
                <Button variant="outline" size="sm" onClick={deselectAll}>
                  Снять выбор
                </Button>
              </div>

              <Button onClick={startBatchProcessing} disabled={selectedGalleries.length === 0} className="w-full">
                <Scan className="mr-2 h-4 w-4" />
                Начать обработку ({selectedGalleries.length} галерей)
              </Button>
            </CardContent>
          </Card>
        </div>
      ) : (
        <Card>
          <CardHeader>
            <CardTitle>Обработка фотографий</CardTitle>
            <CardDescription>{processing ? "Обработка в процессе..." : "Обработка завершена"}</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <div className="flex items-center justify-between text-sm">
                <span>Прогресс</span>
                <span>
                  {currentIndex + 1} / {totalImages}
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
              {results.map((result) => (
                <div key={result.imageId} className="flex items-center justify-between rounded-lg border p-3 text-sm">
                  <div className="flex-1 truncate">
                    <p className="truncate font-medium" title={result.filename}>
                      {result.filename}
                    </p>
                    <p className="text-xs text-muted-foreground">{result.galleryTitle}</p>
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
              <Button
                onClick={() => {
                  setResults([])
                  setSelectedGalleries([])
                }}
                className="w-full"
              >
                Начать новую обработку
              </Button>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  )
}
