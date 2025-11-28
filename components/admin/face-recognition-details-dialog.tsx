"use client"

import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { useEffect, useRef } from "react"

interface FaceRecognitionDetailsDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  faces: DetailedFace[]
  imageUrl?: string // Added imageUrl to crop face previews
}

export interface DetailedFace {
  insightface_bbox: { x: number; y: number; width: number; height: number }
  size: number
  blur_score?: number
  detection_score: number
  recognition_confidence?: number
  embedding_quality?: number
  distance_to_nearest?: number
  top_matches?: Array<{
    person_id: string
    name: string
    similarity: number
  }>
  person_name?: string // Added person_name for recognized faces
  verified?: boolean // Added verified field
}

export function FaceRecognitionDetailsDialog({
  open,
  onOpenChange,
  faces,
  imageUrl,
}: FaceRecognitionDetailsDialogProps) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-5xl max-h-[80vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Детальная информация о распознавании</DialogTitle>
        </DialogHeader>

        <div className="space-y-6">
          {faces.map((face, index) => (
            <Card key={index} className={face.verified ? "border-green-500 border-2" : ""}>
              <CardHeader>
                <CardTitle className="flex items-center justify-between">
                  <span>{face.person_name || `Лицо ${index + 1}`}</span>
                  {face.verified && <span className="text-green-600 text-sm font-bold">✓ ВРУЧНУЮ ПОДТВЕРЖДЕНО</span>}
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex gap-6 items-start">
                  <div className="flex-shrink-0">
                    {imageUrl && <FacePreview imageUrl={imageUrl} insightface_bbox={face.insightface_bbox} />}
                  </div>

                  <div className="flex-1 grid grid-cols-2 gap-x-8 gap-y-4">
                    <div>
                      <p className="text-sm text-muted-foreground mb-1">Размер лица</p>
                      <p className="text-lg font-semibold">
                        {face.size !== undefined && face.size !== null ? face.size.toFixed(2) : "N/A"} px
                      </p>
                    </div>

                    <div>
                      <p className="text-sm text-muted-foreground mb-1">Blur score</p>
                      <p className="text-lg font-semibold">
                        {face.blur_score !== undefined && face.blur_score !== null ? face.blur_score.toFixed(1) : "N/A"}
                      </p>
                    </div>

                    <div>
                      <p className="text-sm text-muted-foreground mb-1">Detection score</p>
                      <p className="text-lg font-semibold">
                        {face.detection_score !== undefined && face.detection_score !== null
                          ? face.detection_score.toFixed(2)
                          : "N/A"}
                      </p>
                    </div>

                    <div>
                      <p className="text-sm text-muted-foreground mb-1">Recognition confidence</p>
                      <p className={`text-lg font-semibold ${face.verified ? "text-green-600" : ""}`}>
                        {face.verified
                          ? "100% (Verified)"
                          : face.recognition_confidence !== undefined &&
                              face.recognition_confidence !== null &&
                              face.recognition_confidence > 0
                            ? face.recognition_confidence >= 0.999
                              ? "Exact Match"
                              : face.recognition_confidence.toFixed(2)
                            : "Unknown face"}
                      </p>
                    </div>

                    <div>
                      <p className="text-sm text-muted-foreground mb-1">Качество эмбеддинга</p>
                      <p className="text-lg font-semibold">
                        {face.embedding_quality !== undefined && face.embedding_quality !== null
                          ? face.embedding_quality.toFixed(2)
                          : "N/A"}
                      </p>
                    </div>

                    <div>
                      <p className="text-sm text-muted-foreground mb-1">Расстояние до ближайшего</p>
                      <p className="text-lg font-semibold">
                        {face.distance_to_nearest !== undefined && face.distance_to_nearest !== null
                          ? face.distance_to_nearest < 0.001
                            ? "Exact Match"
                            : face.distance_to_nearest.toFixed(2)
                          : "N/A"}
                      </p>
                    </div>
                  </div>
                </div>

                {face.top_matches && face.top_matches.length > 0 && (
                  <div className="mt-6 pt-4 border-t">
                    <p className="text-sm text-muted-foreground mb-2 font-medium">Топ-3 похожих лиц:</p>
                    <ol className="list-decimal list-inside space-y-1 text-sm">
                      {face.top_matches.map((match, i) => (
                        <li key={i} className="text-foreground">
                          <span className="font-medium">{match.name}</span>
                          <span className="text-muted-foreground">
                            {" "}
                            (similarity:{" "}
                            {match.similarity !== undefined && match.similarity !== null
                              ? match.similarity.toFixed(2)
                              : "N/A"}
                            )
                          </span>
                        </li>
                      ))}
                    </ol>
                  </div>
                )}
              </CardContent>
            </Card>
          ))}
        </div>
      </DialogContent>
    </Dialog>
  )
}

function FacePreview({
  imageUrl,
  insightface_bbox,
}: { imageUrl: string; insightface_bbox: { x: number; y: number; width: number; height: number } }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    const img = new Image()
    img.crossOrigin = "anonymous"

    img.onload = () => {
      const padding = 0.2
      const paddedWidth = insightface_bbox.width * (1 + padding * 2)
      const paddedHeight = insightface_bbox.height * (1 + padding * 2)
      const paddedX = Math.max(0, insightface_bbox.x - insightface_bbox.width * padding)
      const paddedY = Math.max(0, insightface_bbox.y - insightface_bbox.height * padding)

      const cropX = Math.max(0, paddedX)
      const cropY = Math.max(0, paddedY)
      const cropWidth = Math.min(paddedWidth, img.width - cropX)
      const cropHeight = Math.min(paddedHeight, img.height - cropY)

      const previewSize = 150
      canvas.width = previewSize
      canvas.height = previewSize

      ctx.fillStyle = "#000000"
      ctx.fillRect(0, 0, previewSize, previewSize)

      const scale = Math.min(previewSize / cropWidth, previewSize / cropHeight)
      const scaledWidth = cropWidth * scale
      const scaledHeight = cropHeight * scale

      const offsetX = (previewSize - scaledWidth) / 2
      const offsetY = (previewSize - scaledHeight) / 2

      ctx.drawImage(img, cropX, cropY, cropWidth, cropHeight, offsetX, offsetY, scaledWidth, scaledHeight)
    }

    img.src = imageUrl
  }, [imageUrl, insightface_bbox])

  return <canvas ref={canvasRef} className="border rounded-lg" style={{ width: 150, height: 150 }} />
}
