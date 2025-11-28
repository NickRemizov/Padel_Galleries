"use client"

import { useState, useEffect } from "react"
import Image from "next/image"
import { Button } from "@/components/ui/button"
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Card } from "@/components/ui/card"
import { UserPlus, X, Loader2, RefreshCw, ChevronLeft, ChevronRight, SkipForward } from "lucide-react"
import {
  getUnknownFaceClustersAction,
  rejectFaceClusterAction,
  assignClusterToPersonAction,
  regenerateUnknownDescriptorsAction,
} from "@/app/admin/actions"
import { AddPersonDialog } from "./add-person-dialog"
import type { UnknownFaceCluster, UnknownFace } from "@/lib/types"
import { toast } from "sonner"

interface UnknownFacesReviewDialogProps {
  galleryId: string
  open: boolean
  onOpenChange: (open: boolean) => void
  onComplete?: () => void
}

export function UnknownFacesReviewDialog({ galleryId, open, onOpenChange, onComplete }: UnknownFacesReviewDialogProps) {
  const [clusters, setClusters] = useState<UnknownFaceCluster[]>([])
  const [currentIndex, setCurrentIndex] = useState(0)
  const [loading, setLoading] = useState(false)
  const [processing, setProcessing] = useState(false)
  const [regenerating, setRegenerating] = useState(false)
  const [showAddPerson, setShowAddPerson] = useState(false)
  const [excludedFaces, setExcludedFaces] = useState<Set<string>>(new Set())

  useEffect(() => {
    if (open) {
      loadClusters()
    }
  }, [open, galleryId])

  useEffect(() => {
    setExcludedFaces(new Set())
  }, [currentIndex])

  async function loadClusters() {
    setLoading(true)
    const result = await getUnknownFaceClustersAction(galleryId)
    if (result.success && result.data) {
      const sortedClusters = result.data.sort(
        (a: UnknownFaceCluster, b: UnknownFaceCluster) => b.faces.length - a.faces.length,
      )
      setClusters(sortedClusters)
      setCurrentIndex(0)
    }
    setLoading(false)
  }

  async function handleRegenerateDescriptors() {
    if (!confirm("Регенерировать дескрипторы для всех неизвестных лиц без дескрипторов? Это может занять время.")) {
      return
    }

    setRegenerating(true)
    toast.info("Начинаем регенерацию дескрипторов...")

    try {
      const result = await regenerateUnknownDescriptorsAction(galleryId)

      if (result.success && result.data) {
        const { totalFaces, regenerated, failed, alreadyHadDescriptor } = result.data

        toast.success(
          `Регенерация завершена!\n` +
            `Всего лиц: ${totalFaces}\n` +
            `Регенерировано: ${regenerated}\n` +
            `Уже были дескрипторы: ${alreadyHadDescriptor}\n` +
            `Ошибок: ${failed}`,
        )

        await loadClusters()
      } else {
        toast.error(result.error || "Ошибка при регенерации дескрипторов")
      }
    } catch (error: any) {
      console.error("[v0] Error regenerating descriptors:", error)
      toast.error("Ошибка при регенерации дескрипторов")
    } finally {
      setRegenerating(false)
    }
  }

  function toggleExcludeFace(photoId: string) {
    setExcludedFaces((prev) => {
      const newSet = new Set(prev)
      if (newSet.has(photoId)) {
        newSet.delete(photoId)
      } else {
        newSet.add(photoId)
      }
      return newSet
    })
  }

  function getActiveFaces(cluster: UnknownFaceCluster): UnknownFace[] {
    return cluster.faces.filter((face) => !excludedFaces.has(face.photo_id))
  }

  async function handleReject() {
    const cluster = clusters[currentIndex]
    const activeFaces = getActiveFaces(cluster)

    if (activeFaces.length === 0) {
      toast.error("Все фото исключены из кластера")
      return
    }

    if (!confirm(`Отклонить ${activeFaces.length} фото этого человека?`)) return

    setProcessing(true)
    const faces = activeFaces.map((face) => ({
      photo_id: face.photo_id,
      descriptor: face.descriptor,
    }))

    const result = await rejectFaceClusterAction(faces)
    if (result.success) {
      if (currentIndex < clusters.length - 1) {
        setCurrentIndex(currentIndex + 1)
      } else {
        handleClose()
      }
    } else {
      toast.error("Ошибка при отклонении лиц")
    }
    setProcessing(false)
  }

  function handleOpenAddPerson() {
    const cluster = clusters[currentIndex]
    const activeFaces = getActiveFaces(cluster)

    if (activeFaces.length === 0) {
      toast.error("Все фото исключены из кластера")
      return
    }

    setShowAddPerson(true)
  }

  async function handlePersonCreated(personId: string, personName: string) {
    console.log("[v0] handlePersonCreated called with:", { personId, personName })

    const cluster = clusters[currentIndex]
    const activeFaces = getActiveFaces(cluster)

    console.log("[v0] Active faces count:", activeFaces.length)
    console.log(
      "[v0] Active faces:",
      activeFaces.map((f) => ({ photo_id: f.photo_id, has_descriptor: !!f.descriptor })),
    )

    setProcessing(true)
    const faces = activeFaces.map((face) => ({
      photo_id: face.photo_id,
      descriptor: face.descriptor,
    }))

    console.log("[v0] Calling assignClusterToPersonAction with:", { personId, facesCount: faces.length })

    const result = await assignClusterToPersonAction(personId, faces)

    console.log("[v0] assignClusterToPersonAction result:", result)

    if (result.success) {
      toast.success(`Игрок "${personName}" создан и назначен ${activeFaces.length} фото`)
      if (currentIndex < clusters.length - 1) {
        setCurrentIndex(currentIndex + 1)
      } else {
        handleClose()
      }
    } else {
      toast.error("Ошибка при назначении фото игроку")
    }
    setProcessing(false)
  }

  function handleSkip() {
    if (currentIndex < clusters.length - 1) {
      setCurrentIndex(currentIndex + 1)
    } else {
      handleClose()
    }
  }

  function handlePrevious() {
    if (currentIndex > 0) {
      setCurrentIndex(currentIndex - 1)
    }
  }

  function handleClose() {
    onOpenChange(false)
    if (onComplete) {
      onComplete()
    }
  }

  const currentCluster = clusters[currentIndex]
  const activeFaces = currentCluster ? getActiveFaces(currentCluster) : []

  return (
    <>
      <Dialog open={open} onOpenChange={handleClose}>
        <DialogContent className="max-w-[95vw] w-full h-[95vh] flex flex-col">
          <DialogHeader>
            <div className="flex items-center justify-between">
              <div>
                <DialogTitle>Неизвестные лица</DialogTitle>
                <DialogDescription>
                  {clusters.length > 0
                    ? `Кластер ${currentIndex + 1} из ${clusters.length}`
                    : "Сгруппированы по схожести"}
                </DialogDescription>
              </div>
              {clusters.length === 0 && (
                <Button variant="outline" size="sm" onClick={handleRegenerateDescriptors} disabled={regenerating}>
                  {regenerating ? (
                    <>
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                      Регенерация...
                    </>
                  ) : (
                    <>
                      <RefreshCw className="h-4 w-4 mr-2" />
                      Регенерировать дескрипторы
                    </>
                  )}
                </Button>
              )}
            </div>
          </DialogHeader>

          {loading ? (
            <div className="flex items-center justify-center flex-1">
              <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
            </div>
          ) : clusters.length === 0 ? (
            <Card className="flex-1 flex flex-col items-center justify-center gap-4 p-6">
              <p className="text-center text-muted-foreground">Нет неизвестных лиц для обработки</p>
              <p className="text-center text-sm text-muted-foreground">
                Сначала запустите распознавание фото через кнопку "Распознать фото"
              </p>
            </Card>
          ) : currentCluster ? (
            <div className="flex-1 flex flex-col gap-4 overflow-hidden">
              <div className="flex-1 overflow-y-auto">
                <div className="grid grid-cols-4 gap-4 p-2">
                  {currentCluster.faces.map((face, idx) => {
                    const isExcluded = excludedFaces.has(face.photo_id)
                    const objectPosition = face.bbox
                      ? `${(face.bbox.x + face.bbox.width / 2) * 100}% ${(face.bbox.y + face.bbox.height / 2) * 100}%`
                      : "center"

                    return (
                      <div
                        key={idx}
                        className={`relative aspect-square rounded-lg overflow-hidden border-2 transition-all ${
                          isExcluded ? "border-red-500 opacity-50" : "border-border hover:border-primary"
                        }`}
                      >
                        <Image
                          src={face.photo_url || "/placeholder.svg"}
                          alt={`Face ${idx + 1}`}
                          fill
                          className="object-cover"
                          style={{ objectPosition }}
                        />
                        {face.bbox && (
                          <div
                            className="absolute border-2 border-blue-500"
                            style={{
                              left: `${face.bbox.x * 100}%`,
                              top: `${face.bbox.y * 100}%`,
                              width: `${face.bbox.width * 100}%`,
                              height: `${face.bbox.height * 100}%`,
                            }}
                          />
                        )}
                        <button
                          onClick={() => toggleExcludeFace(face.photo_id)}
                          className={`absolute top-2 right-2 p-1 rounded-full ${
                            isExcluded ? "bg-red-500 hover:bg-red-600" : "bg-black/50 hover:bg-black/70"
                          } text-white transition-colors`}
                          title={isExcluded ? "Вернуть фото" : "Исключить фото"}
                        >
                          <X className="h-4 w-4" />
                        </button>
                        {isExcluded && (
                          <div className="absolute inset-0 flex items-center justify-center bg-black/50">
                            <span className="text-white font-semibold">Исключено</span>
                          </div>
                        )}
                      </div>
                    )
                  })}
                </div>
              </div>

              <div className="border-t pt-4 space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">
                    Всего фото: {currentCluster.faces.length} | Активных: {activeFaces.length} | Исключено:{" "}
                    {excludedFaces.size}
                  </span>
                </div>

                <div className="flex gap-2 justify-end">
                  <Button onClick={handleOpenAddPerson} disabled={processing || activeFaces.length === 0}>
                    {processing ? (
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    ) : (
                      <UserPlus className="h-4 w-4 mr-2" />
                    )}
                    Создать игрока
                  </Button>
                  <Button
                    variant="destructive"
                    onClick={handleReject}
                    disabled={processing || activeFaces.length === 0}
                  >
                    <X className="h-4 w-4 mr-2" />
                    Отклонить
                  </Button>
                  <Button variant="outline" onClick={handleSkip} disabled={processing}>
                    <SkipForward className="h-4 w-4 mr-2" />
                    Пропустить
                  </Button>
                </div>

                <div className="flex justify-between items-center pt-2 border-t">
                  <Button variant="outline" onClick={handlePrevious} disabled={currentIndex === 0 || processing}>
                    <ChevronLeft className="h-4 w-4 mr-2" />
                    Предыдущий
                  </Button>

                  <span className="text-sm text-muted-foreground">
                    Кластер {currentIndex + 1} из {clusters.length}
                  </span>

                  <Button
                    variant="outline"
                    onClick={handleSkip}
                    disabled={currentIndex === clusters.length - 1 || processing}
                  >
                    Следующий
                    <ChevronRight className="h-4 w-4 ml-2" />
                  </Button>
                </div>
              </div>
            </div>
          ) : null}
        </DialogContent>
      </Dialog>

      {showAddPerson && (
        <AddPersonDialogWithCallback
          open={showAddPerson}
          onOpenChange={setShowAddPerson}
          onPersonCreated={handlePersonCreated}
        />
      )}
    </>
  )
}

function AddPersonDialogWithCallback({
  open,
  onOpenChange,
  onPersonCreated,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  onPersonCreated: (personId: string, personName: string) => void
}) {
  return <AddPersonDialog open={open} onOpenChange={onOpenChange} onPersonCreated={onPersonCreated} />
}
