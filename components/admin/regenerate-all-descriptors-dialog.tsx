"use client"

import { useState, useEffect } from "react"
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
import { Progress } from "@/components/ui/progress"
import { Loader2, RefreshCw } from "lucide-react"
import { getPhotosWithExcessDescriptorsAction, saveFaceDescriptorAction } from "@/app/admin/actions"

// Face detection will be handled by backend API instead

export function RegenerateAllDescriptorsDialog() {
  const [open, setOpen] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)
  const [progress, setProgress] = useState({ current: 0, total: 0 })
  const [result, setResult] = useState<string | null>(null)
  const [peopleStats, setPeopleStats] = useState<{ personName: string; photoCount: number }[]>([])

  useEffect(() => {
    if (open) {
      setResult(null)
      setProgress({ current: 0, total: 0 })
      setPeopleStats([])
    }
  }, [open])

  const handleRegenerate = async () => {
    console.log("[v0] ========== REGENERATE ALL BUTTON CLICKED ==========")

    setIsProcessing(true)
    setResult(null)
    setProgress({ current: 0, total: 0 })
    setPeopleStats([])

    try {
      setResult("Запуск регенерации дескрипторов на сервере...")

      // Get photos with excess descriptors
      console.log("[v0] Getting photos with excess descriptors...")
      const response = await getPhotosWithExcessDescriptorsAction()

      if (!response.success || !response.data) {
        setResult(`Ошибка: ${response.error || "Не удалось получить данные"}`)
        return
      }

      const { facesToProcess, peopleStats: stats, totalFaces } = response.data
      console.log(`[v0] Found ${totalFaces} faces to process`)

      if (totalFaces === 0) {
        setResult("Нет фото с избыточными дескрипторами")
        return
      }

      setProgress({ current: 0, total: totalFaces })

      let successCount = 0
      let errorCount = 0
      const processedByPerson = new Map<string, number>()

      for (let i = 0; i < facesToProcess.length; i++) {
        const face = facesToProcess[i]
        setProgress({ current: i + 1, total: totalFaces })

        try {
          console.log(`[v0] Processing ${i + 1}/${totalFaces}: ${face.filename}`)

          // Call backend API to detect and save descriptor
          const detectResponse = await fetch("/api/face-detection/detect", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              imageUrl: face.imageUrl,
              boundingBox: face.boundingBox,
            }),
          })

          if (!detectResponse.ok) {
            console.log(`[v0] Detection failed for ${face.filename}`)
            errorCount++
            continue
          }

          const detectData = await detectResponse.json()

          if (!detectData.descriptor) {
            console.log(`[v0] No descriptor returned for ${face.filename}`)
            errorCount++
            continue
          }

          // Save descriptor
          const saveResult = await saveFaceDescriptorAction(face.personId, detectData.descriptor, face.photoId)

          if (saveResult.success) {
            successCount++
            processedByPerson.set(face.personId, (processedByPerson.get(face.personId) || 0) + 1)
          } else {
            errorCount++
          }
        } catch (error) {
          console.error(`[v0] Error processing ${face.filename}:`, error)
          errorCount++
        }
      }

      // Build final report
      const finalStats = stats.map((stat) => ({
        personName: stat.personName,
        photoCount: processedByPerson.get(stat.personId) || 0,
      }))

      setPeopleStats(finalStats)
      setResult(`Регенерация завершена:\n✓ Успешно: ${successCount}\n✗ Ошибок: ${errorCount}\n📊 Всего: ${totalFaces}`)
      console.log(`[v0] Regeneration complete: ${successCount} success, ${errorCount} errors`)
    } catch (error: any) {
      console.error("[v0] Error during regeneration:", error)
      setResult(`Ошибка: ${error.message || "Неизвестная ошибка"}`)
    } finally {
      setIsProcessing(false)
      console.log("[v0] ========== REGENERATE ALL COMPLETE ==========")
    }
  }

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button variant="outline" size="sm">
          <RefreshCw className="mr-2 h-4 w-4" />
          Регенерировать все дескрипторы
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-[600px]">
        <DialogHeader>
          <DialogTitle>Регенерация всех дескрипторов</DialogTitle>
          <DialogDescription>
            Регенерирует дескрипторы только на тех фото, где количество дескрипторов превышает количество отмеченных
            людей
          </DialogDescription>
        </DialogHeader>
        <div className="grid gap-4 py-4">
          {isProcessing && (
            <div className="space-y-2">
              <div className="text-sm text-muted-foreground">
                Обработка: {progress.current} из {progress.total}
              </div>
              <Progress value={(progress.current / progress.total) * 100} />
            </div>
          )}
          {result && <div className="rounded-md bg-muted p-3 text-sm whitespace-pre-line">{result}</div>}
          {peopleStats.length > 0 && (
            <div className="rounded-md border">
              <div className="p-3 border-b bg-muted/50">
                <h4 className="text-sm font-medium">Отчет по людям</h4>
              </div>
              <div className="max-h-[200px] overflow-y-auto">
                {peopleStats.map((stat, index) => (
                  <div key={index} className="flex justify-between p-3 border-b last:border-0">
                    <span className="text-sm">{stat.personName}</span>
                    <span className="text-sm font-medium">{stat.photoCount} фото</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={() => setOpen(false)} disabled={isProcessing}>
            Закрыть
          </Button>
          <Button onClick={handleRegenerate} disabled={isProcessing}>
            {isProcessing && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
            Начать регенерацию
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
