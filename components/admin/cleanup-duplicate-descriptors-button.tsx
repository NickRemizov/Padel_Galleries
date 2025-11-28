"use client"

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
import { cleanupDuplicateDescriptorsAction } from "@/app/admin/actions"
import { Loader2 } from "lucide-react"

export function CleanupDuplicateDescriptorsButton() {
  const [open, setOpen] = useState(false)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<any>(null)

  const handleCleanup = async () => {
    setLoading(true)
    setResult(null)

    try {
      const response = await cleanupDuplicateDescriptorsAction()

      if (response.error) {
        alert(`Ошибка: ${response.error}`)
      } else {
        setResult(response.data)
        console.log("[v0] Cleanup completed successfully:", response.data)
      }
    } catch (error: any) {
      console.error("[v0] Error during cleanup:", error)
      alert(`Ошибка: ${error.message}`)
    } finally {
      setLoading(false)
    }
  }

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button variant="outline" size="sm">
          Удалить дубликаты дескрипторов
        </Button>
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Удалить дубликаты дескрипторов</DialogTitle>
          <DialogDescription>
            Эта операция удалит дублирующие записи в таблице face_descriptors, оставив только один дескриптор для каждой
            комбинации человек+фото (самый старый по дате создания).
          </DialogDescription>
        </DialogHeader>

        {result && (
          <div className="space-y-2 rounded-lg bg-muted p-4 text-sm">
            <div className="font-semibold">Результаты очистки:</div>
            <div>До очистки: {result.before.descriptors} дескрипторов</div>
            <div>После очистки: {result.after.descriptors} дескрипторов</div>
            <div>Удалено дубликатов: {result.deletedDescriptors}</div>
            <div>Групп с дубликатами: {result.duplicateGroups}</div>
            <div className="mt-2 text-muted-foreground">
              Теперь в базе: {result.after.photoFaces} записей photo_faces и {result.after.descriptors} дескрипторов
            </div>
          </div>
        )}

        <DialogFooter>
          <Button variant="outline" onClick={() => setOpen(false)} disabled={loading}>
            Отмена
          </Button>
          <Button onClick={handleCleanup} disabled={loading} variant="destructive">
            {loading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
            {loading ? "Удаление..." : "Удалить дубликаты"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
