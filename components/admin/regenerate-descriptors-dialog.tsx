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
} from "@/components/ui/dialog"
import { Loader2 } from "lucide-react"
import { regeneratePersonDescriptorsAction } from "@/app/admin/actions"

interface RegenerateDescriptorsDialogProps {
  personId: string
  personName: string
  open: boolean
  onOpenChange: (open: boolean) => void
}

export function RegenerateDescriptorsDialog({
  personId,
  personName,
  open,
  onOpenChange,
}: RegenerateDescriptorsDialogProps) {
  const [isProcessing, setIsProcessing] = useState(false)
  const [result, setResult] = useState<string | null>(null)

  useEffect(() => {
    if (open) {
      setResult(null)
    }
  }, [open])

  const handleRegenerate = async () => {
    console.log("[v0] Regenerating descriptors for person:", personId)

    setIsProcessing(true)
    setResult(null)

    try {
      const response = await regeneratePersonDescriptorsAction(personId)
      console.log("[v0] Response received:", response)

      if (!response.success) {
        setResult(`Ошибка: ${response.error || "Не удалось регенерировать дескрипторы"}`)
        return
      }

      setResult(`Регенерация завершена для ${personName}`)
    } catch (error: any) {
      console.error("[v0] Error during regeneration:", error)
      setResult(`Ошибка: ${error.message || "Неизвестная ошибка"}`)
    } finally {
      setIsProcessing(false)
    }
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[500px]">
        <DialogHeader>
          <DialogTitle>Регенерация дескрипторов лиц</DialogTitle>
          <DialogDescription>
            Удалит все существующие дескрипторы для <strong>{personName}</strong> и создаст новые через FastAPI
          </DialogDescription>
        </DialogHeader>
        <div className="grid gap-4 py-4">
          {result && <div className="rounded-md bg-muted p-3 text-sm whitespace-pre-line">{result}</div>}
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)} disabled={isProcessing}>
            Закрыть
          </Button>
          <Button onClick={handleRegenerate} disabled={isProcessing}>
            {isProcessing && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
            Регенерировать
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
