"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { Loader2, Sparkles } from "lucide-react"
import { generateMissingDescriptorsAction } from "@/app/admin/actions"
import { useToast } from "@/hooks/use-toast"

export function GenerateMissingDescriptorsDialog() {
  const [open, setOpen] = useState(false)
  const [isGenerating, setIsGenerating] = useState(false)
  const { toast } = useToast()

  const handleGenerate = async () => {
    setIsGenerating(true)

    try {
      const result = await generateMissingDescriptorsAction()

      if (result.error) {
        toast({
          title: "Ошибка",
          description: result.error,
          variant: "destructive",
        })
      } else {
        toast({
          title: "Дескрипторы сгенерированы",
          description: result.message,
        })
        setOpen(false)
      }
    } catch (error: any) {
      toast({
        title: "Ошибка",
        description: error.message || "Не удалось сгенерировать дескрипторы",
        variant: "destructive",
      })
    } finally {
      setIsGenerating(false)
    }
  }

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button variant="outline" size="sm">
          <Sparkles className="mr-2 h-4 w-4" />
          Сгенерировать недостающие дескрипторы
        </Button>
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Генерация недостающих дескрипторов</DialogTitle>
          <DialogDescription>
            Эта операция найдет все верифицированные лица без дескрипторов и автоматически создаст для них дескрипторы.
            Это может занять некоторое время.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          <div className="rounded-lg border p-4 space-y-2">
            <p className="text-sm text-muted-foreground">
              Будут обработаны все верифицированные лица, у которых отсутствуют дескрипторы в таблице face_descriptors.
            </p>
            <p className="text-sm text-muted-foreground">
              После генерации дескрипторов эти люди смогут распознаваться на других фото.
            </p>
          </div>

          <div className="flex justify-end gap-2">
            <Button variant="outline" onClick={() => setOpen(false)} disabled={isGenerating}>
              Отмена
            </Button>
            <Button onClick={handleGenerate} disabled={isGenerating}>
              {isGenerating && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
              {isGenerating ? "Генерация..." : "Сгенерировать"}
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}
