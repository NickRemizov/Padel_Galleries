"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { cleanupDuplicateFacesAction } from "@/app/admin/actions"
import { useToast } from "@/hooks/use-toast"
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog"
import { Loader2 } from "lucide-react"

export function CleanupDuplicatesButton() {
  const [isLoading, setIsLoading] = useState(false)
  const { toast } = useToast()

  const handleCleanup = async () => {
    setIsLoading(true)
    try {
      const result = await cleanupDuplicateFacesAction()

      if (result.error) {
        toast({
          title: "Ошибка",
          description: result.error,
          variant: "destructive",
        })
      } else if (result.data) {
        toast({
          title: "Очистка завершена",
          description: `Удалено ${result.data.deleted} дубликатов из ${result.data.duplicateGroups} групп. Осталось ${result.data.after.total} записей (${result.data.after.verified} подтверждено).`,
        })
      }
    } catch (error) {
      toast({
        title: "Ошибка",
        description: "Не удалось выполнить очистку",
        variant: "destructive",
      })
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <AlertDialog>
      <AlertDialogTrigger asChild>
        <Button variant="outline" disabled={isLoading}>
          {isLoading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
          Удалить дубликаты
        </Button>
      </AlertDialogTrigger>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>Удалить дубликаты дескрипторов?</AlertDialogTitle>
          <AlertDialogDescription>
            Эта операция удалит все дублирующие записи в таблице photo_faces (несколько записей для одной комбинации
            человек+фото). Для каждой группы дубликатов будет сохранена лучшая запись (с verified=true или наибольшей
            уверенностью).
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel>Отмена</AlertDialogCancel>
          <AlertDialogAction onClick={handleCleanup}>Удалить дубликаты</AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  )
}
