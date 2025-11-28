"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Trash2 } from "lucide-react"
import { cleanupUnverifiedFacesAction } from "@/app/admin/actions"
import { toast } from "sonner"
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

export function CleanupFacesButton() {
  const [isLoading, setIsLoading] = useState(false)

  const handleCleanup = async () => {
    setIsLoading(true)
    try {
      const result = await cleanupUnverifiedFacesAction()

      if (result.error) {
        toast.error(result.error)
      } else if (result.data) {
        const { before, after, deleted, inconsistentRecords } = result.data

        let message = `Удалено: ${deleted} записей\n`
        message += `До: ${before.total} всего (${before.verified} подтверждено)\n`
        message += `После: ${after.total} всего (${after.verified} подтверждено)`

        if (inconsistentRecords.length > 0) {
          message += `\n\nВНИМАНИЕ: Найдено ${inconsistentRecords.length} записей с verified=true но confidence != 1`
          console.log("[v0] Inconsistent records:", inconsistentRecords)
        }

        toast.success("Очистка завершена", {
          description: message,
          duration: 10000,
        })
      }
    } catch (error) {
      console.error("[v0] Error during cleanup:", error)
      toast.error("Ошибка при очистке базы данных")
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <AlertDialog>
      <AlertDialogTrigger asChild>
        <Button variant="destructive" size="sm" disabled={isLoading}>
          <Trash2 className="h-4 w-4 mr-2" />
          Очистить неподтвержденные лица
        </Button>
      </AlertDialogTrigger>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>Вы уверены?</AlertDialogTitle>
          <AlertDialogDescription>
            Это действие удалит ВСЕ неподтвержденные дескрипторы лиц из базы данных. Останутся только записи с ручным
            подтверждением администратора (verified=true).
            <br />
            <br />
            <strong>Это действие необратимо!</strong>
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel>Отмена</AlertDialogCancel>
          <AlertDialogAction onClick={handleCleanup} disabled={isLoading}>
            {isLoading ? "Очистка..." : "Удалить"}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  )
}
