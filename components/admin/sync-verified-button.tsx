"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { syncVerifiedAndConfidenceAction } from "@/app/admin/actions"
import { Loader2, RefreshCw } from "lucide-react"
import { useToast } from "@/hooks/use-toast"

export function SyncVerifiedButton() {
  const [isLoading, setIsLoading] = useState(false)
  const { toast } = useToast()

  const handleSync = async () => {
    setIsLoading(true)
    try {
      const result = await syncVerifiedAndConfidenceAction()

      if (result.error) {
        toast({
          title: "Ошибка",
          description: result.error,
          variant: "destructive",
        })
      } else {
        toast({
          title: "Синхронизация завершена",
          description: `Обновлено: ${result.data?.updatedVerified} verified→confidence, ${result.data?.updatedConfidence} confidence→verified`,
        })
      }
    } catch (error) {
      toast({
        title: "Ошибка",
        description: "Не удалось выполнить синхронизацию",
        variant: "destructive",
      })
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <Button onClick={handleSync} disabled={isLoading} variant="outline" size="sm">
      {isLoading ? (
        <>
          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
          Синхронизация...
        </>
      ) : (
        <>
          <RefreshCw className="mr-2 h-4 w-4" />
          Синхронизировать verified/confidence
        </>
      )}
    </Button>
  )
}
