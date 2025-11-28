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
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import { BarChart3 } from "lucide-react"
import { getRecognitionStatsAction } from "@/app/admin/actions"

export function RecognitionStatsDialog() {
  const [open, setOpen] = useState(false)
  const [loading, setLoading] = useState(false)
  const [stats, setStats] = useState<any>(null)

  const loadStats = async () => {
    setLoading(true)
    const result = await getRecognitionStatsAction()
    if (result.success && result.data) {
      setStats(result.data)
    }
    setLoading(false)
  }

  const handleOpen = (isOpen: boolean) => {
    setOpen(isOpen)
    if (isOpen && !stats) {
      loadStats()
    }
  }

  return (
    <Dialog open={open} onOpenChange={handleOpen}>
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>
            <DialogTrigger asChild>
              <Button variant="outline" size="sm">
                <BarChart3 className="h-4 w-4" />
              </Button>
            </DialogTrigger>
          </TooltipTrigger>
          <TooltipContent>
            <p>Статистика распознавания</p>
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
      <DialogContent className="max-w-4xl max-h-[80vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Статистика распознавания лиц</DialogTitle>
          <DialogDescription>Общая информация о накопленных данных для распознавания</DialogDescription>
        </DialogHeader>

        {loading && <div className="text-center py-8">Загрузка статистики...</div>}

        {stats && (
          <div className="space-y-6">
            {/* Summary */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="border rounded-lg p-4">
                <div className="text-sm text-muted-foreground">Всего людей</div>
                <div className="text-2xl font-bold">{stats.summary.totalPeople}</div>
              </div>
              <div className="border rounded-lg p-4">
                <div className="text-sm text-muted-foreground">Подтвержденных лиц</div>
                <div className="text-2xl font-bold">{stats.summary.totalVerifiedFaces}</div>
              </div>
              <div className="border rounded-lg p-4">
                <div className="text-sm text-muted-foreground">Высокая уверенность</div>
                <div className="text-2xl font-bold">{stats.summary.totalHighConfidenceFaces}</div>
              </div>
              <div className="border rounded-lg p-4">
                <div className="text-sm text-muted-foreground">Дескрипторов</div>
                <div className="text-2xl font-bold">{stats.summary.totalDescriptors}</div>
              </div>
            </div>

            {/* Per-person stats */}
            <div>
              <h3 className="text-lg font-semibold mb-3">Все игроки по количеству подтвержденных фото</h3>
              <div className="border rounded-lg overflow-hidden">
                <table className="w-full">
                  <thead className="bg-muted">
                    <tr>
                      <th className="text-left p-3 font-medium">Имя</th>
                      <th className="text-right p-3 font-medium">Подтверждено</th>
                      <th className="text-right p-3 font-medium">Высокая уверенность</th>
                      <th className="text-right p-3 font-medium">Дескрипторов</th>
                      <th className="text-right p-3 font-medium">Всего</th>
                    </tr>
                  </thead>
                  <tbody>
                    {stats.peopleStats.map((person: any, index: number) => {
                      const totalPhotos = person.verifiedPhotos + person.highConfidencePhotos
                      const hasDescriptorMismatch = person.descriptors !== totalPhotos

                      return (
                        <tr key={person.id} className={index % 2 === 0 ? "bg-background" : "bg-muted/50"}>
                          <td className="p-3">
                            {person.name}
                            {person.telegramName && (
                              <span className="text-sm text-muted-foreground ml-2">({person.telegramName})</span>
                            )}
                          </td>
                          <td className="text-right p-3">{person.verifiedPhotos}</td>
                          <td className="text-right p-3">{person.highConfidencePhotos}</td>
                          <td
                            className={`text-right p-3 ${hasDescriptorMismatch ? "text-red-600 dark:text-red-400 font-semibold" : ""}`}
                          >
                            {person.descriptors}
                          </td>
                          <td className="text-right p-3 font-semibold">{person.totalConfirmed}</td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Recommendations */}
            <div className="border-l-4 border-blue-500 bg-blue-50 dark:bg-blue-950 p-4 rounded">
              <h4 className="font-semibold mb-2">Рекомендации</h4>
              <ul className="text-sm space-y-1 list-disc list-inside">
                <li>Для улучшения качества распознавания подтвердите 15-20 фото для каждого активного игрока</li>
                <li>Выбирайте фото с разными углами, освещением и выражениями лица</li>
                <li>После накопления дескрипторов перезапустите пакетную обработку</li>
                <li>
                  Средняя точность:{" "}
                  {stats.summary.totalDescriptors > 0
                    ? Math.round((stats.summary.totalDescriptors / stats.summary.totalPeople) * 5)
                    : 0}
                  %
                </li>
              </ul>
            </div>
          </div>
        )}
      </DialogContent>
    </Dialog>
  )
}
