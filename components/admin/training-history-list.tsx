"use client"

import { Badge } from "@/components/ui/badge"
import { CheckCircle2, XCircle, Loader2 } from "lucide-react"

interface TrainingSession {
  id: string
  created_at: string
  training_mode: string
  faces_count: number
  people_count: number
  metrics: {
    accuracy?: number
    precision?: number
    recall?: number
  }
  status: string
}

interface TrainingHistoryListProps {
  sessions: TrainingSession[]
}

export function TrainingHistoryList({ sessions }: TrainingHistoryListProps) {
  if (sessions.length === 0) {
    return (
      <div className="flex min-h-[200px] items-center justify-center text-sm text-muted-foreground">
        Нет истории обучений
      </div>
    )
  }

  return (
    <div className="space-y-2">
      {sessions.map((session) => (
        <div key={session.id} className="flex items-center justify-between rounded-lg border p-3">
          <div className="flex-1 space-y-1">
            <div className="flex items-center gap-2">
              <span className="text-sm font-medium">{new Date(session.created_at).toLocaleString("ru-RU")}</span>
              <Badge variant="outline" className="text-xs">
                {session.training_mode === "full" ? "Полное" : "Инкрементальное"}
              </Badge>
            </div>
            <div className="flex gap-4 text-xs text-muted-foreground">
              <span>Людей: {session.people_count}</span>
              <span>Лиц: {session.faces_count}</span>
              {session.metrics.accuracy && <span>Accuracy: {(session.metrics.accuracy * 100).toFixed(1)}%</span>}
            </div>
          </div>
          <div>
            {session.status === "completed" && (
              <Badge variant="default">
                <CheckCircle2 className="mr-1 h-3 w-3" />
                Завершено
              </Badge>
            )}
            {session.status === "running" && (
              <Badge variant="secondary">
                <Loader2 className="mr-1 h-3 w-3 animate-spin" />В процессе
              </Badge>
            )}
            {session.status === "failed" && (
              <Badge variant="destructive">
                <XCircle className="mr-1 h-3 w-3" />
                Ошибка
              </Badge>
            )}
          </div>
        </div>
      ))}
    </div>
  )
}
