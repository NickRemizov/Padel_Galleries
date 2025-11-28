"use client"

import { useEffect } from "react"
import { Button } from "@/components/ui/button"
import Link from "next/link"

export default function PlayerError({
  error,
  reset,
}: {
  error: Error & { digest?: string }
  reset: () => void
}) {
  useEffect(() => {
    console.error("[v0] Player page error:", error)
  }, [error])

  return (
    <div className="flex min-h-screen flex-col items-center justify-center bg-background px-4">
      <div className="mx-auto max-w-md text-center">
        <h1 className="mb-4 font-serif text-4xl font-bold text-foreground">Ошибка загрузки игрока</h1>
        <p className="mb-8 text-muted-foreground">
          Не удалось загрузить информацию об игроке. Возможно, профиль был удален или произошла ошибка.
        </p>
        <div className="flex gap-4 justify-center">
          <Button onClick={reset} variant="default">
            Попробовать снова
          </Button>
          <Button asChild variant="outline">
            <Link href="/players">К списку игроков</Link>
          </Button>
        </div>
      </div>
    </div>
  )
}
