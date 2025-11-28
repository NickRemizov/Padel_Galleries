"use client"

import { useEffect, useRef } from "react"
import { PlayerCard } from "@/components/player-card"
import type { Person } from "@/lib/types"

interface PlayersGridProps {
  players: Person[]
}

export function PlayersGrid({ players }: PlayersGridProps) {
  const gridRef = useRef<HTMLDivElement>(null)
  const isotopeRef = useRef<any>(null)

  useEffect(() => {
    const initIsotope = async () => {
      if (typeof window === "undefined" || !gridRef.current) return

      const Isotope = (await import("isotope-layout")).default

      isotopeRef.current = new Isotope(gridRef.current, {
        itemSelector: ".player-item",
        layoutMode: "masonry",
        masonry: {
          columnWidth: ".player-item-sizer",
          gutter: 16,
        },
        percentPosition: true,
        transitionDuration: "0.4s",
        hiddenStyle: {
          opacity: 0,
        },
        visibleStyle: {
          opacity: 1,
        },
      })
    }

    initIsotope()

    return () => {
      if (isotopeRef.current) {
        isotopeRef.current.destroy()
      }
    }
  }, [players.length])

  if (players.length === 0) {
    return (
      <div className="flex min-h-[400px] items-center justify-center">
        <div className="text-center">
          <p className="text-xl text-muted-foreground">Пока нет игроков</p>
          <p className="mt-2 text-sm text-muted-foreground">Добавьте игроков через админ-панель</p>
        </div>
      </div>
    )
  }

  return (
    <div className="w-full px-[5%]">
      <div ref={gridRef} className="isotope-grid">
        <div className="player-item-sizer w-[calc(50%-8px)] md:w-[calc(25%-12px)] lg:w-[calc(20%-13px)] xl:w-[calc(16.666%-13px)] 2xl:w-[calc(12.5%-14px)]"></div>
        {players.map((player) => (
          <div
            key={player.id}
            className="player-item mb-4 w-[calc(50%-8px)] md:w-[calc(25%-12px)] lg:w-[calc(20%-13px)] xl:w-[calc(16.666%-13px)] 2xl:w-[calc(12.5%-14px)]"
          >
            <PlayerCard player={player} />
          </div>
        ))}
      </div>
    </div>
  )
}
