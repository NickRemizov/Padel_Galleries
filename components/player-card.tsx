"use client"

import Image from "next/image"
import { useRouter } from "next/navigation"
import { Card, CardContent } from "@/components/ui/card"
import type { Person } from "@/lib/types"
import { ExternalLink, ImageIcon } from "lucide-react"

interface PlayerCardProps {
  player: Person
}

export function PlayerCard({ player }: PlayerCardProps) {
  const router = useRouter()

  const handleCardClick = () => {
    router.push(`/players/${player.id}`)
  }

  const nameParts = player.real_name.trim().split(/\s+/)
  const firstName = nameParts[0] || ""
  const lastName = nameParts.slice(1).join(" ") || "\u00A0"

  const photoCount = player._count?.photo_faces || 0

  return (
    <div onClick={handleCardClick} className="group cursor-pointer">
      <Card className="overflow-hidden transition-all duration-300 hover:shadow-xl mx-0 my-0 bg-slate-200 p-0 leading-6">
        <div className="relative aspect-[3/4] overflow-hidden bg-muted">
          <Image
            src={player.avatar_url || "/placeholder.svg?height=400&width=400"}
            alt={player.real_name}
            fill
            className="object-contain transition-transform duration-500 group-hover:scale-105 border-0 mx-0 my-0"
            sizes="(max-width: 640px) 50vw, (max-width: 768px) 33vw, (max-width: 1024px) 25vw, (max-width: 1280px) 16.666vw, 12.5vw"
          />
          {photoCount > 0 && (
            <div className="absolute top-1 left-1 sm:top-2 sm:left-2 flex items-center gap-1 sm:gap-1.5 bg-black/30 text-white px-1.5 py-0.5 sm:px-2 sm:py-1 rounded-md backdrop-blur-sm">
              <ImageIcon className="h-2.5 w-2.5 sm:h-3 sm:w-3" />
              <span className="text-[10px] sm:text-xs font-medium">{photoCount} фото</span>
            </div>
          )}
        </div>

        <CardContent className="px-3 sm:px-4 pt-0 pb-3 sm:pb-4">
          <div className="mb-1">
            <h3 className="font-semibold text-lg sm:text-xl leading-tight">{firstName}</h3>
            <h3 className="font-semibold text-lg sm:text-xl leading-tight">{lastName}</h3>
          </div>

          {player.telegram_nickname ? (
            <a
              href={player.telegram_profile_url || `https://t.me/${player.telegram_nickname.replace("@", "")}`}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground transition-colors"
              onClick={(e) => e.stopPropagation()}
            >
              <span className="underline decoration-dotted">{player.telegram_nickname}</span>
              <ExternalLink className="h-3 w-3" />
            </a>
          ) : (
            <div className="text-sm text-muted-foreground">@</div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
