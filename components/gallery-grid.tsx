"use client"

import { useState, useMemo } from "react"
import { GalleryCard } from "@/components/gallery-card"
import { Button } from "@/components/ui/button"
import type { Gallery } from "@/lib/types"

interface GalleryGridProps {
  galleries: Gallery[]
}

function createSafeClassName(name: string): string {
  return name
    .toLowerCase()
    .replace(/\s+/g, "-")
    .replace(/[^a-z0-9-]/g, "")
}

export function GalleryGrid({ galleries }: GalleryGridProps) {
  const [locationFilter, setLocationFilter] = useState("*")
  const [organizerFilter, setOrganizerFilter] = useState("*")

  const locations = Array.from(new Set(galleries.map((g) => g.locations?.name).filter(Boolean)))
  const organizers = Array.from(new Set(galleries.map((g) => g.organizers?.name).filter(Boolean)))

  const availableLocations = useMemo(() => {
    if (organizerFilter === "*") {
      return new Set(locations)
    }

    return new Set(
      galleries
        .filter((g) => g.organizers?.name === organizerFilter)
        .map((g) => g.locations?.name)
        .filter(Boolean),
    )
  }, [galleries, organizerFilter, locations])

  const availableOrganizers = useMemo(() => {
    if (locationFilter === "*") {
      return new Set(organizers)
    }

    return new Set(
      galleries
        .filter((g) => g.locations?.name === locationFilter)
        .map((g) => g.organizers?.name)
        .filter(Boolean),
    )
  }, [galleries, locationFilter, organizers])

  const filteredGalleries = useMemo(() => {
    return galleries.filter((gallery) => {
      const matchesLocation = locationFilter === "*" || gallery.locations?.name === locationFilter
      const matchesOrganizer = organizerFilter === "*" || gallery.organizers?.name === organizerFilter
      return matchesLocation && matchesOrganizer
    })
  }, [galleries, locationFilter, organizerFilter])

  if (galleries.length === 0) {
    return (
      <div className="flex min-h-[400px] items-center justify-center">
        <div className="text-center">
          <p className="text-xl text-muted-foreground">Пока нет галерей</p>
          <p className="mt-2 text-sm text-muted-foreground">Добавьте первую галерею через админ-панель</p>
        </div>
      </div>
    )
  }

  return (
    <div>
      {locations.length > 0 && (
        <div className="mb-4">
          <div className="flex flex-wrap justify-center gap-2">
            <Button
              variant={locationFilter === "*" ? "default" : "outline"}
              onClick={() => setLocationFilter("*")}
              className="rounded-full"
            >
              Все площадки
            </Button>
            {locations.map((location) => {
              const isAvailable = availableLocations.has(location)
              return (
                <Button
                  key={location}
                  variant={locationFilter === location ? "default" : "outline"}
                  onClick={() => setLocationFilter(location!)}
                  className="rounded-full"
                  disabled={!isAvailable}
                >
                  {location}
                </Button>
              )
            })}
          </div>
        </div>
      )}

      {organizers.length > 0 && (
        <div className="mb-8">
          <div className="flex flex-wrap justify-center gap-2">
            <Button
              variant={organizerFilter === "*" ? "default" : "outline"}
              onClick={() => setOrganizerFilter("*")}
              className="rounded-full"
            >
              Все игры
            </Button>
            {organizers.map((organizer) => {
              const isAvailable = availableOrganizers.has(organizer)
              return (
                <Button
                  key={organizer}
                  variant={organizerFilter === organizer ? "default" : "outline"}
                  onClick={() => setOrganizerFilter(organizer!)}
                  className="rounded-full"
                  disabled={!isAvailable}
                >
                  {organizer}
                </Button>
              )
            })}
          </div>
        </div>
      )}

      <div className="w-full px-[5%]">
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          {filteredGalleries.map((gallery) => (
            <div key={gallery.id} className="animate-in fade-in duration-300">
              <GalleryCard
                gallery={gallery}
                onOrganizerClick={setOrganizerFilter}
                onLocationClick={setLocationFilter}
              />
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
