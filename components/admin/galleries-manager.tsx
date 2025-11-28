"use client"

import { useState, useEffect, useMemo } from "react"
import { AddGalleryDialog } from "@/components/admin/add-gallery-dialog"
import { GalleryList } from "@/components/admin/gallery-list"
import { Button } from "@/components/ui/button"
import type { Gallery, Photographer, Location, Organizer } from "@/lib/types"

export function GalleriesManager() {
  const [galleries, setGalleries] = useState<Gallery[]>([])
  const [photographers, setPhotographers] = useState<Photographer[]>([])
  const [locations, setLocations] = useState<Location[]>([])
  const [organizers, setOrganizers] = useState<Organizer[]>([])
  const [sortBy, setSortBy] = useState<"created_at" | "shoot_date">("created_at")
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadData()
  }, [])

  const sortedGalleries = useMemo(() => {
    return [...galleries].sort((a, b) => {
      const dateA = sortBy === "shoot_date" ? a.shoot_date : a.created_at
      const dateB = sortBy === "shoot_date" ? b.shoot_date : b.created_at

      if (!dateA && !dateB) return 0
      if (!dateA) return 1
      if (!dateB) return -1

      return new Date(dateB).getTime() - new Date(dateA).getTime()
    })
  }, [galleries, sortBy])

  async function loadData() {
    setLoading(true)

    try {
      const [galleriesRes, photographersRes, locationsRes, organizersRes] = await Promise.all([
        fetch(`/api/data/galleries`),
        fetch(`/api/data/photographers`),
        fetch(`/api/data/locations`),
        fetch(`/api/data/organizers`),
      ])

      const [galleriesData, photographersData, locationsData, organizersData] = await Promise.all([
        galleriesRes.ok ? galleriesRes.json() : [],
        photographersRes.ok ? photographersRes.json() : [],
        locationsRes.ok ? locationsRes.json() : [],
        organizersRes.ok ? organizersRes.json() : [],
      ])

      setGalleries(galleriesData || [])
      setPhotographers(photographersData || [])
      setLocations(locationsData || [])
      setOrganizers(organizersData || [])
    } catch (error) {
      console.error("[v0] Failed to load data:", error)
    } finally {
      setLoading(false)
    }
  }

  function toggleSort() {
    setSortBy((prev) => (prev === "created_at" ? "shoot_date" : "created_at"))
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Управление галереями</h2>
          <p className="text-sm text-muted-foreground">Добавляйте, редактируйте и удаляйте галереи</p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" onClick={toggleSort} disabled={loading}>
            {sortBy === "created_at" ? "По дате добавления" : "По дате события"}
          </Button>
          <AddGalleryDialog
            photographers={photographers}
            locations={locations}
            organizers={organizers}
            onSuccess={loadData}
          />
        </div>
      </div>

      <GalleryList
        galleries={sortedGalleries}
        photographers={photographers}
        locations={locations}
        organizers={organizers}
        onDelete={loadData}
      />
    </div>
  )
}
