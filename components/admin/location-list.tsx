"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Trash2, Pencil } from "lucide-react"
import { deleteLocationAction } from "@/app/admin/actions"
import { EditLocationDialog } from "./edit-location-dialog"
import type { Location } from "@/lib/types"

interface LocationListProps {
  locations: Location[]
}

export function LocationList({ locations }: LocationListProps) {
  const [deletingId, setDeletingId] = useState<string | null>(null)
  const [editingLocation, setEditingLocation] = useState<Location | null>(null)

  async function handleDelete(id: string) {
    if (!confirm("Вы уверены, что хотите удалить это место?")) return

    setDeletingId(id)
    await deleteLocationAction(id)
    setDeletingId(null)
  }

  if (locations.length === 0) {
    return (
      <Card>
        <CardContent className="flex min-h-[200px] items-center justify-center">
          <p className="text-muted-foreground">Нет мест</p>
        </CardContent>
      </Card>
    )
  }

  return (
    <>
      <div className="grid gap-2">
        {locations.map((location) => (
          <Card key={location.id}>
            <CardContent className="flex items-center justify-between p-4">
              <span className="font-medium">{location.name}</span>
              <div className="flex gap-2">
                <Button variant="outline" size="sm" onClick={() => setEditingLocation(location)}>
                  <Pencil className="h-4 w-4" />
                </Button>
                <Button
                  variant="destructive"
                  size="sm"
                  onClick={() => handleDelete(location.id)}
                  disabled={deletingId === location.id}
                >
                  <Trash2 className="h-4 w-4" />
                </Button>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {editingLocation && (
        <EditLocationDialog
          location={editingLocation}
          open={!!editingLocation}
          onOpenChange={(open) => !open && setEditingLocation(null)}
        />
      )}
    </>
  )
}
