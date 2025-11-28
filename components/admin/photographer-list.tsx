"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Trash2, Pencil } from "lucide-react"
import { deletePhotographerAction } from "@/app/admin/actions"
import { EditPhotographerDialog } from "./edit-photographer-dialog"
import type { Photographer } from "@/lib/types"

interface PhotographerListProps {
  photographers: Photographer[]
}

export function PhotographerList({ photographers }: PhotographerListProps) {
  const [deletingId, setDeletingId] = useState<string | null>(null)
  const [editingPhotographer, setEditingPhotographer] = useState<Photographer | null>(null)

  async function handleDelete(id: string) {
    if (!confirm("Вы уверены, что хотите удалить этого фотографа?")) return

    setDeletingId(id)
    await deletePhotographerAction(id)
    setDeletingId(null)
  }

  if (photographers.length === 0) {
    return (
      <Card>
        <CardContent className="flex min-h-[200px] items-center justify-center">
          <p className="text-muted-foreground">Нет фотографов</p>
        </CardContent>
      </Card>
    )
  }

  return (
    <>
      <div className="grid gap-2">
        {photographers.map((photographer) => (
          <Card key={photographer.id}>
            <CardContent className="flex items-center justify-between p-4">
              <span className="font-medium">{photographer.name}</span>
              <div className="flex gap-2">
                <Button variant="outline" size="sm" onClick={() => setEditingPhotographer(photographer)}>
                  <Pencil className="h-4 w-4" />
                </Button>
                <Button
                  variant="destructive"
                  size="sm"
                  onClick={() => handleDelete(photographer.id)}
                  disabled={deletingId === photographer.id}
                >
                  <Trash2 className="h-4 w-4" />
                </Button>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {editingPhotographer && (
        <EditPhotographerDialog
          photographer={editingPhotographer}
          open={!!editingPhotographer}
          onOpenChange={(open) => !open && setEditingPhotographer(null)}
        />
      )}
    </>
  )
}
