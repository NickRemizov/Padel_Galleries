"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Trash2, Pencil } from "lucide-react"
import { deleteOrganizerAction } from "@/app/admin/actions"
import { EditOrganizerDialog } from "./edit-organizer-dialog"
import type { Organizer } from "@/lib/types"

interface OrganizerListProps {
  organizers: Organizer[]
}

export function OrganizerList({ organizers }: OrganizerListProps) {
  const [deletingId, setDeletingId] = useState<string | null>(null)
  const [editingOrganizer, setEditingOrganizer] = useState<Organizer | null>(null)

  async function handleDelete(id: string) {
    if (!confirm("Вы уверены, что хотите удалить этого организатора?")) return

    setDeletingId(id)
    await deleteOrganizerAction(id)
    setDeletingId(null)
  }

  if (organizers.length === 0) {
    return (
      <Card>
        <CardContent className="flex min-h-[200px] items-center justify-center">
          <p className="text-muted-foreground">Нет организаторов</p>
        </CardContent>
      </Card>
    )
  }

  return (
    <>
      <div className="grid gap-2">
        {organizers.map((organizer) => (
          <Card key={organizer.id}>
            <CardContent className="flex items-center justify-between p-4">
              <span className="font-medium">{organizer.name}</span>
              <div className="flex gap-2">
                <Button variant="outline" size="sm" onClick={() => setEditingOrganizer(organizer)}>
                  <Pencil className="h-4 w-4" />
                </Button>
                <Button
                  variant="destructive"
                  size="sm"
                  onClick={() => handleDelete(organizer.id)}
                  disabled={deletingId === organizer.id}
                >
                  <Trash2 className="h-4 w-4" />
                </Button>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {editingOrganizer && (
        <EditOrganizerDialog
          organizer={editingOrganizer}
          open={!!editingOrganizer}
          onOpenChange={(open) => !open && setEditingOrganizer(null)}
        />
      )}
    </>
  )
}
