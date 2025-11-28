"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { updatePersonAction } from "@/app/admin/actions"
import type { Person } from "@/lib/types"

interface EditPersonDialogProps {
  person: Person
  open: boolean
  onOpenChange: (open: boolean) => void
}

export function EditPersonDialog({ person, open, onOpenChange }: EditPersonDialogProps) {
  const [loading, setLoading] = useState(false)

  async function handleSubmit(formData: FormData) {
    setLoading(true)
    const result = await updatePersonAction(person.id, formData)
    setLoading(false)

    if (result.success) {
      onOpenChange(false)
    }
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-h-[90vh] overflow-y-auto sm:max-w-[600px]">
        <form action={handleSubmit}>
          <DialogHeader>
            <DialogTitle>Редактировать человека</DialogTitle>
            <DialogDescription>Измените информацию о человеке</DialogDescription>
          </DialogHeader>

          <div className="grid gap-4 py-4">
            <div className="grid gap-2">
              <Label htmlFor="real_name">Реальное имя *</Label>
              <Input id="real_name" name="real_name" defaultValue={person.real_name} required />
            </div>

            <div className="grid gap-2">
              <Label htmlFor="telegram_name">Имя в Telegram</Label>
              <Input id="telegram_name" name="telegram_name" defaultValue={person.telegram_name || ""} />
            </div>

            <div className="grid gap-2">
              <Label htmlFor="telegram_nickname">Ник в Telegram</Label>
              <Input
                id="telegram_nickname"
                name="telegram_nickname"
                defaultValue={person.telegram_nickname || ""}
                placeholder="@username"
              />
            </div>

            <div className="grid gap-2">
              <Label htmlFor="telegram_profile_url">Ссылка на профиль Telegram</Label>
              <Input
                id="telegram_profile_url"
                name="telegram_profile_url"
                type="url"
                defaultValue={person.telegram_profile_url || ""}
                placeholder="https://t.me/username"
              />
            </div>

            <div className="grid gap-2">
              <Label htmlFor="facebook_profile_url">Ссылка на профиль Facebook</Label>
              <Input
                id="facebook_profile_url"
                name="facebook_profile_url"
                type="url"
                defaultValue={person.facebook_profile_url || ""}
                placeholder="https://facebook.com/username"
              />
            </div>

            <div className="grid gap-2">
              <Label htmlFor="instagram_profile_url">Ссылка на профиль Instagram</Label>
              <Input
                id="instagram_profile_url"
                name="instagram_profile_url"
                type="url"
                defaultValue={person.instagram_profile_url || ""}
                placeholder="https://instagram.com/username"
              />
            </div>

            <div className="grid gap-2">
              <Label htmlFor="paddle_ranking">Рейтинг в падел</Label>
              <Input
                id="paddle_ranking"
                name="paddle_ranking"
                type="number"
                min="0"
                defaultValue={person.paddle_ranking || ""}
              />
            </div>

            <div className="grid gap-2">
              <Label htmlFor="tournament_results">Результаты турниров (JSON)</Label>
              <Textarea
                id="tournament_results"
                name="tournament_results"
                defaultValue={JSON.stringify(person.tournament_results, null, 2)}
                placeholder='[{"tournament": "Название турнира", "place": 1, "date": "2024-01-01"}]'
                rows={6}
              />
              <p className="text-xs text-muted-foreground">Формат: массив объектов с полями tournament, place, date</p>
            </div>
          </div>

          <DialogFooter>
            <Button type="submit" disabled={loading}>
              {loading ? "Сохранение..." : "Сохранить"}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  )
}
