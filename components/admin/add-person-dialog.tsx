"use client"

import type React from "react"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Plus } from "lucide-react"
import { addPersonAction } from "@/app/admin/actions"

interface AddPersonDialogProps {
  open?: boolean
  onOpenChange?: (open: boolean) => void
  onPersonCreated?: (personId: string, personName: string) => void
}

export function AddPersonDialog({
  open: controlledOpen,
  onOpenChange: controlledOnOpenChange,
  onPersonCreated,
}: AddPersonDialogProps = {}) {
  const [internalOpen, setInternalOpen] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const open = controlledOpen !== undefined ? controlledOpen : internalOpen
  const setOpen = controlledOnOpenChange || setInternalOpen

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const form = event.currentTarget
    const formData = new FormData(form)
    setLoading(true)
    setError(null)
    console.log("[v0] AddPersonDialog: handleSubmit called")

    const result = await addPersonAction(formData)
    console.log("[v0] AddPersonDialog: result =", result)
    setLoading(false)

    if (result.success) {
      setOpen(false)
      if (onPersonCreated && result.data) {
        onPersonCreated(result.data.id, result.data.real_name)
      }
    } else {
      console.log("[v0] AddPersonDialog: error =", result.error)
      setError(result.error || "Неизвестная ошибка")
    }
  }

  const content = (
    <DialogContent className="max-h-[90vh] overflow-y-auto sm:max-w-[600px]">
      <form onSubmit={handleSubmit}>
        <DialogHeader>
          <DialogTitle>Добавить человека</DialogTitle>
          <DialogDescription>Заполните информацию о человеке</DialogDescription>
        </DialogHeader>

        {error && <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">{error}</div>}

        <div className="grid gap-4 py-4">
          <div className="grid gap-2">
            <Label htmlFor="real_name">Реальное имя *</Label>
            <Input id="real_name" name="real_name" required />
          </div>

          <div className="grid gap-2">
            <Label htmlFor="telegram_name">Имя в Telegram</Label>
            <Input id="telegram_name" name="telegram_name" />
          </div>

          <div className="grid gap-2">
            <Label htmlFor="telegram_nickname">Ник в Telegram</Label>
            <Input id="telegram_nickname" name="telegram_nickname" placeholder="@username" />
          </div>

          <div className="grid gap-2">
            <Label htmlFor="telegram_profile_url">Ссылка на профиль Telegram</Label>
            <Input
              id="telegram_profile_url"
              name="telegram_profile_url"
              type="url"
              placeholder="https://t.me/username"
            />
          </div>

          <div className="grid gap-2">
            <Label htmlFor="facebook_profile_url">Ссылка на профиль Facebook</Label>
            <Input
              id="facebook_profile_url"
              name="facebook_profile_url"
              type="url"
              placeholder="https://facebook.com/username"
            />
          </div>

          <div className="grid gap-2">
            <Label htmlFor="instagram_profile_url">Ссылка на профиль Instagram</Label>
            <Input
              id="instagram_profile_url"
              name="instagram_profile_url"
              type="url"
              placeholder="https://instagram.com/username"
            />
          </div>

          <div className="grid gap-2">
            <Label htmlFor="paddle_ranking">Рейтинг в падел</Label>
            <Input id="paddle_ranking" name="paddle_ranking" type="number" min="0" />
          </div>
        </div>

        <DialogFooter>
          <Button type="submit" disabled={loading}>
            {loading ? "Добавление..." : "Добавить"}
          </Button>
        </DialogFooter>
      </form>
    </DialogContent>
  )

  if (controlledOpen !== undefined) {
    return (
      <Dialog open={open} onOpenChange={setOpen}>
        {content}
      </Dialog>
    )
  }

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button>
          <Plus className="mr-2 h-4 w-4" />
          Добавить человека
        </Button>
      </DialogTrigger>
      {content}
    </Dialog>
  )
}
