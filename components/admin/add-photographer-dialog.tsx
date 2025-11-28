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
  DialogTrigger,
} from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Plus } from "lucide-react"
import { addPhotographerAction } from "@/app/admin/actions"

export function AddPhotographerDialog() {
  const [open, setOpen] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function handleSubmit(formData: FormData) {
    console.log("[v0] AddPhotographerDialog: handleSubmit called")
    setLoading(true)
    setError(null)

    try {
      const result = await addPhotographerAction(formData)
      console.log("[v0] AddPhotographerDialog: result =", result)

      if (result.success) {
        setOpen(false)
      } else if (result.error) {
        console.error("[v0] AddPhotographerDialog: error =", result.error)
        setError(result.error)
      }
    } catch (err: any) {
      console.error("[v0] AddPhotographerDialog: exception =", err)
      setError(err.message || "Неизвестная ошибка")
    } finally {
      setLoading(false)
    }
  }

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button>
          <Plus className="mr-2 h-4 w-4" />
          Добавить фотографа
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-[425px]">
        <form action={handleSubmit}>
          <DialogHeader>
            <DialogTitle>Добавить фотографа</DialogTitle>
            <DialogDescription>Введите имя фотографа</DialogDescription>
          </DialogHeader>

          <div className="grid gap-4 py-4">
            <div className="grid gap-2">
              <Label htmlFor="name">Имя</Label>
              <Input id="name" name="name" required />
            </div>
            {error && <div className="text-sm text-red-500 bg-red-50 p-2 rounded">Ошибка: {error}</div>}
          </div>

          <DialogFooter>
            <Button type="submit" disabled={loading}>
              {loading ? "Добавление..." : "Добавить"}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  )
}
