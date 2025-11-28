import { env } from "@/lib/env"
import { AddPhotographerDialog } from "@/components/admin/add-photographer-dialog"
import { PhotographerList } from "@/components/admin/photographer-list"

async function getPhotographers() {
  const res = await fetch(`${env.FASTAPI_URL}/api/crud/photographers`, {
    cache: "no-store",
    headers: {
      "X-API-Key": env.API_SECRET_KEY,
    },
  })
  if (!res.ok) {
    console.error("[v0] Failed to fetch photographers:", res.status)
    return []
  }
  return res.json()
}

export async function PhotographersManager() {
  const photographers = await getPhotographers()

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Управление фотографами</h2>
          <p className="text-sm text-muted-foreground">Добавляйте, редактируйте и удаляйте фотографов</p>
        </div>
        <AddPhotographerDialog />
      </div>

      <PhotographerList photographers={photographers || []} />
    </div>
  )
}
