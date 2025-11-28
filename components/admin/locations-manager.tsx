import { env } from "@/lib/env"
import { AddLocationDialog } from "@/components/admin/add-location-dialog"
import { LocationList } from "@/components/admin/location-list"

async function getLocations() {
  const res = await fetch(`${env.FASTAPI_URL}/api/crud/locations`, {
    cache: "no-store",
    headers: {
      "X-API-Key": env.API_SECRET_KEY,
    },
  })
  if (!res.ok) {
    console.error("[v0] Failed to fetch locations:", res.status)
    return []
  }
  return res.json()
}

export async function LocationsManager() {
  const locations = await getLocations()

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Управление местами съёмки</h2>
          <p className="text-sm text-muted-foreground">Добавляйте, редактируйте и удаляйте места съёмки</p>
        </div>
        <AddLocationDialog />
      </div>
      <LocationList locations={locations || []} />
    </div>
  )
}
