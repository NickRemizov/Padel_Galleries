import { env } from "@/lib/env"
import { AddOrganizerDialog } from "@/components/admin/add-organizer-dialog"
import { OrganizerList } from "@/components/admin/organizer-list"

async function getOrganizers() {
  const res = await fetch(`${env.FASTAPI_URL}/api/crud/organizers`, {
    cache: "no-store",
    headers: {
      "X-API-Key": env.API_SECRET_KEY,
    },
  })
  if (!res.ok) {
    console.error("[v0] Failed to fetch organizers:", res.status)
    return []
  }
  return res.json()
}

export async function OrganizersManager() {
  const organizers = await getOrganizers()

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Управление организаторами</h2>
          <p className="text-sm text-muted-foreground">Добавляйте, редактируйте и удаляйте организаторов</p>
        </div>
        <AddOrganizerDialog />
      </div>

      <OrganizerList organizers={organizers || []} />
    </div>
  )
}
