import { getPeopleWithStatsAction } from "@/app/admin/actions"
import { PersonList } from "./person-list"
import { AddPersonDialog } from "./add-person-dialog"
import { GenerateMissingDescriptorsDialog } from "./generate-missing-descriptors-dialog"

export type PersonWithStats = {
  id: string
  real_name: string
  photos_count: number
  descriptor_count: number
}

export async function PeopleManager() {
  const result = await getPeopleWithStatsAction()
  const peopleWithStats = result.success && result.data ? result.data : []

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Люди</h2>
          <p className="text-muted-foreground">Управление базой людей для распознавания лиц</p>
        </div>
        <div className="flex gap-2">
          <GenerateMissingDescriptorsDialog />
          <AddPersonDialog />
        </div>
      </div>

      <PersonList people={peopleWithStats} />
    </div>
  )
}
