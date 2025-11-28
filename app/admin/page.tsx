import { createClient } from "@/lib/supabase/server"
import { redirect } from "next/navigation"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { GalleriesManager } from "@/components/admin/galleries-manager"
import { PhotographersManager } from "@/components/admin/photographers-manager"
import { LocationsManager } from "@/components/admin/locations-manager"
import { OrganizersManager } from "@/components/admin/organizers-manager"
import { PeopleManager } from "@/components/admin/people-manager"
// UNUSED: Batch recognition disabled - code kept for future reference
// import { BatchRecognitionManager } from "@/components/admin/batch-recognition-manager"
import { RecognitionStatsDialog } from "@/components/admin/recognition-stats-dialog"
import { LogOut } from "lucide-react"
import { Button } from "@/components/ui/button"
import { signOutAction } from "@/app/admin/actions"
import { FaceTrainingManager } from "@/components/admin/face-training-manager"
import { ServiceManager } from "@/components/admin/service-manager"

const APP_VERSION = "0.8.2" // Updated version to 0.8.2 after documentation update

export const dynamic = "force-dynamic"

export default async function AdminPage() {
  const supabase = await createClient()

  const {
    data: { user },
  } = await supabase.auth.getUser()

  if (!user) {
    redirect("/admin/login")
  }

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b bg-card">
        <div className="container mx-auto flex items-center justify-between px-4 py-4">
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-bold">Админ-панель</h1>
            <span className="rounded-md bg-primary/10 px-2 py-1 text-xs font-medium text-primary">v{APP_VERSION}</span>
          </div>
          <div className="flex items-center gap-2">
            <RecognitionStatsDialog />
            <form action={signOutAction}>
              <Button variant="outline" size="sm">
                <LogOut className="mr-2 h-4 w-4" />
                Выйти
              </Button>
            </form>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8">
        <Tabs defaultValue="galleries" className="w-full">
          <TabsList className="grid w-full grid-cols-7">
            <TabsTrigger value="galleries">Галереи</TabsTrigger>
            <TabsTrigger value="people">Люди</TabsTrigger>
            <TabsTrigger value="organizers">Организаторы</TabsTrigger>
            <TabsTrigger value="locations">Места съемки</TabsTrigger>
            <TabsTrigger value="photographers">Фотографы</TabsTrigger>
            <TabsTrigger value="face-training">Настройки распознавания</TabsTrigger>
            <TabsTrigger value="service">Сервис</TabsTrigger>
          </TabsList>

          <TabsContent value="galleries" className="mt-6">
            <GalleriesManager />
          </TabsContent>

          <TabsContent value="people" className="mt-6">
            <PeopleManager />
          </TabsContent>

          <TabsContent value="organizers" className="mt-6">
            <OrganizersManager />
          </TabsContent>

          <TabsContent value="locations" className="mt-6">
            <LocationsManager />
          </TabsContent>

          {/* UNUSED: Batch recognition disabled - code kept for future reference */}
          {/* <TabsContent value="batch" className="mt-6">
            <BatchRecognitionManager />
          </TabsContent> */}

          <TabsContent value="photographers" className="mt-6">
            <PhotographersManager />
          </TabsContent>

          <TabsContent value="face-training" className="mt-6">
            <FaceTrainingManager />
          </TabsContent>

          <TabsContent value="service" className="mt-6">
            <ServiceManager />
          </TabsContent>
        </Tabs>
      </main>
    </div>
  )
}
