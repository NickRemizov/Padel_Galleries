"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Separator } from "@/components/ui/separator"
import { RegenerateAllDescriptorsDialog } from "@/components/admin/regenerate-all-descriptors-dialog"
import { CleanupDuplicateDescriptorsButton } from "@/components/admin/cleanup-duplicate-descriptors-button"
import { CleanupDuplicatesButton } from "@/components/admin/cleanup-duplicates-button"
import { CleanupFacesButton } from "@/components/admin/cleanup-faces-button"
import { SyncVerifiedButton } from "@/components/admin/sync-verified-button"
import { DebugPersonPhotos } from "@/components/admin/debug-person-photos"
import { DatabaseIntegrityChecker } from "@/components/admin/database-integrity-checker"
import { Button } from "@/components/ui/button"
import { Database, Bug, Wrench, Shield } from "lucide-react"

export function ServiceManager() {
  const handleComingSoon = (feature: string) => {
    alert(`${feature} - в разработке`)
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <Shield className="h-5 w-5" />
            <CardTitle>Проверка целостности базы данных</CardTitle>
          </div>
          <CardDescription>
            Диагностика и исправление нарушений целостности данных в системе распознавания лиц
          </CardDescription>
        </CardHeader>
        <CardContent>
          <DatabaseIntegrityChecker />
        </CardContent>
      </Card>

      {/* Обслуживание базы данных */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <Database className="h-5 w-5" />
            <CardTitle>Обслуживание базы данных</CardTitle>
          </div>
          <CardDescription>Инструменты для очистки и оптимизации базы данных распознавания лиц</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-3">
            <div className="flex items-center justify-between">
              <div>
                <div className="font-medium">Регенерация дескрипторов</div>
                <div className="text-sm text-muted-foreground">Пересоздать дескрипторы лиц для всех людей</div>
              </div>
              <RegenerateAllDescriptorsDialog />
            </div>

            <Separator />

            <div className="flex items-center justify-between">
              <div>
                <div className="font-medium">Очистка дублирующихся дескрипторов</div>
                <div className="text-sm text-muted-foreground">Удалить дубликаты в таблице face_descriptors</div>
              </div>
              <CleanupDuplicateDescriptorsButton />
            </div>

            <Separator />

            <div className="flex items-center justify-between">
              <div>
                <div className="font-medium">Удаление дубликатов записей</div>
                <div className="text-sm text-muted-foreground">
                  Удалить несколько записей для одной комбинации человек+фото
                </div>
              </div>
              <CleanupDuplicatesButton />
            </div>

            <Separator />

            <div className="flex items-center justify-between">
              <div>
                <div className="font-medium">Очистка неподтверждённых лиц</div>
                <div className="text-sm text-muted-foreground">
                  Удалить все записи с verified=false (опасная операция!)
                </div>
              </div>
              <CleanupFacesButton />
            </div>

            <Separator />

            <div className="flex items-center justify-between">
              <div>
                <div className="font-medium">Синхронизация verified/confidence</div>
                <div className="text-sm text-muted-foreground">
                  Исправить несоответствия в полях verified и confidence
                </div>
              </div>
              <SyncVerifiedButton />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Отладка */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <Bug className="h-5 w-5" />
            <CardTitle>Отладка</CardTitle>
          </div>
          <CardDescription>Инструменты для диагностики и отладки системы распознавания</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-between">
            <div>
              <div className="font-medium">Проверка фото человека</div>
              <div className="text-sm text-muted-foreground">
                Отладочная информация о фото конкретного человека или файла
              </div>
            </div>
            <DebugPersonPhotos />
          </div>
        </CardContent>
      </Card>

      {/* Инструменты (в разработке) */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <Wrench className="h-5 w-5" />
            <CardTitle>Инструменты</CardTitle>
          </div>
          <CardDescription>Дополнительные инструменты для работы с системой (в разработке)</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-3">
            <div className="flex items-center justify-between">
              <div>
                <div className="font-medium">Экспорт данных</div>
                <div className="text-sm text-muted-foreground">Экспорт списка людей и статистики в CSV/JSON</div>
              </div>
              <Button variant="outline" size="sm" onClick={() => handleComingSoon("Экспорт данных")}>
                Экспорт
              </Button>
            </div>

            <Separator />

            <div className="flex items-center justify-between">
              <div>
                <div className="font-medium">Логи системы</div>
                <div className="text-sm text-muted-foreground">Просмотр последних операций и событий</div>
              </div>
              <Button variant="outline" size="sm" onClick={() => handleComingSoon("Логи системы")}>
                Открыть логи
              </Button>
            </div>

            <Separator />

            <div className="flex items-center justify-between">
              <div>
                <div className="font-medium">Backup/Restore</div>
                <div className="text-sm text-muted-foreground">Резервное копирование и восстановление базы данных</div>
              </div>
              <Button variant="outline" size="sm" onClick={() => handleComingSoon("Backup/Restore")}>
                Управление
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
