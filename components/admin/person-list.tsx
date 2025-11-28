"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import { Switch } from "@/components/ui/switch"
import { Label } from "@/components/ui/label"
import { Trash2, Pencil, Images, RefreshCw } from "lucide-react"
import { deletePersonAction, updatePersonVisibilityAction } from "@/app/admin/actions"
import { EditPersonDialog } from "./edit-person-dialog"
import { PersonGalleryDialog } from "./person-gallery-dialog"
import { RegenerateDescriptorsDialog } from "./regenerate-descriptors-dialog"
import { getInitials } from "@/lib/utils/get-initials"
import type { Person } from "@/lib/types"

type PersonWithStats = Person & {
  verified_photos_count?: number
  high_confidence_photos_count?: number
  descriptor_count?: number
}

interface PersonListProps {
  people: PersonWithStats[]
}

export function PersonList({ people }: PersonListProps) {
  const [deletingId, setDeletingId] = useState<string | null>(null)
  const [editingPerson, setEditingPerson] = useState<PersonWithStats | null>(null)
  const [galleryPerson, setGalleryPerson] = useState<PersonWithStats | null>(null)
  const [regeneratePerson, setRegeneratePerson] = useState<PersonWithStats | null>(null)
  const [updatingVisibility, setUpdatingVisibility] = useState<string | null>(null)

  async function handleDelete(id: string) {
    if (!confirm("Вы уверены, что хотите удалить этого человека?")) return

    setDeletingId(id)
    await deletePersonAction(id)
    setDeletingId(null)
  }

  async function handleVisibilityToggle(
    personId: string,
    field: "show_in_players_gallery" | "show_photos_in_galleries",
    value: boolean,
  ) {
    setUpdatingVisibility(personId)
    await updatePersonVisibilityAction(personId, field, value)
    setUpdatingVisibility(null)
  }

  if (people.length === 0) {
    return (
      <Card>
        <CardContent className="flex min-h-[200px] items-center justify-center">
          <p className="text-muted-foreground">Нет людей в базе</p>
        </CardContent>
      </Card>
    )
  }

  return (
    <>
      <div className="grid gap-3">
        {people.map((person) => {
          const socialLinks = []
          if (person.telegram_profile_url) {
            socialLinks.push(
              <span key="telegram">
                Telegram:{" "}
                <a
                  href={person.telegram_profile_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-primary hover:underline"
                >
                  {person.telegram_nickname || "профиль"}
                </a>
              </span>,
            )
          }
          if (person.instagram_profile_url) {
            socialLinks.push(
              <span key="instagram">
                Instagram:{" "}
                <a
                  href={person.instagram_profile_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-primary hover:underline"
                >
                  профиль
                </a>
              </span>,
            )
          }
          if (person.facebook_profile_url) {
            socialLinks.push(
              <span key="facebook">
                Facebook:{" "}
                <a
                  href={person.facebook_profile_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-primary hover:underline"
                >
                  профиль
                </a>
              </span>,
            )
          }

          const totalPhotoCount = (person.verified_photos_count || 0) + (person.high_confidence_photos_count || 0)
          const descriptorCount = person.descriptor_count || 0
          const needsRegeneration = descriptorCount !== totalPhotoCount || totalPhotoCount === 0

          return (
            <Card key={person.id}>
              <CardContent className="p-4">
                <div className="flex items-start justify-between gap-4">
                  <div className="flex flex-1 gap-3">
                    {person.avatar_url ? (
                      <div
                        className="relative h-20 w-15 flex-shrink-0 overflow-hidden rounded-md cursor-pointer hover:opacity-80 transition-opacity"
                        onClick={() => setGalleryPerson(person)}
                      >
                        <img
                          src={person.avatar_url || "/placeholder.svg"}
                          alt={person.real_name}
                          className="h-full w-full object-cover"
                        />
                      </div>
                    ) : (
                      <Avatar
                        className="h-15 w-15 flex-shrink-0 cursor-pointer hover:opacity-80 transition-opacity"
                        onClick={() => setGalleryPerson(person)}
                      >
                        <AvatarFallback className="bg-primary/10 text-primary text-lg">
                          {getInitials(person.real_name)}
                        </AvatarFallback>
                      </Avatar>
                    )}
                    <div className="flex-1 space-y-2">
                      <div className="flex items-center gap-2">
                        <h3 className="font-semibold">
                          {person.real_name}
                          {person.telegram_name && ` (${person.telegram_name})`}
                        </h3>
                        {person.paddle_ranking && <Badge variant="secondary">Рейтинг: {person.paddle_ranking}</Badge>}
                      </div>
                      <div className="space-y-1 text-sm text-muted-foreground">
                        <p>
                          {socialLinks.length > 0 ? (
                            <>
                              {socialLinks.map((link, index) => (
                                <span key={index}>
                                  {link}
                                  {index < socialLinks.length - 1 && " • "}
                                </span>
                              ))}
                            </>
                          ) : (
                            "Профили не указаны"
                          )}
                        </p>
                        {(person.verified_photos_count !== undefined ||
                          person.high_confidence_photos_count !== undefined) && (
                          <p>
                            Подтвержденные фото: {person.verified_photos_count || 0} • Высокая уверенность:{" "}
                            {person.high_confidence_photos_count || 0}
                            {person.descriptor_count !== undefined && ` • Дескрипторов: ${person.descriptor_count}`}
                          </p>
                        )}
                        {person.tournament_results && person.tournament_results.length > 0 && (
                          <p>Турниров: {person.tournament_results.length}</p>
                        )}
                      </div>
                    </div>
                  </div>
                  <div className="flex flex-col items-end gap-3">
                    <div className="flex gap-2">
                      <Button variant="outline" size="sm" onClick={() => setGalleryPerson(person)}>
                        <Images className="h-4 w-4" />
                      </Button>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => setRegeneratePerson(person)}
                        disabled={!needsRegeneration}
                        title={
                          needsRegeneration
                            ? "Регенерировать дескрипторы"
                            : "Дескрипторы в порядке (количество совпадает с количеством фото)"
                        }
                      >
                        <RefreshCw className="h-4 w-4" />
                      </Button>
                      <Button variant="outline" size="sm" onClick={() => setEditingPerson(person)}>
                        <Pencil className="h-4 w-4" />
                      </Button>
                      <Button
                        variant="destructive"
                        size="sm"
                        onClick={() => handleDelete(person.id)}
                        disabled={deletingId === person.id}
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </div>
                    <div className="flex flex-col gap-2 items-end">
                      <div className="flex items-center gap-2">
                        <Label htmlFor={`show-in-gallery-${person.id}`} className="text-xs cursor-pointer">
                          Показывать в галерее игроков
                        </Label>
                        <Switch
                          id={`show-in-gallery-${person.id}`}
                          checked={person.show_in_players_gallery}
                          onCheckedChange={(checked) =>
                            handleVisibilityToggle(person.id, "show_in_players_gallery", checked)
                          }
                          disabled={updatingVisibility === person.id}
                        />
                      </div>
                      <div className="flex items-center gap-2">
                        <Label htmlFor={`show-photos-${person.id}`} className="text-xs cursor-pointer">
                          Показывать фото этого игрока в галереях
                        </Label>
                        <Switch
                          id={`show-photos-${person.id}`}
                          checked={person.show_photos_in_galleries}
                          onCheckedChange={(checked) =>
                            handleVisibilityToggle(person.id, "show_photos_in_galleries", checked)
                          }
                          disabled={updatingVisibility === person.id}
                        />
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          )
        })}
      </div>

      {editingPerson && (
        <EditPersonDialog
          person={editingPerson}
          open={!!editingPerson}
          onOpenChange={(open) => !open && setEditingPerson(null)}
        />
      )}

      {galleryPerson && (
        <PersonGalleryDialog
          personId={galleryPerson.id}
          personName={galleryPerson.real_name}
          open={!!galleryPerson}
          onOpenChange={(open) => !open && setGalleryPerson(null)}
        />
      )}

      {regeneratePerson && (
        <RegenerateDescriptorsDialog
          personId={regeneratePerson.id}
          personName={regeneratePerson.real_name}
          open={!!regeneratePerson}
          onOpenChange={(open) => !open && setRegeneratePerson(null)}
        />
      )}
    </>
  )
}
