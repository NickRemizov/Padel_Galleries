"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import { AlertCircle, CheckCircle2, Loader2, RefreshCw, Wrench, ChevronDown, ChevronRight } from "lucide-react"
import { checkDatabaseIntegrityAction, fixIntegrityIssueAction } from "@/app/admin/actions"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"

interface IntegrityReport {
  photoFaces: {
    verifiedWithoutPerson: number
    verifiedWithWrongConfidence: number
    personWithoutConfidence: number
    nonExistentPerson: number
    nonExistentPhoto: number
    withoutDescriptors: number
    inconsistentPersonId: number
  }
  faceDescriptors: {
    orphaned: number
    nonExistentPerson: number
    withoutPerson: number
    withoutEmbedding: number
    duplicates: number
    inconsistentPersonId: number
  }
  people: {
    withoutDescriptors: number
    withoutFaces: number
    duplicateNames: number
  }
  totalIssues: number
  details: Record<string, any[]>
}

export function DatabaseIntegrityChecker() {
  const [isChecking, setIsChecking] = useState(false)
  const [report, setReport] = useState<IntegrityReport | null>(null)
  const [fixingIssue, setFixingIssue] = useState<string | null>(null)
  const [expandedIssues, setExpandedIssues] = useState<Set<string>>(new Set())
  const [checkingIssue, setCheckingIssue] = useState<string | null>(null)

  const handleCheck = async () => {
    setIsChecking(true)
    try {
      const result = await checkDatabaseIntegrityAction()
      if (result.success && result.data) {
        setReport(result.data)
      } else {
        alert(`Ошибка проверки: ${result.error}`)
      }
    } catch (error: any) {
      alert(`Ошибка: ${error.message}`)
    } finally {
      setIsChecking(false)
    }
  }

  const handleFix = async (issueType: string) => {
    if (!confirm(`Исправить проблему "${issueType}"? Это действие необратимо.`)) {
      return
    }

    setFixingIssue(issueType)
    try {
      const result = await fixIntegrityIssueAction(issueType)
      if (result.success) {
        alert(`Исправлено: ${result.data?.fixed} записей`)
        // Перезапустить проверку
        await handleCheck()
      } else {
        alert(`Ошибка исправления: ${result.error}`)
      }
    } catch (error: any) {
      alert(`Ошибка: ${error.message}`)
    } finally {
      setFixingIssue(null)
    }
  }

  const handleCheckIssue = async (issueType: string) => {
    setCheckingIssue(issueType)
    try {
      setExpandedIssues((prev) => {
        const newSet = new Set(prev)
        if (newSet.has(issueType)) {
          newSet.delete(issueType)
        } else {
          newSet.add(issueType)
        }
        return newSet
      })
    } finally {
      setCheckingIssue(null)
    }
  }

  const IssueRow = ({
    title,
    count,
    issueType,
    description,
    severity = "medium",
  }: {
    title: string
    count: number
    issueType: string
    description: string
    severity?: "critical" | "high" | "medium" | "low"
  }) => {
    if (count === 0) return null

    const isExpanded = expandedIssues.has(issueType)
    const details = report?.details?.[issueType] || []
    const hasDetails = details.length > 0

    const severityVariant = {
      critical: "destructive" as const,
      high: "destructive" as const,
      medium: "default" as const,
      low: "secondary" as const,
    }

    return (
      <div className="space-y-2">
        <div className="flex items-center justify-between py-2">
          <div className="flex-1">
            <div className="flex items-center gap-2">
              <span className="font-medium">{title}</span>
              <Badge variant={severityVariant[severity]}>{count}</Badge>
            </div>
            <p className="text-sm text-muted-foreground">{description}</p>
          </div>
          <div className="flex items-center gap-2">
            {hasDetails && (
              <Button
                variant="outline"
                size="sm"
                onClick={() => handleCheckIssue(issueType)}
                disabled={checkingIssue !== null}
              >
                {checkingIssue === issueType ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Проверка...
                  </>
                ) : (
                  <>
                    {isExpanded ? <ChevronDown className="mr-2 h-4 w-4" /> : <ChevronRight className="mr-2 h-4 w-4" />}
                    Проверить
                  </>
                )}
              </Button>
            )}
            <Button variant="outline" size="sm" onClick={() => handleFix(issueType)} disabled={fixingIssue !== null}>
              {fixingIssue === issueType ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Исправление...
                </>
              ) : (
                <>
                  <Wrench className="mr-2 h-4 w-4" />
                  Исправить
                </>
              )}
            </Button>
          </div>
        </div>
        {isExpanded && hasDetails && (
          <div className="ml-4 p-4 bg-muted rounded-lg space-y-2">
            <div className="text-sm font-medium">Найдено записей: {details.length}</div>
            <div className="text-xs text-muted-foreground">Показаны первые {Math.min(10, details.length)} записей:</div>
            <div className="space-y-1 max-h-60 overflow-y-auto">
              {details.slice(0, 10).map((item: any, index: number) => (
                <div key={index} className="text-xs font-mono bg-background p-2 rounded border">
                  {JSON.stringify(item, null, 2)}
                </div>
              ))}
            </div>
            {details.length > 10 && (
              <div className="text-xs text-muted-foreground">... и еще {details.length - 10} записей</div>
            )}
          </div>
        )}
        <Separator />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Проверка целостности базы данных</CardTitle>
          <CardDescription>
            Проверка и исправление нарушений целостности данных в системе распознавания лиц
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <Button onClick={handleCheck} disabled={isChecking} className="w-full">
            {isChecking ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Проверка...
              </>
            ) : (
              <>
                <RefreshCw className="mr-2 h-4 w-4" />
                Запустить проверку
              </>
            )}
          </Button>

          {report && (
            <Alert variant={report.totalIssues > 0 ? "destructive" : "default"}>
              {report.totalIssues > 0 ? (
                <>
                  <AlertCircle className="h-4 w-4" />
                  <AlertTitle>Обнаружено проблем: {report.totalIssues}</AlertTitle>
                  <AlertDescription>
                    Рекомендуется исправить проблемы для корректной работы системы распознавания
                  </AlertDescription>
                </>
              ) : (
                <>
                  <CheckCircle2 className="h-4 w-4" />
                  <AlertTitle>Проблем не обнаружено</AlertTitle>
                  <AlertDescription>База данных в порядке</AlertDescription>
                </>
              )}
            </Alert>
          )}
        </CardContent>
      </Card>

      {report && report.totalIssues > 0 && (
        <>
          {(report.photoFaces.verifiedWithoutPerson > 0 ||
            report.photoFaces.verifiedWithWrongConfidence > 0 ||
            report.photoFaces.personWithoutConfidence > 0 ||
            report.photoFaces.nonExistentPerson > 0 ||
            report.photoFaces.nonExistentPhoto > 0 ||
            report.photoFaces.withoutDescriptors > 0 ||
            report.photoFaces.inconsistentPersonId > 0) && (
            <Card>
              <CardHeader>
                <CardTitle>Проблемы с лицами на фото (Photo Faces)</CardTitle>
                <CardDescription>
                  Всего проблем:{" "}
                  {report.photoFaces.verifiedWithoutPerson +
                    report.photoFaces.verifiedWithWrongConfidence +
                    report.photoFaces.personWithoutConfidence +
                    report.photoFaces.nonExistentPerson +
                    report.photoFaces.nonExistentPhoto +
                    report.photoFaces.withoutDescriptors +
                    report.photoFaces.inconsistentPersonId}
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <IssueRow
                    title="Верифицированные лица без игрока"
                    count={report.photoFaces.verifiedWithoutPerson}
                    issueType="verifiedWithoutPerson"
                    description="Лица с verified=true, но person_id=null (игрок удален)"
                    severity="critical"
                  />
                  <IssueRow
                    title="Верифицированные лица с неправильным confidence"
                    count={report.photoFaces.verifiedWithWrongConfidence}
                    issueType="verifiedWithWrongConfidence"
                    description="Лица с verified=true, но recognition_confidence != 1.0"
                    severity="high"
                  />
                  <IssueRow
                    title="Лица с игроком без confidence"
                    count={report.photoFaces.personWithoutConfidence}
                    issueType="personWithoutConfidence"
                    description="Лица с person_id, но recognition_confidence = null или 0"
                    severity="high"
                  />
                  <IssueRow
                    title="Лица с несуществующим игроком"
                    count={report.photoFaces.nonExistentPerson}
                    issueType="nonExistentPersonFaces"
                    description="person_id ссылается на удаленного игрока"
                    severity="critical"
                  />
                  <IssueRow
                    title="Лица с несуществующим фото"
                    count={report.photoFaces.nonExistentPhoto}
                    issueType="nonExistentPhotoFaces"
                    description="photo_id ссылается на удаленное фото"
                    severity="critical"
                  />
                  <IssueRow
                    title="Лица без дескрипторов"
                    count={report.photoFaces.withoutDescriptors}
                    issueType="facesWithoutDescriptors"
                    description="Лица без записей в face_descriptors"
                    severity="medium"
                  />
                  <IssueRow
                    title="Несогласованность person_id"
                    count={report.photoFaces.inconsistentPersonId}
                    issueType="inconsistentPersonIds"
                    description="person_id в photo_faces != person_id в face_descriptors"
                    severity="high"
                  />
                </div>
              </CardContent>
            </Card>
          )}

          {(report.faceDescriptors.orphaned > 0 ||
            report.faceDescriptors.nonExistentPerson > 0 ||
            report.faceDescriptors.withoutPerson > 0 ||
            report.faceDescriptors.withoutEmbedding > 0 ||
            report.faceDescriptors.duplicates > 0) && (
            <Card>
              <CardHeader>
                <CardTitle>Проблемы с дескрипторами (Face Descriptors)</CardTitle>
                <CardDescription>
                  Всего проблем:{" "}
                  {report.faceDescriptors.orphaned +
                    report.faceDescriptors.nonExistentPerson +
                    report.faceDescriptors.withoutPerson +
                    report.faceDescriptors.withoutEmbedding +
                    report.faceDescriptors.duplicates}
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <IssueRow
                    title="Потерянные дескрипторы"
                    count={report.faceDescriptors.orphaned}
                    issueType="orphanedDescriptors"
                    description="source_image_id ссылается на несуществующее лицо"
                    severity="critical"
                  />
                  <IssueRow
                    title="Дескрипторы с несуществующим игроком"
                    count={report.faceDescriptors.nonExistentPerson}
                    issueType="nonExistentPersonDescriptors"
                    description="person_id ссылается на удаленного игрока"
                    severity="critical"
                  />
                  <IssueRow
                    title="Дескрипторы без игрока"
                    count={report.faceDescriptors.withoutPerson}
                    issueType="descriptorsWithoutPerson"
                    description="person_id = null (информационное)"
                    severity="low"
                  />
                  <IssueRow
                    title="Дескрипторы без embedding"
                    count={report.faceDescriptors.withoutEmbedding}
                    issueType="descriptorsWithoutEmbedding"
                    description="descriptor = null (ошибка создания)"
                    severity="medium"
                  />
                  <IssueRow
                    title="Дубликаты дескрипторов"
                    count={report.faceDescriptors.duplicates}
                    issueType="duplicateDescriptors"
                    description="Несколько дескрипторов для одного лица"
                    severity="medium"
                  />
                </div>
              </CardContent>
            </Card>
          )}

          {(report.people.withoutDescriptors > 0 ||
            report.people.withoutFaces > 0 ||
            report.people.duplicateNames > 0) && (
            <Card>
              <CardHeader>
                <CardTitle>Проблемы с игроками (People)</CardTitle>
                <CardDescription>
                  Всего проблем:{" "}
                  {report.people.withoutDescriptors + report.people.withoutFaces + report.people.duplicateNames}
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <IssueRow
                    title="Игроки без дескрипторов"
                    count={report.people.withoutDescriptors}
                    issueType="peopleWithoutDescriptors"
                    description="Игроки без записей в face_descriptors (информационное)"
                    severity="low"
                  />
                  <IssueRow
                    title="Игроки без фото"
                    count={report.people.withoutFaces}
                    issueType="peopleWithoutFaces"
                    description="Игроки без лиц в photo_faces (информационное)"
                    severity="low"
                  />
                  <IssueRow
                    title="Дубликаты имен"
                    count={report.people.duplicateNames}
                    issueType="duplicateNames"
                    description="Несколько игроков с одинаковым именем"
                    severity="medium"
                  />
                </div>
              </CardContent>
            </Card>
          )}
        </>
      )}
    </div>
  )
}
