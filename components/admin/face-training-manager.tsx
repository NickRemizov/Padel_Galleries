"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { Badge } from "@/components/ui/badge"
import { Label } from "@/components/ui/label"
import { Slider } from "@/components/ui/slider"
import { RefreshCw, Loader2, CheckCircle2, AlertCircle, TrendingUp, AlertTriangle } from "lucide-react"
import { TrainingHistoryList } from "./training-history-list"
import { TrainingStatsCard } from "./training-stats-card"

const FASTAPI_URL = process.env.NEXT_PUBLIC_FASTAPI_URL!

interface TrainingSession {
  id: string
  created_at: string
  training_mode: string
  faces_count: number
  people_count: number
  metrics: {
    accuracy?: number
    precision?: number
    recall?: number
  }
  status: string
}

interface Config {
  confidence_thresholds: {
    low_data: number
    medium_data: number
    high_data: number
  }
  context_weight: number
  min_faces_per_person: number
  auto_retrain_threshold: number
  auto_retrain_percentage: number
  quality_filters?: {
    min_detection_score: number
    min_face_size: number
    min_blur_score: number
    verified_threshold: number
  }
}

interface DatasetStats {
  total_people: number
  total_faces: number
  faces_per_person: {
    min: number
    max: number
    avg: number
  }
  people_by_face_count: Record<string, number>
}

const DEFAULT_CONFIG: Config = {
  confidence_thresholds: {
    low_data: 0.75,
    medium_data: 0.65,
    high_data: 0.55,
  },
  context_weight: 0.1,
  min_faces_per_person: 3,
  auto_retrain_threshold: 25,
  auto_retrain_percentage: 0.1,
  quality_filters: {
    min_detection_score: 0.7,
    min_face_size: 80,
    min_blur_score: 100.0,
    verified_threshold: 0.99,
  },
}

export function FaceTrainingManager() {
  const [config, setConfig] = useState<Config>(DEFAULT_CONFIG)
  const [sessions, setSessions] = useState<TrainingSession[]>([])
  const [datasetStats, setDatasetStats] = useState<DatasetStats | null>(null)
  const [loading, setLoading] = useState(true)
  const [preparing, setPreparing] = useState(false)
  const [training, setTraining] = useState(false)
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null)
  const [trainingProgress, setTrainingProgress] = useState(0)
  const [fastapiError, setFastapiError] = useState(false)
  const [httpsRequired, setHttpsRequired] = useState(false)

  const [localConfig, setLocalConfig] = useState<Config>(DEFAULT_CONFIG)

  useEffect(() => {
    loadData()
    const interval = setInterval(() => {
      if (currentSessionId) {
        checkTrainingStatus(currentSessionId)
      }
    }, 2000)
    return () => clearInterval(interval)
  }, [currentSessionId])

  async function loadData() {
    setLoading(true)
    setFastapiError(false)
    setHttpsRequired(false)
    try {
      console.log("[v0] Loading training data from API routes")

      const [configRes, historyRes] = await Promise.all([
        fetch("/api/admin/training/config"),
        fetch("/api/admin/training/history?limit=10"),
      ])

      console.log("[v0] Config response status:", configRes.status)
      console.log("[v0] History response status:", historyRes.status)

      if (configRes.ok) {
        const configData = await configRes.json()
        console.log("[v0] Config data received:", configData)
        console.log("[v0] quality_filters from backend:", configData.quality_filters)
        console.log("[v0] min_blur_score from backend:", configData.quality_filters?.min_blur_score)

        if (configData.error === "https_required") {
          setHttpsRequired(true)
          setFastapiError(true)
          setConfig(DEFAULT_CONFIG)
          setLocalConfig(DEFAULT_CONFIG)
        } else if (configData.error === "connection_failed" || configData.error === "server_error") {
          // Backend error - use default config
          setFastapiError(true)
          setConfig(DEFAULT_CONFIG)
          setLocalConfig(DEFAULT_CONFIG)
        } else if (configData.confidence_thresholds && typeof configData.confidence_thresholds === "object") {
          // Valid config received - merge with defaults to ensure all fields exist
          const validConfig = {
            ...DEFAULT_CONFIG,
            ...configData,
            confidence_thresholds: {
              ...DEFAULT_CONFIG.confidence_thresholds,
              ...configData.confidence_thresholds,
            },
            quality_filters: {
              ...DEFAULT_CONFIG.quality_filters,
              ...(configData.quality_filters || {}),
            },
          }
          console.log("[v0] Merged config:", validConfig)
          console.log("[v0] Final min_blur_score:", validConfig.quality_filters.min_blur_score)
          setFastapiError(false)
          setConfig(validConfig)
          setLocalConfig(validConfig)
        } else {
          // Invalid config structure - use defaults
          console.warn("[v0] Invalid config structure, using defaults")
          setFastapiError(true)
          setConfig(DEFAULT_CONFIG)
          setLocalConfig(DEFAULT_CONFIG)
        }
      } else {
        const errorText = await configRes.text()
        console.error("[v0] Config request failed:", configRes.status, errorText)
        setFastapiError(true)
        setConfig(DEFAULT_CONFIG)
        setLocalConfig(DEFAULT_CONFIG)
      }

      if (historyRes.ok) {
        const historyData = await historyRes.json()
        setSessions(historyData.sessions || [])
      } else {
        console.error("[v0] History request failed")
        setSessions([])
      }
    } catch (error) {
      console.error("[v0] Error loading data:", error)
      setFastapiError(true)
      setConfig(DEFAULT_CONFIG)
      setLocalConfig(DEFAULT_CONFIG)
      setSessions([])
    } finally {
      setLoading(false)
    }
  }

  async function prepareDataset() {
    setPreparing(true)
    try {
      console.log("[v0] Preparing dataset...")

      const response = await fetch("/api/admin/training/prepare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          filters: {},
          options: {
            min_faces_per_person: config?.min_faces_per_person || 3,
            include_co_occurring: true,
            context_weight: config?.context_weight || 0.1,
            quality_filters: config?.quality_filters,
          },
        }),
      })

      console.log("[v0] Prepare response status:", response.status)

      if (!response.ok) {
        const errorData = await response.json()
        console.error("[v0] Prepare dataset error:", errorData)
        alert(`Ошибка при подготовке датасета: ${errorData.error || "Неизвестная ошибка"}`)
        return
      }

      const data = await response.json()
      console.log("[v0] Dataset prepared:", data)
      setDatasetStats(data.dataset_stats)
      alert("Датасет успешно подготовлен!")
    } catch (error) {
      console.error("[v0] Error preparing dataset:", error)
      alert(`Ошибка при подготовке датасета: ${error instanceof Error ? error.message : "Неизвестная ошибка"}`)
    } finally {
      setPreparing(false)
    }
  }

  async function startTraining() {
    if (!datasetStats) {
      alert("Сначала подготовьте датасет")
      await prepareDataset()
      return
    }

    setTraining(true)
    setTrainingProgress(0)

    try {
      console.log("[v0] Starting training...")

      const response = await fetch("/api/admin/training/execute", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          mode: "full",
          filters: {},
          options: {
            min_faces_per_person: config?.min_faces_per_person || 3,
            context_weight: config?.context_weight || 0.1,
            model_version: "v1.0",
            quality_filters: config?.quality_filters,
          },
        }),
      })

      console.log("[v0] Training response status:", response.status)

      if (!response.ok) {
        const errorData = await response.json()
        console.error("[v0] Training error:", errorData)
        alert(`Ошибка при запуске обучения: ${errorData.error || "Неизвестная ошибка"}`)
        setTraining(false)
        return
      }

      const data = await response.json()
      console.log("[v0] Training started:", data)

      setCurrentSessionId(data.session_id)
      alert("Обучение запущено!")
    } catch (error) {
      console.error("[v0] Error starting training:", error)
      alert(`Ошибка при запуске обучения: ${error instanceof Error ? error.message : "Неизвестная ошибка"}`)
      setTraining(false)
    }
  }

  async function checkTrainingStatus(sessionId: string) {
    try {
      const response = await fetch(`/api/admin/training/status/${sessionId}`)
      const data = await response.json()

      if (data.progress) {
        setTrainingProgress(data.progress.percentage)
      }

      if (data.status === "completed" || data.status === "failed") {
        setTraining(false)
        setCurrentSessionId(null)
        setTrainingProgress(0)
        loadData()
      }
    } catch (error) {
      console.error("[v0] Error checking training status:", error)
    }
  }

  async function saveConfig() {
    try {
      console.log("[v0] Saving config:", localConfig)

      const response = await fetch("/api/admin/training/config", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(localConfig),
      })

      console.log("[v0] Save config response status:", response.status)

      if (!response.ok) {
        const errorText = await response.text()
        console.error("[v0] Save config failed:", response.status, errorText)
        throw new Error(`HTTP ${response.status}: ${errorText}`)
      }

      const data = await response.json()
      console.log("[v0] Save config response:", data)

      if (data.error) {
        throw new Error(data.error)
      }

      setConfig(localConfig)
      setFastapiError(false)
      alert("Настройки сохранены успешно!")
    } catch (error) {
      console.error("[v0] Error saving config:", error)
      const errorMessage = error instanceof Error ? error.message : "Неизвестная ошибка"

      if (errorMessage.includes("503") || errorMessage.includes("Service Unavailable")) {
        alert(
          "Ошибка 503: FastAPI сервер недоступен. Проверьте, что сервер запущен и доступен по адресу " + FASTAPI_URL,
        )
      } else if (errorMessage.includes("connection_failed")) {
        alert("Ошибка подключения: Не удалось подключиться к FastAPI серверу. Проверьте URL и доступность сервера.")
      } else {
        alert(`Ошибка при сохранении настроек: ${errorMessage}`)
      }
    }
  }

  function resetConfig() {
    setLocalConfig(config)
  }

  if (loading) {
    return (
      <div className="flex min-h-[400px] items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    )
  }

  const lastSession = sessions[0]
  const needsRetraining = lastSession && lastSession.metrics.accuracy && lastSession.metrics.accuracy < 0.85

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold">Обучение модели распознавания лиц</h2>
        <p className="text-sm text-muted-foreground">Управление обучением InsightFace модели на verified faces</p>
      </div>

      {httpsRequired && (
        <div className="flex items-start gap-2 rounded-lg border border-destructive/50 bg-destructive/10 p-4">
          <AlertCircle className="h-5 w-5 text-destructive" />
          <div className="flex-1 space-y-2">
            <p className="text-sm font-medium text-destructive">FastAPI сервер требует HTTPS</p>
            <p className="text-xs text-muted-foreground">
              Сервер отклоняет HTTP запросы с сообщением "Invalid request, only https is supported"
            </p>
            <div className="mt-2 space-y-1 text-xs">
              <p className="font-medium">Решение 1: Измените переменную окружения</p>
              <p className="text-muted-foreground">
                В разделе "Vars" боковой панели измените FASTAPI_URL на https://23.88.61.20:8001
              </p>
              <p className="mt-2 font-medium">Решение 2: Настройте FastAPI сервер</p>
              <p className="text-muted-foreground">
                Отключите проверку HTTPS в настройках FastAPI сервера для разработки
              </p>
            </div>
          </div>
        </div>
      )}

      {fastapiError && !httpsRequired && (
        <div className="flex items-start gap-2 rounded-lg border border-amber-500/50 bg-amber-500/10 p-4">
          <AlertTriangle className="h-5 w-5 text-amber-500" />
          <div className="flex-1 space-y-2">
            <p className="text-sm font-medium text-amber-500">FastAPI сервер недоступен</p>
            <p className="text-xs text-muted-foreground">
              Не удалось подключиться к серверу обучения. Проверьте консоль браузера (F12) для деталей.
            </p>
            <div className="mt-2 space-y-1 text-xs">
              <p className="font-medium">Возможные причины:</p>
              <ul className="list-inside list-disc text-muted-foreground">
                <li>FastAPI сервер не запущен</li>
                <li>Неверный URL в переменной FASTAPI_URL</li>
                <li>Сервер недоступен из интернета (для деплоя)</li>
                <li>Порт закрыт файрволом</li>
              </ul>
              <p className="mt-2 font-medium">Текущий URL: {process.env.NEXT_PUBLIC_FASTAPI_URL || "не установлен"}</p>
            </div>
          </div>
        </div>
      )}

      {/* Training Control Section */}
      <Card>
        <CardHeader>
          <CardTitle>Переобучение модели</CardTitle>
          <CardDescription>Запустите обучение модели на подтвержденных лицах из базы данных</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {lastSession && (
            <div className="space-y-2 rounded-lg border p-4">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Последнее обучение:</span>
                <Badge variant={lastSession.status === "completed" ? "default" : "secondary"}>
                  {lastSession.status === "completed" ? "Завершено" : lastSession.status}
                </Badge>
              </div>
              <div className="grid grid-cols-2 gap-2 text-sm text-muted-foreground">
                <div>Дата: {new Date(lastSession.created_at).toLocaleString("ru-RU")}</div>
                <div>Режим: {lastSession.training_mode === "full" ? "Полное" : "Инкрементальное"}</div>
                <div>Людей: {lastSession.people_count}</div>
                <div>Лиц: {lastSession.faces_count}</div>
              </div>
              {lastSession.metrics.accuracy && (
                <div className="mt-2 space-y-1">
                  <div className="flex items-center justify-between text-sm">
                    <span>Accuracy:</span>
                    <span className="font-medium">{(lastSession.metrics.accuracy * 100).toFixed(1)}%</span>
                  </div>
                  <Progress value={lastSession.metrics.accuracy * 100} />
                </div>
              )}
            </div>
          )}

          {needsRetraining && (
            <div className="flex items-start gap-2 rounded-lg border border-destructive/50 bg-destructive/10 p-4">
              <AlertCircle className="h-5 w-5 text-destructive" />
              <div className="flex-1">
                <p className="text-sm font-medium text-destructive">Рекомендуется переобучение</p>
                <p className="text-xs text-muted-foreground">
                  Accuracy модели ниже 85%. Запустите переобучение для улучшения качества распознавания.
                </p>
              </div>
            </div>
          )}

          {datasetStats && (
            <div className="space-y-2 rounded-lg border bg-muted/50 p-4">
              <h4 className="text-sm font-medium">Статистика датасета:</h4>
              <div className="grid grid-cols-2 gap-2 text-sm text-muted-foreground">
                <div>Людей: {datasetStats.total_people}</div>
                <div>Лиц: {datasetStats.total_faces}</div>
                <div>Мин. лиц: {datasetStats.faces_per_person.min}</div>
                <div>Макс. лиц: {datasetStats.faces_per_person.max}</div>
                <div>Средн. лиц: {datasetStats.faces_per_person.avg.toFixed(1)}</div>
              </div>
            </div>
          )}

          {training && (
            <div className="space-y-2">
              <div className="flex items-center justify-between text-sm">
                <span>Обучение в процессе...</span>
                <span>{trainingProgress.toFixed(0)}%</span>
              </div>
              <Progress value={trainingProgress} />
            </div>
          )}

          <div className="flex gap-2">
            <Button onClick={prepareDataset} disabled={preparing || training || fastapiError} variant="outline">
              {preparing ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Подготовка...
                </>
              ) : (
                <>
                  <TrendingUp className="mr-2 h-4 w-4" />
                  Подготовить датасет
                </>
              )}
            </Button>
            <Button onClick={startTraining} disabled={preparing || training || fastapiError}>
              {training ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Обучение...
                </>
              ) : (
                <>
                  <CheckCircle2 className="mr-2 h-4 w-4" />
                  Запустить обучение
                </>
              )}
            </Button>
            <Button onClick={loadData} variant="outline" size="icon">
              <RefreshCw className="h-4 w-4" />
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Configuration Section */}
      <Card>
        <CardHeader>
          <CardTitle>Настройки распознавания</CardTitle>
          <CardDescription>Настройте пороги confidence и вес контекста для распознавания лиц</CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Quality Filtering section */}
          <div className="space-y-4">
            <h4 className="text-sm font-medium">Фильтрация качества</h4>
            <p className="text-xs text-muted-foreground">
              Отсеивайте лица низкого качества при детекции и распознавании
            </p>

            <div className="space-y-3">
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label className="text-sm">Минимальный det_score</Label>
                  <span className="text-sm font-medium">
                    {(localConfig.quality_filters?.min_detection_score || 0.7).toFixed(2)}
                  </span>
                </div>
                <Slider
                  value={[localConfig.quality_filters?.min_detection_score || 0.7]}
                  onValueChange={([value]) =>
                    setLocalConfig({
                      ...localConfig,
                      quality_filters: {
                        ...localConfig.quality_filters!,
                        min_detection_score: value,
                      },
                    })
                  }
                  min={0.5}
                  max={0.9}
                  step={0.05}
                />
                <p className="text-xs text-muted-foreground">
                  0.50 - очень мягкий | 0.70 - рекомендуемый | 0.90 - строгий
                </p>
              </div>

              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label className="text-sm">Минимальный размер лица (px)</Label>
                  <span className="text-sm font-medium">
                    {Math.round(localConfig.quality_filters?.min_face_size || 80)}
                  </span>
                </div>
                <Slider
                  value={[localConfig.quality_filters?.min_face_size || 80]}
                  onValueChange={([value]) =>
                    setLocalConfig({
                      ...localConfig,
                      quality_filters: {
                        ...localConfig.quality_filters!,
                        min_face_size: value,
                      },
                    })
                  }
                  min={30}
                  max={200}
                  step={10}
                />
                <p className="text-xs text-muted-foreground">
                  30px - очень мелкие лица | 80px - рекомендуемый | 200px - только крупные
                </p>
              </div>

              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label className="text-sm">Минимальная резкость (blur score)</Label>
                  <span className="text-sm font-medium">
                    {Math.round(localConfig.quality_filters?.min_blur_score || 100)}
                  </span>
                </div>
                <Slider
                  value={[localConfig.quality_filters?.min_blur_score || 100]}
                  onValueChange={([value]) =>
                    setLocalConfig({
                      ...localConfig,
                      quality_filters: {
                        ...localConfig.quality_filters!,
                        min_blur_score: value,
                      },
                    })
                  }
                  min={10}
                  max={150}
                  step={10}
                />
                <p className="text-xs text-muted-foreground">
                  10 - размытые лица | 60-80 - рекомендуемый | 150 - только четкие
                </p>
              </div>

              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label className="text-sm">Порог автоверификации (verified threshold)</Label>
                  <span className="text-sm font-medium">
                    {((localConfig.quality_filters?.verified_threshold || 0.99) * 100).toFixed(0)}%
                  </span>
                </div>
                <Slider
                  value={[(localConfig.quality_filters?.verified_threshold || 0.99) * 100]}
                  onValueChange={([value]) =>
                    setLocalConfig({
                      ...localConfig,
                      quality_filters: {
                        ...localConfig.quality_filters!,
                        verified_threshold: value / 100,
                      },
                    })
                  }
                  min={60}
                  max={99}
                  step={1}
                />
                <p className="text-xs text-muted-foreground">
                  60% - мягкий | 85% - средний | 99% - строгий (практически идентичные фото)
                </p>
              </div>
            </div>
          </div>

          {/* Confidence Thresholds */}
          <div className="space-y-4 pt-4 border-t">
            <h4 className="text-sm font-medium">Пороги Confidence для обучения</h4>

            <div className="space-y-3">
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label className="text-sm">3-4 лица (Low Data)</Label>
                  <span className="text-sm font-medium">{localConfig.confidence_thresholds.low_data.toFixed(2)}</span>
                </div>
                <Slider
                  value={[localConfig.confidence_thresholds.low_data]}
                  onValueChange={([value]) =>
                    setLocalConfig({
                      ...localConfig,
                      confidence_thresholds: { ...localConfig.confidence_thresholds, low_data: value },
                    })
                  }
                  min={0.5}
                  max={1.0}
                  step={0.05}
                />
              </div>

              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label className="text-sm">5-9 лиц (Medium Data)</Label>
                  <span className="text-sm font-medium">
                    {localConfig.confidence_thresholds.medium_data.toFixed(2)}
                  </span>
                </div>
                <Slider
                  value={[localConfig.confidence_thresholds.medium_data]}
                  onValueChange={([value]) =>
                    setLocalConfig({
                      ...localConfig,
                      confidence_thresholds: { ...localConfig.confidence_thresholds, medium_data: value },
                    })
                  }
                  min={0.5}
                  max={1.0}
                  step={0.05}
                />
              </div>

              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label className="text-sm">10+ лиц (High Data)</Label>
                  <span className="text-sm font-medium">{localConfig.confidence_thresholds.high_data.toFixed(2)}</span>
                </div>
                <Slider
                  value={[localConfig.confidence_thresholds.high_data]}
                  onValueChange={([value]) =>
                    setLocalConfig({
                      ...localConfig,
                      confidence_thresholds: { ...localConfig.confidence_thresholds, high_data: value },
                    })
                  }
                  min={0.5}
                  max={1.0}
                  step={0.05}
                />
              </div>
            </div>
          </div>

          {/* Save Threshold */}
          <div className="space-y-4 pt-4 border-t">
            <h4 className="text-sm font-medium">Порог сохранения в БД</h4>
            <p className="text-xs text-muted-foreground">
              При пакетном распознавании лица с уверенностью ниже порога сохраняются как "Неизвестные"
            </p>
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label className="text-sm">Минимальная уверенность для сохранения person_id</Label>
                <span className="text-sm font-medium">
                  {((localConfig.confidence_thresholds.high_data || 0.6) * 100).toFixed(0)}%
                </span>
              </div>
              <Slider
                value={[(localConfig.confidence_thresholds.high_data || 0.6) * 100]}
                onValueChange={([value]) =>
                  setLocalConfig({
                    ...localConfig,
                    confidence_thresholds: { ...localConfig.confidence_thresholds, high_data: value / 100 },
                  })
                }
                min={30}
                max={80}
                step={5}
              />
              <p className="text-xs text-muted-foreground">30% - очень мягкий | 60% - рекомендуемый | 80% - строгий</p>
            </div>
          </div>

          {/* Context Weight */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label>Context Weight</Label>
              <span className="text-sm font-medium">{localConfig.context_weight.toFixed(2)}</span>
            </div>
            <Slider
              value={[localConfig.context_weight]}
              onValueChange={([value]) => setLocalConfig({ ...localConfig, context_weight: value })}
              min={0}
              max={0.5}
              step={0.05}
            />
            <p className="text-xs text-muted-foreground">
              0.00 - контекст не учитывается | 0.10 - минимальное влияние | 0.30 - среднее влияние | 0.50 - максимальное
              влияние
            </p>
          </div>

          <div className="flex gap-2">
            <Button onClick={saveConfig} disabled={fastapiError}>
              Сохранить настройки
            </Button>
            <Button onClick={resetConfig} variant="outline">
              Сбросить
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Training History */}
      <Card>
        <CardHeader>
          <CardTitle>История обучений</CardTitle>
          <CardDescription>Последние 10 сессий обучения модели</CardDescription>
        </CardHeader>
        <CardContent>
          <TrainingHistoryList sessions={sessions} />
        </CardContent>
      </Card>

      {/* Statistics */}
      <TrainingStatsCard />
    </div>
  )
}
