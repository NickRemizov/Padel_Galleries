"use client"

import { useEffect, useState } from "react"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { ArrowLeft, RefreshCw } from "lucide-react"
import Link from "next/link"

interface Statistics {
  overall: {
    total_people: number
    total_verified_faces: number
    avg_faces_per_person: string
    min_faces: number
    max_faces: number
  }
  distribution: Array<{
    threshold: string
    people_count: number
    total_faces: number
    percentage: string
  }>
  top_people: Array<{
    real_name: string | null
    telegram_name: string | null
    count: number
  }>
  histogram: Array<{
    face_range: string
    people_count: number
    total_faces: number
  }>
}

export default function FaceStatisticsPage() {
  const [stats, setStats] = useState<Statistics | null>(null)
  const [loading, setLoading] = useState(true)

  const fetchStats = async () => {
    setLoading(true)
    try {
      const response = await fetch("/api/admin/face-statistics")
      const data = await response.json()
      setStats(data)
    } catch (error) {
      console.error("Error fetching statistics:", error)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchStats()
  }, [])

  if (loading) {
    return (
      <div className="container mx-auto p-6">
        <div className="flex items-center justify-center min-h-[400px]">
          <RefreshCw className="h-8 w-8 animate-spin text-muted-foreground" />
        </div>
      </div>
    )
  }

  if (!stats) {
    return (
      <div className="container mx-auto p-6">
        <div className="text-center text-muted-foreground">Failed to load statistics</div>
      </div>
    )
  }

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <Link href="/admin">
            <Button variant="ghost" size="icon">
              <ArrowLeft className="h-5 w-5" />
            </Button>
          </Link>
          <h1 className="text-3xl font-bold">Face Training Statistics</h1>
        </div>
        <Button onClick={fetchStats} variant="outline" size="sm">
          <RefreshCw className="h-4 w-4 mr-2" />
          Refresh
        </Button>
      </div>

      {/* Overall Statistics */}
      <Card className="p-6">
        <h2 className="text-xl font-semibold mb-4">Overall Statistics</h2>
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
          <div>
            <div className="text-sm text-muted-foreground">Total People</div>
            <div className="text-2xl font-bold">{stats.overall.total_people}</div>
          </div>
          <div>
            <div className="text-sm text-muted-foreground">Total Faces</div>
            <div className="text-2xl font-bold">{stats.overall.total_verified_faces}</div>
          </div>
          <div>
            <div className="text-sm text-muted-foreground">Avg per Person</div>
            <div className="text-2xl font-bold">{stats.overall.avg_faces_per_person}</div>
          </div>
          <div>
            <div className="text-sm text-muted-foreground">Min Faces</div>
            <div className="text-2xl font-bold">{stats.overall.min_faces}</div>
          </div>
          <div>
            <div className="text-sm text-muted-foreground">Max Faces</div>
            <div className="text-2xl font-bold">{stats.overall.max_faces}</div>
          </div>
        </div>
      </Card>

      {/* Distribution by Thresholds */}
      <Card className="p-6">
        <h2 className="text-xl font-semibold mb-4">Distribution by Minimum Faces</h2>
        <div className="space-y-3">
          {stats.distribution.map((item) => (
            <div key={item.threshold} className="flex items-center justify-between p-3 bg-muted/50 rounded-lg">
              <div className="flex items-center gap-4">
                <div className="font-medium min-w-[120px]">{item.threshold}</div>
                <div className="text-sm text-muted-foreground">
                  {item.people_count} people ({item.percentage}%)
                </div>
              </div>
              <div className="text-sm text-muted-foreground">{item.total_faces} total faces</div>
            </div>
          ))}
        </div>
      </Card>

      {/* Histogram */}
      <Card className="p-6">
        <h2 className="text-xl font-semibold mb-4">Distribution Histogram</h2>
        <div className="space-y-2">
          {stats.histogram.map((item) => (
            <div key={item.face_range} className="flex items-center gap-4">
              <div className="min-w-[100px] text-sm font-medium">{item.face_range}</div>
              <div className="flex-1">
                <div className="flex items-center gap-2">
                  <div
                    className="h-8 bg-primary rounded transition-all"
                    style={{
                      width: `${(item.people_count / stats.overall.total_people) * 100}%`,
                      minWidth: item.people_count > 0 ? "20px" : "0",
                    }}
                  />
                  <div className="text-sm text-muted-foreground">
                    {item.people_count} people ({item.total_faces} faces)
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </Card>

      {/* Top People */}
      <Card className="p-6">
        <h2 className="text-xl font-semibold mb-4">Top 20 People by Face Count</h2>
        <div className="space-y-2">
          {stats.top_people.map((person, index) => (
            <div key={index} className="flex items-center justify-between p-2 hover:bg-muted/50 rounded">
              <div className="flex items-center gap-3">
                <div className="text-sm text-muted-foreground w-8">#{index + 1}</div>
                <div className="font-medium">{person.real_name || person.telegram_name || "Unknown"}</div>
                {person.telegram_name && person.real_name && (
                  <div className="text-sm text-muted-foreground">@{person.telegram_name}</div>
                )}
              </div>
              <div className="text-sm font-medium">{person.count} faces</div>
            </div>
          ))}
        </div>
      </Card>

      {/* Recommendations */}
      <Card className="p-6 bg-blue-50 dark:bg-blue-950/20 border-blue-200 dark:border-blue-900">
        <h2 className="text-xl font-semibold mb-4">Training Recommendations</h2>
        <div className="space-y-2 text-sm">
          <div>
            <strong>Minimum 3 faces:</strong>{" "}
            {stats.distribution.find((d) => d.threshold === ">= 3 faces")?.people_count} people (
            {stats.distribution.find((d) => d.threshold === ">= 3 faces")?.percentage}%) - Good coverage, lower accuracy
          </div>
          <div>
            <strong>Minimum 5 faces:</strong>{" "}
            {stats.distribution.find((d) => d.threshold === ">= 5 faces")?.people_count} people (
            {stats.distribution.find((d) => d.threshold === ">= 5 faces")?.percentage}%) - Balanced approach
            (recommended)
          </div>
          <div>
            <strong>Minimum 10 faces:</strong>{" "}
            {stats.distribution.find((d) => d.threshold === ">= 10 faces")?.people_count} people (
            {stats.distribution.find((d) => d.threshold === ">= 10 faces")?.percentage}%) - High accuracy, limited
            coverage
          </div>
        </div>
      </Card>
    </div>
  )
}
