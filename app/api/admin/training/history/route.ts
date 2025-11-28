import { type NextRequest, NextResponse } from "next/server"
import { apiFetch, ApiError } from "@/lib/apiClient"
import { requireAdmin } from "@/lib/auth/serverGuard"

export async function GET(request: NextRequest) {
  const { error: authError } = await requireAdmin()
  if (authError) return authError

  try {
    const { searchParams } = new URL(request.url)
    const limit = searchParams.get("limit") || "10"
    const offset = searchParams.get("offset") || "0"

    const data = await apiFetch(`/api/v2/train/history?limit=${limit}&offset=${offset}`)

    return NextResponse.json(data)
  } catch (error) {
    console.error("[v0] Error fetching training history:", error)

    if (error instanceof ApiError) {
      return NextResponse.json({ sessions: [], total: 0 })
    }

    return NextResponse.json({ sessions: [], total: 0 })
  }
}
