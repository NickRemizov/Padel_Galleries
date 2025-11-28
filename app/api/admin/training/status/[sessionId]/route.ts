import { type NextRequest, NextResponse } from "next/server"
import { apiFetch, ApiError } from "@/lib/apiClient"
import { requireAdmin } from "@/lib/auth/serverGuard"

export async function GET(request: NextRequest, { params }: { params: { sessionId: string } }) {
  const { error: authError } = await requireAdmin()
  if (authError) return authError

  try {
    const { sessionId } = params

    const data = await apiFetch(`/api/v2/train/status/${sessionId}`)

    return NextResponse.json(data)
  } catch (error) {
    console.error("[v0] Error fetching training status:", error)

    if (error instanceof ApiError) {
      return NextResponse.json(
        { error: "Failed to fetch training status", details: error.message },
        { status: error.status },
      )
    }

    return NextResponse.json({ error: "Failed to fetch training status" }, { status: 500 })
  }
}
