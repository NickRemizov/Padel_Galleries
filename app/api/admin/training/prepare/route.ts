import { type NextRequest, NextResponse } from "next/server"
import { apiFetch, ApiError } from "@/lib/apiClient"
import { requireAdmin } from "@/lib/auth/serverGuard"

export async function POST(request: NextRequest) {
  const { error: authError } = await requireAdmin()
  if (authError) return authError

  try {
    const body = await request.json()

    console.log("[v0] Preparing dataset")
    console.log("[v0] Request body:", JSON.stringify(body))

    const data = await apiFetch("/api/v2/train/prepare", {
      method: "POST",
      body: JSON.stringify(body),
    })

    console.log("[v0] Successfully prepared dataset")
    return NextResponse.json(data)
  } catch (error) {
    console.error("[v0] Error preparing dataset:", error)

    if (error instanceof ApiError) {
      if (error.status === 503 || error.status === 404) {
        return NextResponse.json(
          {
            error:
              "FastAPI training endpoints not available. Please deploy the updated FastAPI code with training endpoints.",
          },
          { status: 503 },
        )
      }

      return NextResponse.json({ error: error.message }, { status: error.status })
    }

    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Failed to prepare dataset" },
      { status: 500 },
    )
  }
}
