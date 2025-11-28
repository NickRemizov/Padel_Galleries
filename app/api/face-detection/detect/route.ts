import { type NextRequest, NextResponse } from "next/server"
import { apiFetch, ApiError } from "@/lib/apiClient"

export async function POST(request: NextRequest) {
  try {
    console.log("[v2.21] Proxy: Received detect-faces request")

    const body = await request.json()
    console.log("[v2.21] Proxy: Image URL:", body.image_url)
    console.log("[v2.21] Proxy: Apply quality filters:", body.apply_quality_filters)

    const data = await apiFetch("/detect-faces", {
      method: "POST",
      body: JSON.stringify(body),
    })

    console.log("[v2.21] Proxy: Detected faces:", data.faces?.length || 0)

    return NextResponse.json(data)
  } catch (error) {
    console.error("[v2.21] Proxy: Error:", error)

    if (error instanceof ApiError) {
      return NextResponse.json({ error: "Failed to detect faces", details: error.message }, { status: error.status })
    }

    return NextResponse.json(
      { error: "Internal server error", details: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 },
    )
  }
}
