import { type NextRequest, NextResponse } from "next/server"
import { apiFetch, ApiError } from "@/lib/apiClient"

export async function POST(request: NextRequest) {
  try {
    console.log("[v2.20] Proxy: Received recognize-face request")

    const body = await request.json()
    console.log("[v2.20] Proxy: Embedding length:", body.embedding?.length || 0)
    console.log("[v2.20] Proxy: Confidence threshold:", body.confidence_threshold)

    const data = await apiFetch("/recognize-face", {
      method: "POST",
      body: JSON.stringify(body),
    })

    console.log("[v2.20] Proxy: Recognition result:", data)

    return NextResponse.json(data)
  } catch (error) {
    console.error("[v2.20] Proxy: Error:", error)

    if (error instanceof ApiError) {
      return NextResponse.json({ error: "Failed to recognize face", details: error.message }, { status: error.status })
    }

    return NextResponse.json(
      { error: "Internal server error", details: error instanceof Error ? error.message : "Unknown error" },
      { status: 500 },
    )
  }
}
