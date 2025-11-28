import { env } from "@/lib/env"
import { NextResponse } from "next/server"

export async function GET() {
  try {
    const res = await fetch(`${env.FASTAPI_URL}/api/crud/organizers`, {
      cache: "no-store",
      headers: {
        "X-API-Key": env.API_SECRET_KEY,
      },
    })

    if (!res.ok) {
      console.error("[v0] Python API error:", res.status)
      return NextResponse.json([], { status: res.status })
    }

    const data = await res.json()
    return NextResponse.json(data)
  } catch (error) {
    console.error("[v0] Failed to fetch organizers:", error)
    return NextResponse.json([], { status: 500 })
  }
}
