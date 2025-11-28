import { NextResponse } from "next/server"
import { createClient } from "@/lib/supabase/server"
import { cookies } from "next/headers"

// GET /api/favorites - Get all favorites for current user
export async function GET() {
  try {
    const cookieStore = await cookies()
    const userId = cookieStore.get("user_id")?.value

    if (!userId) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 })
    }

    const supabase = await createClient()

    const { data: favorites, error } = await supabase
      .from("favorites")
      .select(
        `
        *,
        gallery_images (
          id,
          gallery_id,
          image_url,
          original_url,
          original_filename,
          file_size,
          width,
          height,
          created_at
        )
      `,
      )
      .eq("user_id", userId)
      .order("created_at", { ascending: false })

    if (error) {
      console.error("[v0] Error fetching favorites:", error)
      return NextResponse.json({ error: "Failed to fetch favorites" }, { status: 500 })
    }

    return NextResponse.json({ favorites: favorites || [] })
  } catch (error) {
    console.error("[v0] Error in GET /api/favorites:", error)
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}
