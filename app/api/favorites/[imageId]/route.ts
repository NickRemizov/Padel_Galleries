import { type NextRequest, NextResponse } from "next/server"
import { createClient } from "@/lib/supabase/server"
import { cookies } from "next/headers"

// GET /api/favorites/[imageId] - Check if image is favorited
export async function GET(request: NextRequest, { params }: { params: Promise<{ imageId: string }> }) {
  try {
    const { imageId } = await params
    const cookieStore = await cookies()
    const userId = cookieStore.get("user_id")?.value

    if (!userId) {
      return NextResponse.json({ isFavorited: false })
    }

    const supabase = await createClient()

    const { data, error } = await supabase
      .from("favorites")
      .select("id")
      .eq("user_id", userId)
      .eq("gallery_image_id", imageId)
      .single()

    if (error && error.code !== "PGRST116") {
      console.error("[v0] Error checking favorite:", error)
      return NextResponse.json({ error: "Failed to check favorite" }, { status: 500 })
    }

    return NextResponse.json({ isFavorited: !!data })
  } catch (error) {
    console.error("[v0] Error in GET /api/favorites:", error)
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}

// POST /api/favorites/[imageId] - Toggle favorite
export async function POST(request: NextRequest, { params }: { params: Promise<{ imageId: string }> }) {
  try {
    const { imageId } = await params
    const cookieStore = await cookies()
    const userId = cookieStore.get("user_id")?.value

    if (!userId) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 })
    }

    const supabase = await createClient()

    // Check if already favorited
    const { data: existing } = await supabase
      .from("favorites")
      .select("id")
      .eq("user_id", userId)
      .eq("gallery_image_id", imageId)
      .single()

    if (existing) {
      // Remove from favorites
      const { error } = await supabase.from("favorites").delete().eq("id", existing.id)

      if (error) {
        console.error("[v0] Error removing favorite:", error)
        return NextResponse.json({ error: "Failed to remove favorite" }, { status: 500 })
      }

      return NextResponse.json({ isFavorited: false })
    } else {
      // Add to favorites
      const { error } = await supabase.from("favorites").insert({
        user_id: userId,
        gallery_image_id: imageId,
      })

      if (error) {
        console.error("[v0] Error adding favorite:", error)
        return NextResponse.json({ error: "Failed to add favorite" }, { status: 500 })
      }

      return NextResponse.json({ isFavorited: true })
    }
  } catch (error) {
    console.error("[v0] Error in POST /api/favorites:", error)
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}
