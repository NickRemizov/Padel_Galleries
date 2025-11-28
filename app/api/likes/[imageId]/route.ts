import { type NextRequest, NextResponse } from "next/server"
import { createClient } from "@/lib/supabase/server"

// Get likes for an image
export async function GET(request: NextRequest, { params }: { params: Promise<{ imageId: string }> }) {
  try {
    const { imageId } = await params
    const supabase = await createClient()

    // Get total likes count
    const { count, error: countError } = await supabase
      .from("likes")
      .select("*", { count: "exact", head: true })
      .eq("image_id", imageId)

    if (countError) throw countError

    // Check if current user liked this image
    const userCookie = request.cookies.get("telegram_user")
    let isLiked = false

    if (userCookie) {
      try {
        const user = JSON.parse(userCookie.value)
        const { data: userLike } = await supabase
          .from("likes")
          .select("id")
          .eq("image_id", imageId)
          .eq("user_id", user.id)
          .single()

        isLiked = !!userLike
      } catch {
        // User not authenticated or error parsing cookie
      }
    }

    return NextResponse.json({
      count: count || 0,
      isLiked,
    })
  } catch (error) {
    console.error("[v0] Error fetching likes:", error)
    return NextResponse.json({ error: "Failed to fetch likes" }, { status: 500 })
  }
}

// Toggle like for an image
export async function POST(request: NextRequest, { params }: { params: Promise<{ imageId: string }> }) {
  try {
    const { imageId } = await params
    const supabase = await createClient()

    // Get user from cookie
    const userCookie = request.cookies.get("telegram_user")
    if (!userCookie) {
      return NextResponse.json({ error: "Authentication required" }, { status: 401 })
    }

    const user = JSON.parse(userCookie.value)

    // Check if user already liked this image
    const { data: existingLike } = await supabase
      .from("likes")
      .select("id")
      .eq("image_id", imageId)
      .eq("user_id", user.id)
      .single()

    if (existingLike) {
      // Unlike - remove the like
      const { error } = await supabase.from("likes").delete().eq("id", existingLike.id)

      if (error) throw error

      // Get updated count
      const { count } = await supabase.from("likes").select("*", { count: "exact", head: true }).eq("image_id", imageId)

      return NextResponse.json({
        isLiked: false,
        count: count || 0,
      })
    } else {
      // Like - add the like
      const { error } = await supabase.from("likes").insert({
        image_id: imageId,
        user_id: user.id,
      })

      if (error) throw error

      // Get updated count
      const { count } = await supabase.from("likes").select("*", { count: "exact", head: true }).eq("image_id", imageId)

      return NextResponse.json({
        isLiked: true,
        count: count || 0,
      })
    }
  } catch (error) {
    console.error("[v0] Error toggling like:", error)
    return NextResponse.json({ error: "Failed to toggle like" }, { status: 500 })
  }
}
