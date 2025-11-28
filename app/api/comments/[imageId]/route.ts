import { type NextRequest, NextResponse } from "next/server"
import { createClient } from "@/lib/supabase/server"
import { cookies } from "next/headers"

// GET /api/comments/[imageId] - Get all comments for an image
export async function GET(request: NextRequest, { params }: { params: Promise<{ imageId: string }> }) {
  try {
    const { imageId } = await params
    const supabase = await createClient()

    const { data: comments, error } = await supabase
      .from("comments")
      .select(
        `
        *,
        users (
          id,
          telegram_id,
          username,
          first_name,
          last_name,
          photo_url
        )
      `,
      )
      .eq("gallery_image_id", imageId)
      .order("created_at", { ascending: true })

    if (error) {
      console.error("[v0] Error fetching comments:", error)
      return NextResponse.json({ error: "Failed to fetch comments" }, { status: 500 })
    }

    return NextResponse.json({ comments: comments || [] })
  } catch (error) {
    console.error("[v0] Error in GET /api/comments:", error)
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}

// POST /api/comments/[imageId] - Add a new comment
export async function POST(request: NextRequest, { params }: { params: Promise<{ imageId: string }> }) {
  try {
    const { imageId } = await params
    const cookieStore = await cookies()
    const userId = cookieStore.get("user_id")?.value

    if (!userId) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 })
    }

    const { content } = await request.json()

    if (!content || content.trim().length === 0) {
      return NextResponse.json({ error: "Comment content is required" }, { status: 400 })
    }

    if (content.length > 1000) {
      return NextResponse.json({ error: "Comment is too long (max 1000 characters)" }, { status: 400 })
    }

    const supabase = await createClient()

    const { data: comment, error } = await supabase
      .from("comments")
      .insert({
        gallery_image_id: imageId,
        user_id: userId,
        content: content.trim(),
      })
      .select(
        `
        *,
        users (
          id,
          telegram_id,
          username,
          first_name,
          last_name,
          photo_url
        )
      `,
      )
      .single()

    if (error) {
      console.error("[v0] Error creating comment:", error)
      return NextResponse.json({ error: "Failed to create comment" }, { status: 500 })
    }

    return NextResponse.json({ comment })
  } catch (error) {
    console.error("[v0] Error in POST /api/comments:", error)
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}
