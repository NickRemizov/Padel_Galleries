import { type NextRequest, NextResponse } from "next/server"
import { createClient } from "@/lib/supabase/server"
import { cookies } from "next/headers"

// DELETE /api/comments/[imageId]/[commentId] - Delete a comment
export async function DELETE(
  request: NextRequest,
  { params }: { params: Promise<{ imageId: string; commentId: string }> },
) {
  try {
    const { commentId } = await params
    const cookieStore = await cookies()
    const userId = cookieStore.get("user_id")?.value

    if (!userId) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 })
    }

    const supabase = await createClient()

    // Check if comment belongs to user
    const { data: comment } = await supabase.from("comments").select("user_id").eq("id", commentId).single()

    if (!comment || comment.user_id !== userId) {
      return NextResponse.json({ error: "Forbidden" }, { status: 403 })
    }

    const { error } = await supabase.from("comments").delete().eq("id", commentId)

    if (error) {
      console.error("[v0] Error deleting comment:", error)
      return NextResponse.json({ error: "Failed to delete comment" }, { status: 500 })
    }

    return NextResponse.json({ success: true })
  } catch (error) {
    console.error("[v0] Error in DELETE /api/comments:", error)
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}
