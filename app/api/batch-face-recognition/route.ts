import { createClient } from "@/lib/supabase/server"
import { NextResponse } from "next/server"

export async function POST(request: Request) {
  try {
    const supabase = await createClient()

    // Check authentication
    const {
      data: { user },
    } = await supabase.auth.getUser()

    if (!user) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 })
    }

    const { galleryIds } = await request.json()

    // Get all images from specified galleries
    let query = supabase.from("gallery_images").select("id, image_url, original_filename, gallery_id")

    if (galleryIds && galleryIds.length > 0) {
      query = query.in("gallery_id", galleryIds)
    }

    const { data: images, error } = await query

    if (error) {
      console.error("[v0] Error fetching images:", error)
      return NextResponse.json({ error: error.message }, { status: 500 })
    }

    return NextResponse.json({
      success: true,
      images: images || [],
      count: images?.length || 0,
    })
  } catch (error) {
    console.error("[v0] Batch recognition error:", error)
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}
