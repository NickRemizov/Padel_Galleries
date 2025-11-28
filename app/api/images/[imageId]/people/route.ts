import { createClient } from "@/lib/supabase/server"
import { NextResponse } from "next/server"

export async function GET(request: Request, { params }: { params: { imageId: string } }) {
  try {
    const supabase = await createClient()

    const { data, error } = await supabase
      .from("photo_faces")
      .select(
        `
        person_id,
        people!inner (
          id,
          real_name,
          telegram_name
        )
      `,
      )
      .eq("photo_id", params.imageId)
      .eq("verified", true)

    if (error) {
      console.error("[v0] Error fetching verified people:", error)
      return NextResponse.json({ error: error.message }, { status: 500 })
    }

    const people = data.map((item: any) => ({
      id: item.people.id,
      name: item.people.real_name || item.people.telegram_name || "Unknown",
    }))

    return NextResponse.json(people)
  } catch (error) {
    console.error("[v0] Error in people API:", error)
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}
