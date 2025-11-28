import { type NextRequest, NextResponse } from "next/server"
import { createClient } from "@/lib/supabase/server"
import { verifyTelegramAuth, isTelegramAuthDataValid } from "@/lib/telegram-auth"

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { id, first_name, last_name, username, photo_url, auth_date, hash } = body

    // Verify bot token exists
    const botToken = process.env.TELEGRAM_BOT_TOKEN
    if (!botToken) {
      return NextResponse.json({ error: "Telegram bot token not configured" }, { status: 500 })
    }

    // Verify Telegram auth data
    const authData = { id: String(id), first_name, auth_date: String(auth_date), hash }
    if (last_name) authData.last_name = last_name
    if (username) authData.username = username
    if (photo_url) authData.photo_url = photo_url

    if (!verifyTelegramAuth(authData, botToken)) {
      return NextResponse.json({ error: "Invalid authentication data" }, { status: 401 })
    }

    // Check if auth data is not expired
    if (!isTelegramAuthDataValid(auth_date)) {
      return NextResponse.json({ error: "Authentication data expired" }, { status: 401 })
    }

    // Create or update user in database
    const supabase = await createClient()

    const { data: existingUser } = await supabase.from("users").select("*").eq("telegram_id", id).single()

    let user
    if (existingUser) {
      // Update existing user
      const { data, error } = await supabase
        .from("users")
        .update({
          username,
          first_name,
          last_name,
          photo_url,
          updated_at: new Date().toISOString(),
        })
        .eq("telegram_id", id)
        .select()
        .single()

      if (error) throw error
      user = data
    } else {
      // Create new user
      const { data, error } = await supabase
        .from("users")
        .insert({
          telegram_id: id,
          username,
          first_name,
          last_name,
          photo_url,
        })
        .select()
        .single()

      if (error) throw error
      user = data
    }

    // Create session cookie
    const response = NextResponse.json({ user })
    response.cookies.set("telegram_user", JSON.stringify(user), {
      httpOnly: true,
      secure: process.env.NODE_ENV === "production",
      sameSite: "lax",
      maxAge: 60 * 60 * 24 * 30, // 30 days
      path: "/",
    })

    return response
  } catch (error) {
    console.error("[v0] Telegram auth error:", error)
    return NextResponse.json({ error: "Authentication failed" }, { status: 500 })
  }
}
