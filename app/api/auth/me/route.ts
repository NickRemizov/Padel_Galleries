import { type NextRequest, NextResponse } from "next/server"

export async function GET(request: NextRequest) {
  const userCookie = request.cookies.get("telegram_user")

  if (!userCookie) {
    return NextResponse.json({ user: null })
  }

  try {
    const user = JSON.parse(userCookie.value)
    return NextResponse.json({ user })
  } catch {
    return NextResponse.json({ user: null })
  }
}
