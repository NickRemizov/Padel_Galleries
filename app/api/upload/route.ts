export const runtime = "nodejs"
export const maxDuration = 60 // 60 seconds for upload
export const bodyParser = false // Disable default body parser

import { put } from "@vercel/blob"
import { type NextRequest, NextResponse } from "next/server"

export const config = {
  api: {
    bodyParser: {
      sizeLimit: "50mb",
    },
  },
}

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData()
    const file = formData.get("file") as File

    if (!file) {
      return NextResponse.json({ error: "No file provided" }, { status: 400 })
    }

    const MAX_FILE_SIZE = 50 * 1024 * 1024 // 50 MB
    if (file.size > MAX_FILE_SIZE) {
      return NextResponse.json(
        { error: `Файл слишком большой. Максимальный размер: ${MAX_FILE_SIZE / (1024 * 1024)} МБ` },
        { status: 413 },
      )
    }

    console.log("[v0] Uploading file:", file.name, "size:", file.size)

    // Upload to Vercel Blob
    const blob = await put(file.name, file, {
      access: "public",
      addRandomSuffix: true,
    })

    console.log("[v0] File uploaded:", blob.url)

    return NextResponse.json({
      url: blob.url,
      filename: file.name,
      size: file.size,
      type: file.type,
    })
  } catch (error) {
    console.error("Upload error:", error)

    if (error instanceof Error) {
      // Check if it's a quota exceeded error
      if (error.message.includes("Storage quota exceeded")) {
        return NextResponse.json(
          {
            error: "Квота хранилища исчерпана. Пожалуйста, удалите старые изображения или обновите план.",
            code: "QUOTA_EXCEEDED",
          },
          { status: 507 },
        )
      }

      return NextResponse.json({ error: error.message || "Upload failed" }, { status: 500 })
    }

    return NextResponse.json({ error: "Upload failed" }, { status: 500 })
  }
}
