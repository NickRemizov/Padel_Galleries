"use client"

import { useEffect, useRef } from "react"

interface TelegramLoginButtonProps {
  botName: string
  onAuth: (user: any) => void
  buttonSize?: "large" | "medium" | "small"
  cornerRadius?: number
  requestAccess?: boolean
}

export function TelegramLoginButton({
  botName,
  onAuth,
  buttonSize = "large",
  cornerRadius = 10,
  requestAccess = false,
}: TelegramLoginButtonProps) {
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    // Create callback function
    window.TelegramLoginWidget = {
      dataOnauth: async (user: any) => {
        try {
          const response = await fetch("/api/auth/telegram", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(user),
          })

          if (response.ok) {
            const data = await response.json()
            onAuth(data.user)
          }
        } catch (error) {
          console.error("[v0] Telegram login error:", error)
        }
      },
    }

    // Load Telegram widget script
    const script = document.createElement("script")
    script.src = "https://telegram.org/js/telegram-widget.js?22"
    script.setAttribute("data-telegram-login", botName)
    script.setAttribute("data-size", buttonSize)
    script.setAttribute("data-radius", String(cornerRadius))
    script.setAttribute("data-onauth", "TelegramLoginWidget.dataOnauth(user)")
    script.setAttribute("data-request-access", requestAccess ? "write" : "")
    script.async = true

    if (containerRef.current) {
      containerRef.current.appendChild(script)
    }

    return () => {
      if (containerRef.current) {
        containerRef.current.innerHTML = ""
      }
      delete window.TelegramLoginWidget
    }
  }, [botName, buttonSize, cornerRadius, requestAccess, onAuth])

  return <div ref={containerRef} />
}

declare global {
  interface Window {
    TelegramLoginWidget?: {
      dataOnauth: (user: any) => void
    }
  }
}
