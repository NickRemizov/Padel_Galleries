"use client"

import { useAuth } from "@/lib/auth-context"
import { TelegramLoginButton } from "./telegram-login-button"
import { UserMenu } from "./user-menu"
import { Button } from "./ui/button"
import { LogIn } from "lucide-react"
import { useState } from "react"
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "./ui/dialog"

export function AuthButton() {
  const { user, loading, login } = useAuth()
  const [showLoginDialog, setShowLoginDialog] = useState(false)

  const botName = process.env.NEXT_PUBLIC_TELEGRAM_BOT_USERNAME

  if (loading) {
    return <div className="h-10 w-10 animate-pulse rounded-full bg-muted" />
  }

  if (user) {
    return <UserMenu />
  }

  if (!botName) {
    return null
  }

  return (
    <>
      <Button variant="outline" size="sm" onClick={() => setShowLoginDialog(true)}>
        <LogIn className="mr-2 h-4 w-4" />
        Войти
      </Button>

      <Dialog open={showLoginDialog} onOpenChange={setShowLoginDialog}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>Вход через Telegram</DialogTitle>
            <DialogDescription>
              Войдите с помощью вашего аккаунта Telegram, чтобы оставлять комментарии и лайки
            </DialogDescription>
          </DialogHeader>
          <div className="flex justify-center py-4">
            <TelegramLoginButton
              botName={botName}
              onAuth={(user) => {
                login()
                setShowLoginDialog(false)
              }}
              requestAccess={false}
            />
          </div>
        </DialogContent>
      </Dialog>
    </>
  )
}
