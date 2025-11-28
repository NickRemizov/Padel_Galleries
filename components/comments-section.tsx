"use client"

import type React from "react"

import { useState, useEffect } from "react"
import { useAuth } from "@/lib/auth-context"
import { Button } from "./ui/button"
import { Textarea } from "./ui/textarea"
import { Avatar, AvatarFallback, AvatarImage } from "./ui/avatar"
import { MessageCircle, Send, Trash2 } from "lucide-react"
import type { Comment } from "@/lib/types"
import { cn } from "@/lib/utils"

interface CommentsSectionProps {
  imageId: string
  className?: string
}

export function CommentsSection({ imageId, className }: CommentsSectionProps) {
  const { user } = useAuth()
  const [comments, setComments] = useState<Comment[]>([])
  const [newComment, setNewComment] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [isSubmitting, setIsSubmitting] = useState(false)

  useEffect(() => {
    fetchComments()
  }, [imageId])

  async function fetchComments() {
    setIsLoading(true)
    try {
      const response = await fetch(`/api/comments/${imageId}`)
      const data = await response.json()
      setComments(data.comments || [])
    } catch (error) {
      console.error("[v0] Error fetching comments:", error)
    } finally {
      setIsLoading(false)
    }
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()

    if (!user) {
      alert("Войдите через Telegram, чтобы оставлять комментарии")
      return
    }

    if (!newComment.trim()) return

    setIsSubmitting(true)
    try {
      const response = await fetch(`/api/comments/${imageId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content: newComment }),
      })

      if (response.ok) {
        const data = await response.json()
        setComments([...comments, data.comment])
        setNewComment("")
      } else {
        const error = await response.json()
        alert(error.error || "Не удалось добавить комментарий")
      }
    } catch (error) {
      console.error("[v0] Error submitting comment:", error)
      alert("Произошла ошибка при добавлении комментария")
    } finally {
      setIsSubmitting(false)
    }
  }

  async function handleDelete(commentId: string) {
    if (!confirm("Удалить комментарий?")) return

    try {
      const response = await fetch(`/api/comments/${imageId}/${commentId}`, {
        method: "DELETE",
      })

      if (response.ok) {
        setComments(comments.filter((c) => c.id !== commentId))
      }
    } catch (error) {
      console.error("[v0] Error deleting comment:", error)
    }
  }

  function formatDate(dateString: string) {
    const date = new Date(dateString)
    const now = new Date()
    const diffMs = now.getTime() - date.getTime()
    const diffMins = Math.floor(diffMs / 60000)
    const diffHours = Math.floor(diffMs / 3600000)
    const diffDays = Math.floor(diffMs / 86400000)

    if (diffMins < 1) return "только что"
    if (diffMins < 60) return `${diffMins} мин назад`
    if (diffHours < 24) return `${diffHours} ч назад`
    if (diffDays < 7) return `${diffDays} д назад`

    return date.toLocaleDateString("ru-RU", {
      day: "numeric",
      month: "short",
      year: date.getFullYear() !== now.getFullYear() ? "numeric" : undefined,
    })
  }

  return (
    <div className={cn("flex flex-col gap-4", className)}>
      <div className="flex items-center gap-2">
        <MessageCircle className="h-5 w-5" />
        <h3 className="font-semibold">Комментарии ({comments.length})</h3>
      </div>

      {/* Comment form */}
      {user ? (
        <form onSubmit={handleSubmit} className="flex gap-2">
          <Avatar className="h-8 w-8 shrink-0">
            <AvatarImage src={user.photo_url || undefined} />
            <AvatarFallback>{user.first_name?.charAt(0) || "?"}</AvatarFallback>
          </Avatar>
          <div className="flex-1 flex gap-2">
            <Textarea
              value={newComment}
              onChange={(e) => setNewComment(e.target.value)}
              placeholder="Добавить комментарий..."
              className="min-h-[60px] resize-none"
              maxLength={1000}
              disabled={isSubmitting}
            />
            <Button type="submit" size="icon" disabled={isSubmitting || !newComment.trim()}>
              <Send className="h-4 w-4" />
            </Button>
          </div>
        </form>
      ) : (
        <p className="text-sm text-muted-foreground">Войдите через Telegram, чтобы оставлять комментарии</p>
      )}

      {/* Comments list */}
      <div className="flex flex-col gap-4">
        {isLoading ? (
          <p className="text-sm text-muted-foreground">Загрузка комментариев...</p>
        ) : comments.length === 0 ? (
          <p className="text-sm text-muted-foreground">Пока нет комментариев</p>
        ) : (
          comments.map((comment) => {
            const commentUser = comment.users
            const displayName = commentUser?.first_name || commentUser?.username || "Пользователь"

            return (
              <div key={comment.id} className="flex gap-3">
                <Avatar className="h-8 w-8 shrink-0">
                  <AvatarImage src={commentUser?.photo_url || undefined} />
                  <AvatarFallback>{displayName.charAt(0)}</AvatarFallback>
                </Avatar>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="font-medium text-sm">{displayName}</span>
                    <span className="text-xs text-muted-foreground">{formatDate(comment.created_at)}</span>
                    {user?.id === comment.user_id && (
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-6 w-6 ml-auto"
                        onClick={() => handleDelete(comment.id)}
                      >
                        <Trash2 className="h-3 w-3" />
                      </Button>
                    )}
                  </div>
                  <p className="text-sm whitespace-pre-wrap break-words">{comment.content}</p>
                </div>
              </div>
            )
          })
        )}
      </div>
    </div>
  )
}
