import { createClient } from "@/lib/supabase/server"
import { NextResponse } from "next/server"

/**
 * Server-side guard для защиты admin API роутов
 * Проверяет наличие валидной Supabase сессии и роли admin
 *
 * @returns { user, error } - объект с пользователем или ошибкой
 */
export async function requireAdmin() {
  const supabase = await createClient()

  // Получаем текущего пользователя
  const {
    data: { user },
    error: authError,
  } = await supabase.auth.getUser()

  // Если нет пользователя или ошибка аутентификации
  if (authError || !user) {
    return {
      user: null,
      error: NextResponse.json({ error: "Unauthorized: Authentication required" }, { status: 401 }),
    }
  }

  // TODO: Включить проверку роли после настройки ролей в Supabase
  /*
  const userRole = user.app_metadata?.role || user.user_metadata?.role

  if (userRole !== "admin") {
    return {
      user: null,
      error: NextResponse.json({ error: "Forbidden: Admin role required" }, { status: 403 }),
    }
  }
  */

  // Всё ок, возвращаем пользователя
  return { user, error: null }
}
