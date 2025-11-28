/**
 * Safely executes a Supabase query with retry logic for rate limiting errors.
 * Handles the case where Supabase returns "Too Many Requests" as plain text
 * instead of JSON, which causes SyntaxError in the SDK.
 */
export async function safeSupabaseCall<T>(
  fn: () => Promise<{ data: T | null; error: any }>,
  maxRetries = 5,
  initialDelay = 2000,
): Promise<{ data: T | null; error: any }> {
  let lastError: any

  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      const result = await fn()

      // Check if we got a rate limit error in the error field
      if (result.error && (result.error.message?.includes("Too Many") || result.error.code === "429")) {
        throw new Error("Rate limit exceeded")
      }

      return result
    } catch (error: any) {
      lastError = error

      // Check if this is a rate limiting error
      const isRateLimitError =
        error instanceof SyntaxError ||
        error.message?.includes("Too Many") ||
        error.message?.includes("Rate limit") ||
        error.code === "429" ||
        error.status === 429

      if (!isRateLimitError || attempt === maxRetries - 1) {
        // Not a rate limit error or last attempt, throw immediately
        return { data: null, error: lastError }
      }

      // Calculate delay with exponential backoff
      const delay = initialDelay * Math.pow(2, attempt)
      console.log(`[v0] Rate limit hit, retrying in ${delay}ms (attempt ${attempt + 1}/${maxRetries})`)

      // Wait before retrying
      await new Promise((resolve) => setTimeout(resolve, delay))
    }
  }

  return { data: null, error: lastError }
}
