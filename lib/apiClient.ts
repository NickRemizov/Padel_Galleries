import { env } from "./env"
import { randomUUID } from "crypto"

export class ApiError extends Error {
  constructor(
    public status: number,
    public code?: string,
    message?: string,
  ) {
    super(message || `API Error: ${status}`)
    this.name = "ApiError"
  }
}

interface ApiFetchOptions extends RequestInit {
  timeout?: number
  retries?: number
}

export async function apiFetch<T = any>(path: string, options: ApiFetchOptions = {}): Promise<T> {
  const { timeout = 30000, retries = 2, headers = {}, ...fetchOptions } = options

  const normalizedPath = path.startsWith("/") ? path : `/${path}`
  const url = `${env.FASTAPI_URL}${normalizedPath}`

  const requestId = randomUUID()

  const fetchWithTimeout = async (attemptNumber: number): Promise<T> => {
    const controller = new AbortController()
    const timeoutId = setTimeout(() => controller.abort(), timeout)

    try {
      console.log(`[apiClient] Request ${requestId} (attempt ${attemptNumber}): ${fetchOptions.method || "GET"} ${url}`)

      const response = await fetch(url, {
        ...fetchOptions,
        headers: {
          "Content-Type": "application/json",
          "x-request-id": requestId,
          "X-API-Key": env.API_SECRET_KEY,
          ...headers,
        },
        signal: controller.signal,
      })

      clearTimeout(timeoutId)

      if (!response.ok) {
        const contentType = response.headers.get("content-type")
        let errorMessage = `HTTP ${response.status}`
        let errorCode: string | undefined

        if (contentType?.includes("application/json")) {
          try {
            const errorData = await response.json()
            errorMessage = errorData.message || errorData.detail || errorMessage
            errorCode = errorData.code
          } catch {
            // Failed to parse error JSON, use status text
            errorMessage = response.statusText || errorMessage
          }
        } else {
          const text = await response.text()
          errorMessage = text || response.statusText || errorMessage
        }

        if (response.status >= 500 && attemptNumber < retries) {
          console.warn(`[apiClient] Request ${requestId} failed with ${response.status}, retrying...`)
          await new Promise((resolve) => setTimeout(resolve, 1000 * attemptNumber))
          return fetchWithTimeout(attemptNumber + 1)
        }

        throw new ApiError(response.status, errorCode, errorMessage)
      }

      const contentType = response.headers.get("content-type")
      if (contentType?.includes("application/json")) {
        const data = await response.json()
        console.log(`[apiClient] Request ${requestId} succeeded`)
        return data
      }

      return (await response.text()) as any
    } catch (error) {
      clearTimeout(timeoutId)

      if (error instanceof Error && error.name === "AbortError") {
        if (attemptNumber < retries) {
          console.warn(`[apiClient] Request ${requestId} timed out, retrying...`)
          await new Promise((resolve) => setTimeout(resolve, 1000 * attemptNumber))
          return fetchWithTimeout(attemptNumber + 1)
        }
        throw new ApiError(408, "TIMEOUT", `Request timed out after ${timeout}ms`)
      }

      if (error instanceof ApiError) {
        throw error
      }

      throw new ApiError(500, "FETCH_ERROR", error instanceof Error ? error.message : "Unknown error")
    }
  }

  return fetchWithTimeout(1)
}
