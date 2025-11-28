"use client"

import { useState, useCallback } from "react"
import { getAllFaceDescriptorsAction } from "@/app/admin/actions"

type CachedDescriptor = {
  id: string
  personId: string
  personName: string
  descriptor: number[]
  sourceImageId: string | null
}

export function useFaceDetection() {
  const [isDetecting, setIsDetecting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const loadDescriptors = useCallback(async (): Promise<CachedDescriptor[]> => {
    try {
      const result = await getAllFaceDescriptorsAction()

      if (!result.success || !result.data) {
        throw new Error(result.error || "Failed to get face descriptors")
      }

      return result.data
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : "Failed to load descriptors"
      setError(errorMessage)
      console.error("[v0] Error loading descriptors:", err)
      throw err
    }
  }, [])

  return {
    loadDescriptors,
    isDetecting,
    error,
  }
}
