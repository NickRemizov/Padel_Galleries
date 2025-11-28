import { z } from "zod"

const envSchema = z.object({
  // Server-side only
  FASTAPI_URL: z.string().url("FASTAPI_URL must be a valid URL"),
  API_SECRET_KEY: z.string().min(1, "API_SECRET_KEY is required for Python API auth"),

  // Supabase (required)
  NEXT_PUBLIC_SUPABASE_URL: z.string().url("NEXT_PUBLIC_SUPABASE_URL must be a valid URL"),
  NEXT_PUBLIC_SUPABASE_ANON_KEY: z.string().min(1, "NEXT_PUBLIC_SUPABASE_ANON_KEY is required"),

  // Optional server-side variables
  SUPABASE_SERVICE_ROLE_KEY: z.string().optional(),
  BLOB_READ_WRITE_TOKEN: z.string().optional(),
  TELEGRAM_BOT_TOKEN: z.string().optional(),
  NEXT_PUBLIC_TELEGRAM_BOT_USERNAME: z.string().optional(),
  NEXT_PUBLIC_FASTAPI_URL: z.string().optional(),
  NEXT_PUBLIC_DEV_SUPABASE_REDIRECT_URL: z.string().optional(),

  // Database
  POSTGRES_URL: z.string().optional(),
  POSTGRES_PRISMA_URL: z.string().optional(),
  POSTGRES_URL_NON_POOLING: z.string().optional(),
  POSTGRES_USER: z.string().optional(),
  POSTGRES_PASSWORD: z.string().optional(),
  POSTGRES_DATABASE: z.string().optional(),
  POSTGRES_HOST: z.string().optional(),

  // Auth
  SUPABASE_URL: z.string().optional(),
  SUPABASE_JWT_SECRET: z.string().optional(),
  SUPABASE_ANON_KEY: z.string().optional(),

  // Server config
  SERVER_IP: z.string().optional(),
  SERVER_PORT: z.string().optional(),
  SERVER_HOST: z.string().optional(),
  JWT_SECRET_KEY: z.string().optional(),
  ALLOWED_ORIGINS: z.string().optional(),

  // OAuth
  GOOGLE_CLIENT_ID: z.string().optional(),
  GOOGLE_CLIENT_SECRET: z.string().optional(),
})

function validateEnv() {
  try {
    return envSchema.parse(process.env)
  } catch (error) {
    if (error instanceof z.ZodError) {
      const missingVars = error.errors.map((err) => `${err.path.join(".")}: ${err.message}`).join("\n")

      console.error("❌ Invalid environment variables:")
      console.error(missingVars)

      throw new Error(
        `Environment validation failed. Missing or invalid variables:\n${missingVars}\n\n` +
          `Please check your environment variables in Vercel dashboard or .env file.`,
      )
    }
    throw error
  }
}

export const env = validateEnv()

export type Env = z.infer<typeof envSchema>
