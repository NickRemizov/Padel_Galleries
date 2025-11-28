import crypto from "node:crypto"

export function verifyTelegramAuth(data: Record<string, string>, botToken: string): boolean {
  const { hash, ...authData } = data

  // Create data-check-string
  const checkString = Object.keys(authData)
    .sort()
    .map((key) => `${key}=${authData[key]}`)
    .join("\n")

  // Create secret key from bot token
  const secretKey = crypto.createHash("sha256").update(botToken).digest()

  // Create hash
  const hmac = crypto.createHmac("sha256", secretKey).update(checkString).digest("hex")

  // Compare hashes
  return hmac === hash
}

export function isTelegramAuthDataValid(authDate: number): boolean {
  const currentTime = Math.floor(Date.now() / 1000)
  const timeDiff = currentTime - authDate

  // Auth data is valid for 24 hours
  return timeDiff < 86400
}
