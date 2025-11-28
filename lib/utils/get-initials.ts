/**
 * Get initials from a person's name
 * @param name - Full name of the person
 * @returns Initials (up to 2 characters)
 */
export function getInitials(name: string): string {
  if (!name) return "?"

  const parts = name.trim().split(/\s+/)

  if (parts.length === 1) {
    return parts[0].charAt(0).toUpperCase()
  }

  return (parts[0].charAt(0) + parts[parts.length - 1].charAt(0)).toUpperCase()
}
