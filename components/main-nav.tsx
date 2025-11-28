"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { cn } from "@/lib/utils"

export function MainNav() {
  const pathname = usePathname()

  const links = [
    {
      href: "/",
      label: "События",
    },
    {
      href: "/players",
      label: "Игроки",
    },
    {
      href: "/favorites",
      label: "Избранное",
    },
  ]

  return (
    <nav className="flex items-center justify-center gap-6 mb-8">
      {links.map((link) => (
        <Link
          key={link.href}
          href={link.href}
          className={cn(
            "text-lg font-medium transition-colors hover:text-primary",
            pathname === link.href ? "text-foreground border-b-2 border-primary" : "text-muted-foreground",
          )}
        >
          {link.label}
        </Link>
      ))}
    </nav>
  )
}
