import { LucideIcon } from "lucide-react"

export interface HeaderLink {
  href: string
  label: string
  icon?: LucideIcon
  description?: string
}

export interface HeaderConfig {
  brand: {
    title: string
    icon: string
  }
  navigationLinks: HeaderLink[]
}

export const headerConfig: HeaderConfig = {
  brand: {
    title: "NextJS Template",
    icon: "/globe.svg"
  },
  navigationLinks: [
    { 
    href: "/", 
    label: "Home" 
  },
  { 
    href: "/about", 
    label: "About" 
  },
  { 
    href: "/activity", 
    label: "Activity" 
  },
  { 
    href: "/files", 
    label: "Files" 
  },
  { 
    href: "/settings", 
    label: "Settings" 
  },
  { 
    href: "/contact", 
    label: "Contact" 
  }
  ]
}