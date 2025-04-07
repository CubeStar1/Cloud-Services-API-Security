"use client"

import { ChevronRight, LucideIcon } from "lucide-react"
import Link from "next/link"
import { useState } from "react"
import {
  SidebarGroup,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarMenuSub,
  SidebarMenuSubButton,
  SidebarMenuSubItem,
} from "@/components/ui/sidebar"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"

interface MenuItem {
  title: string
  href: string
  icon: LucideIcon
  description?: string
}

interface NavSectionProps {
  label: string
  items: MenuItem[]
  collapsible?: boolean
  defaultOpen?: boolean
  icon?: LucideIcon
  collapsibleTitle?: string
}

export function NavSection({
  label,
  items,
  collapsible = false,
  defaultOpen = true,
  icon: SectionIcon,
  collapsibleTitle,
}: NavSectionProps) {
  const [isOpen, setIsOpen] = useState(defaultOpen)

  const MenuItems = () => (
    <>
      {items.map((item) => (
        <SidebarMenuItem key={item.title}>
          <SidebarMenuButton asChild>
            <Link href={item.href}>
              <item.icon className="mr-2 h-4 w-4" />
              <span>{item.title}</span>
            </Link>
          </SidebarMenuButton>
        </SidebarMenuItem>
      ))}
    </>
  )

  const CollapsibleItems = () => (
    <SidebarMenuSub>
      {items.map((item) => (
        <SidebarMenuSubItem key={item.title}>
          <SidebarMenuSubButton asChild>
            <Link href={item.href}>
              <item.icon className="mr-2 h-4 w-4" />
              <span>{item.title}</span>
            </Link>
          </SidebarMenuSubButton>
        </SidebarMenuSubItem>
      ))}
    </SidebarMenuSub>
  )

  return (
    <SidebarGroup>
      <SidebarGroupLabel>{label}</SidebarGroupLabel>
      <SidebarMenu>
        {collapsible ? (
          <Collapsible open={isOpen} onOpenChange={setIsOpen}>
            <CollapsibleTrigger asChild>
              <SidebarMenuButton>
                {SectionIcon && <SectionIcon className="mr-2 h-4 w-4" />}
                <span>{collapsibleTitle}</span>
                <ChevronRight 
                  className={`ml-auto h-4 w-4 transition-transform duration-200 ${
                    isOpen ? 'rotate-90' : ''
                  }`} 
                />
              </SidebarMenuButton>
            </CollapsibleTrigger>
            <CollapsibleContent>
              <CollapsibleItems />
            </CollapsibleContent>
          </Collapsible>
        ) : (
          <MenuItems />
        )}
      </SidebarMenu>
    </SidebarGroup>
  )
} 