"use client"

import { ModeToggle } from "@/components/global/theme-switcher"
import { SidebarTrigger } from "@/components/ui/sidebar"
import { Separator } from "@/components/ui/separator"
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from "@/components/ui/breadcrumb"
import { usePathname } from "next/navigation"

const routes = {
  "/": {
    name: "Home",
    path: "/",
  },
  "/dashboard": {
    name: "Dashboard",
    path: "/dashboard",
    description: "Proxy control and traffic monitoring",
  },
  "/flowchart": {
    name: "System Flow",
    path: "/flowchart",
    description: "Complete system workflow visualization",
  },
  "/logs": {
    name: "Raw Logs",
    path: "/logs",
    description: "View and analyze raw traffic logs",
  },
  "/labelling": {
    name: "Data Labelling",
    path: "/labelling",
    description: "Manual data labelling",
  },
  "/zsl": {
    name: "Zero-Shot Learning",
    path: "/zsl",
    description: "Zero-shot learning models",
    subRoutes: {
      "/zsl/deberta": {
        name: "DeBERTa Model",
        path: "/zsl/deberta",
        description: "DeBERTa-based classification",
      },
      "/zsl/codebert": {
        name: "CodeBERT Model",
        path: "/zsl/codebert",
        description: "CodeBERT-based classification",
      }
    }
  },
  "/random-forest": {
    name: "Random Forest",
    path: "/random-forest",
    description: "Random Forest classification",
  },
  "/settings": {
    name: "Settings",
    path: "/settings",
    description: "System configuration",
    subRoutes: {
      "/settings/analysis": {
        name: "Analysis Configuration",
        path: "/settings/analysis",
      },
      "/settings/scanning": {
        name: "Scanning Parameters",
        path: "/settings/scanning",
      },
      "/settings/thresholds": {
        name: "Detection Thresholds",
        path: "/settings/thresholds",
      }
    }
  }
}

export function Breadcrumbs({ 
  action 
}: { 
  action?: React.ReactNode 
}) {
  const pathname = usePathname()
  
  const getBreadcrumbs = () => {
    const segments = pathname.split('/').filter(Boolean)
    const items = []
    let currentPath = ""
    
    // Always add home
    items.push(
      <BreadcrumbItem key="home">
        <BreadcrumbLink href="/">Home</BreadcrumbLink>
      </BreadcrumbItem>
    )

    // Add intermediate segments
    segments.forEach((segment, index) => {
      currentPath += `/${segment}`
      const route = routes[currentPath as keyof typeof routes]
      
      if (route) {
        items.push(<BreadcrumbSeparator key={`separator-${index}`} />)
        
        if (index === segments.length - 1) {
          // Last segment - show as current page
          items.push(
            <BreadcrumbItem key={currentPath}>
              <BreadcrumbPage>{route.name}</BreadcrumbPage>
            </BreadcrumbItem>
          )
        } else {
          // Intermediate segment - show as link
          items.push(
            <BreadcrumbItem key={currentPath}>
              <BreadcrumbLink href={route.path}>{route.name}</BreadcrumbLink>
            </BreadcrumbItem>
          )
        }
      }
    })

    return items
  }

  return (
    <header className="flex h-20 z-50 shrink-0 sticky top-0 items-center gap-2 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="flex items-center gap-2 px-4 w-full">
        <SidebarTrigger className="-ml-1" />
        <Separator orientation="vertical" className="mr-2 h-4" />
        <Breadcrumb>
          <BreadcrumbList>
            {getBreadcrumbs()}
          </BreadcrumbList>
        </Breadcrumb>
        <div className="ml-auto flex items-center gap-4">
          {action}
          <ModeToggle />
        </div>
      </div>
    </header>
  )
} 