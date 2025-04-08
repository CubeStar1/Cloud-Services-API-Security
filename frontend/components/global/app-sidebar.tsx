"use client"

import { 
  LayoutDashboard,
  Cloud,
  Shield,
  Network,
  Settings,
  Brain,
  FileJson,
  History,
  Route,
  Scan,
  Lock,
  Activity,
  FileCheck,
  TreesIcon,
  BrainCircuit,
  Code2,
  Binary,
  GitBranch,
  FileIcon
} from "lucide-react"
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarHeader,
  SidebarRail,
} from "@/components/ui/sidebar"
import { NavSection } from "@/components/navigation/nav-section"
import Link from "next/link"
import { cn } from "@/lib/utils"
import { NavProfile } from "@/components/navigation/nav-profile";
// import useUser from "@/hooks/use-user";


const navigationItems = [
  {
    title: "Dashboard",
    href: "/dashboard",
    icon: LayoutDashboard,
    description: "Proxy control and traffic monitoring",
  },
  {
    title: "System Flow",
    href: "/flowchart",
    icon: GitBranch,
    description: "Complete system workflow visualization",
  },
  {
    title: "Data Collection",
    href: "/dashboard",
    icon: Binary,
    description: "Data collection and analysis",
  },
  {
    title: "Raw Logs",
    href: "/logs",
    icon: FileJson,
    description: "View and analyze raw traffic logs",
  },
  {
    title: "Labelling",
    href: "/labelling",
    icon: FileCheck,
    description: "Labelling of data",
  },
  {
    title: "Deberta",
    href: "/zsl/deberta",
    icon: BrainCircuit,
    description: "Deberta model",
  },
  {
    title: "Codebert",
    href: "/zsl/codebert",
    icon: Code2,
    description: "Codebert model",
  },
  {
    title: "Random Forest",
    href: "/rfc",
    icon: TreesIcon,
    description: "Random Forest model",
  },
  {
    title: "File Browser",
    href: "/files",
    icon: FileIcon,
    description: "File browser",
  },

]

const settingsItems = [
  {
    title: "Settings",
    href: "/settings",
    icon: Settings,
    description: "System configuration",
  },
]

export function AppSidebar() {
  // const { data: user } = useUser()

  // if (!user) return null
  return (
    <Sidebar className="">
      <SidebarHeader>
        <div className="relative border-b border-border/10 bg-gradient-to-br from-background/90 via-background/50 to-background/90 px-6 py-5 backdrop-blur-xl">
          <Link href="/" className="relative flex items-center gap-4">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-blue-600 via-indigo-600 to-violet-600 shadow-lg ring-2 ring-blue-500/20 dark:from-blue-500 dark:via-indigo-500 dark:to-violet-500">
              <Shield className="h-5 w-5 text-white shadow-sm" />
            </div>
            <div className="flex flex-col gap-0.5">
              <h1 className="text-xl font-semibold tracking-tight text-foreground">
                Cloud API Security
              </h1>
            </div>
          </Link>
        </div>
      </SidebarHeader>
      <SidebarContent className="bg-gradient-to-b from-background/80 to-background/20 dark:from-background/60 dark:to-background/0">
        <div className="space-y-4 py-4">
          <NavSection 
            label="Navigation"
            items={navigationItems}
          />
          <NavSection 
            label="Preferences"
            items={settingsItems}
          />
        </div>
      </SidebarContent>
      <SidebarRail className="" />
      <SidebarFooter className="border-t border-border/20 bg-gradient-to-t from-background/90 to-background/40 px-6 py-3 backdrop-blur-xl dark:from-background/80 dark:to-background/20">
        
        {/* <NavProfile user={user} /> */}
      </SidebarFooter>
    </Sidebar>
  );
} 