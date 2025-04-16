import Icons from "@/components/global/icons";
import { SidebarConfig } from "@/components/global/app-sidebar";

const sidebarConfig: SidebarConfig = {
  brand: {
    title: "Cloud API Security",
    icon: Icons.shield,
    href: "/"
  },
  sections: [
    {
      label: "Training",
      items: [
        {
          title: "Dashboard",
          href: "/dashboard",
          icon: Icons.layoutDashboard
        },
        {
          title: "Data Collection",
          href: "/anyproxy",
          icon: Icons.activity
        },
        {
          title: "Raw Logs",
          href: "/logs",
          icon: Icons.fileJson
        },
        {
          title: "Labelling",
          href: "/labelling",
          icon: Icons.fileCheck
        },
        {
          title: "Deberta",
          href: "/zsl/deberta",
          icon: Icons.brainCircuit
        },
        {
          title: "Codebert",
          href: "/zsl/codebert",
          icon: Icons.code2
        },
        {
          title: "Random Forest",
          href: "/rfc",
          icon: Icons.treesIcon
        },
        {
          title: "File Browser",
          href: "/files",
          icon: Icons.fileIcon
        }
      ]
    },
    {
        label: "Secure File Sharing",
        items: [
          {
            title: "Dashboard",
            href: "/secure-file-sharing",
            icon: Icons.layoutDashboard
          }
        ]
      }
    
  ]
}

export default sidebarConfig