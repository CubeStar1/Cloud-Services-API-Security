'use client'

import { 
  Terminal, FileText, Brain, Code2, GitMerge, FileCode,
  Database, FolderTree, BarChart4, GitBranch
} from 'lucide-react'
import { FeatureCard } from '@/components/landing/feature-card'
import { motion } from 'framer-motion'

// Main workflow features in the exact flow order
const mainFeatures = [
  {
    title: "Data Collection",
    description: "Capture network traffic using AnyProxy to collect HTTP requests and responses from cloud services.",
    icon: Terminal,
    href: "/dashboard",
  },
  {
    title: "Labelling",
    description: "Process raw logs and convert them into structured CSV datasets for training and analysis.",
    icon: FileText,
    href: "/labelling",
  },
  {
    title: "DeBERTa Analysis",
    description: "Zero-shot classification of traffic logs using DeBERTa for initial service and activity identification.",
    icon: Brain,
    href: "/zsl/deberta",
  },
  {
    title: "CodeBERT Processing",
    description: "Advanced pattern recognition and dual-head classification for service and activity types.",
    icon: Code2,
    href: "/zsl/codebert",
  },
  {
    title: "Random Forest Training",
    description: "Fine-tuned machine learning models for high-accuracy classification using feature extraction.",
    icon: GitMerge,
    href: "/rfc",
  },
  {
    title: "C Code Generation",
    description: "Convert trained Random Forest models to efficient C code for deployment in security systems.",
    icon: FileCode,
    href: "/rfc",
  },
]

// Supporting tools and features
const supportingFeatures = [
  {
    title: "File Browser",
    description: "Browse, view and manage all data files including CSVs, logs, and generated code.",
    icon: FolderTree,
    href: "/files",
  },
  {
    title: "Dashboard",
    description: "View real-time metrics and summaries of API traffic and security analysis.",
    icon: BarChart4,
    href: "/dashboard",
  },
  {
    title: "System Flow",
    description: "Interactive visualization of the complete security analysis workflow.",
    icon: GitBranch,
    href: "/flowchart",
  },
  {
    title: "Logs Browser",
    description: "Detailed examination of collected traffic logs with filtering and search capabilities.",
    icon: Database,
    href: "/logs",
  },
]

export function FeatureGrid() {
  return (
    <div className="space-y-12">
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.3 }}
      >
        <motion.h3 
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
          className="text-xl font-medium mb-6 border-l-4 border-primary pl-3"
        >
          Core Workflow Pipeline
        </motion.h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {mainFeatures.map((feature, index) => (
            <FeatureCard 
              key={feature.title}
              title={feature.title}
              description={feature.description}
              icon={feature.icon}
              href={feature.href}
              delay={index}
            />
          ))}
        </div>
      </motion.div>
      
      {supportingFeatures.length > 0 && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.3, delay: 0.2 }}
        >
          <motion.h3 
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.3 }}
            className="text-xl font-medium mb-6 border-l-4 border-secondary pl-3"
          >
            Supporting Tools
          </motion.h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {supportingFeatures.map((feature, index) => (
              <FeatureCard 
                key={feature.title}
                title={feature.title}
                description={feature.description}
                icon={feature.icon}
                href={feature.href}
                delay={index}
              />
            ))}
          </div>
        </motion.div>
      )}
    </div>
  )
} 