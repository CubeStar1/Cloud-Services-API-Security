'use client'

import { HeroSection } from '@/components/landing/hero-section'
import { FeatureGrid } from '@/components/landing/feature-grid'
import { WorkflowAnimation } from '@/components/landing/workflow-animation'
import { QuickStart } from '@/components/landing/quick-start'
import { ContextSection } from '@/components/landing/context-section'
import { StatsSection } from '@/components/landing/stats-section'
import { FlowDiagram } from '@/components/landing/flow-diagram'
import { motion } from 'framer-motion'
import {Header} from '@/components/global/header'

export default function Home() {
  return (
    <div className="min-h-screen px-4 py-12 md:py-16">
      <div className="max-w-7xl mx-auto">
        <Header />
        <HeroSection />
        
        
        <ContextSection />
        
        <FlowDiagram />
        
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="mb-8"
        >
          <h2 className="text-2xl font-semibold mb-6 text-center">Platform Features</h2>
        </motion.div>
        
        <FeatureGrid />
        
        <QuickStart />
      </div>
    </div>
  );
}
