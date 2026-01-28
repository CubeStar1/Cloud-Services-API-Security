'use client'

import { Card } from '@/components/ui/card'
import { LucideIcon } from 'lucide-react'
import { motion } from 'framer-motion'
import Link from 'next/link'

interface FeatureCardProps {
  title: string
  description: string
  icon: LucideIcon
  href: string
  delay: number
}

export function FeatureCard({ title, description, icon: Icon, href, delay }: FeatureCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ 
        duration: 0.5,
        delay: delay * 0.1,
        ease: [0.4, 0, 0.2, 1]
      }}
    >
      <Link href={href} className="block h-full">
        <Card className="p-6 h-full transition-all duration-200 hover:shadow-md hover:scale-[1.01] border-2 border-transparent hover:border-primary/20">
          <div className="flex flex-col h-full">
            <div className="mb-4 bg-primary/10 p-3 w-fit rounded-lg">
              <Icon className="h-6 w-6 text-primary" />
            </div>
            
            <h3 className="text-xl font-semibold mb-2">{title}</h3>
            <p className="text-muted-foreground text-sm flex-grow">{description}</p>
            
            <div className="mt-4 text-primary text-sm font-medium">
              Explore →
            </div>
          </div>
        </Card>
      </Link>
    </motion.div>
  )
} 