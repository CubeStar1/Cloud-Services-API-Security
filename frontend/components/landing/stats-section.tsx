'use client'

import { motion } from 'framer-motion'
import { Card } from '@/components/ui/card'
import { CheckCircle2, Database, RefreshCw, Package } from 'lucide-react'
import { useEffect, useState } from 'react'

export function StatsSection() {
  const stats = [
    {
      icon: <CheckCircle2 className="h-6 w-6 text-emerald-500" />,
      endValue: 99,
      label: "Service Classification Accuracy",
      color: "#10b981" // emerald-500
    },
    {
      icon: <Database className="h-6 w-6 text-indigo-500" />,
      endValue: 100,
      label: "Activity Classification Accuracy",
      color: "#6366f1" // indigo-500
    },
    {
      icon: <RefreshCw className="h-6 w-6 text-blue-500" />,
      endValue: 78,
      label: "Avg. Service Confidence Score",
      color: "#0ea5e9" // blue-500
    },
    {
      icon: <Package className="h-6 w-6 text-pink-500" />,
      endValue: 98,
      label: "Pattern Recognition Score",
      color: "#ec4899" // pink-500
    }
  ]

  // State for animated values
  const [values, setValues] = useState<number[]>(stats.map(() => 0))

  useEffect(() => {
    // Start the animation after component mounts
    const interval = setInterval(() => {
      setValues(prev => 
        prev.map((value, index) => {
          const endValue = stats[index].endValue
          if (value < endValue) {
            // Gradually increase up to the target value
            // Increase faster for larger targets
            const increment = Math.max(1, Math.floor(endValue / 50))
            return Math.min(value + increment, endValue)
          }
          return value
        })
      )
    }, 30)

    // Cleanup interval on unmount
    return () => clearInterval(interval)
  }, [])

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
      className="py-16"
    >
      <h2 className="text-2xl font-semibold mb-10 text-center">Performance Metrics</h2>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 max-w-6xl mx-auto">
        {stats.map((stat, index) => (
          <Card key={index} className="border overflow-hidden">
            <div className="p-6 flex flex-col items-center text-center">
              <div className="mb-4">
                {stat.icon}
              </div>
              
              <motion.div
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                className="text-5xl font-bold mb-2"
              >
                {values[index]}%
              </motion.div>
              
              <p className="text-sm text-muted-foreground">
                {stat.label}
              </p>
            </div>
          </Card>
        ))}
      </div>
    </motion.div>
  )
} 