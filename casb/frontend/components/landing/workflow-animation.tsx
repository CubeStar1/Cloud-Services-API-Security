'use client'

import { motion } from 'framer-motion'
import { useState, useEffect } from 'react'

export function WorkflowAnimation() {
  const [step, setStep] = useState(0)
  const steps = [
    { label: "Data Collection", color: "#0ea5e9" },
    { label: "DeBERTa Analysis", color: "#8b5cf6" },
    { label: "CodeBERT Embedding", color: "#ec4899" },
    { label: "RFC Generation", color: "#f97316" },
    { label: "Security Testing", color: "#10b981" }
  ]

  useEffect(() => {
    const interval = setInterval(() => {
      setStep((prev) => (prev + 1) % steps.length)
    }, 2000)
    
    return () => clearInterval(interval)
  }, [steps.length])

  return (
    <motion.div 
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5, delay: 0.6 }}
      className="w-full max-w-2xl mx-auto my-12 px-4"
    >
      <div className="h-2 bg-gray-200 rounded-full mb-8 relative">
        {steps.map((s, i) => (
          <motion.div
            key={i}
            className="absolute h-2 rounded-full"
            style={{
              left: `${(i / steps.length) * 100}%`,
              width: `${100 / steps.length}%`,
              backgroundColor: s.color,
              opacity: i <= step ? 1 : 0.3,
            }}
            animate={{
              scale: i === step ? [1, 1.2, 1] : 1,
            }}
            transition={{ duration: 0.5, repeat: i === step ? Infinity : 0, repeatDelay: 1.5 }}
          />
        ))}
      </div>

      <div className="flex justify-between">
        {steps.map((s, i) => (
          <motion.div 
            key={i} 
            className="flex flex-col items-center"
            animate={{
              y: i === step ? [0, -10, 0] : 0,
              scale: i === step ? 1.1 : 1,
            }}
            transition={{ duration: 0.5 }}
          >
            <motion.div 
              className="w-4 h-4 rounded-full mb-2" 
              style={{ backgroundColor: s.color }}
              animate={{
                scale: i === step ? [1, 1.3, 1] : 1,
              }}
              transition={{ duration: 0.5, repeat: i === step ? Infinity : 0, repeatDelay: 1.5 }}
            />
            <motion.span 
              className={`text-xs md:text-sm font-medium ${i === step ? 'text-primary' : 'text-muted-foreground'}`}
              style={{ 
                textAlign: 'center',
                maxWidth: '80px',
              }}
              animate={{
                opacity: i === step ? 1 : 0.7,
              }}
            >
              {s.label}
            </motion.span>
          </motion.div>
        ))}
      </div>
    </motion.div>
  )
} 