'use client'

import { motion } from 'framer-motion'
import { Card } from '@/components/ui/card'
import Image from 'next/image'
import { AlertCircle } from 'lucide-react'
import { Alert, AlertDescription } from '@/components/ui/alert'

export function FlowDiagram() {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5, delay: 0.2 }}
      className="my-16"
    >
      <h2 className="text-2xl font-semibold mb-8 text-center">System Workflow</h2>
      
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Left side - Diagram */}
        <Card className="p-6 h-full flex flex-col">
          <h3 className="text-lg font-medium mb-4">Visual Pipeline</h3>
          
          <div className="relative flex-grow w-full bg-muted/20 rounded-md overflow-hidden">
            <div className="relative w-full h-full min-h-[300px]">
              <Image 
                src="/flow-diagram.png"
                alt="API Security Workflow" 
                fill 
                className="object-contain"
                priority
              />
            </div>
          </div>
          
          <Alert className="mt-4 bg-muted/30">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription className="text-sm">
              The workflow illustrates the complete process from data collection 
              through analysis and model deployment.
            </AlertDescription>
          </Alert>
        </Card>
        
        {/* Right side - Explanation */}
        <Card className="p-6 h-full">
          <h3 className="text-lg font-medium mb-4">Process Steps</h3>
          
          <ol className="space-y-5 list-none pl-0">
            <ProcessStep 
              step={1}
              title="Data Collection" 
              description="AnyProxy captures network traffic and saves it in JSON format for each monitored service."
              color="#0ea5e9"
              delay={0.3}
            />
            
            <ProcessStep 
              step={2}
              title="Unsupervised Learning" 
              description="DeBERTa performs zero-shot classification to label the data without requiring manual annotation."
              color="#8b5cf6"
              delay={0.4}
            />
            
            <ProcessStep 
              step={3}
              title="Confidence Check" 
              description="If confidence is above 0.5, the system proceeds to CodeBERT; otherwise, it collects additional data."
              color="#f97316"
              delay={0.5}
            />
            
            <ProcessStep 
              step={4}
              title="Supervised Learning" 
              description="CodeBERT labels services and activities with higher precision using the pre-labeled data."
              color="#ec4899"
              delay={0.6}
            />
            
            <ProcessStep 
              step={5}
              title="Model Training" 
              description="Random Forest classifier is trained using scikit-learn and Python based on the labeled data."
              color="#10b981"
              delay={0.7}
            />
            
            <ProcessStep 
              step={6}
              title="Deployment" 
              description="The trained Random Forest model is converted to efficient C code for integration into security decision trees."
              color="#6366f1"
              delay={0.8}
            />
          </ol>
        </Card>
      </div>
    </motion.div>
  )
}

interface ProcessStepProps {
  step: number;
  title: string;
  description: string;
  color: string;
  delay: number;
}

function ProcessStep({ step, title, description, color, delay }: ProcessStepProps) {
  return (
    <motion.li
      initial={{ opacity: 0, x: -10 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.3, delay }}
      className="relative pl-9"
    >
      <motion.div 
        className="absolute left-0 top-0 w-7 h-7 rounded-full flex items-center justify-center text-white font-medium text-sm"
        style={{ backgroundColor: color }}
        whileHover={{ scale: 1.1 }}
        transition={{ duration: 0.2 }}
      >
        {step}
      </motion.div>
      <h4 className="font-medium text-base">{title}</h4>
      <p className="text-sm text-muted-foreground mt-1">{description}</p>
    </motion.li>
  )
} 