'use client'

import { motion } from 'framer-motion'
import { Card } from '@/components/ui/card'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { 
  Cloud, Server, Cpu, History, Microscope, 
  Network, LineChart, BarChart, GitMerge, BrainCircuit
} from 'lucide-react'

export function ContextSection() {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.6, delay: 0.3 }}
      className="my-16"
    >
      <h2 className="text-2xl font-semibold mb-6 text-center">Project Overview</h2>
      
      <Card className="p-6 mb-8">
        <div className="flex items-start gap-4">
          <div className="mt-1 bg-primary/10 p-3 rounded-lg">
            <Cloud className="h-6 w-6 text-primary" />
          </div>
          <div>
            <h3 className="text-lg font-medium mb-2">Cloud Services API Security</h3>
            <p className="text-muted-foreground">
              This platform identifies cloud service providers, API endpoints, and user activities 
              through machine learning algorithms based on live traffic analysis. The project 
              focuses initially on SAAS public cloud services like DropBox, SalesForce, 
              Google Docs, OneDrive, and Box before expanding to other API providers.
            </p>
          </div>
        </div>
      </Card>
      
      <Tabs defaultValue="data" className="w-full">
        <TabsList className="grid grid-cols-4 mb-4">
          <TabsTrigger value="data">Data Collection</TabsTrigger>
          <TabsTrigger value="models">ML Models</TabsTrigger>
          <TabsTrigger value="analysis">Analysis</TabsTrigger>
          <TabsTrigger value="workflow">Workflow</TabsTrigger>
        </TabsList>
        
        <TabsContent value="data" className="space-y-4">
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
          >
            <Card className="p-4">
              <div className="flex items-start gap-3">
                <Server className="h-5 w-5 text-blue-500 mt-1" />
                <div>
                  <h4 className="font-medium">AnyProxy</h4>
                  <p className="text-sm text-muted-foreground">
                    The platform uses AnyProxy to intercept and collect HTTP traffic, capturing 
                    details about requests and responses from cloud services. Custom rules 
                    process this data to extract valuable features for analysis.
                  </p>
                </div>
              </div>
            </Card>
          </motion.div>
          
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: 0.1 }}
          >
            <Card className="p-4">
              <div className="flex items-start gap-3">
                <History className="h-5 w-5 text-green-500 mt-1" />
                <div>
                  <h4 className="font-medium">Log Processing</h4>
                  <p className="text-sm text-muted-foreground">
                    Raw JSON logs are transformed into structured CSV files that capture details 
                    such as URL patterns, HTTP methods, headers, and content types. These provide 
                    the foundation for machine learning model training.
                  </p>
                </div>
              </div>
            </Card>
          </motion.div>
        </TabsContent>
        
        <TabsContent value="models" className="space-y-4">
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
          >
            <Card className="p-4">
              <div className="flex items-start gap-3">
                <BrainCircuit className="h-5 w-5 text-purple-500 mt-1" />
                <div>
                  <h4 className="font-medium">DeBERTa</h4>
                  <p className="text-sm text-muted-foreground">
                    The platform leverages DeBERTa transformer models for zero-shot classification, 
                    allowing service and activity identification without task-specific training. These 
                    models provide high accuracy and flexibility for unseen data.
                  </p>
                </div>
              </div>
            </Card>
          </motion.div>
          
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: 0.1 }}
          >
            <Card className="p-4">
              <div className="flex items-start gap-3">
                <Cpu className="h-5 w-5 text-pink-500 mt-1" />
                <div>
                  <h4 className="font-medium">CodeBERT</h4>
                  <p className="text-sm text-muted-foreground">
                    CodeBERT is employed for network traffic classification with dual classification 
                    heads for both service types and activity types. Its strong pattern recognition in 
                    structured data makes it ideal for API traffic analysis.
                  </p>
                </div>
              </div>
            </Card>
          </motion.div>
          
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: 0.2 }}
          >
            <Card className="p-4">
              <div className="flex items-start gap-3">
                <Network className="h-5 w-5 text-orange-500 mt-1" />
                <div>
                  <h4 className="font-medium">Random Forest</h4>
                  <p className="text-sm text-muted-foreground">
                    Random Forest classifiers provide robust prediction capabilities based on feature 
                    extraction from HTTP headers, URLs, and content types. These models achieve high 
                    accuracy for service and activity classification.
                  </p>
                </div>
              </div>
            </Card>
          </motion.div>
        </TabsContent>
        
        <TabsContent value="analysis" className="space-y-4">
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
          >
            <Card className="p-4">
              <div className="flex items-start gap-3">
                <Microscope className="h-5 w-5 text-indigo-500 mt-1" />
                <div>
                  <h4 className="font-medium">API Endpoint Signatures</h4>
                  <p className="text-sm text-muted-foreground">
                    The system identifies unique signatures within API traffic to classify endpoints 
                    and activities. Features like host headers, URL patterns, and content types 
                    serve as strong indicators for service identification.
                  </p>
                </div>
              </div>
            </Card>
          </motion.div>
          
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: 0.1 }}
          >
            <Card className="p-4">
              <div className="flex items-start gap-3">
                <BarChart className="h-5 w-5 text-emerald-500 mt-1" />
                <div>
                  <h4 className="font-medium">Classification Results</h4>
                  <p className="text-sm text-muted-foreground">
                    Model performance shows exceptional accuracy, with service classification 
                    reaching 99% accuracy and activity type classification at nearly 100%. 
                    The system effectively handles services like OneDrive, Dropbox, and Box.
                  </p>
                </div>
              </div>
            </Card>
          </motion.div>
        </TabsContent>
        
        <TabsContent value="workflow" className="space-y-4">
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
          >
            <Card className="p-4">
              <div className="flex items-start gap-3">
                <GitMerge className="h-5 w-5 text-blue-500 mt-1" />
                <div>
                  <h4 className="font-medium">End-to-End Pipeline</h4>
                  <p className="text-sm text-muted-foreground">
                    The workflow begins with data collection via AnyProxy, followed by DeBERTa zero-shot 
                    learning for initial classification. CodeBERT processes the data for deeper insights, 
                    leading to Random Forest training and ultimately C code generation for security rules.
                  </p>
                </div>
              </div>
            </Card>
          </motion.div>
          
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: 0.1 }}
          >
            <Card className="p-4">
              <div className="flex items-start gap-3">
                <LineChart className="h-5 w-5 text-rose-500 mt-1" />
                <div>
                  <h4 className="font-medium">Performance Metrics</h4>
                  <p className="text-sm text-muted-foreground">
                    The combined model approach achieves impressive metrics across all service types. 
                    High confidence scores are maintained for both high and low-volume categories, with 
                    activity classification reaching near-perfect accuracy for login, download, and upload.
                  </p>
                </div>
              </div>
            </Card>
          </motion.div>
        </TabsContent>
      </Tabs>
    </motion.div>
  )
} 