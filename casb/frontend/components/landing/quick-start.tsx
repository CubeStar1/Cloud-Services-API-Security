'use client'

import { motion } from 'framer-motion'
import { Button } from '@/components/ui/button'
import { CodeBlock, CodeBlockCode, CodeBlockGroup } from '@/components/ui/code-block'
import { Copy, Check } from 'lucide-react'
import { useState } from 'react'

export function QuickStart() {
  const [copied, setCopied] = useState(false)

  const handleCopy = (code: string) => {
    navigator.clipboard.writeText(code)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const installCode = `
# Clone the repository
git clone https://github.com/CubeStar1/Cloud-Services-API-Security.git

# Navigate to the project directory
cd Cloud-Services-API-Security

# Create and activate virtual environment
python -m venv venv

# Activate virtual environment
./venv/Scripts/activate

# Install dependencies
pip install -r requirements.txt

# Navigate to the frontend directory
cd frontend

# Install frontend dependencies
npm install

# Start the development server
npm run dev

# Start AnyProxy for traffic collection (in a separate terminal)
Use the GUI to collect traffic by going to http://localhost:3000
`.trim()

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.8 }}
      className="my-12 max-w-3xl mx-auto"
    >
      <h2 className="text-2xl font-semibold mb-4 text-center">Quick Start Guide</h2>
      
      <div className="bg-card border rounded-lg overflow-hidden">
        <CodeBlock>
          <CodeBlockGroup className="border-border border-b py-2 px-4 flex justify-between items-center">
            <span className="text-sm font-medium">Installation</span>
            <Button
              variant="ghost"
              size="icon"
              className="h-8 w-8"
              onClick={() => handleCopy(installCode)}
            >
              {copied ? (
                <Check className="h-4 w-4 text-green-500" />
              ) : (
                <Copy className="h-4 w-4" />
              )}
            </Button>
          </CodeBlockGroup>
          <CodeBlockCode code={installCode} language="bash" />
        </CodeBlock>
      </div>
      
      <div className="mt-6 text-center">
        <Button asChild>
          <a href="https://github.com/yourusername/Cloud-Services-API-Security" target="_blank" rel="noopener noreferrer">
            View Documentation
          </a>
        </Button>
      </div>
    </motion.div>
  )
} 