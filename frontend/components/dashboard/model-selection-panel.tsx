'use client'

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Label } from '@/components/ui/label'
import { Settings2 } from 'lucide-react'

interface ModelSelectionPanelProps {
    selectedModel: string;
    onModelChange: (value: string) => void;
    disabled?: boolean; // Allow disabling when running
}

export function ModelSelectionPanel({ 
    selectedModel, 
    onModelChange,
    disabled = false
}: ModelSelectionPanelProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
            <Settings2 className="h-5 w-5" /> Model Selection
        </CardTitle>
        <CardDescription>Choose the classification model.</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-2">
          <Label htmlFor="model-select">Active Model</Label>
          <Select 
            value={selectedModel} 
            onValueChange={onModelChange}
            disabled={disabled}
           >
            <SelectTrigger id="model-select">
              <SelectValue placeholder="Select a model" />
            </SelectTrigger>
            <SelectContent>
              {/* Simplified Model Names */}
              <SelectItem value="deberta">DeBERTa (Zero-Shot)</SelectItem>
              <SelectItem value="codebert">CodeBERT (Supervised)</SelectItem>
              <SelectItem value="random-forest">Random Forest (Live)</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </CardContent>
    </Card>
  )
} 