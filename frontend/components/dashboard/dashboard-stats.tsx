'use client'

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Activity, BarChart3, ServerCrash } from 'lucide-react'

// Define props for the component
interface DashboardStatsProps {
  totalRequests: number;
  anomaliesCount: number;
  selectedModel: string;
  isRunning: boolean; // To potentially adjust display based on status
}

// Mapping for display names
const modelDisplayNames: { [key: string]: string } = {
  'deberta': 'DeBERTa',
  'codebert': 'CodeBERT',
  'random-forest': 'Random Forest'
};

export function DashboardStats({ 
  totalRequests, 
  anomaliesCount, 
  selectedModel,
  isRunning
}: DashboardStatsProps) {
  const displayModelName = modelDisplayNames[selectedModel] || selectedModel; // Fallback to key if not found

  return (
    <div className="grid gap-4 md:grid-cols-3">
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          {/* Changed title slightly */}
          <CardTitle className="text-sm font-medium">Total Requests Captured</CardTitle>
          <Activity className="h-4 w-4 text-muted-foreground" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">{totalRequests}</div>
          <p className={`text-xs ${isRunning ? 'text-green-600' : 'text-muted-foreground'}`}>
            {isRunning ? 'Processing live...' : 'Process stopped'}
          </p>
        </CardContent>
      </Card>
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Classified Requests</CardTitle>
          <ServerCrash className="h-4 w-4 text-muted-foreground" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">{anomaliesCount}</div>
          <p className="text-xs text-muted-foreground">Since process started</p> 
        </CardContent>
      </Card>
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Active Model</CardTitle>
          <BarChart3 className="h-4 w-4 text-muted-foreground" />
        </CardHeader>
        <CardContent>
          {/* Display the selected model name */}
          <div className="text-2xl font-bold">{displayModelName}</div>
          <p className="text-xs text-muted-foreground">
            {isRunning ? 'Currently classifying' : 'Selected for next run'}
          </p>
        </CardContent>
      </Card>
    </div>
  )
} 