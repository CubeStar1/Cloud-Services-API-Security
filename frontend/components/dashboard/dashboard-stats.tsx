'use client'

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Activity, BarChart3, ServerCrash, Zap, Play, Square } from 'lucide-react'
import { Button } from '@/components/ui/button'

// Define props for the component
interface DashboardStatsProps {
  totalRequests: number;
  anomaliesCount: number;
  mostFrequentService: { service: string; count: number } | null;
  isRunning: boolean;
  onToggleRun: () => void;
}

export function DashboardStats({ 
  totalRequests, 
  anomaliesCount, 
  mostFrequentService,
  isRunning,
  onToggleRun
}: DashboardStatsProps) {
  return (
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
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
          <CardTitle className="text-sm font-medium">Total Requests Classified</CardTitle>
          <ServerCrash className="h-4 w-4 text-muted-foreground" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">{anomaliesCount}</div>
          <p className="text-xs text-muted-foreground">Since process started</p> 
        </CardContent>
      </Card>
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Top Application</CardTitle>
          <BarChart3 className="h-4 w-4 text-muted-foreground" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">
            {mostFrequentService ? mostFrequentService.service : 'N/A'}
          </div>
          <p className="text-xs text-muted-foreground">
            {mostFrequentService ? `(${mostFrequentService.count} requests)` : 'No classifications yet'}
          </p>
        </CardContent>
      </Card>
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Start/Stop Classification</CardTitle>
          <Zap className="h-4 w-4 text-muted-foreground" />
        </CardHeader>
        <CardContent className="flex items-center justify-center pt-4 pb-2">
          <Button onClick={onToggleRun} variant={isRunning ? "destructive" : "default"}>
            {isRunning ? <Square className="mr-2 h-4 w-4" /> : <Play className="mr-2 h-4 w-4" />}
            {isRunning ? 'Stop Process' : 'Start Process'}
          </Button>
        </CardContent>
      </Card>
    </div>
  )
} 