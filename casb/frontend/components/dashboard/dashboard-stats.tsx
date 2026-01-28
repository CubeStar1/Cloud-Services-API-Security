'use client'

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Activity, BarChart3, ServerCrash, Zap, Play, Square, Settings2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'

// Define props for the component
interface DashboardStatsProps {
  totalRequests: number;
  totalClassified: number;
  anomaliesCount: number;
  mostFrequentService: { service: string; count: number } | null;
  isRunning: boolean;
  onToggleRun: () => void;
  engine: string;
  onEngineChange: (value: string) => void;
  avgLatency: number;
}

export function DashboardStats({ 
  totalRequests, 
  totalClassified,
  anomaliesCount, 
  mostFrequentService,
  isRunning,
  onToggleRun,
  engine,
  onEngineChange,
  avgLatency
}: DashboardStatsProps) {
  const rps = avgLatency > 0 ? (1000 / avgLatency).toFixed(1) : '0';

  return (
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-5">
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
          <Zap className="h-4 w-4 text-muted-foreground" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">{totalClassified}</div>
          <p className="text-xs text-muted-foreground">
            Detected {anomaliesCount} anomalies
          </p> 
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
          <CardTitle className="text-sm font-medium">Inference Performance</CardTitle>
          <BarChart3 className="h-4 w-4 text-muted-foreground" />
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">{rps} req/s</div>
          <p className="text-xs text-muted-foreground">
            Avg Latency: {avgLatency.toFixed(2)} ms
          </p>
        </CardContent>
      </Card>
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Process Control</CardTitle>
          <Settings2 className="h-4 w-4 text-muted-foreground" />
        </CardHeader>
        <CardContent className="space-y-4 pt-2">
          <div className="flex items-center gap-2">
            <Select value={engine} onValueChange={onEngineChange} disabled={isRunning}>
              <SelectTrigger className="w-full h-8">
                <SelectValue placeholder="Engine" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="python">Python Engine</SelectItem>
                <SelectItem value="c">C Engine (Compiled)</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <Button 
            onClick={onToggleRun} 
            variant={isRunning ? "destructive" : "default"} 
            className="w-full"
          >
            {isRunning ? <Square className="mr-2 h-4 w-4" /> : <Play className="mr-2 h-4 w-4" />}
            {isRunning ? 'Stop Process' : 'Start Process'}
          </Button>
        </CardContent>
      </Card>
    </div>
  )
} 