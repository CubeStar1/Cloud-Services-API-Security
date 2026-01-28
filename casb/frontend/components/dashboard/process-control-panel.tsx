'use client'

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Play, Square, Power } from 'lucide-react'

interface ProcessControlPanelProps {
    isRunning: boolean;
    onToggleRun: () => void;
}

export function ProcessControlPanel({ isRunning, onToggleRun }: ProcessControlPanelProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
            <Power className="h-5 w-5" /> Process Control
        </CardTitle>
        <CardDescription>Start or stop live data collection & classification.</CardDescription>
      </CardHeader>
      <CardContent className="flex justify-center pt-4">
        <Button 
            onClick={onToggleRun} 
            variant={isRunning ? 'destructive' : 'default'}
            className="w-full md:w-auto"
        >
          {isRunning ? (
            <><Square className="mr-2 h-4 w-4" /> Stop Process</>
          ) : (
            <><Play className="mr-2 h-4 w-4" /> Start Process</>
          )}
        </Button>
      </CardContent>
    </Card>
  )
} 