'use client'

import { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Input } from '@/components/ui/input'
import { toast } from 'sonner'

export function ProxyControl() {
    const [isRunning, setIsRunning] = useState(false)
    const [isLoading, setIsLoading] = useState(false)
    const [filename, setFilename] = useState('box-traffic-logs.json')

    const handleProxyAction = async (action: 'start' | 'stop') => {
        setIsLoading(true)
        try {
            const endpoint = action === 'start' ? 'start' : 'stop'
            const body = action === 'start' ? { filename } : {}
            const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000'
            
            const response = await fetch(`${BACKEND_URL}/anyproxy/${endpoint}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body)
            })
            const data = await response.json()
            
            if (data.status === 'success') {
                setIsRunning(action === 'start')
                toast.success(data.message)
            } else {
                toast.error(data.message)
            }
        } catch (error) {
            console.error(`Failed to ${action} proxy:`, error)
            toast.error(`Failed to ${action} proxy`)
        } finally {
            setIsLoading(false)
        }
    }

    // Check proxy status on component mount
    useState(() => {
        const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000'
        fetch(`${BACKEND_URL}/anyproxy/status`)
        .then(res => res.json())
        .then(data => {
            if (data.status === 'success') {
                setIsRunning(data.isRunning)
                if (data.currentLogFile) {
                    setFilename(data.currentLogFile.replace('.json', ''))
                }
            }
        })
        .catch(console.error)
    })

    return (
        <Card className="w-full">
            <CardHeader>
                <CardTitle className="flex items-center justify-between">
                    Proxy Control
                    <Badge variant={isRunning ? "default" : "destructive"}>
                        {isRunning ? "Running" : "Stopped"}
                    </Badge>
                </CardTitle>
                <CardDescription>
                    Control the AnyProxy instance for traffic monitoring
                </CardDescription>
            </CardHeader>
            <CardContent>
                <div className="space-y-4">
                    <div className="flex items-center gap-4">
                        <Input
                            placeholder="Enter log filename"
                            value={filename}
                            onChange={(e) => setFilename(e.target.value)}
                            disabled={isRunning || isLoading}
                            className="flex-1"
                        />
                        <span className="text-sm text-muted-foreground">.json</span>
                    </div>
                    <div className="flex gap-4">
                        <Button
                            variant={isRunning ? "outline" : "default"}
                            onClick={() => handleProxyAction('start')}
                            disabled={isLoading || isRunning || !filename.trim()}
                            className="flex-1"
                        >
                            Start Proxy
                        </Button>
                        <Button
                            variant={isRunning ? "destructive" : "outline"}
                            onClick={() => handleProxyAction('stop')}
                            disabled={isLoading || !isRunning}
                            className="flex-1"
                        >
                            Stop Proxy
                        </Button>
                    </div>
                </div>
            </CardContent>
        </Card>
    )
}