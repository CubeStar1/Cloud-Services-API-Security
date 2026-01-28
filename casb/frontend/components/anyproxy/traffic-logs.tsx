'use client'

import { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { RefreshCw, Table as TableIcon, LayoutList } from 'lucide-react'
import { TrafficTable } from './traffic-table'
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import { toast } from 'sonner'

interface LogFile {
    name: string
    path: string
    timestamp: number
}

interface LogEntry {
    type: 'request' | 'response'
    url: string
    method: string
    headers_Host: string
    requestHeaders_Origin?: string
    requestHeaders_Content_Type?: string
    requestHeaders_Referer?: string
    requestHeaders_Accept?: string
    responseHeaders_Content_Type?: string
    body: any
}

type ViewMode = 'cards' | 'table'

export function TrafficLogs() {
    const [logs, setLogs] = useState<LogEntry[]>([])
    const [logFiles, setLogFiles] = useState<LogFile[]>([])
    const [selectedFile, setSelectedFile] = useState<string>('')
    const [isLoading, setIsLoading] = useState(false)
    const [viewMode, setViewMode] = useState<ViewMode>('cards')

    const [isLive, setIsLive] = useState(false)
    const [eventSource, setEventSource] = useState<EventSource | null>(null)

    const fetchLogFiles = async () => {
        try {
            // Fetch only JSON files from raw-json directory to avoid CSV crash
            const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000'
            const response = await fetch(`${BACKEND_URL}/files?subdir=logs/raw-json&ext=json`)
            const files = await response.json()
            if (Array.isArray(files)) {
                setLogFiles(files)
                if (files.length > 0 && !selectedFile && !isLive) {
                    setSelectedFile(files[0].name)
                }
            }
        } catch (error) {
            console.error('Error fetching log files:', error)
        }
    }

    const fetchLogs = async () => {
        if (!selectedFile || isLive) return
        
        setIsLoading(true)
        try {
            // Fetch file content directly from backend
            const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000'
            const response = await fetch(`${BACKEND_URL}/files?file=logs/raw-json/${encodeURIComponent(selectedFile)}`)
            const data = await response.json()
            
                try {
                    const cleanContent = data.content.trim().replace(/,\s*$/, '')
                    if (cleanContent) {
                         const parsed = JSON.parse(`[${cleanContent}]`)
                         // File content is directly the request/response objects, not wrapped in { type: 'log', data: ... }
                         // So we just use parsed directly.
                         setLogs(parsed.reverse()) // Reverse to show newest first if file is appended
                    } else {
                        setLogs([])
                    }
                } catch (e) {
                    console.error("Error parsing logs:", e)
                    setLogs([])
                }
        } catch (error) {
            console.error('Error fetching logs:', error)
        } finally {
            setIsLoading(false)
        }
    }

    // Streaming effect
    useEffect(() => {
        if (!isLive) {
            if (eventSource) {
                eventSource.close()
                setEventSource(null)
            }
            return
        }

        const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000'
        const es = new EventSource(`${BACKEND_URL}/anyproxy/logs/stream`)
        
        es.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data)
                // Stream data IS wrapped in { type: 'log', data: ... } from rule.js console.log
                if (data.type === 'log') {
                    setLogs(prev => [data.data, ...prev].slice(0, 1000))
                }
            } catch (e) {
                console.error('Error parsing stream data:', e)
            }
        }

        es.onerror = (error) => {
            console.error('EventSource error:', error)
            es.close()
            setIsLive(false) 
            toast.error("Live stream disconnected")
        }

        setEventSource(es)

        return () => {
            es.close()
        }
    }, [isLive])

    // Initial load only
    useEffect(() => {
        fetchLogFiles()
        const interval = setInterval(fetchLogFiles, 5000) // Keep file list update
        return () => clearInterval(interval)
    }, [])

    useEffect(() => {
        if (selectedFile && !isLive) {
            fetchLogs()
            // Removed interval polling for file content. User can use Refresh button.
        }
    }, [selectedFile, isLive])

    const renderHeaders = (log: LogEntry) => {
        const headers = [
            { label: 'Host', value: log.headers_Host },
            { label: 'Origin', value: log.requestHeaders_Origin },
            { label: 'Content-Type', value: log.requestHeaders_Content_Type || log.responseHeaders_Content_Type },
            { label: 'Referer', value: log.requestHeaders_Referer },
            { label: 'Accept', value: log.requestHeaders_Accept }
        ].filter(header => header.value)

        return headers.map((header, i) => (
            <div key={i} className="text-sm">
                <span className="font-medium text-muted-foreground">{header.label}: </span>
                <span className="text-xs break-all">{header.value}</span>
            </div>
        ))
    }

    const formatDate = (timestamp: number) => {
        return new Date(timestamp).toLocaleString()
    }

    const renderCardView = () => (
        <ScrollArea className="h-[calc(100vh-300px)] rounded-lg border">
            <div className="space-y-4 p-4">
                {logs.map((log, index) => (
                    <div key={index} className="rounded-lg border bg-card p-4">
                        {/* existing card content... reusing same structure but simplified for diff */}
                        <div className="flex items-center gap-2 mb-3">
                            <Badge variant={log.type === 'request' ? "outline" : "default"}>
                                {log.type.toUpperCase()}
                            </Badge>
                            <Badge variant="secondary">
                                {log.method}
                            </Badge>
                             {/* Added detailed timestamp in card if needed, or rely on order */}
                        </div>
                        <div className="space-y-2">
                            <div className="text-sm break-all">
                                <span className="font-medium text-muted-foreground">URL: </span>
                                {log.url}
                            </div>
                            {renderHeaders(log)}
                            {log.body && (
                                <div className="mt-3">
                                    <div className="font-medium text-sm text-muted-foreground mb-1">Body:</div>
                                    <pre className="text-xs bg-muted p-2 rounded overflow-x-auto whitespace-pre-wrap break-all">
                                        {typeof log.body === 'string' ? log.body : JSON.stringify(log.body, null, 2)}
                                    </pre>
                                </div>
                            )}
                        </div>
                    </div>
                ))}
                {(!isLive && (!selectedFile || logs.length === 0)) && (
                    <div className="text-center text-muted-foreground py-8">
                        {!selectedFile ? 'Select a log file or start Live Stream' : 'No logs available in this file'}
                    </div>
                )}
                {isLive && logs.length === 0 && (
                     <div className="text-center text-muted-foreground py-8">
                        Waiting for traffic...
                    </div>
                )}
            </div>
        </ScrollArea>
    )

    return (
        <Card className="w-full">
            <CardHeader>
                <div className="flex items-center justify-between mb-2">
                    <CardTitle>Traffic Logs</CardTitle>
                    <div className="flex items-center gap-2">
                        <ToggleGroup type="single" value={viewMode} onValueChange={(value: ViewMode) => value && setViewMode(value)}>
                            <ToggleGroupItem value="cards" aria-label="View as cards">
                                <LayoutList className="h-4 w-4" />
                            </ToggleGroupItem>
                            <ToggleGroupItem value="table" aria-label="View as table">
                                <TableIcon className="h-4 w-4" />
                            </ToggleGroupItem>
                        </ToggleGroup>
                        
                         <Button
                            variant={isLive ? "destructive" : "default"}
                            size="sm"
                            onClick={() => {
                                setIsLive(!isLive)
                                if (!isLive) setLogs([]) // Clear old logs when starting stream
                            }}
                        >
                            {isLive ? 'Stop Live' : 'Go Live'}
                        </Button>

                        <Select value={selectedFile} onValueChange={setSelectedFile} disabled={isLive}>
                            <SelectTrigger className="w-[200px]">
                                <SelectValue placeholder="Select log file" />
                            </SelectTrigger>
                            <SelectContent>
                                {logFiles.map((file) => (
                                    <SelectItem key={file.name} value={file.name}>
                                        <div className="flex flex-col">
                                            <span>{file.name}</span>
                                            <span className="text-xs text-muted-foreground">
                                                {formatDate(file.timestamp)}
                                            </span>
                                        </div>
                                    </SelectItem>
                                ))}
                            </SelectContent>
                        </Select>
                        <Button
                            variant="outline"
                            size="sm"
                            onClick={fetchLogs}
                            disabled={isLoading || !selectedFile || isLive}
                        >
                            <RefreshCw className={`mr-2 h-4 w-4 ${isLoading ? 'animate-spin' : ''}`} />
                            Refresh
                        </Button>
                    </div>
                </div>
                <CardDescription>
                    View network traffic logs in real-time
                </CardDescription>
            </CardHeader>
            <CardContent>
                {viewMode === 'cards' ? renderCardView() : <TrafficTable logs={logs} />}
            </CardContent>
        </Card>
    )
} 