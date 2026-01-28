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

    const fetchLogFiles = async () => {
        try {
            const response = await fetch('/api/anyproxy/logs')
            const files = await response.json()
            setLogFiles(files)
            if (files.length > 0 && !selectedFile) {
                setSelectedFile(files[0].name)
            }
        } catch (error) {
            console.error('Error fetching log files:', error)
        }
    }

    const fetchLogs = async () => {
        if (!selectedFile) return
        
        setIsLoading(true)
        try {
            const response = await fetch(`/api/anyproxy/logs?file=${encodeURIComponent(selectedFile)}`)
            const data = await response.json()
            setLogs(data)
        } catch (error) {
            console.error('Error fetching logs:', error)
        } finally {
            setIsLoading(false)
        }
    }

    useEffect(() => {
        fetchLogFiles()
        const interval = setInterval(fetchLogFiles, 5000) // Check for new files every 5 seconds
        return () => clearInterval(interval)
    }, [])

    useEffect(() => {
        if (selectedFile) {
            fetchLogs()
            const interval = setInterval(fetchLogs, 2000) // Update logs every 2 seconds
            return () => clearInterval(interval)
        }
    }, [selectedFile])

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
                        <div className="flex items-center gap-2 mb-3">
                            <Badge variant={log.type === 'request' ? "outline" : "default"}>
                                {log.type.toUpperCase()}
                            </Badge>
                            <Badge variant="secondary">
                                {log.method}
                            </Badge>
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
                {(!selectedFile || logs.length === 0) && (
                    <div className="text-center text-muted-foreground py-8">
                        {!selectedFile ? 'Select a log file to view logs' : 'No logs available in this file'}
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
                        <Select value={selectedFile} onValueChange={setSelectedFile}>
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
                            disabled={isLoading || !selectedFile}
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