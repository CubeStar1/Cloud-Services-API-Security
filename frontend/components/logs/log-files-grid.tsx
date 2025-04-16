'use client'

import { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Button } from '@/components/ui/button'
import { FileJson, Trash2, Download } from 'lucide-react'
import { TrafficTable } from '@/components/anyproxy/traffic-table'
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog"

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

export function LogFilesGrid() {
    const [logFiles, setLogFiles] = useState<LogFile[]>([])
    const [selectedFileContent, setSelectedFileContent] = useState<LogEntry[]>([])
    const [isLoading, setIsLoading] = useState(true)

    const fetchLogFiles = async () => {
        try {
            const response = await fetch('/api/anyproxy/logs')
            const files = await response.json()
            setLogFiles(files)
        } catch (error) {
            console.error('Error fetching log files:', error)
        } finally {
            setIsLoading(false)
        }
    }

    const fetchFileContent = async (fileName: string) => {
        try {
            const response = await fetch(`/api/anyproxy/logs?file=${encodeURIComponent(fileName)}`)
            const data = await response.json()
            setSelectedFileContent(data)
        } catch (error) {
            console.error('Error fetching file content:', error)
            setSelectedFileContent([])
        }
    }

    // Fetch log files on component mount
    useState(() => {
        fetchLogFiles()
    })

    const formatDate = (timestamp: number) => {
        return new Date(timestamp).toLocaleString()
    }

    const getRequestCount = (logs: LogEntry[]) => {
        return logs.filter(log => log.type === 'request').length
    }

    const getResponseCount = (logs: LogEntry[]) => {
        return logs.filter(log => log.type === 'response').length
    }

    return (
        <div className="container mx-auto py-6">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {logFiles.map((file) => (
                    <Card key={file.name} className="hover:shadow-lg transition-shadow">
                        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                            <div className="flex items-center space-x-2">
                                <FileJson className="h-4 w-4 text-blue-500" />
                                <CardTitle className="text-sm font-medium">
                                    {file.name}
                                </CardTitle>
                            </div>
                            <div className="flex items-center space-x-2">
                                <Button variant="ghost" size="icon">
                                    <Download className="h-4 w-4 text-gray-500" />
                                </Button>
                                <Button variant="ghost" size="icon">
                                    <Trash2 className="h-4 w-4 text-red-500" />
                                </Button>
                            </div>
                        </CardHeader>
                        <CardContent>
                            <div className="text-xs text-muted-foreground">
                                Last modified: {formatDate(file.timestamp)}
                            </div>
                            <Dialog>
                                <DialogTrigger asChild>
                                    <Button 
                                        variant="outline" 
                                        className="w-full mt-4"
                                        onClick={() => fetchFileContent(file.name)}
                                    >
                                        View Logs
                                    </Button>
                                </DialogTrigger>
                                <DialogContent className="max-w-[90vw] h-[90vh]">
                                    <DialogHeader>
                                        <DialogTitle>{file.name}</DialogTitle>
                                        <DialogDescription>
                                            Traffic logs from {formatDate(file.timestamp)}
                                        </DialogDescription>
                                    </DialogHeader>
                                    <div className="flex-1 overflow-hidden">
                                        <TrafficTable logs={selectedFileContent} />
                                    </div>
                                </DialogContent>
                            </Dialog>
                        </CardContent>
                    </Card>
                ))}
                {logFiles.length === 0 && !isLoading && (
                    <div className="col-span-full text-center text-muted-foreground py-12">
                        No log files available
                    </div>
                )}
            </div>
        </div>
    )
} 