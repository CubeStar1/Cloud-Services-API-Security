'use client'

import { useState, useEffect } from "react"
import { LogFilesGrid, LogFile } from "@/components/logs/log-files-grid"
import { ConvertButton } from "@/components/logs/convert-button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

export default function LogsPage() {
    const [logs, setLogs] = useState<LogFile[]>([])
    const [isLoading, setIsLoading] = useState(true)

    useEffect(() => {
        const fetchLogs = async () => {
            try {
                const response = await fetch('/api/anyproxy/logs')
                const data = await response.json()
                setLogs(data)
            } catch (error) {
                console.error('Error fetching logs:', error)
            } finally {
                setIsLoading(false)
            }
        }
        fetchLogs()
    }, [])

    const rawLogs = logs.filter(l => l.name.endsWith('.json'))
    const csvLogs = logs.filter(l => l.name.endsWith('.csv'))

    return (
        <div className="flex-1 space-y-4 p-8 pt-6">
            <div className="flex items-center justify-between space-y-2">
                <div>
                    <h2 className="text-3xl font-bold tracking-tight">Traffic Logs</h2>
                    <p className="text-muted-foreground">
                        View and manage your proxy traffic logs
                    </p>
                </div>
                <ConvertButton />
            </div>
            <Tabs defaultValue="raw" className="space-y-4">
                <TabsList>
                    <TabsTrigger value="raw">Raw JSON</TabsTrigger>
                    <TabsTrigger value="csv">CSV Files</TabsTrigger>
                </TabsList>
                <TabsContent value="raw" className="space-y-4">
                    <Card>
                        <CardHeader>
                            <CardTitle>Raw JSON Logs</CardTitle>
                            <CardDescription>
                                Raw traffic logs from your proxy sessions. Click on a file to view its contents.
                            </CardDescription>
                        </CardHeader>
                        <CardContent>
                            <LogFilesGrid files={rawLogs} type="json" />
                        </CardContent>
                    </Card>
                </TabsContent>
                <TabsContent value="csv" className="space-y-4">
                    <Card>
                        <CardHeader>
                            <CardTitle>CSV Log Files</CardTitle>
                            <CardDescription>
                                Processed CSV files for analysis. Use the convert button to generate CSVs from raw logs.
                            </CardDescription>
                        </CardHeader>
                        <CardContent>
                            <LogFilesGrid files={csvLogs} type="csv" />
                        </CardContent>
                    </Card>
                </TabsContent>
            </Tabs>
        </div>
    )
}
