"use client"
import { Metadata } from "next"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Badge } from "@/components/ui/badge"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Loader2, AlertCircle, FileText, FileCheck } from "lucide-react"
import { useState, useEffect } from "react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

interface FileInfo {
    name: string
    path: string
    timestamp: number
}

interface LabellingStatus {
    isRunning: boolean
    progress: string[]
    error: string | null
}

export default function LabellingPage() {
    const [csvFiles, setCsvFiles] = useState<FileInfo[]>([])
    const [labelledFiles, setLabelledFiles] = useState<FileInfo[]>([])
    const [status, setStatus] = useState<LabellingStatus>({
        isRunning: false,
        progress: [],
        error: null
    })

    // Fetch files on component mount
    useEffect(() => {
        fetchFiles()
    }, [])

    const fetchFiles = async () => {
        try {
            const [csvResponse, labelledResponse] = await Promise.all([
                fetch('/api/labelling'),
                fetch('/api/labelling/labelled')
            ])
            const csvData = await csvResponse.json()
            const labelledData = await labelledResponse.json()
            setCsvFiles(csvData)
            setLabelledFiles(labelledData)
        } catch (error) {
            console.error('Error fetching files:', error)
        }
    }

    const startLabelling = async () => {
        setStatus({
            isRunning: true,
            progress: ['Starting labelling process...'],
            error: null
        })

        try {
            const response = await fetch('/api/labelling', {
                method: 'POST'
            })
            const data = await response.json()

            if (data.success) {
                const progressLines = data.output.split('\n').filter(Boolean)
                setStatus(prev => ({
                    ...prev,
                    isRunning: false,
                    progress: progressLines
                }))
                // Refresh files after successful labelling
                fetchFiles()
            } else {
                setStatus(prev => ({
                    ...prev,
                    isRunning: false,
                    error: data.error || 'Failed to complete labelling process'
                }))
            }
        } catch (error) {
            setStatus(prev => ({
                ...prev,
                isRunning: false,
                error: 'Failed to start labelling process'
            }))
        }
    }

    const FileList = ({ files, type }: { files: FileInfo[], type: 'csv' | 'labelled' }) => (
        <ScrollArea className="h-[300px] pr-4">
            {files.length === 0 ? (
                <p className="text-sm text-muted-foreground">
                    No {type} files found
                </p>
            ) : (
                <div className="space-y-2">
                    {files.map((file) => (
                        <div
                            key={file.path}
                            className="flex items-center justify-between p-2 rounded-lg border"
                        >
                            <div className="flex items-center space-x-2">
                                {type === 'csv' ? (
                                    <FileText className="h-4 w-4 text-muted-foreground" />
                                ) : (
                                    <FileCheck className="h-4 w-4 text-green-500" />
                                )}
                                <span className="text-sm font-medium">
                                    {file.name}
                                </span>
                            </div>
                            <Badge variant="secondary">
                                {new Date(file.timestamp).toLocaleDateString()}
                            </Badge>
                        </div>
                    ))}
                </div>
            )}
        </ScrollArea>
    )

    return (
        <div className="flex-1 space-y-4 p-4 md:p-8 pt-6">
            <div className="flex items-center justify-between">
                <div>
                    <h2 className="text-3xl font-bold tracking-tight">Data Labelling</h2>
                    <p className="text-muted-foreground">
                        Label your traffic logs using AI models
                    </p>
                </div>
                <Button
                    onClick={startLabelling}
                    disabled={status.isRunning || csvFiles.length === 0}
                >
                    {status.isRunning ? (
                        <>
                            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                            Labelling in Progress
                        </>
                    ) : (
                        'Start Labelling'
                    )}
                </Button>
            </div>

            <Tabs defaultValue="files" className="space-y-4">
                <TabsList>
                    <TabsTrigger value="files">Files</TabsTrigger>
                    <TabsTrigger value="progress">Progress</TabsTrigger>
                </TabsList>

                <TabsContent value="files" className="space-y-4">
                    <div className="grid gap-4 grid-cols-1 md:grid-cols-2">
                        <Card>
                            <CardHeader>
                                <CardTitle>CSV Files</CardTitle>
                                <CardDescription>
                                    Raw CSV files ready for labelling
                                </CardDescription>
                            </CardHeader>
                            <CardContent>
                                <FileList files={csvFiles} type="csv" />
                            </CardContent>
                        </Card>

                        <Card>
                            <CardHeader>
                                <CardTitle>Labelled Files</CardTitle>
                                <CardDescription>
                                    Processed and labelled files
                                </CardDescription>
                            </CardHeader>
                            <CardContent>
                                <FileList files={labelledFiles} type="labelled" />
                            </CardContent>
                        </Card>
                    </div>
                </TabsContent>

                <TabsContent value="progress">
                    <Card>
                        <CardHeader>
                            <CardTitle>Labelling Progress</CardTitle>
                            <CardDescription>
                                Real-time progress of the labelling process
                            </CardDescription>
                        </CardHeader>
                        <CardContent>
                            <ScrollArea className="h-[300px] pr-4">
                                {status.error ? (
                                    <Alert variant="destructive">
                                        <AlertCircle className="h-4 w-4" />
                                        <AlertDescription>
                                            {status.error}
                                        </AlertDescription>
                                    </Alert>
                                ) : (
                                    <div className="space-y-2">
                                        {status.progress.map((line, index) => (
                                            <p
                                                key={index}
                                                className="text-sm text-muted-foreground"
                                            >
                                                {line}
                                            </p>
                                        ))}
                                    </div>
                                )}
                            </ScrollArea>
                        </CardContent>
                    </Card>
                </TabsContent>
            </Tabs>
        </div>
    )
}
