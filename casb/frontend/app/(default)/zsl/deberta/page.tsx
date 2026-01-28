"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Loader2 } from "lucide-react"
import { useState, useEffect } from "react"
import { FileList } from "@/components/deberta/file-list"
import { ProgressDisplay } from "@/components/deberta/progress-display"

interface FileInfo {
    name: string
    path: string
    timestamp: number
}

interface InferenceStatus {
    isRunning: boolean
    progress: string[]
    error: string | null
}

export default function DeBERTaPage() {
    const [inputFiles, setInputFiles] = useState<FileInfo[]>([])
    const [predictionFiles, setPredictionFiles] = useState<FileInfo[]>([])
    const [status, setStatus] = useState<InferenceStatus>({
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
            const [inputResponse, predResponse] = await Promise.all([
                fetch('/api/deberta'),
                fetch('/api/deberta?type=predictions')
            ])
            const inputData = await inputResponse.json()
            const predData = await predResponse.json()
            setInputFiles(inputData)
            setPredictionFiles(predData)
        } catch (error) {
            console.error('Error fetching files:', error)
        }
    }

    const startInference = async () => {
        setStatus({
            isRunning: true,
            progress: ['Starting DeBERTa inference...'],
            error: null
        })

        try {
            const response = await fetch('/api/deberta', {
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
                // Refresh files after successful inference
                fetchFiles()
            } else {
                setStatus(prev => ({
                    ...prev,
                    isRunning: false,
                    error: data.error || 'Failed to complete inference'
                }))
            }
        } catch (error) {
            setStatus(prev => ({
                ...prev,
                isRunning: false,
                error: 'Failed to start inference process'
            }))
        }
    }

    return (
        <div className="flex-1 space-y-4 p-4 md:p-8 pt-6">
            <div className="flex items-center justify-between">
                <div>
                    <h2 className="text-3xl font-bold tracking-tight">DeBERTa Inference</h2>
                    <p className="text-muted-foreground">
                        Zero-shot classification of traffic logs using DeBERTa
                    </p>
                </div>
                <Button
                    onClick={startInference}
                    disabled={status.isRunning || inputFiles.length === 0}
                >
                    {status.isRunning ? (
                        <>
                            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                            Running Inference
                        </>
                    ) : (
                        'Start Inference'
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
                                <CardTitle>Input Files</CardTitle>
                                <CardDescription>
                                    CSV files ready for inference
                                </CardDescription>
                            </CardHeader>
                            <CardContent>
                                <FileList files={inputFiles} type="input" />
                            </CardContent>
                        </Card>

                        <Card>
                            <CardHeader>
                                <CardTitle>Prediction Files</CardTitle>
                                <CardDescription>
                                    Generated predictions from DeBERTa
                                </CardDescription>
                            </CardHeader>
                            <CardContent>
                                <FileList files={predictionFiles} type="prediction" />
                            </CardContent>
                        </Card>
                    </div>
                </TabsContent>

                <TabsContent value="progress">
                    <Card>
                        <CardHeader>
                            <CardTitle>Inference Progress</CardTitle>
                            <CardDescription>
                                Real-time progress of the DeBERTa inference process
                            </CardDescription>
                        </CardHeader>
                        <CardContent>
                            <ProgressDisplay
                                error={status.error}
                                progress={status.progress}
                            />
                        </CardContent>
                    </Card>
                </TabsContent>
            </Tabs>
        </div>
    )
}
