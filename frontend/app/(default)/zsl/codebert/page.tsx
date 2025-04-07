'use client'

import { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { useToast } from '@/components/ui/use-toast'
import { FileList } from '@/components/codebert/file-list'
import { ProgressDisplay } from '@/components/codebert/progress-display'
import { Brain, Loader2 } from 'lucide-react'

interface FileInfo {
    name: string
    path: string
    timestamp: number
}

interface ProcessStatus {
    isRunning: boolean
    error: string | null
    progress: string[]
}

export default function CodeBertPage() {
    const [inputFiles, setInputFiles] = useState<FileInfo[]>([])
    const [predictionFiles, setPredictionFiles] = useState<FileInfo[]>([])
    const [inferenceStatus, setInferenceStatus] = useState<ProcessStatus>({
        isRunning: false,
        error: null,
        progress: []
    })
    const [trainingStatus, setTrainingStatus] = useState<ProcessStatus>({
        isRunning: false,
        error: null,
        progress: []
    })
    const { toast } = useToast()

    const fetchFiles = async () => {
        try {
            const [inputResponse, predResponse] = await Promise.all([
                fetch('/api/codebert'),
                fetch('/api/codebert?type=predictions')
            ])
            const inputData = await inputResponse.json()
            const predData = await predResponse.json()
            setInputFiles(inputData)
            setPredictionFiles(predData)
        } catch (error) {
            console.error('Error fetching files:', error)
            toast({
                title: 'Error',
                description: 'Failed to fetch files',
                variant: 'destructive'
            })
        }
    }

    useEffect(() => {
        fetchFiles()
    }, [])

    const startTraining = async () => {
        try {
            setTrainingStatus({
                isRunning: true,
                error: null,
                progress: ['[*] Starting CodeBERT training...']
            })

            const response = await fetch('/api/codebert/train', {
                method: 'POST'
            })

            const data = await response.json()

            if (response.ok) {
                const progressLines = data.output
                    .split('\n')
                    .filter((line: string) => line.trim() !== '')

                setTrainingStatus(prev => ({
                    ...prev,
                    progress: progressLines
                }))

                toast({
                    title: 'Success',
                    description: 'CodeBERT training completed successfully'
                })
            } else {
                throw new Error(data.error || 'Training failed')
            }
        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred'
            setTrainingStatus(prev => ({
                ...prev,
                error: errorMessage
            }))
            toast({
                title: 'Error',
                description: errorMessage,
                variant: 'destructive'
            })
        } finally {
            setTrainingStatus(prev => ({
                ...prev,
                isRunning: false
            }))
        }
    }

    const startInference = async () => {
        try {
            setInferenceStatus({
                isRunning: true,
                error: null,
                progress: ['[*] Starting CodeBERT inference...']
            })

            const response = await fetch('/api/codebert', {
                method: 'POST'
            })

            const data = await response.json()

            if (response.ok) {
                const progressLines = data.output
                    .split('\n')
                    .filter((line: string) => line.trim() !== '')

                setInferenceStatus(prev => ({
                    ...prev,
                    progress: progressLines
                }))

                toast({
                    title: 'Success',
                    description: 'CodeBERT inference completed successfully'
                })

                // Refresh file lists
                await fetchFiles()
            } else {
                throw new Error(data.error || 'Inference failed')
            }
        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred'
            setInferenceStatus(prev => ({
                ...prev,
                error: errorMessage
            }))
            toast({
                title: 'Error',
                description: errorMessage,
                variant: 'destructive'
            })
        } finally {
            setInferenceStatus(prev => ({
                ...prev,
                isRunning: false
            }))
        }
    }

    return (
        <div className="flex-1 space-y-4 p-4 md:p-8 pt-6">
            <div className="flex items-center justify-between">
                <div>
                    <h2 className="text-3xl font-bold tracking-tight">CodeBERT Zero-Shot Learning</h2>
                    <p className="text-muted-foreground">
                        Process traffic logs using CodeBERT for service and activity classification
                    </p>
                </div>
                <div className="flex space-x-2">
                    <Button
                        onClick={startTraining}
                        disabled={trainingStatus.isRunning || inferenceStatus.isRunning}
                        variant="secondary"
                    >
                        {trainingStatus.isRunning ? (
                            <>
                                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                Training Model
                            </>
                        ) : (
                            <>
                                <Brain className="mr-2 h-4 w-4" />
                                Train Model
                            </>
                        )}
                    </Button>
                    <Button
                        onClick={startInference}
                        disabled={inferenceStatus.isRunning || trainingStatus.isRunning || inputFiles.length === 0}
                    >
                        {inferenceStatus.isRunning ? (
                            <>
                                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                Running Inference
                            </>
                        ) : (
                            'Start Inference'
                        )}
                    </Button>
                </div>
            </div>

            <Tabs defaultValue="files" className="space-y-4">
                <TabsList>
                    <TabsTrigger value="files">Files</TabsTrigger>
                    <TabsTrigger value="training">Training Progress</TabsTrigger>
                    <TabsTrigger value="inference">Inference Progress</TabsTrigger>
                </TabsList>

                <TabsContent value="files" className="space-y-4">
                    <div className="grid gap-4 grid-cols-1 md:grid-cols-2">
                        <Card>
                            <CardHeader>
                                <CardTitle>Input Files</CardTitle>
                                <CardDescription>
                                    CSV files ready for processing
                                </CardDescription>
                            </CardHeader>
                            <CardContent>
                                <FileList
                                    files={inputFiles}
                                    type="input"
                                />
                            </CardContent>
                        </Card>

                        <Card>
                            <CardHeader>
                                <CardTitle>Prediction Files</CardTitle>
                                <CardDescription>
                                    Generated predictions from CodeBERT
                                </CardDescription>
                            </CardHeader>
                            <CardContent>
                                <FileList
                                    files={predictionFiles}
                                    type="prediction"
                                />
                            </CardContent>
                        </Card>
                    </div>
                </TabsContent>

                <TabsContent value="training">
                    <Card>
                        <CardHeader>
                            <CardTitle>Training Progress</CardTitle>
                            <CardDescription>
                                Real-time progress of the CodeBERT training process
                            </CardDescription>
                        </CardHeader>
                        <CardContent>
                            <ProgressDisplay
                                error={trainingStatus.error}
                                progress={trainingStatus.progress}
                            />
                        </CardContent>
                    </Card>
                </TabsContent>

                <TabsContent value="inference">
                    <Card>
                        <CardHeader>
                            <CardTitle>Inference Progress</CardTitle>
                            <CardDescription>
                                Real-time progress of the CodeBERT inference process
                            </CardDescription>
                        </CardHeader>
                        <CardContent>
                            <ProgressDisplay
                                error={inferenceStatus.error}
                                progress={inferenceStatus.progress}
                            />
                        </CardContent>
                    </Card>
                </TabsContent>
            </Tabs>
        </div>
    )
}
