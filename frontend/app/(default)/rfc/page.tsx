'use client'

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Button } from "@/components/ui/button"
import { FileList, FileInfo } from "@/components/rfc/file-list"
import { ProgressDisplay } from "@/components/rfc/progress-display"
import { useState, useEffect } from "react"
import { TreesIcon, AlertCircle, Code2Icon, Copy, Check, Expand, MinusSquare } from "lucide-react"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { useToast } from "@/components/ui/use-toast"
import { CodeBlock, CodeBlockCode, CodeBlockGroup } from "@/components/ui/code-block"

interface CodeFile {
    name: string;
    path: string;
    timestamp: string;
    content?: string;
}

const MAX_PREVIEW_LINES = 50  // Show first 50 lines by default

export default function RandomForestPage() {
    const [csvFiles, setCsvFiles] = useState<FileInfo[]>([])
    const [modelFiles, setModelFiles] = useState<FileInfo[]>([])
    const [selectedFile, setSelectedFile] = useState<FileInfo | null>(null)
    const [isTraining, setIsTraining] = useState(false)
    const [isGenerating, setIsGenerating] = useState(false)
    const [progress, setProgress] = useState<string[]>([])
    const [generationOutput, setGenerationOutput] = useState<string[]>([])
    const [error, setError] = useState<string | null>(null)
    const [generationError, setGenerationError] = useState<string | null>(null)
    const [metrics, setMetrics] = useState<any>(null)
    const [codeFiles, setCodeFiles] = useState<CodeFile[]>([])
    const [selectedCodeFile, setSelectedCodeFile] = useState<CodeFile | null>(null)
    const [copied, setCopied] = useState(false)
    const { toast } = useToast()
    const [isExpanded, setIsExpanded] = useState(false)

    // Fetch CSV and model files
    useEffect(() => {
        const fetchFiles = async () => {
            try {
                const [csvResponse, modelResponse, codeResponse] = await Promise.all([
                    fetch('/api/rfc'),
                    fetch('/api/rfc?type=models'),
                    fetch('/api/rfc/code')
                ])
                
                if (csvResponse.ok && modelResponse.ok && codeResponse.ok) {
                    const csvData = await csvResponse.json()
                    const modelData = await modelResponse.json()
                    const codeData = await codeResponse.json()
                    setCsvFiles(csvData.files)
                    setModelFiles(modelData.files)
                    setCodeFiles(codeData.files)
                }
            } catch (error) {
                console.error('Error fetching files:', error)
                toast({
                    variant: "destructive",
                    title: "Error",
                    description: "Failed to fetch files"
                })
            }
        }

        fetchFiles()
    }, [toast])

    // Fetch code file content when selected
    useEffect(() => {
        const fetchCodeContent = async () => {
            if (!selectedCodeFile) return

            try {
                const response = await fetch(`/api/rfc/code?file=${selectedCodeFile.name}`)
                if (response.ok) {
                    const data = await response.json()
                    setSelectedCodeFile(prev => prev ? { ...prev, content: data.content } : null)
                }
            } catch (error) {
                console.error('Error fetching code content:', error)
            }
        }

        if (selectedCodeFile && !selectedCodeFile.content) {
            fetchCodeContent()
        }
    }, [selectedCodeFile])

    const startTraining = async () => {
        if (!selectedFile) {
            toast({
                variant: "destructive",
                title: "Error",
                description: "Please select a file to train on"
            })
            return
        }

        setIsTraining(true)
        setProgress([])
        setError(null)
        setMetrics(null)

        try {
            const response = await fetch('/api/rfc', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    file: selectedFile.name
                })
            })

            const data = await response.json()

            if (response.ok) {
                setProgress(data.output || [])
                setMetrics(data.metrics)
                toast({
                    title: "Success",
                    description: "Model training completed successfully"
                })
            } else {
                setError(data.error || 'Training failed')
                toast({
                    variant: "destructive",
                    title: "Error",
                    description: data.error || 'Training failed'
                })
            }
        } catch (error) {
            console.error('Error during training:', error)
            setError('Failed to start training')
            toast({
                variant: "destructive",
                title: "Error",
                description: "Failed to start training"
            })
        } finally {
            setIsTraining(false)
        }
    }

    const startCodeGeneration = async () => {
        if (!selectedFile) {
            toast({
                variant: "destructive",
                title: "Error",
                description: "Please select a model file for code generation"
            })
            return
        }

        setIsGenerating(true)
        setGenerationOutput([])
        setGenerationError(null)

        try {
            const response = await fetch('/api/rfc/codegen', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    file: selectedFile.name
                })
            })

            const data = await response.json()

            if (response.ok) {
                setGenerationOutput(data.output || [])
                // Refresh code files after generation
                const codeResponse = await fetch('/api/rfc/code')
                if (codeResponse.ok) {
                    const codeData = await codeResponse.json()
                    setCodeFiles(codeData.files)
                }
                toast({
                    title: "Success",
                    description: "Code generation completed successfully"
                })
            } else {
                setGenerationError(data.error || 'Code generation failed')
                toast({
                    variant: "destructive",
                    title: "Error",
                    description: data.error || 'Code generation failed'
                })
            }
        } catch (error) {
            console.error('Error during code generation:', error)
            setGenerationError('Failed to start code generation')
            toast({
                variant: "destructive",
                title: "Error",
                description: "Failed to start code generation"
            })
        } finally {
            setIsGenerating(false)
        }
    }

    const handleCopy = (code: string) => {
        navigator.clipboard.writeText(code)
        setCopied(true)
        setTimeout(() => setCopied(false), 2000)
    }

    const getTruncatedCode = (code: string) => {
        const lines = code.split('\n')
        if (lines.length <= MAX_PREVIEW_LINES || isExpanded) {
            return code
        }
        return lines.slice(0, MAX_PREVIEW_LINES).join('\n') + '\n// ... truncated ...'
    }

    return (
        <div className="flex-1 space-y-6 p-4 md:p-8 pt-6">
            <div className="flex items-center justify-between">
                <div>
                    <h2 className="text-3xl font-bold tracking-tight">Random Forest Classifier</h2>
                    <p className="text-muted-foreground">
                        Train Random Forest models on CodeBERT processed data for final classification
                    </p>
                </div>
            </div>

            <Tabs defaultValue="train" className="space-y-4">
                <TabsList>
                    <TabsTrigger value="train">Training</TabsTrigger>
                    <TabsTrigger value="models">Models</TabsTrigger>
                    <TabsTrigger value="codegen">Code Generation</TabsTrigger>
                </TabsList>

                <TabsContent value="train" className="space-y-4">
                    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-7">
                        <Card className="lg:col-span-4">
                            <CardHeader>
                                <CardTitle>Input Files</CardTitle>
                                <CardDescription>
                                    Select a CodeBERT output file to train the Random Forest model
                                </CardDescription>
                            </CardHeader>
                            <CardContent>
                                <FileList
                                    files={csvFiles}
                                    selectedFile={selectedFile}
                                    onSelect={setSelectedFile}
                                />
                            </CardContent>
                        </Card>

                        <Card className="lg:col-span-3">
                            <CardHeader>
                                <CardTitle>Training Progress</CardTitle>
                                <CardDescription>
                                    Monitor the training process and view metrics
                                </CardDescription>
                            </CardHeader>
                            <CardContent className="space-y-4">
                                <Button
                                    onClick={startTraining}
                                    disabled={!selectedFile || isTraining}
                                    className="w-full"
                                >
                                    <TreesIcon className="mr-2 h-4 w-4" />
                                    {isTraining ? 'Training...' : 'Start Training'}
                                </Button>

                                <ProgressDisplay
                                    progress={progress}
                                    error={error}
                                    metrics={metrics}
                                />
                            </CardContent>
                        </Card>
                    </div>
                </TabsContent>

                <TabsContent value="models" className="space-y-4">
                    <Card>
                        <CardHeader>
                            <CardTitle>Trained Models</CardTitle>
                            <CardDescription>
                                View and manage your trained Random Forest models
                            </CardDescription>
                        </CardHeader>
                        <CardContent>
                            {modelFiles.length > 0 ? (
                                <div className="space-y-4">
                                    {modelFiles.map((file) => (
                                        <Alert key={file.path}>
                                            <TreesIcon className="h-4 w-4" />
                                            <AlertDescription className="flex items-center justify-between">
                                                <span>{file.name}</span>
                                            </AlertDescription>
                                        </Alert>
                                    ))}
                                </div>
                            ) : (
                                <Alert>
                                    <AlertCircle className="h-4 w-4" />
                                    <AlertDescription>
                                        No trained models found. Train a model first.
                                    </AlertDescription>
                                </Alert>
                            )}
                        </CardContent>
                    </Card>
                </TabsContent>

                <TabsContent value="codegen" className="space-y-4">
                    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-7">
                        <Card className="lg:col-span-4">
                            <CardHeader>
                                <CardTitle>Generated Code</CardTitle>
                                <CardDescription>
                                    View and copy the generated C code implementation
                                </CardDescription>
                            </CardHeader>
                            <CardContent className="space-y-4">
                                <div className="grid grid-cols-2 gap-2 mb-4">
                                    {codeFiles.map((file) => (
                                        <Button
                                            key={file.path}
                                            variant={selectedCodeFile?.name === file.name ? "default" : "outline"}
                                            className="w-full justify-start"
                                            onClick={() => setSelectedCodeFile(file)}
                                        >
                                            <Code2Icon className="mr-2 h-4 w-4" />
                                            {file.name}
                                        </Button>
                                    ))}
                                </div>
                                
                                {selectedCodeFile?.content ? (
                                    <div className="space-y-2">
                                        <CodeBlock>
                                            <CodeBlockGroup className="border-border border-b py-2 pr-2 pl-4">
                                                <div className="flex items-center gap-2">
                                                    <div className="bg-primary/10 text-primary rounded px-2 py-1 text-xs font-medium">
                                                        C
                                                    </div>
                                                    <span className="text-muted-foreground text-sm">
                                                        {selectedCodeFile.name}
                                                    </span>
                                                </div>
                                                <div className="flex gap-2">
                                                    <Button
                                                        variant="ghost"
                                                        size="icon"
                                                        className="h-8 w-8"
                                                        onClick={() => handleCopy(selectedCodeFile.content!)}
                                                    >
                                                        {copied ? (
                                                            <Check className="h-4 w-4 text-green-500" />
                                                        ) : (
                                                            <Copy className="h-4 w-4" />
                                                        )}
                                                    </Button>
                                                    <Button
                                                        variant="ghost"
                                                        size="icon"
                                                        className="h-8 w-8"
                                                        onClick={() => setIsExpanded(!isExpanded)}
                                                    >
                                                        {isExpanded ? (
                                                            <MinusSquare className="h-4 w-4" />
                                                        ) : (
                                                            <Expand className="h-4 w-4" />
                                                        )}
                                                    </Button>
                                                </div>
                                            </CodeBlockGroup>
                                            <CodeBlockCode 
                                                code={getTruncatedCode(selectedCodeFile.content)} 
                                                language="c" 
                                            />
                                        </CodeBlock>
                                        {!isExpanded && selectedCodeFile.content.split('\n').length > MAX_PREVIEW_LINES && (
                                            <Button
                                                variant="outline"
                                                className="w-full"
                                                onClick={() => setIsExpanded(true)}
                                            >
                                                <Expand className="mr-2 h-4 w-4" />
                                                Show Full Code ({selectedCodeFile.content.split('\n').length} lines)
                                            </Button>
                                        )}
                                    </div>
                                ) : (
                                    <Alert>
                                        <AlertCircle className="h-4 w-4" />
                                        <AlertDescription>
                                            {codeFiles.length === 0 
                                                ? "No generated code files found. Generate code first."
                                                : "Select a file to view its content."}
                                        </AlertDescription>
                                    </Alert>
                                )}
                            </CardContent>
                        </Card>

                        <Card className="lg:col-span-3">
                            <CardHeader>
                                <CardTitle>Generation Progress</CardTitle>
                                <CardDescription>
                                    Monitor the code generation process
                                </CardDescription>
                            </CardHeader>
                            <CardContent className="space-y-4">
                                <Button
                                    onClick={startCodeGeneration}
                                    disabled={isGenerating}
                                    className="w-full"
                                >
                                    <Code2Icon className="mr-2 h-4 w-4" />
                                    {isGenerating ? 'Generating...' : 'Generate Code'}
                                </Button>

                                <ProgressDisplay
                                    progress={generationOutput}
                                    error={generationError}
                                    metrics={null}
                                />
                            </CardContent>
                        </Card>
                    </div>
                </TabsContent>
            </Tabs>
        </div>
    )
}
