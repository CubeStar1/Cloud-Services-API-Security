'use client'

import { File, Folder, Tree, TreeViewElement } from "@/components/ui/file-tree"
import { useEffect, useState } from "react"
import { Card } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import { FileIcon, FolderIcon, Copy, Check } from "lucide-react"
import { CodeBlock, CodeBlockCode, CodeBlockGroup } from "@/components/ui/code-block"
import { Button } from "@/components/ui/button"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { AlertCircle } from "lucide-react"
import { CSVViewer } from "@/components/ui/csv-viewer"

interface FileContent {
    content: string;
    name: string;
    language?: string;
    type?: 'csv';
}

function renderFileTree(elements: TreeViewElement[], onSelect: (id: string) => void) {
    return elements.map((element) => {
        if (element.children) {
            return (
                <Folder 
                    key={element.id} 
                    value={element.id} 
                    element={element.name}
                >
                    {renderFileTree(element.children, onSelect)}
                </Folder>
            )
        }
        return (
            <div 
                key={element.id}
                className={`flex items-center gap-2 px-2 py-1 cursor-pointer hover:bg-accent rounded-sm`}
                onClick={() => {
                    console.log('File clicked:', element.id);
                    onSelect(element.id);
                }}
            >
                <FileIcon className="h-4 w-4" />
                <span>{element.name}</span>
            </div>
        )
    })
}

export function FileBrowser() {
    const [elements, setElements] = useState<TreeViewElement[]>([])
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState<string | null>(null)
    const [selectedFile, setSelectedFile] = useState<FileContent | null>(null)
    const [fileLoading, setFileLoading] = useState(false)
    const [fileError, setFileError] = useState<string | null>(null)
    const [copied, setCopied] = useState(false)

    useEffect(() => {
        const fetchFiles = async () => {
            try {
                const response = await fetch('/api/files')
                if (response.ok) {
                    const data = await response.json()
                    setElements(data.tree)
                } else {
                    setError('Failed to fetch files')
                }
            } catch (error) {
                setError('Error loading files')
                console.error('Error:', error)
            } finally {
                setLoading(false)
            }
        }

        fetchFiles()
    }, [])

    const handleFileSelect = async (id: string) => {
        console.log('File selected:', id);
        setFileLoading(true)
        setFileError(null)
        try {
            const response = await fetch(`/api/files?file=${encodeURIComponent(id)}`)
            console.log('API Response status:', response.status);
            if (response.ok) {
                const data = await response.json()
                console.log('File data received:', data);
                setSelectedFile(data)
            } else {
                const data = await response.json()
                console.log('API Error:', data);
                setFileError(data.error || 'Failed to load file')
            }
        } catch (error) {
            console.error('Error in handleFileSelect:', error);
            setFileError('Error loading file')
        } finally {
            setFileLoading(false)
        }
    }

    const handleCopy = (code: string) => {
        navigator.clipboard.writeText(code)
        setCopied(true)
        setTimeout(() => setCopied(false), 2000)
    }

    const renderFileContent = () => {
        if (fileLoading) {
            return (
                <div className="p-4 space-y-2">
                    <Skeleton className="h-4 w-full" />
                    <Skeleton className="h-4 w-[90%]" />
                    <Skeleton className="h-4 w-[80%]" />
                </div>
            )
        }

        if (fileError) {
            return (
                <Alert className="m-4">
                    <AlertCircle className="h-4 w-4" />
                    <AlertDescription>{fileError}</AlertDescription>
                </Alert>
            )
        }

        if (!selectedFile) {
            return (
                <div className="h-full flex items-center justify-center text-muted-foreground">
                    Select a file to view its contents
                </div>
            )
        }

        // If it's a CSV file
        if (selectedFile.type === 'csv') {
            return (
                <div className="h-full p-4">
                    <div className="flex items-center justify-between mb-4">
                        <h3 className="text-lg font-semibold">{selectedFile.name}</h3>
                        <Button
                            variant="ghost"
                            size="icon"
                            className="h-8 w-8"
                            onClick={() => handleCopy(selectedFile.content)}
                        >
                            {copied ? (
                                <Check className="h-4 w-4 text-green-500" />
                            ) : (
                                <Copy className="h-4 w-4" />
                            )}
                        </Button>
                    </div>
                    <CSVViewer 
                        data={selectedFile.content} 
                        className="h-[calc(100%-3rem)]"
                    />
                </div>
            )
        }

        // For other files
        return (
            <div className="h-full overflow-auto">
                <CodeBlock>
                    <CodeBlockGroup className="border-border border-b py-2 pr-2 pl-4">
                        <div className="flex items-center gap-2">
                            <div className="bg-primary/10 text-primary rounded px-2 py-1 text-xs font-medium">
                                {selectedFile.language || 'text'}
                            </div>
                            <span className="text-muted-foreground text-sm">
                                {selectedFile.name}
                            </span>
                        </div>
                        <Button
                            variant="ghost"
                            size="icon"
                            className="h-8 w-8"
                            onClick={() => handleCopy(selectedFile.content)}
                        >
                            {copied ? (
                                <Check className="h-4 w-4 text-green-500" />
                            ) : (
                                <Copy className="h-4 w-4" />
                            )}
                        </Button>
                    </CodeBlockGroup>
                    <CodeBlockCode 
                        code={selectedFile.content} 
                        language={selectedFile.language || 'text'} 
                    />
                </CodeBlock>
            </div>
        )
    }

    if (loading) {
        return (
            <Card className="p-4">
                <div className="space-y-2">
                    <Skeleton className="h-4 w-[200px]" />
                    <Skeleton className="h-4 w-[180px]" />
                    <Skeleton className="h-4 w-[160px]" />
                </div>
            </Card>
        )
    }

    if (error) {
        return (
            <Card className="p-4">
                <p className="text-destructive">Error: {error}</p>
            </Card>
        )
    }

    return (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Card className="relative h-[600px] overflow-hidden">
                <Tree
                    className="p-4"
                    elements={elements}
                    initialExpandedItems={elements.map(e => e.id)}
                >
                    {renderFileTree(elements, handleFileSelect)}
                </Tree>
            </Card>

            <Card className="relative h-[600px] overflow-hidden">
                {renderFileContent()}
            </Card>
        </div>
    )
} 