'use client'

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import { useState } from "react"
import { Binary, AlertCircle, CheckCircle2, Info } from "lucide-react"
import { useToast } from "@/components/ui/use-toast"

interface CodeGenProps {
    className?: string
}

export function CodeGeneration({ className }: CodeGenProps) {
    const [isGenerating, setIsGenerating] = useState(false)
    const [progress, setProgress] = useState<string[]>([])
    const [error, setError] = useState<string | null>(null)
    const [success, setSuccess] = useState(false)
    const { toast } = useToast()

    const startCodeGeneration = async () => {
        setIsGenerating(true)
        setProgress([])
        setError(null)
        setSuccess(false)

        try {
            const response = await fetch('/api/rfc/codegen', {
                method: 'POST'
            })

            const data = await response.json()

            if (response.ok) {
                setProgress(data.output || [])
                setSuccess(true)
                toast({
                    title: "Success",
                    description: "C code generation completed successfully"
                })
            } else {
                setError(data.error || 'Code generation failed')
                toast({
                    variant: "destructive",
                    title: "Error",
                    description: data.error || 'Code generation failed'
                })
            }
        } catch (error) {
            console.error('Error during code generation:', error)
            setError('Failed to start code generation')
            toast({
                variant: "destructive",
                title: "Error",
                description: "Failed to start code generation"
            })
        } finally {
            setIsGenerating(false)
        }
    }

    return (
        <div className={className}>
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-7">
                <Card className="lg:col-span-4">
                    <CardHeader>
                        <CardTitle>C Code Generation</CardTitle>
                        <CardDescription>
                            Generate optimized C code from trained Random Forest models
                        </CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                        <Alert>
                            <Info className="h-4 w-4" />
                            <AlertDescription>
                                This will generate C code that implements the trained Random Forest models
                                for efficient real-time classification.
                            </AlertDescription>
                        </Alert>

                        <Button
                            onClick={startCodeGeneration}
                            disabled={isGenerating}
                            className="w-full"
                        >
                            <Binary className="mr-2 h-4 w-4" />
                            {isGenerating ? 'Generating Code...' : 'Generate C Code'}
                        </Button>
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
                        {error && (
                            <Alert variant="destructive">
                                <AlertCircle className="h-4 w-4" />
                                <AlertDescription>{error}</AlertDescription>
                            </Alert>
                        )}

                        {success && (
                            <Alert className="border-emerald-500/50 bg-emerald-500/10 text-emerald-500">
                                <CheckCircle2 className="h-4 w-4" />
                                <AlertDescription>
                                    C code generated successfully! Check the output directory.
                                </AlertDescription>
                            </Alert>
                        )}

                        <ScrollArea className="h-[300px] rounded-md border">
                            <div className="p-4 font-mono text-sm">
                                {progress.map((line, i) => {
                                    if (line.toLowerCase().includes("error")) {
                                        return (
                                            <div key={i} className="flex items-center gap-2 text-red-500">
                                                <AlertCircle className="h-4 w-4" />
                                                <span>{line}</span>
                                            </div>
                                        )
                                    }
                                    if (line.toLowerCase().includes("success") || line.toLowerCase().includes("completed")) {
                                        return (
                                            <div key={i} className="flex items-center gap-2 text-emerald-500">
                                                <CheckCircle2 className="h-4 w-4" />
                                                <span>{line}</span>
                                            </div>
                                        )
                                    }
                                    return (
                                        <div key={i} className="flex items-center gap-2 text-blue-500">
                                            <Info className="h-4 w-4" />
                                            <span>{line}</span>
                                        </div>
                                    )
                                })}
                            </div>
                        </ScrollArea>
                    </CardContent>
                </Card>
            </div>
        </div>
    )
} 