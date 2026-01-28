'use client'

import { Alert, AlertDescription } from "@/components/ui/alert"
import { ScrollArea } from "@/components/ui/scroll-area"
import { AlertCircle, CheckCircle2, Info } from "lucide-react"

interface ProgressDisplayProps {
    progress: string[]
    metrics: {
        service_accuracy?: number
        activity_accuracy?: number
        unique_services?: number
        unique_activities?: number
    } | null
    error?: string | null
    className?: string
}

export function ProgressDisplay({ progress, metrics, error, className }: ProgressDisplayProps) {
    return (
        <div className={className}>
            {error && (
                <Alert variant="destructive" className="mb-4">
                    <AlertCircle className="h-4 w-4" />
                    <AlertDescription>{error}</AlertDescription>
                </Alert>
            )}

            {metrics && (
                <Alert className="mb-4 border-emerald-500/50 bg-emerald-500/10 text-emerald-500">
                    <CheckCircle2 className="h-4 w-4" />
                    <AlertDescription className="space-y-2">
                        <p>Training completed successfully!</p>
                        <div className="mt-2 grid gap-2 text-sm">
                            <div>Service Classification Accuracy: {(metrics.service_accuracy! * 100).toFixed(2)}%</div>
                            <div>Activity Classification Accuracy: {(metrics.activity_accuracy! * 100).toFixed(2)}%</div>
                            <div>Unique Services: {metrics.unique_services}</div>
                            <div>Unique Activities: {metrics.unique_activities}</div>
                        </div>
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
        </div>
    )
} 