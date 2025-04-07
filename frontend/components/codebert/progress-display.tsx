import { ScrollArea } from "@/components/ui/scroll-area"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { AlertCircle } from "lucide-react"

interface ProgressDisplayProps {
    error: string | null
    progress: string[]
    className?: string
}

export function ProgressDisplay({ error, progress, className }: ProgressDisplayProps) {
    return (
        <ScrollArea className={`h-[500px] pr-4 ${className}`}>
            {error ? (
                <Alert variant="destructive">
                    <AlertCircle className="h-4 w-4" />
                    <AlertDescription>
                        {error}
                    </AlertDescription>
                </Alert>
            ) : (
                <div className="space-y-2">
                    {progress.map((line, index) => {
                        // Determine line type based on prefix
                        const getLineStyle = (text: string) => {
                            if (text.startsWith('[+]')) return 'text-green-500'
                            if (text.startsWith('[!]')) return 'text-red-500'
                            if (text.startsWith('[*]')) return 'text-blue-500'
                            return 'text-muted-foreground'
                        }

                        return (
                            <p
                                key={index}
                                className={`text-sm font-mono ${getLineStyle(line)}`}
                            >
                                {line}
                            </p>
                        )
                    })}
                </div>
            )}
        </ScrollArea>
    )
} 