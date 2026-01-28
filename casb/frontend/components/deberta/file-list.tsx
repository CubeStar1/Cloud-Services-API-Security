import { ScrollArea } from "@/components/ui/scroll-area"
import { Badge } from "@/components/ui/badge"
import { FileText, FileCheck } from "lucide-react"

interface FileInfo {
    name: string
    path: string
    timestamp: number
}

interface FileListProps {
    files: FileInfo[]
    type: 'input' | 'prediction'
    className?: string
}

export function FileList({ files, type, className }: FileListProps) {
    return (
        <ScrollArea className={`h-[300px] pr-4 ${className}`}>
            {files.length === 0 ? (
                <p className="text-sm text-muted-foreground">
                    No {type === 'input' ? 'CSV' : 'prediction'} files found
                </p>
            ) : (
                <div className="space-y-2">
                    {files.map((file) => (
                        <div
                            key={file.path}
                            className="flex items-center justify-between p-2 rounded-lg border"
                        >
                            <div className="flex items-center space-x-2">
                                {type === 'input' ? (
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
} 