'use client'

import { ScrollArea } from "@/components/ui/scroll-area"
import { FileIcon } from "lucide-react"
import { formatDistanceToNow } from "date-fns"

export interface FileInfo {
    name: string
    path: string
    timestamp: number
}

interface FileListProps {
    files: FileInfo[]
    onSelect?: (file: FileInfo) => void
    selectedFile?: FileInfo | null
}

export function FileList({ files, onSelect, selectedFile }: FileListProps) {
    if (!files.length) {
        return (
            <div className="flex h-[450px] items-center justify-center rounded-md border border-dashed">
                <div className="mx-auto flex max-w-[420px] flex-col items-center justify-center text-center">
                    <FileIcon className="h-10 w-10 text-muted-foreground" />
                    <h3 className="mt-4 text-lg font-semibold">No files found</h3>
                    <p className="mb-4 mt-2 text-sm text-muted-foreground">
                        No CodeBERT output files found. Process your data through CodeBERT first.
                    </p>
                </div>
            </div>
        )
    }

    return (
        <ScrollArea className="h-[450px] rounded-md border">
            <div className="p-4">
                <div className="space-y-4">
                    {files.map((file) => (
                        <button
                            key={file.path}
                            onClick={() => onSelect?.(file)}
                            className={`w-full rounded-lg border p-4 text-left transition-colors hover:bg-muted/50 ${
                                selectedFile?.path === file.path ? 'bg-muted' : ''
                            }`}
                        >
                            <div className="flex items-center gap-4">
                                <FileIcon className="h-8 w-8 text-blue-500" />
                                <div className="flex-1 space-y-1">
                                    <p className="text-sm font-medium leading-none">{file.name}</p>
                                    <p className="text-sm text-muted-foreground">
                                        {formatDistanceToNow(file.timestamp, { addSuffix: true })}
                                    </p>
                                </div>
                            </div>
                        </button>
                    ))}
                </div>
            </div>
        </ScrollArea>
    )
} 