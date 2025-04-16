'use client'

import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { Badge } from "@/components/ui/badge"
import { ScrollArea } from "@/components/ui/scroll-area"

interface LogEntry {
    type: 'request' | 'response'
    url: string
    method: string
    headers_Host: string
    requestHeaders_Origin?: string
    requestHeaders_Content_Type?: string
    requestHeaders_Referer?: string
    requestHeaders_Accept?: string
    responseHeaders_Content_Type?: string
    body: any
}

interface TrafficTableProps {
    logs: LogEntry[]
}

export function TrafficTable({ logs }: TrafficTableProps) {
    // Helper function to clean and truncate text
    const formatText = (text: string | null | undefined, maxLength: number = 50) => {
        if (!text) return '-'
        const cleaned = text.trim().replace(/\s+/g, ' ')
        if (cleaned.length <= maxLength) return cleaned
        return `${cleaned.substring(0, maxLength)}...`
    }

    // Helper function to format JSON/string body
    const formatBody = (body: any) => {
        if (!body) return '-'
        try {
            const text = typeof body === 'string' ? body : JSON.stringify(body)
            return text.trim().replace(/\s+/g, ' ')
        } catch (error) {
            return 'Invalid body format'
        }
    }

    return (
        <div className="rounded-lg border">
            <ScrollArea className="h-[calc(100vh-300px)]">
                <div className="max-w-[calc(100vw-32px)]">
                    <Table>
                        <TableHeader>
                            <TableRow>
                                <TableHead className="w-[90px]">Type</TableHead>
                                <TableHead className="w-[80px]">Method</TableHead>
                                <TableHead className="w-[200px]">URL</TableHead>
                                <TableHead className="w-[120px]">Host</TableHead>
                                <TableHead className="w-[120px]">Origin</TableHead>
                                <TableHead className="w-[120px]">Content-Type</TableHead>
                                <TableHead className="w-[120px]">Referer</TableHead>
                                <TableHead className="w-[120px]">Accept</TableHead>
                                <TableHead className="w-[200px]">Body</TableHead>
                            </TableRow>
                        </TableHeader>
                        <TableBody>
                            {logs.map((log, index) => (
                                <TableRow key={index}>
                                    <TableCell className="px-2">
                                        <Badge variant={log.type === 'request' ? "outline" : "default"}>
                                            {log.type.toUpperCase()}
                                        </Badge>
                                    </TableCell>
                                    <TableCell className="px-2">
                                        <Badge variant="secondary">
                                            {log.method}
                                        </Badge>
                                    </TableCell>
                                    <TableCell className="font-mono text-xs truncate max-w-[200px]">
                                        <div className="hover:overflow-visible hover:whitespace-normal hover:break-all" title={log.url}>
                                            {formatText(log.url, 100)}
                                        </div>
                                    </TableCell>
                                    <TableCell className="text-xs truncate max-w-[120px]">
                                        <div className="hover:overflow-visible hover:whitespace-normal" title={log.headers_Host}>
                                            {formatText(log.headers_Host)}
                                        </div>
                                    </TableCell>
                                    <TableCell className="text-xs truncate max-w-[120px]">
                                        <div className="hover:overflow-visible hover:whitespace-normal" title={log.requestHeaders_Origin}>
                                            {formatText(log.requestHeaders_Origin)}
                                        </div>
                                    </TableCell>
                                    <TableCell className="text-xs truncate max-w-[120px]">
                                        <div className="hover:overflow-visible hover:whitespace-normal" title={log.requestHeaders_Content_Type || log.responseHeaders_Content_Type}>
                                            {formatText(log.requestHeaders_Content_Type || log.responseHeaders_Content_Type)}
                                        </div>
                                    </TableCell>
                                    <TableCell className="text-xs truncate max-w-[120px]">
                                        <div className="hover:overflow-visible hover:whitespace-normal" title={log.requestHeaders_Referer}>
                                            {formatText(log.requestHeaders_Referer)}
                                        </div>
                                    </TableCell>
                                    <TableCell className="text-xs truncate max-w-[120px]">
                                        <div className="hover:overflow-visible hover:whitespace-normal" title={log.requestHeaders_Accept}>
                                            {formatText(log.requestHeaders_Accept)}
                                        </div>
                                    </TableCell>
                                    <TableCell className="font-mono text-xs max-w-[200px]">
                                        <div className="truncate hover:overflow-visible hover:whitespace-normal hover:break-all" title={typeof log.body === 'string' ? log.body : JSON.stringify(log.body)}>
                                            {formatBody(log.body)}
                                        </div>
                                    </TableCell>
                                </TableRow>
                            ))}
                            {logs.length === 0 && (
                                <TableRow>
                                    <TableCell colSpan={9} className="text-center h-24 text-muted-foreground">
                                        No logs available
                                    </TableCell>
                                </TableRow>
                            )}
                        </TableBody>
                    </Table>
                </div>
            </ScrollArea>
        </div>
    )
} 