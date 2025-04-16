'use client'

// Remove useState, useEffect imports if no longer needed directly
// import { useState, useEffect } from 'react' 
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Badge } from '@/components/ui/badge'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Rss } from 'lucide-react'

// Keep the interface
interface LogEntry {
    id: string;
    timestamp: string;
    method: string;
    host: string;
    url: string; 
    referer: string | null;
    accept: string | null;
    status: number;
}

// Keep the badge logic function
const getStatusBadgeVariant = (status: number): "default" | "secondary" | "destructive" | "outline" => {
    if (status >= 500) return "destructive";
    if (status >= 400) return "secondary";
    if (status >= 300) return "outline" 
    if (status >= 200) return "default";
    return "outline";
}

// Define props for the component
interface LiveDataFeedProps {
    logs: LogEntry[];
    // isRunning prop is useful if you want to show a visual indicator, but interval is controlled by parent
    // isRunning: boolean; 
}

// Make it a simpler display component
export function LiveDataFeed({ logs }: LiveDataFeedProps) {
    return (
        <Card className="h-[400px] flex flex-col">
            <CardHeader>
                <CardTitle className="flex items-center gap-2">
                    <Rss className="h-5 w-5" /> Live Data Feed
                </CardTitle>
                <CardDescription>Incoming network requests being captured.</CardDescription>
            </CardHeader>
            <CardContent className="flex-grow overflow-hidden">
                <ScrollArea className="h-full w-full"> 
                    <div className="overflow-x-auto">
                        <Table className="min-w-full">
                            <TableHeader>
                                <TableRow>
                                    <TableHead className="w-[100px]">Timestamp</TableHead>
                                    <TableHead className="w-[80px]">Method</TableHead>
                                    <TableHead className="w-[200px]">Host</TableHead>
                                    <TableHead className="w-[300px]">URL</TableHead>
                                    <TableHead className="w-[250px]">Referer</TableHead>
                                    <TableHead className="w-[200px]">Accept</TableHead>
                                    <TableHead className="w-[80px] text-right">Status</TableHead>
                                </TableRow>
                            </TableHeader>
                            <TableBody>
                                {logs.map((log) => (
                                    <TableRow key={log.id}>
                                        <TableCell className="font-mono text-xs">{log.timestamp}</TableCell>
                                        <TableCell>
                                            <Badge variant={log.method === 'GET' ? 'outline' : ['POST','PUT','DELETE'].includes(log.method) ? 'default' : 'secondary'} className="text-xs font-semibold">{log.method}</Badge>
                                        </TableCell>
                                        <TableCell className="font-mono text-xs truncate" title={log.host}>{log.host}</TableCell>
                                        <TableCell className="font-mono text-xs truncate" title={log.url}>{log.url}</TableCell>
                                        <TableCell className="font-mono text-xs truncate" title={log.referer ?? '-'}>{log.referer ?? '-'}</TableCell>
                                        <TableCell className="font-mono text-xs truncate" title={log.accept ?? '-'}>{log.accept ?? '-'}</TableCell>
                                        <TableCell className="text-right">
                                            <Badge variant={getStatusBadgeVariant(log.status)} className="text-xs">{log.status}</Badge>
                                        </TableCell>
                                    </TableRow>
                                ))}
                                {/* Add a message if logs are empty */}
                                {logs.length === 0 && (
                                    <TableRow>
                                        <TableCell colSpan={7} className="h-24 text-center">
                                            Process stopped or no data yet.
                                        </TableCell>
                                    </TableRow>
                                )}
                            </TableBody>
                        </Table>
                    </div>
                </ScrollArea>
            </CardContent>
        </Card>
    );
}

// Remove the mock data generation function from here, move it to parent page or a utility file
// export const generateMockLog = (): LogEntry => { ... } 