'use client'

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { Badge } from '@/components/ui/badge'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Bot, ShieldAlert, CheckCircle } from 'lucide-react'
import { Progress } from "@/components/ui/progress"

interface ClassificationResult {
    id: string;
    timestamp: string;
    requestSnippet: string; // e.g., "POST /api/users"
    predictedService: string;
    predictedActivity: string;
    confidence: number; // 0 to 1
    isAnomaly: boolean;
}

const getConfidenceColor = (confidence: number): string => {
    if (confidence > 0.8) return "text-green-600";
    if (confidence > 0.6) return "text-yellow-600";
    return "text-red-600";
}

interface LiveClassificationResultsProps {
    results: ClassificationResult[];
}

export function LiveClassificationResults({ results }: LiveClassificationResultsProps) {
    return (
        <Card className="h-[400px] flex flex-col">
            <CardHeader>
                <CardTitle className="flex items-center gap-2">
                    <Bot className="h-5 w-5" /> Live Classification
                </CardTitle>
                <CardDescription>Real-time predictions from the active model.</CardDescription>
            </CardHeader>
            <CardContent className="flex-grow overflow-hidden">
                <ScrollArea className="h-full w-full">
                    <div className="overflow-x-auto">
                        <Table className="min-w-full">
                            <TableHeader>
                                <TableRow>
                                    <TableHead className="w-[100px]">Timestamp</TableHead>
                                    <TableHead>Request</TableHead>
                                    <TableHead>Service</TableHead>
                                    <TableHead>Activity</TableHead>
                                    <TableHead className="w-[120px]">Confidence</TableHead>
                                    {/* <TableHead className="w-[80px] text-right">Status</TableHead> */}
                                </TableRow>
                            </TableHeader>
                            <TableBody>
                                {results.map((result) => (
                                    <TableRow key={result.id}>
                                        <TableCell className="font-mono text-xs">{result.timestamp}</TableCell>
                                        <TableCell className="font-mono text-xs">{result.requestSnippet}</TableCell>
                                        <TableCell className="text-xs font-medium">{result.predictedService}</TableCell>
                                        <TableCell className="text-xs">{result.predictedActivity}</TableCell>
                                        <TableCell>
                                            <div className="flex items-center gap-2">
                                                <Progress value={result.confidence * 100} className="h-2 w-[60px]" />
                                                <span className={`text-xs font-semibold ${getConfidenceColor(result.confidence)}`}>
                                                    {(result.confidence * 100).toFixed(0)}%
                                                </span>
                                            </div>
                                        </TableCell>
                                        {/* <TableCell className="text-right">
                                            {result.isAnomaly ? (
                                                <Badge variant="destructive" className="text-xs">
                                                    <ShieldAlert className="h-3 w-3 mr-1" /> Anomaly
                                                </Badge>
                                            ) : (
                                                <Badge variant="outline" className="text-xs">
                                                    <CheckCircle className="h-3 w-3 mr-1" /> Normal
                                                </Badge>
                                            )}
                                        </TableCell> */}
                                    </TableRow>
                                ))}
                                {results.length === 0 && (
                                    <TableRow>
                                        <TableCell colSpan={6} className="h-24 text-center">
                                            Process stopped or no results yet.
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