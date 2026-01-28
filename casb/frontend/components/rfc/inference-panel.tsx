"use client";

import { useState, useEffect } from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { FileList, FileInfo } from "@/components/rfc/file-list";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { useToast } from "@/components/ui/use-toast";
import { ScrollArea } from "@/components/ui/scroll-area";
import { ResultsTable } from "@/components/rfc/results-table";
import { CsvDownloadButton } from "@/components/rfc/csv-download-button";
import { ApiOutput } from "./api-output";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { ChevronDown, Loader2 } from "lucide-react"

interface SingleInputState {
  headers_Host?: string;
  url?: string;
  method?: string;
  requestHeaders_Origin?: string;
  requestHeaders_Content_Type?: string;
  responseHeaders_Content_Type?: string;
  requestHeaders_Referer?: string;
  requestHeaders_Accept?: string;
}

export default function RfcInferencePanel() {
  const { toast } = useToast();
  const [mode, setMode] = useState<"single" | "file">("single");
  const [singleInput, setSingleInput] = useState<SingleInputState>({});
  const [testFiles, setTestFiles] = useState<FileInfo[]>([]);
  const [selectedFile, setSelectedFile] = useState<FileInfo | null>(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<Array<{ output?: string[]; success?: boolean; [key: string]: any }>>([]);
  const [engine, setEngine] = useState<"python" | "c">("python");

  // fetch test files once
  useEffect(() => {
    if (mode !== "file") return;
    const fetchFiles = async () => {
      try {
        const res = await fetch("/api/rfc?type=test"); // backend /files via existing RFC route
        if (res.ok) {
          const data = await res.json();
          setTestFiles(data.files);
        }
      } catch (e) {
        toast({ variant: "destructive", title: "Error", description: "Failed to fetch test files" });
      }
    };
    fetchFiles();
  }, [mode, toast]);

  const handlePredict = async () => {
    setLoading(true);
    setResults([]);
    try {
      let body: any;
      if (mode === "single") {
        body = singleInput;
      } else if (selectedFile) {
        body = { file: selectedFile.name };
      }
      const res = await fetch(`/api/rfc/inference?engine=${engine}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await res.json();
      if (res.ok && data.success) {
        const rawResults = data.results ? data.results : [data];
        const filtered = rawResults.filter((r: any) =>
          (r.service_prediction ?? r.service) !== "Unknown Service" &&
          (r.activity_prediction ?? r.activity) !== "Unknown Activity"
        );
        
        // Add output metadata to the last result for display
        if (filtered.length > 0) {
          filtered[filtered.length - 1] = {
            ...filtered[filtered.length - 1],
            output: data.output,
            success: true,
            time: data.time
          };
        } else {
          // If no valid predictions, still show output
          filtered.push({ output: data.output, success: true, time: data.time });
        }
        
        setResults(filtered);
      } else {
        setResults([{ output: data.output, success: false, error: data.error }]);
        toast({ variant: "destructive", title: "Error", description: data.error ?? "Inference failed" });
      }
    } catch (e: any) {
      toast({ variant: "destructive", title: "Error", description: e.message });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <Tabs defaultValue="single" value={mode} onValueChange={(v) => setMode(v as any)} className="flex-1">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-6">
          <TabsList>
            <TabsTrigger value="single">Single Prediction</TabsTrigger>
            <TabsTrigger value="file">File Batch</TabsTrigger>
          </TabsList>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline">
                Engine: {engine === 'python' ? 'Python' : 'C'}
                <ChevronDown className="ml-2 h-4 w-4" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuItem onClick={() => setEngine('python')}>
                Python
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => setEngine('c')}>
                C
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
        <TabsContent value="single" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle>Single Prediction</CardTitle>
                <CardDescription>Enter request features to predict service & activity.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {[
                  "headers_Host",
                  "url",
                  "method",
                  "requestHeaders_Origin",
                  "requestHeaders_Content_Type",
                  "responseHeaders_Content_Type",
                  "requestHeaders_Referer",
                  "requestHeaders_Accept",
                ].map((field) => (
                  <Input
                    key={field}
                    placeholder={field}
                    value={(singleInput as any)[field] || ""}
                    onChange={(e) => setSingleInput({ ...singleInput, [field]: e.target.value })}
                  />
                ))}
              </CardContent>
              <CardFooter>
                <Button onClick={handlePredict} disabled={loading} className="w-full">
                  {loading ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Predicting...
                    </>
                  ) : (
                    "Predict"
                  )}
                </Button>
              </CardFooter>
            </Card>
            
            <ApiOutput 
              output={results[results.length - 1]?.output} 
              success={results[results.length - 1]?.success} 
              time={results[results.length - 1]?.time}
            />
          </div>
          
          {results.length > 0 && results.some(r => r.service_prediction || r.service || r.activity_prediction || r.activity) && (
            <Card>
              <CardHeader>
                <div className="flex justify-between items-center">
                  <CardTitle>Prediction Results</CardTitle>
                  <CsvDownloadButton rows={results} />
                </div>
              </CardHeader>
              <CardContent>
                <ResultsTable rows={results} />
              </CardContent>
            </Card>
          )}
        </TabsContent>
        <TabsContent value="file" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle>Test CSV Files</CardTitle>
                <CardDescription>Select a CSV for batch inference.</CardDescription>
              </CardHeader>
              <CardContent>
                <FileList files={testFiles} selectedFile={selectedFile} onSelect={setSelectedFile} />
                
              </CardContent>
              <CardFooter>
              <Button 
                  onClick={handlePredict} 
                  disabled={!selectedFile || loading} 
                  className="w-full mt-4"
                >
                  {loading ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Predicting...
                    </>
                  ) : (
                    "Predict"
                  )}
                </Button>
              </CardFooter>
            </Card>
            
            <ApiOutput 
              output={results[results.length - 1]?.output} 
              success={results[results.length - 1]?.success} 
              time={results[results.length - 1]?.time}
            />
          </div>
          <Card>
            <CardHeader>
              <div className="flex justify-between items-center">
                <CardTitle>Results</CardTitle>
                {results.length > 0 && <CsvDownloadButton rows={results} />}
              </div>
            </CardHeader>
            <CardContent>
              {results.length > 0 ? (
                <div className="space-y-2">
                  <ResultsTable rows={results} />
                </div>
              ) : (
                <Alert>
                  <AlertDescription>
                    {loading ? (
                      <div className="flex items-center gap-2">
                        <Loader2 className="h-4 w-4 animate-spin" />
                        Processing...
                      </div>
                    ) : (
                      "No results yet. Select a file and click Predict to see results."
                    )}
                  </AlertDescription>
                </Alert>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
