"use client";

import { useState, useEffect } from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { FileList, FileInfo } from "@/components/rfc/file-list";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { useToast } from "@/components/ui/use-toast";
import { ScrollArea } from "@/components/ui/scroll-area";
import { ResultsTable } from "@/components/rfc/results-table"
import { CsvDownloadButton } from "@/components/rfc/csv-download-button"

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
  const [results, setResults] = useState<any[]>([]);

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
      const res = await fetch("/api/rfc/inference", {
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
          setResults(filtered);
      } else {
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
      <Tabs defaultValue="single" value={mode} onValueChange={(v) => setMode(v as any)}>
        <TabsList>
          <TabsTrigger value="single">Single Prediction</TabsTrigger>
          <TabsTrigger value="file">File Batch</TabsTrigger>
        </TabsList>
        <TabsContent value="single">
          <Card>
            <CardHeader>
              <CardTitle>Single Prediction</CardTitle>
              <CardDescription>Enter request features to predict service & activity.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {Object.keys(singleInput).map((key) => key)}
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
              <Button onClick={handlePredict} disabled={loading} className="w-full">
                {loading ? "Predicting..." : "Predict"}
              </Button>
            </CardContent>
          </Card>
        </TabsContent>
        <TabsContent value="file">
          <div className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Test CSV Files</CardTitle>
                <CardDescription>Select a CSV for batch inference.</CardDescription>
              </CardHeader>
              <CardContent>
                <FileList files={testFiles} selectedFile={selectedFile} onSelect={setSelectedFile} />
                <Button onClick={handlePredict} disabled={!selectedFile || loading} className="w-full mt-4">
                  {loading ? "Predicting..." : "Predict"}
                </Button>
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle>Results</CardTitle>
              </CardHeader>
              <CardContent>
                {results.length ? (
                    <div className="space-y-2">
                      <div className="flex justify-end">
                        <CsvDownloadButton rows={results} />
                      </div>
                  <ResultsTable rows={results} />
                      </div>
                ) : (
                  <Alert>
                    <AlertDescription>No results yet.</AlertDescription>
                  </Alert>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
