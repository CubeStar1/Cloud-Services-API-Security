"use client";

import { useState, useEffect, useCallback, useMemo } from "react";
import { Toaster, toast } from "sonner";
import { DashboardStats } from "@/components/dashboard/dashboard-stats";
import { LiveDataFeed } from "@/components/dashboard/live-data-feed";
import { LiveClassificationResults } from "@/components/dashboard/live-classification-results";
import { ServiceFrequencyChart } from "@/components/dashboard/service-frequency-chart";
import { ActivityFrequencyChart } from "@/components/dashboard/activity-frequency-chart";

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

interface ClassificationResult {
  id: string;
  timestamp: string;
  requestSnippet: string;
  predictedService: string;
  predictedActivity: string;
  confidence: number;
  isAnomaly: boolean;
}

const MAX_ITEMS = 1000;

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

export default function DashboardPage() {
  const [isRunning, setIsRunning] = useState(false);
  const [liveLogs, setLiveLogs] = useState<LogEntry[]>([]);
  const [classificationResults, setClassificationResults] = useState<ClassificationResult[]>([]);
  const [eventSource, setEventSource] = useState<EventSource | null>(null);
  const [engine, setEngine] = useState("python");
  const [avgLatency, setAvgLatency] = useState(0);
  const [totalLatency, setTotalLatency] = useState(0);
  const [processedCount, setProcessedCount] = useState(0);

  // Check initial status
  useEffect(() => {
    fetch(`${BACKEND_URL}/anyproxy/status`)
      .then((res) => res.json())
      .then((data) => {
        if (data.status === "success" && data.isRunning) {
          setIsRunning(true);
        }
      })
      .catch((err) => console.error("Failed to check status", err));
  }, []);

  const handleInference = async (logData: any) => {
    try {
      const input = {
        headers_Host: logData.headers_Host || "",
        url: logData.url || "",
        method: logData.method || "",
        requestHeaders_Origin: logData.requestHeaders_Origin || "",
        requestHeaders_Content_Type: logData.requestHeaders_Content_Type || "",
        requestHeaders_Referer: logData.requestHeaders_Referer || "",
        requestHeaders_Accept: logData.requestHeaders_Accept || "",
        responseHeaders_Content_Type: "", 
      };

      const startTime = performance.now();
      const res = await fetch(`/api/rfc/inference?engine=${engine}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(input),
      });

      const data = await res.json();
      
      if (res.ok && data.success) {
        const resultItem = data.results ? data.results[0] : data;
        if (!resultItem) return;

        // Use backend inference time if available, else fall back to RTT
        const actualLatency = resultItem.inference_time ?? (performance.now() - startTime);

        setTotalLatency((prev) => prev + actualLatency);
        setProcessedCount((prev) => prev + 1);
        setAvgLatency((total) => (totalLatency + actualLatency) / (processedCount + 1));


        const predService = resultItem.service_prediction || resultItem.service || "Unknown Service";
        const predActivity = resultItem.activity_prediction || resultItem.activity || "Unknown Activity";
        let confidence = resultItem.service_confidence ?? 0.0;
        const isAnomaly = predService === "Unknown Service" || predActivity === "Unknown Activity" || confidence < 0.5;

        const newResult: ClassificationResult = {
          id: crypto.randomUUID(),
          timestamp: new Date().toLocaleTimeString(),
          requestSnippet: `${input.method} ${new URL(input.url).pathname}`,
          predictedService: predService,
          predictedActivity: predActivity,
          confidence: confidence,
          isAnomaly: isAnomaly,
        };

        if (newResult.requestSnippet.length > 50) {
            newResult.requestSnippet = newResult.requestSnippet.substring(0, 47) + "...";
        }

        setClassificationResults((prev) => [newResult, ...prev].slice(0, MAX_ITEMS));
      }
    } catch (e) {
      console.error("Inference failed", e);
    }
  };

  // Manage EventSource
  useEffect(() => {
    if (isRunning) {
      if (eventSource) return; // Already connected

      const es = new EventSource(`${BACKEND_URL}/anyproxy/logs/stream`);
      
      es.onmessage = (event) => {
        try {
          const parsed = JSON.parse(event.data);
          if (parsed.type === "log" && parsed.data) {
            const rawLog = parsed.data;
            
            if (rawLog.type === "request") {
              const newLog: LogEntry = {
                id: crypto.randomUUID(),
                timestamp: new Date().toLocaleTimeString(),
                method: rawLog.method,
                host: rawLog.headers_Host || new URL(rawLog.url).hostname,
                url: rawLog.url,
                referer: rawLog.requestHeaders_Referer || itemOrNull(rawLog.headers, 'Referer'),
                accept: rawLog.requestHeaders_Accept || itemOrNull(rawLog.headers, 'Accept'),
                status: 0,
              };
              
              setLiveLogs((prev) => [newLog, ...prev].slice(0, MAX_ITEMS));
              handleInference(rawLog);
            }
          }
        } catch (e) {
            // ignore parsing errors
        }
      };

      es.onerror = () => {
        console.error("EventSource failed");
        es.close();
        setEventSource(null);
        // Optional: setIsRunning(false) if connection dies permanently
      };

      setEventSource(es);
    } else {
      if (eventSource) {
        eventSource.close();
        setEventSource(null);
      }
    }
    
    return () => {
      if (eventSource) eventSource.close();
    };
  }, [isRunning]);

  const toggleProcess = async () => {
    try {
      if (isRunning) {
        await fetch(`${BACKEND_URL}/anyproxy/stop`, { method: "POST" });
        setIsRunning(false);
        toast.info("Stopped live process");
      } else {
        // Start

        await fetch(`${BACKEND_URL}/anyproxy/start`, { 
            method: "POST", 
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ filename: "live_dashboard_session" }) 
        });
        setIsRunning(true);
        toast.success("Started live process");
        setLiveLogs([]);
        setClassificationResults([]);
        setAvgLatency(0);
        setTotalLatency(0);
        setProcessedCount(0);
      }
    } catch (e) {
      toast.error("Failed to toggle process");
      console.error(e);
    }
  };

  const totalRequests = liveLogs.length;
  const classifiedCount = classificationResults.length;
  const anomaliesCount = useMemo(() => {
    return classificationResults.filter((r) => r.isAnomaly).length;
  }, [classificationResults]);

  const serviceFrequency = useMemo(() => {
    const counts: { [key: string]: number } = {};
    classificationResults.forEach((r) => {
      counts[r.predictedService] = (counts[r.predictedService] || 0) + 1;
    });
    return Object.entries(counts)
      .map(([service, count]) => ({ service, count }))
      .sort((a, b) => b.count - a.count);
  }, [classificationResults]);

  const activityFrequency = useMemo(() => {
    const counts: { [key: string]: number } = {};
    classificationResults.forEach((r) => {
      counts[r.predictedActivity] = (counts[r.predictedActivity] || 0) + 1;
    });
    return Object.entries(counts)
      .map(([activity, count]) => ({ activity, count }))
      .sort((a, b) => b.count - a.count);
  }, [classificationResults]);

  const mostFrequentService = useMemo(() => 
    serviceFrequency.length > 0 ? serviceFrequency[0] : null
  , [serviceFrequency]);

  return (
    <div className="container mx-auto p-6">
      <div className="space-y-6">
        <h1 className="text-3xl font-bold">Live Dashboard (Real-time)</h1>

        <DashboardStats
          totalRequests={totalRequests}
          totalClassified={classifiedCount}
          anomaliesCount={anomaliesCount}
          mostFrequentService={mostFrequentService}
          isRunning={isRunning}
          onToggleRun={toggleProcess}
          engine={engine}
          onEngineChange={setEngine}
          avgLatency={avgLatency}
        />

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <ServiceFrequencyChart data={serviceFrequency} />
          <ActivityFrequencyChart data={activityFrequency} />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <LiveDataFeed logs={liveLogs} />
          <LiveClassificationResults results={classificationResults} />
        </div>
      </div>
      <Toaster richColors />
    </div>
  );
}

function itemOrNull(obj: any, key: string) {
    return null; 
}
