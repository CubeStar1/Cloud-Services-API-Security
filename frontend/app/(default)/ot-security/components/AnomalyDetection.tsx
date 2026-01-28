import { useState, useEffect } from "react";
import axios from "axios";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  AlertTriangle,
  CheckCircle,
  Loader2,
  FileSearch,
  RefreshCw,
  Wifi,
  WifiOff,
} from "lucide-react";
import { useToast } from "@/hooks/use-toast";

// API Base URL (proxied by Vite)
const API_BASE_URL = '/api/anomaly';

// --- TYPE DEFINITIONS ---
interface AnomalyResult {
  device_id: string;
  device_name: string;
  device_ip: string;
  status: "normal" | "anomaly";
  confidence: number;
  anomaly_type: string | null;
  severity: string;
  fused_score: number;
  logbert_score: number;
  ae_score: number;
  detection_source: string;
  logs: string[];
}

interface AnalysisResponse {
  results: AnomalyResult[];
  total_devices: number;
  anomalies_detected: number;
}

interface DetectorStatus {
  fusion_detector_available: boolean;
  logbert_available: boolean;
  autoencoder_available: boolean;
}

export const AnomalyDetection = () => {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState<AnomalyResult[]>([]);
  const [detectorStatus, setDetectorStatus] = useState<DetectorStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const { toast } = useToast();

  // Check detector status on mount
  useEffect(() => {
    checkDetectorStatus();
  }, []);

  const checkDetectorStatus = async () => {
    try {
      const response = await axios.get<DetectorStatus>(`${API_BASE_URL}/status`);
      setDetectorStatus(response.data);
      setError(null);
    } catch (err: any) {
      console.error("Failed to check detector status:", err);
      setDetectorStatus(null);
      setError("Backend not available");
    }
  };

  const handleAnalyze = async () => {
    setIsAnalyzing(true);
    setResults([]);
    setError(null);

    try {
      const response = await axios.post<AnalysisResponse>(`${API_BASE_URL}/analyze`, {
        devices: null // Use default devices from backend
      });

      setResults(response.data.results);
      
      const anomalyCount = response.data.anomalies_detected;
      toast({
        title: "Analysis Complete",
        description: `Analyzed ${response.data.total_devices} devices. ${anomalyCount > 0 ? `${anomalyCount} anomal${anomalyCount > 1 ? 'ies' : 'y'} detected!` : 'All systems normal.'}`,
        variant: anomalyCount > 0 ? "destructive" : "default",
      });
    } catch (err: any) {
      const errorMessage = err.response?.data?.detail || err.message || "Failed to analyze devices";
      setError(errorMessage);
      toast({
        title: "Analysis Failed",
        description: errorMessage,
        variant: "destructive",
      });
    } finally {
      setIsAnalyzing(false);
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity.toLowerCase()) {
      case 'critical': return 'bg-red-600 text-white';
      case 'high': return 'bg-orange-500 text-white';
      case 'medium': return 'bg-yellow-500 text-black';
      default: return 'bg-green-500 text-white';
    }
  };

  return (
    <Card>
      <CardHeader>
        <div className="flex justify-between items-center">
          <div>
            <CardTitle>ML-Based Anomaly Detection</CardTitle>
            <p className="text-sm text-muted-foreground pt-1">
              Analyze OT devices using LogBERT + Autoencoder fusion detection.
            </p>
            {/* Detector Status Indicator */}
            <div className="flex items-center gap-2 mt-2">
              {detectorStatus ? (
                <>
                  <Wifi className="h-4 w-4 text-green-500" />
                  <span className="text-xs text-green-600">Backend Connected</span>
                  {detectorStatus.logbert_available && (
                    <Badge variant="outline" className="text-xs">LogBERT ✓</Badge>
                  )}
                  {detectorStatus.autoencoder_available && (
                    <Badge variant="outline" className="text-xs">AE ✓</Badge>
                  )}
                </>
              ) : (
                <>
                  <WifiOff className="h-4 w-4 text-red-500" />
                  <span className="text-xs text-red-600">{error || "Checking backend..."}</span>
                  <Button variant="ghost" size="sm" onClick={checkDetectorStatus}>
                    <RefreshCw className="h-3 w-3" />
                  </Button>
                </>
              )}
            </div>
          </div>
          <Button onClick={handleAnalyze} disabled={isAnalyzing || !detectorStatus}>
            {isAnalyzing ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Analyzing...
              </>
            ) : (
              <>
                <FileSearch className="mr-2 h-4 w-4" />
                Analyze Devices
              </>
            )}
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        {error && !results.length && (
          <div className="p-4 mb-4 bg-red-50 border border-red-200 rounded-lg text-red-700">
            <p className="text-sm">{error}</p>
          </div>
        )}

        {results.length === 0 && !isAnalyzing && !error && (
          <div className="p-8 text-center text-muted-foreground border-2 border-dashed rounded-lg">
            <FileSearch className="h-12 w-12 mx-auto mb-4 opacity-50" />
            <p>Click "Analyze Devices" to run ML-based anomaly detection.</p>
            <p className="text-xs mt-2">Uses LogBERT for log analysis and Autoencoder for sensor data.</p>
          </div>
        )}

        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {results.map((result) => (
             <Card key={result.device_id} className="overflow-hidden">
               <CardHeader className="pb-3">
                 <div className="flex items-start justify-between">
                   <div>
                     <CardTitle className="text-base">{result.device_name}</CardTitle>
                     <p className="text-xs text-muted-foreground mt-1">
                       {result.device_ip} • ID: {result.device_id}
                     </p>
                   </div>
                   <Badge
                     variant={result.status === "anomaly" ? "destructive" : "default"}
                     className={
                       result.status === "anomaly"
                         ? "bg-red-100 text-red-800 border-red-300"
                         : "bg-green-100 text-green-800 border-green-300"
                     }
                   >
                     {result.status === "anomaly" ? <AlertTriangle className="mr-1 h-3 w-3" /> : <CheckCircle className="mr-1 h-3 w-3" />}
                     {result.status.toUpperCase()}
                   </Badge>
                 </div>
               </CardHeader>
               <CardContent className="space-y-3">
                 {/* Severity Badge */}
                 <div className="flex items-center gap-2">
                   <span className="text-sm text-muted-foreground">Severity:</span>
                   <Badge className={getSeverityColor(result.severity)}>
                     {result.severity.toUpperCase()}
                   </Badge>
                 </div>

                 {/* Scores */}
                 <div className="space-y-2">
                   <div className="flex justify-between text-sm">
                     <span className="text-muted-foreground">Fused Score</span>
                     <span className="font-mono font-semibold">{result.fused_score.toFixed(4)}</span>
                   </div>
                   <div className="w-full bg-secondary rounded-full h-2">
                     <div
                       className={`h-2 rounded-full ${result.status === "anomaly" ? "bg-destructive" : "bg-green-500"}`}
                       style={{ width: `${Math.min(result.fused_score * 50, 100)}%` }}
                     />
                   </div>
                   <div className="flex justify-between text-xs text-muted-foreground">
                     <span>LogBERT: {result.logbert_score.toFixed(4)}</span>
                     <span>AE: {result.ae_score.toFixed(4)}</span>
                   </div>
                 </div>

                 {/* Detection Source */}
                 {result.anomaly_type && (
                   <div className="text-sm">
                     <span className="text-muted-foreground">Source:</span>{" "}
                     <span className="text-destructive font-medium">{result.anomaly_type}</span>
                   </div>
                 )}

                 {/* Confidence */}
                 <div className="text-sm">
                   <span className="text-muted-foreground">Confidence:</span>{" "}
                   <span className="font-semibold">{result.confidence.toFixed(1)}%</span>
                 </div>

                 {/* Logs */}
                 <div className="space-y-1">
                   <p className="text-sm font-medium text-muted-foreground">Generated Logs:</p>
                   <div className="bg-muted/50 border rounded p-2 space-y-1 max-h-32 overflow-y-auto">
                     {result.logs.length > 0 ? result.logs.map((log, idx) => (
                       <p key={idx} className={`text-xs font-mono ${log.includes('CRITICAL') || log.includes('ALERT') || log.includes('Error') ? 'text-red-600' : 'text-muted-foreground'}`}>
                         {log}
                       </p>
                     )) : (
                       <p className="text-xs text-muted-foreground italic">No logs generated</p>
                     )}
                   </div>
                 </div>
               </CardContent>
             </Card>
           ))}
        </div>
      </CardContent>
    </Card>
  );
};