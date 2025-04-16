'use client'

import { useState, useEffect, useCallback, useMemo } from 'react'
import { Toaster, toast } from 'sonner'
import { DashboardStats } from '@/components/dashboard/dashboard-stats'
import { ModelSelectionPanel } from '@/components/dashboard/model-selection-panel'
import { ProcessControlPanel } from '@/components/dashboard/process-control-panel'
import { LiveDataFeed } from '@/components/dashboard/live-data-feed'
import { LiveClassificationResults } from '@/components/dashboard/live-classification-results'
import { ServiceFrequencyChart } from '@/components/dashboard/service-frequency-chart'
import { ActivityFrequencyChart } from '@/components/dashboard/activity-frequency-chart'
import { Play, Square } from 'lucide-react'

// Define interfaces for data types (can be moved to a types file)
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

// Mock data generation functions (moved from child components)
const generateMockLog = (): LogEntry => {
    const methods = ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'];
    const hosts = ['api.example.com', 'auth.example.com', 'cdn.example.net', 'internal.service.local', 'data.processing.svc', 'app.box.com', 'colab.research.google.com'];
    const paths = ['/users', '/products/123', '/orders?limit=10', '/auth/token', '/inventory/check', '/api/v2/items', '/api/content', '/files/abc'];
    const referers = ['https://app.example.com/', 'https://dashboard.example.com/analytics', null, 'https://shop.example.com/cart', 'https://colab.research.google.com/'];
    const accepts = ['application/json', 'text/html,application/xhtml+xml', '*/*', 'image/png', null, 'text/css'];
    const statuses = [200, 201, 204, 400, 401, 403, 404, 500, 503];
    const host = hosts[Math.floor(Math.random() * hosts.length)];
    return {
        id: crypto.randomUUID(), timestamp: new Date().toLocaleTimeString(), method: methods[Math.floor(Math.random() * methods.length)],
        host: host, url: `https://${host}${paths[Math.floor(Math.random() * paths.length)]}`, referer: referers[Math.floor(Math.random() * referers.length)],
        accept: accepts[Math.floor(Math.random() * accepts.length)], status: statuses[Math.floor(Math.random() * statuses.length)],
    };
};

const generateMockResult = (): ClassificationResult => {
    const services = ['Box', 'Box', 'Box', 'Unknown Service', 'Google', 'BugZilla', 'Adobe'];
    const activities = ['Download', 'Upload', 'Login', 'Download', 'Unknown Activity', 'Sync', 'Preview', 'Read', 'Write'];
    const methods = ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'];
    const paths = ['/users', '/files/abc', '/api/2.0/folders/0', '/login', '/app', '/api/content', '/data'];
    const confidence = Math.random();
    const predictedService = services[Math.floor(Math.random() * services.length)];
    const predictedActivity = activities[Math.floor(Math.random() * activities.length)];
    const effectiveConfidence = (predictedService === 'Unknown Service' || predictedActivity === 'Unknown Activity') ? Math.min(confidence, Math.random() * 0.7) : confidence;
    const isAnomaly = effectiveConfidence < 0.5 || (predictedService === 'Unknown Service' && Math.random() > 0.6);
    return {
        id: crypto.randomUUID(), timestamp: new Date().toLocaleTimeString(), requestSnippet: `${methods[Math.floor(Math.random() * methods.length)]} /api${paths[Math.floor(Math.random() * paths.length)]}`,
        predictedService: predictedService, predictedActivity: predictedActivity, confidence: parseFloat(effectiveConfidence.toFixed(2)), isAnomaly: isAnomaly,
    };
};

const MAX_ITEMS = 50; // Max items to keep in live lists

export default function DashboardPage() {
    const [isRunning, setIsRunning] = useState(false);
    const [selectedModel, setSelectedModel] = useState('random-forest');
    const [liveLogs, setLiveLogs] = useState<LogEntry[]>([]);
    const [classificationResults, setClassificationResults] = useState<ClassificationResult[]>([]);

    // Use useEffect to manage the intervals based on isRunning state
    useEffect(() => {
        let logIntervalId: NodeJS.Timeout | null = null;
        let resultIntervalId: NodeJS.Timeout | null = null;

        if (isRunning) {
            logIntervalId = setInterval(() => {
                setLiveLogs(prevLogs => [generateMockLog(), ...prevLogs].slice(0, MAX_ITEMS));
            }, 1500); // Log interval

            resultIntervalId = setInterval(() => {
                setClassificationResults(prevResults => [generateMockResult(), ...prevResults].slice(0, MAX_ITEMS));
            }, 1800); // Classification interval (slightly slower)
        } else {
            // Clear intervals if running is stopped
            if (logIntervalId) clearInterval(logIntervalId);
            if (resultIntervalId) clearInterval(resultIntervalId);
             // Optionally clear data on stop:
             // setLiveLogs([]);
             // setClassificationResults([]);
        }

        // Cleanup function to clear intervals when component unmounts or isRunning changes
        return () => {
            if (logIntervalId) clearInterval(logIntervalId);
            if (resultIntervalId) clearInterval(resultIntervalId);
        };
    }, [isRunning]); // Re-run effect when isRunning changes

    // Calculate derived stats for DashboardStats
    const totalRequests = liveLogs.length;
    const anomaliesCount = useMemo(() => {
        return classificationResults.filter(result => result.isAnomaly).length;
    }, [classificationResults]);

    // Calculate frequencies for charts using useMemo for optimization
    const serviceFrequency = useMemo(() => {
        const counts: { [key: string]: number } = {};
        classificationResults.forEach(result => {
            counts[result.predictedService] = (counts[result.predictedService] || 0) + 1;
        });
        return Object.entries(counts).map(([service, count]) => ({ service, count })).sort((a, b) => b.count - a.count); // Sort descending
    }, [classificationResults]);

    const activityFrequency = useMemo(() => {
        const counts: { [key: string]: number } = {};
        classificationResults.forEach(result => {
            counts[result.predictedActivity] = (counts[result.predictedActivity] || 0) + 1;
        });
        return Object.entries(counts).map(([activity, count]) => ({ activity, count })).sort((a, b) => b.count - a.count); // Sort descending
    }, [classificationResults]);

    // Handlers
    const handleToggleRun = useCallback(() => {
        const newState = !isRunning;
        setIsRunning(newState);
        toast(newState ? 'Started live process' : 'Stopped live process', {
          description: `Using model: ${selectedModel}`,
          icon: newState ? <Play className="h-4 w-4" /> : <Square className="h-4 w-4" />
        })
    }, [isRunning, selectedModel]);

    const handleModelChange = useCallback((value: string) => {
        setSelectedModel(value);
        if(isRunning) {
            toast('Model selection changed', { description: `Will use ${value} for new classifications.` })
        }
    }, [isRunning]);

    return (
        <div className="container mx-auto p-6">
            <div className="space-y-6">
                <h1 className="text-3xl font-bold">Live Dashboard</h1>
                
                {/* Pass calculated stats and state to DashboardStats */}
                <DashboardStats 
                    isRunning={isRunning}
                    selectedModel={selectedModel}
                    totalRequests={totalRequests}
                    anomaliesCount={anomaliesCount}
                />

                {/* Control Panels Side-by-Side */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <ModelSelectionPanel 
                        selectedModel={selectedModel} 
                        onModelChange={handleModelChange} 
                        disabled={isRunning} // Disable model change while running
                    />
                    <ProcessControlPanel 
                        isRunning={isRunning} 
                        onToggleRun={handleToggleRun} 
                    />
                </div>

                {/* Frequency Charts Side-by-Side */}
                 <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <ServiceFrequencyChart data={serviceFrequency} />
                    <ActivityFrequencyChart data={activityFrequency} />
                 </div>

                {/* Live Feeds (now just display components) Side-by-Side */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <LiveDataFeed logs={liveLogs} />
                    <LiveClassificationResults results={classificationResults} />
                </div>
            </div>
            <Toaster richColors /> 
        </div>
    )
}
