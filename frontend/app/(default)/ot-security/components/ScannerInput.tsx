import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import { Loader2, Search } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useToast } from "@/hooks/use-toast";

// Use the relative path handled by Vite proxy
const API_BASE_URL = '/api/scan';

// Define a type for the data structure coming back from FastAPI
import { ScanResult } from "../types";

interface ScannerInputProps {
    isScanning: boolean;
    // New props for handling scan state and completion
    onScanStart: () => void;
    onScanComplete: (results: ScanResult[], scanTime: string) => void;
    onStatusUpdate: (status: string) => void;
}

// ⚠️ We use a named export as per your original file structure
export const ScannerInput = ({ isScanning, onScanStart, onScanComplete, onStatusUpdate }: ScannerInputProps) => {
    const [subnet, setSubnet] = useState("192.168.1.0/24"); // Use subnet for scan
    const [taskId, setTaskId] = useState<string | null>(null);
    const [pollIntervalId, setPollIntervalId] = useState<NodeJS.Timeout | null>(null);
    const { toast } = useToast();

    const handleScanStart = async () => {
        if (pollIntervalId) clearInterval(pollIntervalId);

        if (!subnet.trim()) {
            toast({
                title: "Error",
                description: "Please enter a valid network subnet (e.g., 192.168.1.0/24)",
                variant: "destructive",
            });
            return;
        }

        onScanStart(); // Notify Index.tsx to set isScanning=true
        onStatusUpdate("Starting scan request...");

        try {
            // 1. START SCAN REQUEST
            const response = await axios.post(`${API_BASE_URL}/start`, {
                subnet: subnet,
                scan_type: "2", // Full scan
                shodan_api_key: null
            });

            const newTaskId = response.data.task_id;
            setTaskId(newTaskId);

            onStatusUpdate(`Scan started. Task ID: ${newTaskId}. Polling...`);

            // 2. Begin Polling
            const interval = setInterval(() => pollScanStatus(newTaskId), 3000);
            setPollIntervalId(interval);

        } catch (error: any) {
            const errorDetail = error.response?.data?.detail || error.message;
            onStatusUpdate(`Error: Failed to start scan. ${errorDetail}`);
            onScanStart(); // Call again to set isScanning=false through Index.tsx
            toast({ title: "API Error", description: `Could not start scan: ${errorDetail}`, variant: "destructive" });
        }
    };

    const pollScanStatus = useCallback(async (id: string) => {
        try {
            // 3. POLL STATUS
            const statusResponse = await axios.get(`${API_BASE_URL}/status/${id}`);
            const statusData = statusResponse.data;
            const elapsed = (new Date().getTime() / 1000 - statusData.timestamp).toFixed(1);

            if (statusData.status === 'completed') {
                // SUCCESS
                if (pollIntervalId) clearInterval(pollIntervalId);
                setPollIntervalId(null);

                onStatusUpdate(`Scan Completed in ${statusData.duration_seconds.toFixed(2)}s`);

                // Convert to ISO string for consistency
                const scanTime = new Date().toISOString();
                onScanComplete(statusData.results || [], scanTime);

                toast({ title: "Scan Complete", description: `Found ${statusData.results.length} hosts.`, variant: "default" });

            } else if (statusData.status === 'running') {
                // RUNNING
                onStatusUpdate(`Scanning... Time elapsed: ${elapsed}s. Waiting for results.`);
            }

        } catch (error: any) {
            // FAILURE
            if (pollIntervalId) clearInterval(pollIntervalId);
            setPollIntervalId(null);

            const errorDetail = error.response?.data?.detail || error.message;
            onStatusUpdate(`Scan Failed: ${errorDetail}`);
            onScanComplete([], ""); // Pass empty array to stop loading state
            console.error("Polling error:", error);
        }
    }, [pollIntervalId, onScanComplete, onStatusUpdate, toast]);

    // Cleanup interval on component unmount
    useEffect(() => {
        return () => {
            if (pollIntervalId) clearInterval(pollIntervalId);
        };
    }, [pollIntervalId]);

    return (
        <div className="flex gap-3 items-end">
            <div className="flex-1">
                <label htmlFor="ip-input" className="text-sm font-medium text-muted-foreground block mb-2">
                    Target Network Subnet (CIDR)
                </label>
                <Input
                    id="ip-input"
                    type="text"
                    placeholder="192.168.1.0/24"
                    value={subnet}
                    onChange={(e) => setSubnet(e.target.value)}
                    disabled={isScanning}
                    className="font-mono bg-card border-border focus:border-primary transition-colors"
                />
            </div>
            <Button
                onClick={handleScanStart} // 👈 Call the new API function
                disabled={isScanning}
                className="bg-primary hover:bg-primary/90 text-primary-foreground glow-primary"
            >
                {isScanning ? (
                    <>
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        Scanning...
                    </>
                ) : (
                    <>
                        <Search className="mr-2 h-4 w-4" />
                        Start Scan
                    </>
                )}
            </Button>
        </div>
    );
};
