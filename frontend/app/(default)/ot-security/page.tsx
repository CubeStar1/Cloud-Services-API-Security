"use client";

import { useState, useCallback, useMemo } from "react";
// --- UI Imports ---
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Activity } from "lucide-react";
// --- Custom Component Imports ---
import { ScannerInput } from "./components/ScannerInput";
import { DeviceTable } from "./components/DeviceTable";
import { NetworkTopology } from "./components/NetworkTopology";
import { AnomalyDetection } from "./components/AnomalyDetection";
import { Summary } from "./components/Summary";

// --- Data Structure Definitions ---
import { ScanResult } from "./types";

function OTSecurityPage() {
  const [isScanning, setIsScanning] = useState(false);
  const [devices, setDevices] = useState<ScanResult[]>([]);
  const [lastScanTime, setLastScanTime] = useState<string>("");
  const [scanStatus, setScanStatus] = useState<string>("Ready to start network scan.");

  const handleScanStart = useCallback(() => {
    setIsScanning(true);
    setDevices([]);
    setScanStatus("Scan initiated...");
  }, []);

  const handleScanComplete = useCallback((results: ScanResult[], scanTime: string) => {
    setIsScanning(false);
    setLastScanTime(scanTime);
    setDevices(results);
    setScanStatus(`Scan completed. Found ${results.length} devices.`);
  }, []);

  const handleStatusUpdate = useCallback((status: string) => {
      setScanStatus(status);
  }, []);

  // --- DERIVED STATE & DATA TRANSFORMATION ---

  // ✅ FIX: Transform the API data into the format NetworkTopology expects.
  // We use useMemo to prevent this calculation from running on every render.
  const topologyDevices = useMemo(() => {
    return devices.map(device => {
      // Determine the primary protocol to display. Prioritize OT.
      const primaryProtocol = device.ot_services.length > 0
        ? device.ot_services[0][1] // Get the name of the first OT service
        : (device.it_services.length > 0 ? device.it_services[0][1] : 'N/A');

      return {
        id: device.mac || device.ip, // Use MAC or IP as a unique ID
        ip: device.ip,
        name: device.vendor || 'Unknown Device', // Use vendor as the display name
        protocol: primaryProtocol,
      };
    });
  }, [devices]); // This will only re-run when the 'devices' state changes.

  const anomaliesCount = devices.filter(d => d.risk !== 'Low').length;

  const uniqueProtocols = [...new Set(
    devices.flatMap(d => [
      ...d.ot_services.map(([_, protocol]) => protocol),
      ...d.it_services.map(([_, protocol]) => protocol)
    ])
  )];

  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto px-4 py-8 max-w-7xl">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-2">
            <div className="p-2 rounded-lg bg-primary/10">
              <Activity className="h-6 w-6 text-primary" />
            </div>
            <h1 className="text-3xl font-bold text-glow">OT Device Discovery</h1>
          </div>
          <p className="text-muted-foreground">
            Advanced network scanning and anomaly detection for operational technology devices
          </p>
          <p className="text-sm text-gray-500 mt-2">Current Activity: <strong>{scanStatus}</strong></p>
        </div>

        {/* Scanner Input */}
        <Card className="mb-6 shadow-lg">
          <CardHeader>
            <CardTitle>Network Scanner</CardTitle>
            <CardDescription>
              Enter target network in CIDR format (e.g., 192.168.1.0/24) to scan and discover OT devices
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ScannerInput
              isScanning={isScanning}
              onScanStart={handleScanStart}
              onScanComplete={handleScanComplete}
              onStatusUpdate={handleStatusUpdate}
            />
          </CardContent>
        </Card>

        {/* Main Content Tabs */}
        <Tabs defaultValue="devices" className="space-y-6">
          <TabsList className="grid w-full grid-cols-4 lg:w-auto lg:inline-grid">
            <TabsTrigger value="devices">Devices ({devices.length})</TabsTrigger>
            <TabsTrigger value="topology">Topology</TabsTrigger>
            <TabsTrigger value="anomaly">Anomaly Detection</TabsTrigger>
            <TabsTrigger value="summary">Summary</TabsTrigger>
          </TabsList>

          <TabsContent value="devices" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Discovered Devices</CardTitle>
                <CardDescription>
                  List of all OT devices found during network scan
                </CardDescription>
              </CardHeader>
              <CardContent>
                <DeviceTable devices={devices} />
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="topology" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Network Topology</CardTitle>
                <CardDescription>
                  Visual representation of discovered devices and their connections
                </CardDescription>
              </CardHeader>
              <CardContent className="p-0">
                {/* ✅ FIX: Pass the newly transformed 'topologyDevices' array */}
                <NetworkTopology devices={topologyDevices} />
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="anomaly" className="space-y-4">
            <AnomalyDetection />
          </TabsContent>

          <TabsContent value="summary" className="space-y-4">
            <Summary
              totalDevices={devices.length}
              anomaliesDetected={anomaliesCount}
              protocolsUsed={uniqueProtocols}
              lastScanTime={lastScanTime}
            />
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}

export default OTSecurityPage;
