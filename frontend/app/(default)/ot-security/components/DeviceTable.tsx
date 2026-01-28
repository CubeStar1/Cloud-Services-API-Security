import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Network } from "lucide-react";

// Updated interface to match your API response
import { ScanResult } from "../types";

interface DeviceTableProps {
  devices: ScanResult[];
  isScanning?: boolean; // Optional prop for loading state
}

export const DeviceTable = ({ devices }: DeviceTableProps) => {
  console.log('DeviceTable received devices:', devices);
  console.log('First device structure:', devices[0]);

  if (devices.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-16 text-muted-foreground">
        <Network className="h-16 w-16 mb-4 opacity-50" />
        <p className="text-lg">No devices discovered yet</p>
        <p className="text-sm">Start a network scan to discover OT devices</p>
      </div>
    );
  }

  return (
    <div className="rounded-lg border border-border overflow-hidden">
      <Table>
        <TableHeader>
          <TableRow className="bg-card/50">
            <TableHead className="font-semibold">IP Address</TableHead>
            <TableHead className="font-semibold">Device Name</TableHead>

            <TableHead className="font-semibold">MAC Address</TableHead>
            <TableHead className="font-semibold">OT/IT Protocol</TableHead>
            <TableHead className="font-semibold">Open Ports</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {devices.map((device, index) => (
            <TableRow key={device.ip || index} className="hover:bg-card/50 transition-colors">
              <TableCell className="font-mono text-primary">{device.ip}</TableCell>
              <TableCell className="font-medium">
                {device.vendor !== "Unknown" ? device.vendor : device.ip}
              </TableCell>

              <TableCell className="font-mono text-sm">{device.mac}</TableCell>
              <TableCell>
                <div className="flex gap-1 flex-wrap">
                  {/* Display OT protocols with red styling */}
                  {device.ot_services && Array.isArray(device.ot_services) && device.ot_services.map(([port, protocol]) => (
                    <Badge
                      key={`ot-${port}`}
                      variant="outline"
                      className="border-red-500 text-red-600 bg-red-50"
                    >
                      {protocol}
                    </Badge>
                  ))}
                  {/* Display IT protocols with blue styling */}
                  {device.it_services && Array.isArray(device.it_services) && device.it_services.map(([port, protocol]) => (
                    <Badge
                      key={`it-${port}`}
                      variant="outline"
                      className="border-blue-500 text-blue-600 bg-blue-50"
                    >
                      {protocol}
                    </Badge>
                  ))}
                </div>
              </TableCell>
              <TableCell>
                <div className="flex gap-1 flex-wrap">
                  {device.open_ports && Array.isArray(device.open_ports) && device.open_ports.map((port) => (
                    <Badge key={port} variant="secondary" className="font-mono text-xs">
                      {port}
                    </Badge>
                  ))}
                </div>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}