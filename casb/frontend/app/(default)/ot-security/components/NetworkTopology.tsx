import { useEffect, useState, useCallback } from "react";
import ReactFlow, {
  Background,
  Controls,
  Node,
  Edge,
  MarkerType,
  useNodesState,
  useEdgesState,
} from "react-flow-renderer";
import { Card } from "@/components/ui/card";

interface NetworkTopologyProps {
  devices: Array<{
    id: string;
    ip: string;
    name: string;
    protocol: string;
  }>;
}

// Protocol color mapping
const getProtocolColor = (protocol: string) => {
  const colors: Record<string, string> = {
    modbus: "#3b82f6",
    bacnet: "#8b5cf6", 
    mqtt: "#22c55e",
    http: "#f59e0b",
    ssh: "#ef4444",
    unknown: "#6b7280",
  };
  return colors[protocol?.toLowerCase()] || colors.unknown;
};

export const NetworkTopology = ({ devices }: NetworkTopologyProps) => {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);

  useEffect(() => {
    if (devices.length === 0) return;

    const containerWidth = 800;
    const containerHeight = 500;
    
    // Create gateway node at top center
    const gatewayNode: Node = {
      id: "gateway",
      type: "default",
      data: {
        label: (
          <div className="text-center">
            <div className="font-bold text-primary">🌐 Gateway</div>
            <div className="text-xs text-muted-foreground font-mono">172.20.0.1</div>
          </div>
        ),
      },
      position: { x: containerWidth / 2 - 75, y: 30 },
      style: {
        background: "hsl(var(--card))",
        border: "3px solid hsl(var(--primary))",
        borderRadius: "12px",
        padding: "12px",
        width: 150,
        boxShadow: "0 4px 12px rgba(0,0,0,0.15)",
      },
    };

    // Arrange devices in rows to prevent overlapping
    const devicesPerRow = Math.min(4, Math.ceil(Math.sqrt(devices.length)));
    const rowCount = Math.ceil(devices.length / devicesPerRow);
    const nodeWidth = 160;
    const nodeHeight = 80;
    const horizontalGap = 40;
    const verticalGap = 100;
    const startY = 150;

    const deviceNodes: Node[] = devices.map((device, index) => {
      const row = Math.floor(index / devicesPerRow);
      const col = index % devicesPerRow;
      
      // Center each row
      const devicesInThisRow = Math.min(devicesPerRow, devices.length - row * devicesPerRow);
      const rowWidth = devicesInThisRow * nodeWidth + (devicesInThisRow - 1) * horizontalGap;
      const startX = (containerWidth - rowWidth) / 2;
      
      const x = startX + col * (nodeWidth + horizontalGap);
      const y = startY + row * (nodeHeight + verticalGap);

      const protocolColor = getProtocolColor(device.protocol);

      return {
        id: device.id,
        type: "default",
        data: {
          label: (
            <div className="text-center">
              <div className="font-semibold text-sm">{device.name}</div>
              <div className="text-xs text-muted-foreground font-mono">{device.ip}</div>
              <div 
                className="text-xs mt-1 px-2 py-0.5 rounded-full inline-block"
                style={{ 
                  backgroundColor: `${protocolColor}20`,
                  color: protocolColor,
                  border: `1px solid ${protocolColor}40`
                }}
              >
                {device.protocol || "unknown"}
              </div>
            </div>
          ),
        },
        position: { x, y },
        style: {
          background: "hsl(var(--card))",
          border: `2px solid ${protocolColor}`,
          borderRadius: "10px",
          padding: "10px",
          width: nodeWidth,
          boxShadow: "0 2px 8px rgba(0,0,0,0.1)",
        },
      };
    });

    // Create edges from gateway to devices
    const deviceEdges: Edge[] = devices.map((device) => ({
      id: `gateway-${device.id}`,
      source: "gateway",
      target: device.id,
      type: "smoothstep",
      animated: true,
      style: { 
        stroke: getProtocolColor(device.protocol), 
        strokeWidth: 2,
        opacity: 0.7,
      },
      markerEnd: {
        type: MarkerType.ArrowClosed,
        color: getProtocolColor(device.protocol),
      },
    }));

    setNodes([gatewayNode, ...deviceNodes]);
    setEdges(deviceEdges);
  }, [devices]);

  if (devices.length === 0) {
    return (
      <Card className="p-8 text-center text-muted-foreground h-[500px] flex items-center justify-center">
        <div>
          <div className="text-4xl mb-4">🔍</div>
          <p>Scan the network to visualize topology</p>
        </div>
      </Card>
    );
  }

  return (
    <Card className="p-0 overflow-hidden h-[500px]">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        fitView
        fitViewOptions={{ padding: 0.2 }}
        className="bg-background"
        minZoom={0.5}
        maxZoom={1.5}
        // defaultViewport={{ x: 0, y: 0, zoom: 0.9 }}
      >
        <Background color="hsl(var(--border))" gap={20} size={1} />
        <Controls className="bg-card border-border" showInteractive={false} />
      </ReactFlow>
    </Card>
  );
};