import { useEffect, useState } from "react";
import ReactFlow, {
  Background,
  Controls,
  Node,
  Edge,
  MarkerType,
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

export const NetworkTopology = ({ devices }: NetworkTopologyProps) => {
  const [nodes, setNodes] = useState<Node[]>([]);
  const [edges, setEdges] = useState<Edge[]>([]);

  useEffect(() => {
    if (devices.length === 0) return;

    // Create gateway node
    const gatewayNode: Node = {
      id: "gateway",
      type: "default",
      data: {
        label: (
          <div className="text-center">
            <div className="font-semibold text-primary">Gateway</div>
            <div className="text-xs text-muted-foreground font-mono">Router</div>
          </div>
        ),
      },
      position: { x: 400, y: 50 },
      style: {
        background: "hsl(var(--card))",
        border: "2px solid hsl(var(--primary))",
        borderRadius: "8px",
        padding: "10px",
        width: 150,
      },
    };

    // Create device nodes in a circular layout
    const deviceNodes: Node[] = devices.map((device, index) => {
      const angle = (index / devices.length) * 2 * Math.PI;
      const radius = 250;
      const x = 400 + radius * Math.cos(angle);
      const y = 300 + radius * Math.sin(angle);

      return {
        id: device.id,
        type: "default",
        data: {
          label: (
            <div className="text-center">
              <div className="font-semibold">{device.name}</div>
              <div className="text-xs text-muted-foreground font-mono">{device.ip}</div>

            </div>
          ),
        },
        position: { x, y },
        style: {
          background: "hsl(var(--card))",
          border: "1px solid hsl(var(--border))",
          borderRadius: "8px",
          padding: "8px",
          width: 140,
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
      style: { stroke: "hsl(var(--primary))", strokeWidth: 2 },
      markerEnd: {
        type: MarkerType.ArrowClosed,
        color: "hsl(var(--primary))",
      },
    }));

    setNodes([gatewayNode, ...deviceNodes]);
    setEdges(deviceEdges);
  }, [devices]);

  if (devices.length === 0) {
    return (
      <Card className="p-8 text-center text-muted-foreground">
        <p>Scan the network to visualize topology</p>
      </Card>
    );
  }

  return (
    <Card className="p-0 overflow-hidden h-[600px]">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        fitView
        className="bg-background"
      >
        <Background color="hsl(var(--border))" gap={16} />
        <Controls className="bg-card border-border" />
      </ReactFlow>
    </Card>
  );
};
