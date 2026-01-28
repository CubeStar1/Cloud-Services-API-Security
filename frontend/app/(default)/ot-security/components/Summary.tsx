import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Download, Activity, Shield, AlertTriangle, Network } from "lucide-react";
import { useToast } from "@/hooks/use-toast";

interface SummaryProps {
  totalDevices: number;
  anomaliesDetected: number;
  protocolsUsed: string[];
  lastScanTime?: string;
}

export const Summary = ({
  totalDevices,
  anomaliesDetected,
  protocolsUsed,
  lastScanTime,
}: SummaryProps) => {
  const { toast } = useToast();

  const handleDownloadReport = () => {
    // Simulate report generation
    toast({
      title: "Generating Report",
      description: "Your network security report is being prepared...",
    });

    setTimeout(() => {
      toast({
        title: "Report Ready",
        description: "Your report has been downloaded successfully",
      });
    }, 2000);
  };

  const stats = [
    {
      title: "Total Devices",
      value: totalDevices,
      icon: Network,
      color: "text-primary",
      bgColor: "bg-primary/10",
    },
    {
      title: "Anomalies Detected",
      value: anomaliesDetected,
      icon: AlertTriangle,
      color: "text-destructive",
      bgColor: "bg-destructive/10",
    },
    {
      title: "Protocols Used",
      value: protocolsUsed.length,
      icon: Activity,
      color: "text-warning",
      bgColor: "bg-warning/10",
    },
    {
      title: "Security Score",
      value: totalDevices > 0 ? Math.round(((totalDevices - anomaliesDetected) / totalDevices) * 100) : 100,
      icon: Shield,
      color: "text-success",
      bgColor: "bg-success/10",
      suffix: "%",
    },
  ];

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-2xl font-bold">Network Summary</h2>
          {lastScanTime && (
            <p className="text-sm text-muted-foreground mt-1">
              Last scan: {new Date(lastScanTime).toLocaleString()}
            </p>
          )}
        </div>
        <Button onClick={handleDownloadReport} className="glow-primary">
          <Download className="mr-2 h-4 w-4" />
          Download Report
        </Button>
      </div>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        {stats.map((stat) => (
          <Card key={stat.title} className="overflow-hidden">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground">
                {stat.title}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-center justify-between">
                <div className="text-3xl font-bold">
                  {stat.value}
                  {stat.suffix}
                </div>
                <div className={`${stat.bgColor} p-3 rounded-lg`}>
                  <stat.icon className={`h-6 w-6 ${stat.color}`} />
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {protocolsUsed.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Detected OT Protocols</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-2">
              {protocolsUsed.map((protocol) => (
                <div
                  key={protocol}
                  className="px-3 py-1 bg-primary/10 text-primary rounded-full text-sm font-medium"
                >
                  {protocol}
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};
