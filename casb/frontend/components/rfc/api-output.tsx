import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { CheckCircle2, XCircle, Clock } from "lucide-react";

interface ApiOutputProps {
  output?: string[];
  success?: boolean;
  time?: number;
}

export function ApiOutput({ output = [], success, time }: ApiOutputProps) {
  return (
    <Card className="h-full flex flex-col">
      <CardHeader className="pb-2">
        <div className="flex items-end justify-between">
          <CardTitle>Console Output</CardTitle>
          <div className="flex items-center gap-4 text-sm">
          {success !== undefined && (
            <div className="flex items-center gap-2 text-sm">
              {success ? (
                <span className="text-green-600 flex items-center">
                  <CheckCircle2 className="h-4 w-4 mr-1" /> Success
                </span>
              ) : (
                <span className="text-red-600 flex items-center">
                  <XCircle className="h-4 w-4 mr-1" /> Failed
                </span>
              )}
            </div>
          )}
          {time !== undefined && (
            <div className="flex items-center gap-2 text-sm">
              <span className="text-blue-600 flex items-center">
                <Clock className="h-4 w-4 mr-1" /> {time.toFixed(2)} seconds
              </span>
            </div>
          )}
          </div>
        </div>
      </CardHeader>
      <CardContent className="flex-1 p-0">
        <ScrollArea className="h-[400px] px-4 py-2">
          <div className="space-y-2 font-mono text-sm">
            {output.length > 0 ? (
              output.map((line, i) => (
                <div key={i} className="whitespace-pre-wrap break-words">
                  {line}
                </div>
              ))
            ) : (
              <div className="text-muted-foreground text-center py-8">
                No output yet. Run a prediction to see the results.
              </div>
            )}
          </div>
        </ScrollArea>
      </CardContent>
    </Card>
  );
}
