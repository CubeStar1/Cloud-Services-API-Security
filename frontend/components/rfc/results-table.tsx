'use client'

import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { ScrollArea } from "@/components/ui/scroll-area";

interface ResultsTableProps {
  rows: Record<string, any>[];
}

export function ResultsTable({ rows }: ResultsTableProps) {
  if (!rows.length) {
    return (
      <div className="text-center text-muted-foreground py-8 border rounded-lg">
        No results to display
      </div>
    );
  }

  const headerMap: Record<string, string> = {
    headers_Host: "Host",
    url: "URL",
    method: "Method",
    requestHeaders_Origin: "Origin",
    requestHeaders_Content_Type: "Req Content-Type",
    responseHeaders_Content_Type: "Resp Content-Type",
    requestHeaders_Referer: "Referer",
    requestHeaders_Accept: "Accept",
    source_file: "File",
    service_prediction: "Service",
    service_confidence: "Service Conf.",
    activity_prediction: "Activity",
    activity_confidence: "Activity Conf."
  };

  const cols = Object.keys(rows[0]);

  return (
    <div className="rounded-lg border">
      {/* <ScrollArea className="h-[450px]"> */}
        <div className="max-h-[450px] overflow-y-auto overflow-x-auto">
          <Table className="">
            <TableHeader>
              <TableRow>
                {cols.map((c) => (
                  <TableHead key={c}>{headerMap[c] ?? c}</TableHead>
                ))}
              </TableRow>
            </TableHeader>
            <TableBody>
              {rows.map((row, idx) => (
                <TableRow key={idx}>
                  {cols.map((c) => (
                    <TableCell key={c} className="max-w-[100px] truncate overflow-x-scroll">
                      {String(row[c])}
                    </TableCell>
                  ))}
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      {/* </ScrollArea> */}
    </div>
  );
}
