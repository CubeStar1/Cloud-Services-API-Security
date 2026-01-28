"use client";

import { Button } from "@/components/ui/button";

interface CsvDownloadButtonProps {
  rows: Record<string, any>[];
  filename?: string;
}

export function CsvDownloadButton({ rows, filename = "rfc_predictions.csv" }: CsvDownloadButtonProps) {
  if (!rows.length) return null;

  const handleDownload = () => {
    const cols = Object.keys(rows[0]);
    const csvHeader = cols.join(",");
    const csvRows = rows.map((r) =>
      cols
        .map((c) => {
          const value = String(r[c] ?? "");
          return `"${value.replace(/"/g, '""')}"`;
        })
        .join(",")
    );
    const csvContent = [csvHeader, ...csvRows].join("\n");
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.setAttribute("download", filename);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  return (
    <Button variant="default" size="sm" onClick={handleDownload}>
      Download CSV
    </Button>
  );
}
