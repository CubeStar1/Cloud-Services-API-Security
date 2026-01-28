'use client'

import {
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip as ChartTooltip
} from "recharts"

import {
  Card, 
  CardContent, 
  CardDescription, 
  CardHeader, 
  CardTitle 
} from "@/components/ui/card"
import {
  ChartConfig, 
  ChartContainer, 
  ChartTooltipContent, 
  ChartLegend, 
  ChartLegendContent
} from "@/components/ui/chart"
import { Info } from "lucide-react"

interface ServiceFrequencyData {
  service: string;
  count: number;
  fill?: string; // Optional fill color for dynamic assignment
}

interface ServiceFrequencyChartProps {
  data: ServiceFrequencyData[];
}

const chartConfig = {
  count: {
    label: "Requests",
    color: "hsl(var(--chart-1))", 
  },
} satisfies ChartConfig

export function ServiceFrequencyChart({ data }: ServiceFrequencyChartProps) {
  // Assign colors dynamically based on chartConfig or a default palette if needed
  const coloredData = data.map((item, index) => ({
    ...item,
    fill: `hsl(var(--chart-${(index % 5) + 1}))` // Cycle through 5 chart colors
  }))

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Info className="h-5 w-5" /> Service Frequency
        </CardTitle>
        <CardDescription>Count of requests per predicted service.</CardDescription>
      </CardHeader>
      <CardContent>
        <ChartContainer config={chartConfig} className="h-[250px]"> 
          <BarChart data={coloredData} layout="vertical" margin={{ left: 10, right: 10 }}>
            <CartesianGrid horizontal={false} />
            <YAxis 
              dataKey="service" 
              type="category" 
              tickLine={false} 
              axisLine={false}
              tickMargin={10}
              width={100} // Adjust width for service names
            />
            <XAxis dataKey="count" type="number" hide />
            <ChartTooltip 
              cursor={false} 
              content={<ChartTooltipContent indicator="line" />} 
            />
            <Bar 
              dataKey="count" 
              layout="vertical" 
              radius={5} 
              barSize={30}
            />
          </BarChart>
        </ChartContainer>
      </CardContent>
    </Card>
  )
} 