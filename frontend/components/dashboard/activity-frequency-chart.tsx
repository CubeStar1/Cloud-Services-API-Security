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
  ChartTooltipContent
} from "@/components/ui/chart"
import { ListChecks } from "lucide-react" // Use a different icon

interface ActivityFrequencyData {
  activity: string;
  count: number;
  fill?: string;
}

interface ActivityFrequencyChartProps {
  data: ActivityFrequencyData[];
}

const chartConfig = {
  count: {
    label: "Requests",
    color: "hsl(var(--chart-2))", // Use a different base color
  },
} satisfies ChartConfig

export function ActivityFrequencyChart({ data }: ActivityFrequencyChartProps) {
  const coloredData = data.map((item, index) => ({
    ...item,
    // Cycle through colors, perhaps offsetting from the service chart
    fill: `hsl(var(--chart-${(index % 5) + 1}))` 
  }))

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
            <ListChecks className="h-5 w-5" /> Activity Frequency
        </CardTitle>
        <CardDescription>Count of requests per predicted activity.</CardDescription>
      </CardHeader>
      <CardContent>
        <ChartContainer config={chartConfig} className="h-[250px]"> 
          <BarChart data={coloredData} layout="vertical" margin={{ left: 10, right: 10 }}>
            <CartesianGrid horizontal={false} />
            <YAxis 
              dataKey="activity" 
              type="category" 
              tickLine={false} 
              axisLine={false}
              tickMargin={10}
              width={100} // Adjust width for activity names
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