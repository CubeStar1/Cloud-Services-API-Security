import { ProxyControl } from '@/components/anyproxy/proxy-control'
import { TrafficLogs } from '@/components/anyproxy/traffic-logs'
import { Toaster } from 'sonner'

export default function DashboardPage() {
    return (
        <div className="container mx-auto p-6">
            <div className="grid gap-6">
                <h1 className="text-3xl font-bold">Dashboard</h1>
                <div className="grid gap-6">
                    <ProxyControl />
                    <TrafficLogs />
                </div>
            </div>
            <Toaster />
        </div>
    )
}
