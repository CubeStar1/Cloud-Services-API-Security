'use client'

import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { FileSpreadsheet, CheckCircle2, XCircle, Loader2 } from 'lucide-react'
import { useToast } from '@/components/ui/use-toast'

export function ConvertButton() {
    const [isConverting, setIsConverting] = useState(false)
    const [status, setStatus] = useState<'idle' | 'converting' | 'success' | 'error'>('idle')
    const { toast } = useToast()

    const handleConversion = async () => {
        setIsConverting(true)
        setStatus('converting')

        try {
            const response = await fetch('/api/logs/convert', {
                method: 'POST'
            })
            const data = await response.json()

            if (data.success) {
                setStatus('success')
                toast({
                    title: "Conversion Successful",
                    description: "All log files have been converted to CSV format.",
                    variant: "default",
                })
            } else {
                throw new Error(data.message || 'Conversion failed')
            }
        } catch (error) {
            setStatus('error')
            toast({
                title: "Conversion Failed",
                description: error instanceof Error ? error.message : "Failed to convert logs to CSV",
                variant: "destructive",
            })
        } finally {
            setIsConverting(false)
            // Reset status after 2 seconds
            setTimeout(() => {
                setStatus('idle')
            }, 2000)
        }
    }

    const getButtonContent = () => {
        switch (status) {
            case 'converting':
                return (
                    <>
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        Converting...
                    </>
                )
            case 'success':
                return (
                    <>
                        <CheckCircle2 className="mr-2 h-4 w-4 text-green-500" />
                        Converted!
                    </>
                )
            case 'error':
                return (
                    <>
                        <XCircle className="mr-2 h-4 w-4 text-red-500" />
                        Failed
                    </>
                )
            default:
                return (
                    <>
                        <FileSpreadsheet className="mr-2 h-4 w-4" />
                        Convert to CSV
                    </>
                )
        }
    }

    const getButtonVariant = () => {
        switch (status) {
            case 'success':
                return 'outline'
            case 'error':
                return 'destructive'
            default:
                return 'default'
        }
    }

    return (
        <Button
            variant={getButtonVariant()}
            onClick={handleConversion}
            disabled={isConverting}
            className={`transition-all duration-200 ${
                status === 'success' ? 'bg-green-100 hover:bg-green-200' : ''
            }`}
        >
            {getButtonContent()}
        </Button>
    )
} 