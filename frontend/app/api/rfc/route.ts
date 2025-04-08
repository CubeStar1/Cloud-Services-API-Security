import { NextRequest, NextResponse } from "next/server"
import { spawn } from "child_process"
import path from "path"
import fs from "fs"

// Helper function to get CodeBERT output files
function getCodeBertFiles() {
    const codebertDir = path.join(process.cwd(), 'data', 'output', 'codebert')
    
    if (!fs.existsSync(codebertDir)) {
        return []
    }

    return fs.readdirSync(codebertDir)
        .filter(file => file.endsWith('.csv'))
        .map(file => {
            const filePath = path.join(codebertDir, file)
            const stats = fs.statSync(filePath)
            return {
                name: file,
                path: filePath,
                timestamp: stats.mtimeMs
            }
        })
        .sort((a, b) => b.timestamp - a.timestamp)
}

// Helper function to get model files
function getModelFiles() {
    const modelDir = path.join(process.cwd(), 'data', 'models', 'rfc')
    
    if (!fs.existsSync(modelDir)) {
        return []
    }

    return fs.readdirSync(modelDir)
        .filter(file => file.endsWith('.joblib'))
        .map(file => {
            const filePath = path.join(modelDir, file)
            const stats = fs.statSync(filePath)
            return {
                name: file,
                path: filePath,
                timestamp: stats.mtimeMs
            }
        })
        .sort((a, b) => b.timestamp - a.timestamp)
}

export async function GET(req: NextRequest) {
    try {
        const type = req.nextUrl.searchParams.get('type')
        
        if (type === 'models') {
            return NextResponse.json({
                files: getModelFiles()
            })
        }

        return NextResponse.json({
            files: getCodeBertFiles()
        })
    } catch (error) {
        console.error('Error fetching files:', error)
        return NextResponse.json(
            { error: 'Failed to fetch files' },
            { status: 500 }
        )
    }
}

export async function POST(req: NextRequest) {
    try {
        const { file } = await req.json()
        
        if (!file) {
            return NextResponse.json(
                { error: 'No file specified' },
                { status: 400 }
            )
        }

        const pythonProcess = spawn('python', [
            'scripts/rfc/train.py',
            file
        ])

        let output: string[] = []
        let error: string | null = null
        let metrics: any = null

        return new Promise((resolve) => {
            pythonProcess.stdout.on('data', (data) => {
                const lines = data.toString().split('\n').filter(Boolean)
                output.push(...lines)
            })

            pythonProcess.stderr.on('data', (data) => {
                error = data.toString()
            })

            pythonProcess.on('close', (code) => {
                if (code === 0) {
                    // Extract metrics from the output
                    const serviceAccuracy = output.find(line => line.includes('Service Classification Accuracy:'))
                    const activityAccuracy = output.find(line => line.includes('Activity Classification Accuracy:'))
                    const uniqueServices = output.find(line => line.includes('Found'))?.match(/(\d+) unique services/)?.[1]
                    const uniqueActivities = output.find(line => line.includes('Found'))?.match(/(\d+) unique activities/)?.[1]

                    if (serviceAccuracy && activityAccuracy && uniqueServices && uniqueActivities) {
                        metrics = {
                            service_accuracy: parseFloat(serviceAccuracy.split(':')[1].trim()),
                            activity_accuracy: parseFloat(activityAccuracy.split(':')[1].trim()),
                            unique_services: parseInt(uniqueServices),
                            unique_activities: parseInt(uniqueActivities)
                        }
                    }

                    resolve(NextResponse.json({
                        success: true,
                        output,
                        metrics
                    }))
                } else {
                    resolve(NextResponse.json({
                        success: false,
                        error: error || 'Training failed',
                        output
                    }, { status: 500 }))
                }
            })
        })
    } catch (error) {
        console.error('Error during training:', error)
        return NextResponse.json(
            { error: 'Training failed' },
            { status: 500 }
        )
    }
} 