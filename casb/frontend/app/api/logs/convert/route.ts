import { NextRequest, NextResponse } from 'next/server'
import { spawn } from 'child_process'
import path from 'path'

export async function POST(request: NextRequest) {
    try {
        // Get the absolute path to the Python script
        const scriptPath = path.resolve(process.cwd(), 'scripts', 'csv-creation.py')
        
        // Create a promise to handle the Python script execution
        const conversionPromise = new Promise((resolve, reject) => {
            const pythonProcess = spawn('python', [scriptPath], {
                stdio: 'pipe'
            })

            let output = ''
            let errorOutput = ''

            pythonProcess.stdout.on('data', (data) => {
                output += data.toString()
            })

            pythonProcess.stderr.on('data', (data) => {
                errorOutput += data.toString()
            })

            pythonProcess.on('close', (code) => {
                if (code === 0) {
                    resolve({
                        success: true,
                        message: 'CSV conversion completed successfully',
                        output: output
                    })
                } else {
                    reject({
                        success: false,
                        message: 'CSV conversion failed',
                        error: errorOutput
                    })
                }
            })

            pythonProcess.on('error', (error) => {
                reject({
                    success: false,
                    message: 'Failed to start Python process',
                    error: error.message
                })
            })
        })

        const result = await conversionPromise
        return NextResponse.json(result)

    } catch (error) {
        console.error('Error in CSV conversion:', error)
        return NextResponse.json({
            success: false,
            message: 'Failed to convert logs to CSV',
            error: error instanceof Error ? error.message : 'Unknown error'
        }, { status: 500 })
    }
} 