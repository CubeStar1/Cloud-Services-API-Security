import { NextResponse } from 'next/server'
import { spawn } from 'child_process'
import path from 'path'
import fs from 'fs'

export async function POST() {
    try {
        const scriptPath = path.join(process.cwd(), 'scripts', 'codebert', 'train.py')
        
        // Check if training data exists
        const trainDataPath = path.join(process.cwd(), 'data', 'labelled', 'train_set.xlsx')
        if (!fs.existsSync(trainDataPath)) {
            return NextResponse.json({ 
                success: false, 
                message: 'Training data not found',
                error: `Expected file at: ${trainDataPath}` 
            }, { status: 404 })
        }
        
        // Spawn the Python process
        const pythonProcess = spawn('python', [scriptPath])
        
        let output = ''
        let errorOutput = ''
        
        // Collect stdout data
        pythonProcess.stdout.on('data', (data) => {
            const newData = data.toString()
            output += newData
            console.log(newData) // Log progress in real-time
        })
        
        // Collect stderr data
        pythonProcess.stderr.on('data', (data) => {
            const newData = data.toString()
            errorOutput += newData
            console.error(newData)
        })
        
        // Wait for the process to complete
        const exitCode = await new Promise<number>((resolve) => {
            pythonProcess.on('close', resolve)
        })
        
        if (exitCode === 0) {
            return NextResponse.json({ 
                success: true, 
                message: 'CodeBERT training completed successfully',
                output 
            })
        } else {
            return NextResponse.json({ 
                success: false, 
                message: 'CodeBERT training failed',
                error: errorOutput 
            }, { status: 500 })
        }
    } catch (error) {
        const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred'
        console.error('Error running CodeBERT training:', error)
        return NextResponse.json({ 
            success: false, 
            message: 'Failed to run CodeBERT training',
            error: errorMessage 
        }, { status: 500 })
    }
} 