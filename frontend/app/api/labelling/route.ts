import { NextRequest, NextResponse } from 'next/server'
import { spawn } from 'child_process'
import path from 'path'
import fs from 'fs'

// Get all CSV files from the directory
function getCsvFiles() {
    const csvDir = path.join(process.cwd(), 'data', 'logs', 'csv')
    if (!fs.existsSync(csvDir)) {
        return []
    }
    return fs.readdirSync(csvDir)
        .filter(file => file.endsWith('.csv'))
        .map(file => ({
            name: file,
            path: path.join(csvDir, file),
            timestamp: fs.statSync(path.join(csvDir, file)).mtime.getTime()
        }))
        .sort((a, b) => b.timestamp - a.timestamp)
}

export async function GET() {
    try {
        const files = getCsvFiles()
        return NextResponse.json(files)
    } catch (error) {
        console.error('Error getting CSV files:', error)
        return NextResponse.json({ error: 'Failed to get CSV files' }, { status: 500 })
    }
}

export async function POST(req: NextRequest) {
    try {
        const scriptPath = path.join(process.cwd(), 'scripts', 'labelling.py')
        
        // Spawn the Python process
        const pythonProcess = spawn('python', [scriptPath])
        
        let output = ''
        let error = ''
        
        // Collect stdout data
        pythonProcess.stdout.on('data', (data) => {
            const newData = data.toString()
            output += newData
            console.log(newData) // Log progress in real-time
        })
        
        // Collect stderr data
        pythonProcess.stderr.on('data', (data) => {
            const newData = data.toString()
            error += newData
            console.error(newData)
        })
        
        // Wait for the process to complete
        const exitCode = await new Promise((resolve) => {
            pythonProcess.on('close', resolve)
        })
        
        if (exitCode === 0) {
            return NextResponse.json({ 
                success: true, 
                message: 'Labelling completed successfully',
                output 
            })
        } else {
            return NextResponse.json({ 
                success: false, 
                message: 'Labelling failed',
                error 
            }, { status: 500 })
        }
    } catch (error) {
        console.error('Error running labelling script:', error)
        return NextResponse.json({ 
            success: false, 
            message: 'Failed to run labelling script',
            error: error.message 
        }, { status: 500 })
    }
} 