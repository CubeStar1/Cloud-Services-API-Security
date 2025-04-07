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

// Get all prediction files
function getPredictionFiles() {
    const predDir = path.join(process.cwd(), 'data', 'output', 'deberta', 'predictions')
    if (!fs.existsSync(predDir)) {
        return []
    }
    return fs.readdirSync(predDir)
        .filter(file => file.endsWith('.csv'))
        .map(file => ({
            name: file,
            path: path.join(predDir, file),
            timestamp: fs.statSync(path.join(predDir, file)).mtime.getTime()
        }))
        .sort((a, b) => b.timestamp - a.timestamp)
}

export async function GET(req: NextRequest) {
    try {
        const searchParams = req.nextUrl.searchParams
        const type = searchParams.get('type')

        if (type === 'predictions') {
            const files = getPredictionFiles()
            return NextResponse.json(files)
        } else {
            const files = getCsvFiles()
            return NextResponse.json(files)
        }
    } catch (error) {
        console.error('Error getting files:', error)
        return NextResponse.json({ error: 'Failed to get files' }, { status: 500 })
    }
}

export async function POST() {
    try {
        const scriptPath = path.join(process.cwd(), 'scripts', 'deberta', 'inference.py')
        
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
                message: 'DeBERTa inference completed successfully',
                output 
            })
        } else {
            return NextResponse.json({ 
                success: false, 
                message: 'DeBERTa inference failed',
                error 
            }, { status: 500 })
        }
    } catch (error) {
        console.error('Error running DeBERTa inference:', error)
        return NextResponse.json({ 
            success: false, 
            message: 'Failed to run DeBERTa inference',
            error: error.message 
        }, { status: 500 })
    }
} 