import { NextRequest, NextResponse } from 'next/server'
import fs from 'fs'
import path from 'path'

// Store logs in memory
let logs: any[] = []

// Function to add a new log
export function addLog(log: any) {
    logs.unshift(log) // Add to beginning of array
    if (logs.length > 1000) { // Keep only last 1000 logs
        logs = logs.slice(0, 1000)
    }
}

// Get all log files from the directory
function getLogFiles() {
    const logsDir = path.join(process.cwd(), 'data', 'logs', 'raw-json')
    if (!fs.existsSync(logsDir)) {
        return []
    }
    return fs.readdirSync(logsDir)
        .filter(file => file.endsWith('.json'))
        .map(file => ({
            name: file,
            path: path.join(logsDir, file),
            timestamp: fs.statSync(path.join(logsDir, file)).mtime.getTime()
        }))
        .sort((a, b) => b.timestamp - a.timestamp) // Sort by most recent first
}

// Read logs from a specific file
function readLogsFromFile(filePath: string) {
    try {
        const fileContent = fs.readFileSync(filePath, 'utf-8')
        const cleanContent = fileContent.replace(/,\s*$/, '')
        return JSON.parse(`[${cleanContent}]`)
    } catch (error) {
        console.error(`Error reading file ${filePath}:`, error)
        return []
    }
}

export async function GET(req: NextRequest) {
    try {
        const searchParams = req.nextUrl.searchParams
        const file = searchParams.get('file')

        // If a specific file is requested
        if (file) {
            const filePath = path.join(process.cwd(), 'data', 'logs', 'raw-json', file)
            if (!fs.existsSync(filePath)) {
                return NextResponse.json({ error: 'File not found' }, { status: 404 })
            }
            const logs = readLogsFromFile(filePath)
            return NextResponse.json(logs)
        }

        // Otherwise, return list of available log files
        const files = getLogFiles()
        return NextResponse.json(files)
    } catch (error) {
        console.error('Error handling logs request:', error)
        return NextResponse.json({ error: 'Failed to process request' }, { status: 500 })
    }
} 