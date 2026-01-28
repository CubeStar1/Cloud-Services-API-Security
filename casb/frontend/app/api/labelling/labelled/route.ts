import { NextRequest, NextResponse } from 'next/server'
import fs from 'fs'
import path from 'path'

// Get all labelled files from the directory
function getLabelledFiles() {
    const labelledDir = path.join(process.cwd(), 'data', 'labelled')
    if (!fs.existsSync(labelledDir)) {
        return []
    }
    return fs.readdirSync(labelledDir)
        .filter(file => file.endsWith('.xlsx') || file.endsWith('.csv'))
        .map(file => ({
            name: file,
            path: path.join(labelledDir, file),
            timestamp: fs.statSync(path.join(labelledDir, file)).mtime.getTime()
        }))
        .sort((a, b) => b.timestamp - a.timestamp)
}

export async function GET() {
    try {
        const files = getLabelledFiles()
        return NextResponse.json(files)
    } catch (error) {
        console.error('Error getting labelled files:', error)
        return NextResponse.json({ error: 'Failed to get labelled files' }, { status: 500 })
    }
} 