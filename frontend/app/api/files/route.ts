import { NextResponse } from 'next/server'
import { NextRequest } from 'next/server'
import fs from 'fs'
import path from 'path'

function buildFileTree(dir: string, baseDir: string): any[] {
    const items = fs.readdirSync(dir)
    const tree = []

    for (const item of items) {
        const fullPath = path.join(dir, item)
        const relativePath = path.relative(baseDir, fullPath)
        const stats = fs.statSync(fullPath)

        if (stats.isDirectory()) {
            tree.push({
                id: relativePath,
                name: item,
                isSelectable: true,
                children: buildFileTree(fullPath, baseDir)
            })
        } else {
            // Show CSV files but mark them as non-selectable
            tree.push({
                id: relativePath,
                name: item,
                isSelectable: !item.endsWith('.csv')
            })
        }
    }

    return tree
}

export async function GET(request: NextRequest) {
    try {
        const searchParams = request.nextUrl.searchParams
        const filePath = searchParams.get('file')
        const dataDir = path.join(process.cwd(), 'data')

        // If a specific file is requested, return its content
        if (filePath) {
            const fullPath = path.join(dataDir, filePath)
            // Ensure the file is within the data directory
            if (!fullPath.startsWith(dataDir)) {
                return NextResponse.json({ error: 'Invalid file path' }, { status: 400 })
            }
            
            if (fs.existsSync(fullPath) && fs.statSync(fullPath).isFile()) {
                const content = fs.readFileSync(fullPath, 'utf-8')
                const name = path.basename(fullPath)
                const extension = path.extname(fullPath).slice(1) // Remove the dot
                
                // For CSV files, return with a special type
                if (extension === 'csv') {
                    return NextResponse.json({
                        content,
                        name,
                        type: 'csv'
                    })
                }
                
                // For other files, return as before
                return NextResponse.json({
                    content,
                    name,
                    language: extension
                })
            }
            return NextResponse.json({ error: 'File not found' }, { status: 404 })
        }

        // Otherwise return the file tree
        const tree = buildFileTree(dataDir, dataDir)
        return NextResponse.json({ tree })
    } catch (error) {
        console.error('Error reading directory structure:', error)
        return NextResponse.json({ 
            error: 'Failed to read directory structure' 
        }, { status: 500 })
    }
} 