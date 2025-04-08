import { NextResponse } from 'next/server'
import { NextRequest } from 'next/server'
import fs from 'fs'
import path from 'path'

export async function GET(request: NextRequest) {
    try {
        const codegenDir = path.join(process.cwd(), 'data', 'output', 'rfc', 'codegen')
        const searchParams = request.nextUrl.searchParams
        const fileName = searchParams.get('file')
        
        // Check if directory exists
        if (!fs.existsSync(codegenDir)) {
            return NextResponse.json({ 
                files: [],
                message: "No generated code files found" 
            })
        }

        // If a specific file is requested, return only that file's content
        if (fileName) {
            const filePath = path.join(codegenDir, fileName)
            if (fs.existsSync(filePath)) {
                const content = fs.readFileSync(filePath, 'utf-8')
                return NextResponse.json({
                    content,
                    name: fileName,
                    path: filePath,
                    timestamp: fs.statSync(filePath).mtime.toISOString()
                })
            }
            return NextResponse.json({ error: 'File not found' }, { status: 404 })
        }

        // Otherwise, return list of files without content
        const files = fs.readdirSync(codegenDir)
            .filter(file => file.endsWith('.c') || file.endsWith('.h'))
            .map(file => {
                const filePath = path.join(codegenDir, file)
                return {
                    name: file,
                    path: filePath,
                    timestamp: fs.statSync(filePath).mtime.toISOString()
                }
            })
            .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())

        return NextResponse.json({ files })
    } catch (error) {
        console.error('Error reading code files:', error)
        return NextResponse.json({ 
            error: 'Failed to read code files' 
        }, { status: 500 })
    }
} 