import { NextResponse } from 'next/server'
import fs from 'fs/promises'
import path from 'path'
import { statSync } from 'fs' // For statSync, keep 'fs'

const projectBasePath = path.join(process.cwd()) // Assuming process.cwd() is project root
const normalCodegenDir = path.join(projectBasePath, 'data', 'output', 'rfc', 'codegen')
const emlearnCodegenDir = path.join(projectBasePath, 'data', 'output', 'rfc', 'em-codegen')
const emlearnIncludeDir = path.join(emlearnCodegenDir, 'include')

const codeDirectories = [
    { dir: normalCodegenDir, fileExtensions: ['.c', '.txt'], prefix: '' },
    { dir: emlearnCodegenDir, fileExtensions: ['.c', '.txt'], prefix: '' },
    { dir: emlearnIncludeDir, fileExtensions: ['.h'], prefix: 'include/' }
]

async function listFilesFromDir(dirPath: string, allowedExtensions: string[], pathPrefix: string) {
    const filesInfo: { name: string, path: string, timestamp: string }[] = []
    try {
        await fs.access(dirPath) // Check if directory exists
        const entries = await fs.readdir(dirPath, { withFileTypes: true })
        for (const entry of entries) {
            const ext = path.extname(entry.name)
            if (entry.isFile() && allowedExtensions.includes(ext)) {
                const fullPath = path.join(dirPath, entry.name)
                const stats = statSync(fullPath) // fs.statSync is fine here for simplicity
                filesInfo.push({
                    name: pathPrefix + entry.name, 
                    path: fullPath,
                    timestamp: stats.mtime.toISOString()
                })
            }
        }
    } catch (error) {
        // If a directory doesn't exist or other error, log it but don't fail the whole listing
        console.warn(`Warning: Could not read directory ${dirPath} or it doesn't exist. Error: ${(error as Error).message}`)
    }
    return filesInfo
}

export async function GET(request: Request) {
    const { searchParams } = new URL(request.url)
    const fileName = searchParams.get('file')

    if (fileName) {
        let foundPath = ''
        let actualFileNameForResponse = ''

        for (const dirConfig of codeDirectories) {
            let effectiveFileName: string = fileName
            // If the requested fileName starts with the dirConfig prefix, strip it for path joining
            if (dirConfig.prefix && fileName.startsWith(dirConfig.prefix)) {
                effectiveFileName = fileName.substring(dirConfig.prefix.length)
            }
            
            const potentialPath = path.join(dirConfig.dir, effectiveFileName)
            
            try {
                await fs.access(potentialPath)
                // Ensure the file being accessed actually matches the original request (after prefix reconstruction)
                if ((dirConfig.prefix + effectiveFileName) === fileName) {
                    foundPath = potentialPath
                    actualFileNameForResponse = fileName // Use the original (potentially prefixed) name for response
                    break
                }
            } catch {
                // File not found in this location or under this prefix configuration
            }
        }

        if (foundPath) {
            try {
                const content = await fs.readFile(foundPath, 'utf-8')
                const stats = statSync(foundPath)
                return NextResponse.json({
                    content,
                    name: actualFileNameForResponse, // Return the name as requested by client
                    path: foundPath, // This is the actual FS path
                    timestamp: stats.mtime.toISOString()
                })
            } catch (error) {
                console.error(`Error reading file ${actualFileNameForResponse} (path: ${foundPath}):`, error)
                return NextResponse.json({ error: `Failed to read file: ${actualFileNameForResponse}` }, { status: 500 })
            }
        } else {
            return NextResponse.json({ error: `File not found: ${fileName}` }, { status: 404 })
        }
    } else {
        // List all code files
        let allFiles: { name: string, path: string, timestamp: string }[] = []
        for (const dirConfig of codeDirectories) {
            const files = await listFilesFromDir(dirConfig.dir, dirConfig.fileExtensions, dirConfig.prefix)
            allFiles = allFiles.concat(files)
        }
        // Sort files by name for consistent order in the UI
        allFiles.sort((a, b) => a.name.localeCompare(b.name))

        return NextResponse.json({ files: allFiles })
    }
} 