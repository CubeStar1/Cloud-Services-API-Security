import { spawn } from "child_process"
import path from "path"
import fs from "fs"
import { NextRequest, NextResponse } from "next/server"
import { addLog } from './logs/route'

let proxyProcess: any = null
let currentLogFile: string | null = null

export async function POST(request: NextRequest) {
    const { action, filename } = await request.json()

    if (action === 'start') {
        if (proxyProcess) {
            return NextResponse.json({ status: 'error', message: 'Proxy is already running' })
        }

        try {
            // Ensure data directories exist
            const dataDir = path.resolve(process.cwd(), 'data', 'logs', 'raw-json')
            fs.mkdirSync(dataDir, { recursive: true })

            // Use absolute path for the rule file
            const rulePath = path.resolve(process.cwd(), 'anyproxy', 'rule.js')
            
            if (!fs.existsSync(rulePath)) {
                throw new Error(`Rule file not found at ${rulePath}`)
            }

            // Validate filename
            const logFilename = filename?.trim() || 'box-traffic-logs.json'
            if (!/^[\w-]+\.json$/.test(logFilename)) {
                throw new Error('Invalid filename. Use only letters, numbers, hyphens, and underscores.')
            }

            console.log('Starting proxy with rule file:', rulePath)
            console.log('Using log file:', logFilename)
            
            // Store current log file name
            currentLogFile = logFilename

            // Use npx to run anyproxy with environment variable for filename
            proxyProcess = spawn('npx', [
                'anyproxy',
                '--port', '8001',
                '--rule', rulePath,
                '--intercept'
            ], {
                stdio: 'pipe',
                shell: true, // This is needed for Windows
                env: {
                    ...process.env,
                    LOG_FILE_NAME: logFilename
                }
            })

            proxyProcess.on('error', (err: Error) => {
                console.error('Proxy process error:', err)
                proxyProcess = null
                currentLogFile = null
            })

            // Parse and handle structured logs from stdout
            proxyProcess.stdout?.on('data', (data: Buffer) => {
                const output = data.toString()
                try {
                    // Try to parse each line as JSON
                    output.split('\n').forEach(line => {
                        if (line.trim()) {
                            try {
                                const log = JSON.parse(line)
                                if (log.type === 'log') {
                                    addLog(log.data)
                                }
                            } catch (e) {
                                // Not a JSON log, ignore
                            }
                        }
                    })
                } catch (error) {
                    console.error('Error parsing proxy output:', error)
                }
            })

            // Log stderr for debugging
            proxyProcess.stderr?.on('data', (data: Buffer) => {
                console.error('Proxy error:', data.toString())
            })

            // Handle process exit
            proxyProcess.on('exit', (code: number) => {
                console.log(`Proxy process exited with code ${code}`)
                proxyProcess = null
                currentLogFile = null
            })

            return NextResponse.json({ 
                status: 'success', 
                message: `Proxy started successfully on port 8001, logging to ${logFilename}`
            })
        } catch (error) {
            console.error('Failed to start proxy:', error)
            return NextResponse.json({ 
                status: 'error', 
                message: `Failed to start proxy: ${error instanceof Error ? error.message : 'Unknown error'}`
            })
        }
    }

    if (action === 'stop') {
        if (!proxyProcess) {
            return NextResponse.json({ status: 'error', message: 'Proxy is not running' })
        }

        try {
            // On Windows, we need to kill the process tree
            if (process.platform === 'win32') {
                spawn('taskkill', ['/pid', proxyProcess.pid.toString(), '/f', '/t'])
            } else {
                proxyProcess.kill()
            }
            proxyProcess = null
            currentLogFile = null
            return NextResponse.json({ status: 'success', message: 'Proxy stopped successfully' })
        } catch (error) {
            console.error('Failed to stop proxy:', error)
            return NextResponse.json({ status: 'error', message: 'Failed to stop proxy' })
        }
    }

    if (action === 'status') {
        return NextResponse.json({ 
            status: 'success', 
            isRunning: proxyProcess !== null,
            currentLogFile
        })
    }

    return NextResponse.json({ status: 'error', message: 'Invalid action' })
}