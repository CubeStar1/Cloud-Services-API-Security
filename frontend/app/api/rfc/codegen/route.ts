import { NextResponse } from 'next/server'
import { spawn, ChildProcess } from 'child_process'
import path from 'path'

export async function POST() {
    try {
        const output: string[] = []
        const scriptPath = path.join(process.cwd(), 'scripts', 'rfc', 'run_rfc.bat')

        return new Promise((resolve) => {
            let childProcess: ChildProcess
            childProcess = spawn('cmd.exe', ['/c', scriptPath], {
                cwd: path.join(process.cwd(), 'scripts', 'rfc')
            })

            childProcess.stdout?.on('data', (data: Buffer) => {
                const lines = data.toString().split('\n')
                output.push(...lines.filter((line: string) => line.trim()))
            })

            childProcess.stderr?.on('data', (data: Buffer) => {
                const lines = data.toString().split('\n')
                output.push(...lines.filter((line: string) => line.trim()))
            })

            childProcess.on('close', (code: number | null) => {
                if (code === 0) {
                    resolve(NextResponse.json({ 
                        success: true, 
                        output 
                    }))
                } else {
                    resolve(NextResponse.json({ 
                        success: false, 
                        error: 'Code generation failed', 
                        output 
                    }, { status: 500 }))
                }
            })

            childProcess.on('error', (error: Error) => {
                resolve(NextResponse.json({ 
                    success: false, 
                    error: error.message,
                    output
                }, { status: 500 }))
            })
        })
    } catch (error) {
        console.error('Error during code generation:', error)
        return NextResponse.json({ 
            success: false, 
            error: 'Internal server error' 
        }, { status: 500 })
    }
} 