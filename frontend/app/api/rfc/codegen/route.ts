import { NextResponse } from 'next/server'
import { spawn, ChildProcess } from 'child_process'
import path from 'path'

export async function POST(request: Request) {
    try {
        const body = await request.json()
        const { codegenType } = body 

        const output: string[] = []
        let scriptPath = ''
        let scriptCwd = ''

        const projectRoot = process.cwd() 

        if (codegenType === 'emlearn') {
            scriptPath = path.join(projectRoot,  'scripts', 'rfc', 'em-learn', 'windows-scripts', 'run_rfc_em_inference.bat')
            scriptCwd = path.join(projectRoot,  'scripts', 'rfc', 'em-learn', 'windows-scripts')
        } else { // Default to 'normal'
            scriptPath = path.join(projectRoot,  'scripts', 'rfc', 'run_rfc.bat')
            scriptCwd = path.join(projectRoot,  'scripts', 'rfc')
        }

        return new Promise((resolve) => {
            const childProcess: ChildProcess = spawn('cmd.exe', ['/c', scriptPath], {
                cwd: scriptCwd,
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
                        error: `Script execution failed with code ${code}`,
                        output 
                    }, { status: 500 }))
                }
            })

            childProcess.on('error', (error: Error) => {
                resolve(NextResponse.json({ 
                    success: false, 
                    error: error.message,
                    output // Include output accumulated so far, if any
                }, { status: 500 }))
            })
        })
    } catch (error: any) {
        console.error('Error during code generation request setup:', error)
        return NextResponse.json({ 
            success: false, 
            error: error.message || 'Internal server error' 
        }, { status: 500 })
    }
} 