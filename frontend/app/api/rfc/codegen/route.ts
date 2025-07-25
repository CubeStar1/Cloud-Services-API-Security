import { NextResponse } from 'next/server'

export async function POST(request: Request) {
    try {
        const body = await request.json()
        const { codegenType } = body 

        // Decide which backend endpoint to call
        const backendBase = process.env.BACKEND_URL || 'http://localhost:8000'
        const endpointPath = codegenType === 'emlearn' ? '/rfc/train/c/emlearn' : '/rfc/train/c/manual'
        const url = `${backendBase}${endpointPath}`

        try {
            const backendResp = await fetch(url, { method: 'POST' })
            const data = await backendResp.json()

            // Ensure the response shape mimics the old script behaviour
            if (backendResp.ok && data.success) {
                return NextResponse.json({ success: true, output: data.output ?? [] })
            }
            return NextResponse.json(
                {
                    success: false,
                    error: data.error || `Backend returned status ${backendResp.status}`,
                    output: data.output ?? []
                },
                { status: 500 }
            )
        } catch (error: any) {
            console.error('Error contacting backend:', error)
            return NextResponse.json({ success: false, error: error.message || 'Fetch failed' }, { status: 500 })
        }
    } catch (error: any) {
        console.error('Error parsing request:', error)
        return NextResponse.json({ success: false, error: error.message || 'Request failed' }, { status: 500 })
    }
}
