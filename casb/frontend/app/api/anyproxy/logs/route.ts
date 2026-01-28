import { NextRequest, NextResponse } from 'next/server';
import axios from 'axios';
import path from 'path';

const BACKEND = process.env.BACKEND_URL ?? 'http://localhost:8000';

export async function GET(req: NextRequest) {
    try {
        const searchParams = req.nextUrl.searchParams;
        const file = searchParams.get('file');
        const format = searchParams.get('format'); // 'json' or 'csv'

        // If a specific file is requested
        if (file) {
            // Infer directory from extension if not explicit path
            let subdir = 'logs/raw-json';
            if (file.endsWith('.csv')) {
                subdir = 'logs/csv';
            }

            // Safety check: avoid double prefixing if UI changes to send full relative path
            const relPath = file.includes('/') ? file : `${subdir}/${file}`;
            
            const { data } = await axios.get(`${BACKEND}/files`, {
                params: { file: relPath }
            });
            
            if (data.error) {
                 return NextResponse.json({ error: data.error }, { status: 404 });
            }
            
            let content = data.content;
            
            // Special parsing for our raw JSON logs
            if (file.endsWith('.json') && typeof content === 'string') {
                 try {
                     const cleanContent = content.replace(/,\s*$/, '');
                     const parsed = JSON.parse(`[${cleanContent}]`);
                     return NextResponse.json(parsed);
                 } catch (e) {
                     console.error('Error parsing proxy log content:', e);
                     return NextResponse.json([]);
                 }
            }
            
            // For CSV or other files, return content as is (or handle CSV parsing if needed later)
            return NextResponse.json(content);
        }

        // Otherwise, return list of available log files
        let params: any = { subdir: 'logs' }; // Default: fetch all logs recursively

        if (format === 'json') {
            params = { subdir: 'logs/raw-json', ext: 'json' };
        } else if (format === 'csv') {
            params = { subdir: 'logs/csv', ext: 'csv' };
        }

        const { data } = await axios.get(`${BACKEND}/files`, {
            params
        });
        
        return NextResponse.json(data);
        
    } catch (error: any) {
        console.error('Error handling logs request:', error.message);
        const status = error.response?.status ?? 500;
        return NextResponse.json({ error: 'Failed to process request' }, { status });
    }
}
