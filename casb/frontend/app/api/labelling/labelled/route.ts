import { NextResponse } from 'next/server';
import axios from 'axios';

const BACKEND = process.env.BACKEND_URL ?? 'http://localhost:8000';

export async function GET() {
    try {
        // Request files from the 'labelled' subdirectory
        const { data } = await axios.get(`${BACKEND}/files`, {
            params: { subdir: 'labelled' }
        });
        return NextResponse.json(data);
    } catch (error: any) {
        console.error('Error getting labelled files:', error.message);
        const status = error.response?.status ?? 500;
        return NextResponse.json({ error: 'Failed to get labelled files' }, { status });
    }
}