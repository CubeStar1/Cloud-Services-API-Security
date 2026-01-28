import { NextRequest, NextResponse } from 'next/server';
import axios from 'axios';

const BACKEND = process.env.BACKEND_URL ?? 'http://localhost:8000';
const GROQ_API_KEY = process.env.GROQ_API_KEY!

// GET: List CSV files (proxy to backend listing)
export async function GET() {
  try {
    // Request files with subdir=logs/csv and ext=csv
    const { data } = await axios.get(`${BACKEND}/files`, {
      params: { subdir: 'logs/csv', ext: 'csv' },
    });
    return NextResponse.json(data);
  } catch (error: any) {
    console.error('Error getting CSV files:', error.message);
    return NextResponse.json({ error: 'Failed to get CSV files' }, { status: 500 });
  }
}

// POST: Trigger labelling
export async function POST(req: NextRequest) {
  try {
    const body = await req.json().catch(() => ({})); // Handle empty body if any
    const payload = { 
        ...body, 
        api_key: GROQ_API_KEY,
        services: body.services,
        activities: body.activities
    };
    const { data } = await axios.post(`${BACKEND}/label`, payload);
    return NextResponse.json(data);
  } catch (error: any) {
    console.error('Error running labelling:', error.message);
    const status = error.response?.status ?? 500;
    const message = error.response?.data?.message ?? error.message ?? 'Failed to run labelling';
    return NextResponse.json({ success: false, message, error: message }, { status });
  }
}
