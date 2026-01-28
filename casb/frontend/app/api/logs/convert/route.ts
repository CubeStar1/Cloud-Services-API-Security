import { NextResponse } from 'next/server';
import axios from 'axios';

const BACKEND = process.env.BACKEND_URL ?? 'http://localhost:8000';

export async function POST() {
  try {
    const { data } = await axios.post(`${BACKEND}/convert/json-to-csv`);
    return NextResponse.json(data);
  } catch (error: any) {
    console.error('Error in CSV conversion:', error.message);
    const status = error.response?.status ?? 500;
    const message = error.response?.data?.message ?? error.message ?? 'Failed to convert logs to CSV';
    return NextResponse.json({ success: false, message, error: message }, { status });
  }
}
